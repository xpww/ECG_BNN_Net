import math
from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.common_types import _size_1_t


# -------------------------------------STE-----------------------------------
class BinarizeF(Function):

    @staticmethod
    def forward(ctx, input):
        # output = input.new(input.size())
        # output[input >= 0] = 1
        # output[input < 0] = -1
        ctx.save_for_backward(input)

        return input.sign()

    @staticmethod
    def backward(ctx, grad_output):
        # grad_input = grad_output.clone()
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input.masked_fill_(input.ge(1) | input.le(-1), 0)
        return grad_input


# ---------------------------------(2/coshx) ^2--------------------------
class bcEstimator(Function):  # approx with (2/coshx) ^2 like in binaryConnect?
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = (2 / torch.cosh(input)) ** 2
        return grad_output * grad_input


# ---------------------------------tanhx--------------------------
class irEstimator(Function):  # approx with (2/coshx) ^2 like in binaryConnect?
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = 1 - torch.pow(torch.tanh(input), 2)
        return grad_output * grad_input


# ---------------------------------fft--------------------------
class fft_Estimator(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # grad_input = 0
        # for i in range(0, 10):
        #     grad_input += (4/torch.pi) * torch.cos(input=(2*i+1)*input)
        pi = torch.pi
        # grad_input = (4 / pi) * (torch.cos(input=(2 * 0 + 1) * pi * input) + torch.cos(input=(2 * 1 + 1) * pi * input)
        #                                +torch.cos(input=(2 * 2 + 1) * input) + torch.cos(input=(2 * 3 + 1) * input)
        #                                + torch.cos(input=(2 * 4 + 1) * input) + torch.cos(input=(2 * 5 + 1) * input)
        #                                + torch.cos(input=(2 * 6 + 1) * input) + torch.cos(input=(2 * 7 + 1) * input)
        #                                )
        # grad_input = (4 / pi) * (torch.cos(input=(2 * 0 + 1) * input) + torch.cos(input=(2 * 1 + 1) * input)
        #                          + torch.cos(input=(2 * 2 + 1) * input)
        #                          )
        grad_input = (pi / 2) * torch.cos((pi / 2) * grad_output)
        grad_input.masked_fill_(input.ge(1) | input.le(-1), 0)
        return grad_output * grad_input


# ---------------------------------2x+-x^2--------------------------
class BinaryActivation(nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
        # out_e1 = (x^2 + 2*x)
        # out_e2 = (-x^2 + 2*x)
        # out_e_total = 0
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x * x + 2 * x) * (1 - mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x * x + 2 * x) * (1 - mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1 - mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3

        return out


class IR_Estimator(Function):  # implementing BiRealNet paper approximation
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        mask1 = (input >= -1) * 1
        mask2 = (input >= 0) * 1
        mask3 = (input >= 1) * 1

        grad_input = (mask1 - mask2) * (2 + 2 * input) + (mask2 - mask3) * (2 - 2 * input)
        # grad_input.masked_fill_(input.ge(1) | input.le(-1), 0)
        return grad_output * grad_input


binarize = BinarizeF.apply

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def bin_act(x):
    bin_act = torch.sign(x).detach()
    le_clip = x.lt(-1.0).type(torch.float32)
    ri_clip = x.ge(1.0).type(torch.float32)
    clip_l = torch.bitwise_and(x.ge(-1.0), x.lt(0.0))
    clip_r = torch.bitwise_and(x.ge(0.0), x.lt(1.0))
    cliped = clip_l * (2 + x) * x + clip_r * (2 - x) * x
    out = cliped + ri_clip - le_clip
    # out = torch.tanh(x)
    return bin_act + out - out.detach()


def bin_wei(x):
    bin_wei = torch.sign(x).detach()
    out = torch.tanh(x)
    return bin_wei + out - out.detach()


class BinActivation(nn.Module):
    def __init__(self):
        super(BinActivation, self).__init__()

    def forward(self, x):
        out = bin_wei(x)
        return out
class FocalLoss(nn.Module):

    def __init__(self, weight=None, reduction='mean', gamma=0, alpha=1, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = self.alpha * (1 - p) ** self.gamma * logp
        return loss.mean()

class focal_loss(nn.Module):
    def __init__(self, alpha, gamma):
        super(focal_loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.BCE = nn.BCEWithLogitsLoss(reduction='sum')

    def forward(self, logits, labels):
        BCloss = self.BCE(input=logits.float(), target=labels.float())

        if self.gamma == 0:
            modulator = 1.0
        else:
            modulator = torch.exp(-self.gamma * labels * logits - self.gamma * torch.log(1 + torch.exp(-1.0 * logits)))

        loss = modulator * BCloss

        weighted_loss = self.alpha * loss
        fl = torch.sum(weighted_loss)

        fl /= torch.sum(labels)

        return

import numpy as np
from scipy.stats import ortho_group
def get_ab(N):
    sqrt = int(np.sqrt(N))
    for i in range(sqrt, 0, -1):
        if N % i == 0:
            return i, N // i
class BinaryConv1d_RBNN(nn.Conv1d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_1_t, stride: _size_1_t = 1,
                 padding: Union[str, _size_1_t] = 0,
                 dilation: _size_1_t = 1,
                 groups: int = 1,
                 bias: bool = False,
                 padding_mode: str = 'zeros', ):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                         groups=groups, bias=bias, padding_mode=padding_mode)
        # RBNN
        self.k = torch.tensor([10.]).float()
        self.t = torch.tensor([0.1]).float()
        self.epoch = -1

        w = self.weight
        self.a, self.b = get_ab(np.prod(w.shape[1:]))
        if self.a == 1:
            R1 = torch.tensor([[1.0]]).float().cuda()
        else:
            R1 = torch.tensor(ortho_group.rvs(dim=self.a)).float().cuda()
        R2 = torch.tensor(ortho_group.rvs(dim=self.b)).float().cuda()
        self.register_buffer('R1', R1)

        self.register_buffer('R2', R2)
        self.Rweight = torch.ones_like(w)

        self.w3 = torch.ones_like(w).cuda()

        self.rotate = nn.Parameter(torch.ones(w.size(0), 1, 1).cuda() * np.pi / 2, requires_grad=True)
        self.Rotate = torch.zeros(1)
    def forward(self, input):
        # a = input
        # w = self.weight
        # -------IR-Net--------
        # ba = IR_Estimator().apply(a)
        # bw = IR_Estimator().apply(w)
        # -------STE----------
        # if self.training:
        if self.in_channels == 1:
            ba = input
        else:
            ba = BinarizeF().apply(input)

        # a0 = input
        if self.training:
            w = self.weight
            a, b = self.a, self.b
            X = w.view(w.shape[0], a, b).detach()
            if self.epoch > -1:
                for _ in range(3):
                    # * update B
                    V = self.R1.t() @ X.detach() @ self.R2
                    B = torch.sign(V)
                    # * update R1
                    D1 = sum([Bi @ (self.R2.t()) @ (Xi.t()) for (Bi, Xi) in zip(B, X.detach())])
                    U1, S1, V1h = torch.linalg.svd(D1)
                    V1 = V1h.mH
                    self.R1 = (V1 @ (U1.t()))
                    # * update R2
                    D2 = sum([(Xi.t()) @ self.R1 @ Bi for (Xi, Bi) in zip(X.detach(), B)])
                    U2, S2, V2h = torch.linalg.svd(D2)
                    V2 = V2h.mH
                    self.R2 = (U2 @ (V2.t()))
            self.Rweight = ((self.R1.t()) @ X @ (self.R2)).view_as(w)
            delta = self.Rweight.detach() - w
            self.w3 = w + torch.abs(torch.sin(self.rotate)) * delta
            bw = BinarizeF().apply(self.w3)
        else:
            bw = BinarizeF().apply(self.w3)
        # else:
        #     ba = torch.sign(a)
        #     bw = torch.sign(w)
        # -------2/coshx----------
        # ba = bcEstimator().apply(a)
        # bw = bcEstimator().apply(w)
        # -----------tanhx-----------
        # ba = irEstimator().apply(a)
        # bw = irEstimator().apply(w)
        # -----------fft-----------
        # ba = fft_Estimator().apply(a)
        # bw = fft_Estimator().apply(w)

        out = F.conv1d(input=ba, weight=bw, bias=None, stride=self.stride,
                       padding=self.padding, dilation=self.dilation, groups=self.groups)

        return out



class BinaryConv1d_bw(nn.Conv1d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_1_t, stride: _size_1_t = 1,
                 padding: Union[str, _size_1_t] = 0,
                 dilation: _size_1_t = 1,
                 groups: int = 1,
                 bias: bool = False,
                 padding_mode: str = 'zeros', ):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                         groups=groups, bias=bias, padding_mode=padding_mode)
        self.k = torch.tensor([10.]).float()
        self.t = torch.tensor([0.1]).float()
    def forward(self, input):
        a = input
        w = self.weight
        # -------IR-Net--------
        # ba = IR_Estimator().apply(a)
        # bw = IR_Estimator().apply(w)
        # -------STE----------
        # if self.training:
        # ba = BinarizeF().apply(a)
        # bw = BinarizeF().apply(w)
        # else:
        #     ba = torch.sign(a)
        #     bw = torch.sign(w)
        #-------IE-Net--------
        bw = OwnQuantize().apply(w, self.k.to(w.device), self.t.to(w.device))
        out = F.conv1d(input=a, weight=bw, bias=None, stride=self.stride,
                       padding=self.padding, dilation=self.dilation, groups=self.groups)

        return out
# ---------------------------------------IEE--------------------------------------
class OwnQuantize_a(Function):
    @staticmethod
    def forward(ctx, input, k, t):
        ctx.save_for_backward(input, k, t)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, k, t = ctx.saved_tensors
        k = torch.tensor(1.).to(input.device)
        t = max(t, torch.tensor(1.).to(input.device))
        # grad_input = k * (1.4*t - torch.abs(t**2 * input))
        grad_input = k * (3 * torch.sqrt(t ** 2 / 3) - torch.abs(t ** 2 * input * 3) / 2)
        grad_input = grad_input.clamp(min=0) * grad_output.clone()
        return grad_input, None, None


class OwnQuantize(Function):
    @staticmethod
    def forward(ctx, input, k, t):
        ctx.save_for_backward(input, k, t)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, k, t = ctx.saved_tensors
        # grad_input = k * (1.4*t - torch.abs(t**2 * input))
        grad_input = k * (3 * torch.sqrt(t ** 2 / 3) - torch.abs(t ** 2 * input * 3) / 2)
        grad_input = grad_input.clamp(min=0) * grad_output.clone()
        return grad_input, None, None

class BinaryConv1d_baw(nn.Conv1d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_1_t, stride: _size_1_t = 1,
                 padding: Union[str, _size_1_t] = 0,
                 dilation: _size_1_t = 1,
                 groups: int = 1,
                 bias: bool = False,
                 padding_mode: str = 'zeros', ):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                         groups=groups, bias=bias, padding_mode=padding_mode)
        self.k = torch.tensor([10.]).float()
        self.t = torch.tensor([0.1]).float()

    def forward(self, input):
        a = input
        w = self.weight
        # -------IR-Net--------
        # ba = IR_Estimator().apply(a)
        # bw = IR_Estimator().apply(w)
        # -------STE----------
        # ba = BinarizeF().apply(a)
        # bw = BinarizeF().apply(w)
        # -------IEE----------
        bw = OwnQuantize().apply(w, self.k.to(w.device), self.t.to(w.device))
        ba = OwnQuantize_a().apply(a, self.k.to(w.device), self.t.to(w.device))
        out = F.conv1d(input=ba, weight=bw, bias=None, stride=self.stride,
                       padding=self.padding, dilation=self.dilation, groups=self.groups)

        return out

class BinaryLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__(in_features, out_features, bias)
        # self.binarize_a = BinaryActivation()
        # self.binarize_w = BinaryActivation()

    def forward(self, input):
        a = input
        w = self.weight
        ba = BinarizeF().apply(a)
        bw = BinarizeF().apply(w)
        out = F.linear(input=ba, weight=bw, bias=None)
        return out



class Bn_bin_conv_pool_block_Float(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, padding_value, pool_size,
                 pool_stride, ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bn = nn.BatchNorm1d(out_channels)  # default eps = 1e-5, momentum = 0.1, affine = True
        # 无bias
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                           bias=False,)
        self.relu = nn.ReLU()
        self.prelu = nn.PReLU()
        # self.htanh = nn.Hardtanh()
        # self.tanh = nn.Tanh()
        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride)
        # self.binarize = BinaryActivation()
        # self.pad = nn.ConstantPad1d(padding=padding, value=padding_value)

    def forward(self, I):
        # I = self.pad(I)
        I = self.conv(I)
        I = self.pool(I)
        I = self.relu(I)
        I = self.bn(I)
        return I

class Bn_bin_conv_pool_block_bw(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, padding_value, pool_size,
                 pool_stride, ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bn = nn.BatchNorm1d(out_channels)  # default eps = 1e-5, momentum = 0.1, affine = True
        # 无bias
        self.conv = BinaryConv1d_bw(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0,
                                    bias=False)
        self.relu = nn.ReLU()
        self.prelu = nn.PReLU()
        self.htanh = nn.Hardtanh()
        # self.tanh = nn.Tanh()
        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride)
        # self.binarize = BinaryActivation()
        self.pad = nn.ConstantPad1d(padding=padding, value=padding_value)

    def forward(self, I):
        I = self.pad(I)
        I = self.conv(I)
        I = self.pool(I)
        I = self.prelu(I)
        I = self.bn(I)
        return I


class Bn_bin_conv_pool_block_baw(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, padding_value, pool_size,
                 pool_stride, ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bn = nn.BatchNorm1d(out_channels)  # default eps = 1e-5, momentum = 0.1, affine = True
        # 无bias
        self.conv = BinaryConv1d_baw(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0,
                                    bias=False)
        self.relu = nn.ReLU()
        self.prelu = nn.PReLU()
        self.htanh = nn.Hardtanh()
        # self.tanh = nn.Tanh()
        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride)
        self.pad = nn.ConstantPad1d(padding=padding, value=padding_value)

    def forward(self, I):
        I = self.pad(I)
        I = self.conv(I)
        I = self.pool(I)
        I = self.prelu(I)
        I = self.bn(I)
        return I

class BinActive(nn.Module):
    def __init__(self):
        super(BinActive, self).__init__()
        self.hardtanh = nn.Hardtanh(inplace=True)

    def forward(self, input):
        # output = self.hardtanh(input)
        # output = binarize(output)
        out_forward = torch.sign(input)
        out1 = self.hardtanh(input)
        output = out_forward.detach() - out1.detach() + out1
        return output




class WeightOperation:
    def __init__(self, model):
        self.model = model
        self.count_group_weights = 0
        self.weight = []
        self.saved_weight = []  # 保存weight的原始数据
        self.saved_alpha = []
        self.binarize = BinaryActivation()
        for m in model.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                self.count_group_weights += 1
                self.weight.append(m.weight)
                self.saved_weight.append(m.weight.data)

    def WeightSave(self):
        for index in range(self.count_group_weights):
            self.saved_weight[index].copy_(self.weight[index].data)  # 把weight的值给了saved_weight,而后进行binary等操作

    def alpha_extract(self):
        for m in self.model.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                self.saved_alpha.append(m.alpha)

    def WeightBinarize(self):
        for index in range(self.count_group_weights):
            self.weight[index].data = binarize(self.weight[index].data)  # 对应原model中的值也会跟着改变

    def WeightRestore(self):
        for index in range(self.count_group_weights):
            self.weight[index].data.copy_(self.saved_weight[index])

    def WeightGradient(self):
        for index in range(self.count_group_weights):
            self.weight[index].grad.data = self.weight[index].grad.data * self.saved_alpha[
                index].data.detach().transpose(0, 1)



class BinaryQuantize(Function):
    @staticmethod
    def forward(ctx, input):
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class BinaryQuantize_a(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        grad_input = (2 - torch.abs(2 * input))
        grad_input = grad_input.clamp(min=0) * grad_output.clone()
        return grad_input

