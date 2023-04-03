import torch
import torch.nn as nn

from utils.OP import Bn_bin_conv_pool_block_bw, Bn_bin_conv_pool_block_baw, Bn_bin_conv_pool_block_Float, BinaryLinear




class ECG_XNOR_Full_Bin(nn.Module):
    def __init__(self, block1, block2, block3, block4, block5, block6, block7, device):
        super(ECG_XNOR_Full_Bin, self).__init__()
        self.name = 'Full_Bin_ECG'
        self.device = device
        self.block1 = Bn_bin_conv_pool_block_baw(*block1)
        self.block2 = Bn_bin_conv_pool_block_baw(*block2)
        self.block3 = Bn_bin_conv_pool_block_baw(*block3)
        self.block4 = Bn_bin_conv_pool_block_baw(*block4)

        self.is_block5 = False
        self.is_block6 = False
        self.is_block7 = False
        if block5 is not None:
            self.block5 = Bn_bin_conv_pool_block_baw(*block5)
            self.is_block5 = True
        if block6 is not None:
            self.block6 = Bn_bin_conv_pool_block_baw(*block6)
            self.is_block6 = True
        if block7 is not None:
            self.block7 = Bn_bin_conv_pool_block_baw(*block7)
            self.is_block7 = True
        self.dropout0 = nn.Dropout(p=0.5)
    def forward(self, batch_data):
        batch_data = batch_data.clone().detach().requires_grad_(True).to(self.device)
        batch_data = self.block1(batch_data)
        batch_data = self.block2(batch_data)
        batch_data = self.block3(batch_data)
        batch_data = self.block4(batch_data)
        if self.is_block5:
            batch_data = self.block5(batch_data)
        if self.is_block6:
            batch_data = self.block6(batch_data)
        if self.is_block7:
            batch_data = self.block7(batch_data)
        batch_data = self.dropout0(batch_data)
        batch_data = batch_data.sum(dim=2)
        return batch_data

class ECG_XNOR_Ori(nn.Module):
    def __init__(self, block1, block2, block3, block4, block5, block6, block7, device):
        super(ECG_XNOR_Ori, self).__init__()
        self.name = 'Ori_Bin_ECG'
        self.device = device
        self.block1 = Bn_bin_conv_pool_block_bw(*block1)
        self.block2 = Bn_bin_conv_pool_block_baw(*block2)
        self.block3 = Bn_bin_conv_pool_block_baw(*block3)
        self.block4 = Bn_bin_conv_pool_block_baw(*block4)

        self.is_block5 = False
        self.is_block6 = False
        self.is_block7 = False
        if block5 is not None:
            self.block5 = Bn_bin_conv_pool_block_baw(*block5)
            self.is_block5 = True
        if block6 is not None:
            self.block6 = Bn_bin_conv_pool_block_baw(*block6)
            self.is_block6 = True
        if block7 is not None:
            self.block7 = Bn_bin_conv_pool_block_baw(*block7)
            self.is_block7 = True
        self.dropout0 = nn.Dropout(p=0.5)

    def forward(self, batch_data):
        batch_data = batch_data.clone().detach().requires_grad_(True).to(self.device)
        batch_data = self.block1(batch_data)
        batch_data = self.block2(batch_data)
        batch_data = self.block3(batch_data)
        batch_data = self.block4(batch_data)
        if self.is_block5:
            batch_data = self.block5(batch_data)
        if self.is_block6:
            batch_data = self.block6(batch_data)
        if self.is_block7:
            batch_data = self.block7(batch_data)
        batch_data = self.dropout0(batch_data)
        batch_data = batch_data.sum(dim=2)
        return batch_data



class ECG_Full_Float(nn.Module):
    def __init__(self, block1, block2, block3, block4, block5, block6, block7, device):
        super(ECG_Full_Float, self).__init__()
        self.name = 'Full_Bin_ECG'
        self.device = device
        self.block1 = Bn_bin_conv_pool_block_Float(*block1)
        self.block2 = Bn_bin_conv_pool_block_Float(*block2)
        self.block3 = Bn_bin_conv_pool_block_Float(*block3)
        self.block4 = Bn_bin_conv_pool_block_Float(*block4)

        self.is_block5 = False
        self.is_block6 = False
        self.is_block7 = False
        if block5 is not None:
            self.block5 = Bn_bin_conv_pool_block_Float(*block5)
            self.is_block5 = True
        if block6 is not None:
            self.block6 = Bn_bin_conv_pool_block_Float(*block6)
            self.is_block6 = True
        if block7 is not None:
            self.block7 = Bn_bin_conv_pool_block_Float(*block7)
            self.is_block7 = True
        self.dropout0 = nn.Dropout(p=0.5)
    def forward(self, batch_data):
        batch_data = batch_data.clone().detach().requires_grad_(True).to(self.device)
        batch_data = self.block1(batch_data)
        batch_data = self.block2(batch_data)
        batch_data = self.block3(batch_data)
        batch_data = self.block4(batch_data)
        if self.is_block5:
            batch_data = self.block5(batch_data)
        if self.is_block6:
            batch_data = self.block6(batch_data)
        if self.is_block7:
            batch_data = self.block7(batch_data)
        batch_data = self.dropout0(batch_data)
        batch_data = batch_data.mean(dim=2)
        return batch_data