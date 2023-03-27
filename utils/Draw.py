import torch
import copy
import itertools
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.optim import Adam
from mpl_toolkits import mplot3d
from prettytable import PrettyTable
from scipy.interpolate import make_interp_spline

import numpy
# import loss_landscapes
# import loss_landscapes.metrics
import pandas as pd
def plot_confusion_matrix(cm, classes, cmap,  index,normalize=False, title='Confusion matrix'):
	cm_nor = cm.astype('float') / cm.sum(axis=0)
	# cm_nor = cm.astype('float')
	sum_TP = 0
	for i in range(len(classes)):
		sum_TP += cm[i, i]
	acc = sum_TP / np.sum(cm)
	acc_str = "{:.2f}".format(acc * 100) + '%'
	print("the model accuracy is ", acc)
	table = PrettyTable()
	table.field_names = ["", "ACC", "SEN", "SPE", "PPR", "F1"]
	TP_SUM, FP_SUM, FN_SUM, TN_SUM = 0, 0, 0, 0
	ACC_Tol, SEN_Tol, SPE_Tol, PPR_Tol, F1_Tol = 0, 0, 0, 0, 0
	for i in range(len(classes)):
		TP = cm[i, i]
		FP = np.sum(cm[i, :]) - TP
		FN = np.sum(cm[:, i]) - TP
		TN = np.sum(cm) - TP - FP - FN
		TP_SUM += TP
		FP_SUM += FP
		FN_SUM += FN
		TN_SUM += TN

		ACC = round((TP + TN) / (TP + FP + TN + FN), 3) if TP + FP + TN + FN != 0 else 0.
		SEN = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
		SPE = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
		PPR = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
		F1 = round(2 * TP / (2 * TP + FP + FN), 3) if TP + FP != 0 else 0.
		ACC_Tol += ACC
		SEN_Tol += SEN
		SPE_Tol += SPE
		PPR_Tol += PPR
		F1_Tol += F1
		table.add_row([classes[i], ACC, SEN, SPE, PPR, F1])
	ACC_SUM = round((TP_SUM + TN_SUM) / (TP_SUM + FP_SUM + TN_SUM + FN_SUM),
					3) if TP_SUM + FP_SUM + TN_SUM + FN_SUM != 0 else 0.
	SEN_SUM = round(TP_SUM / (TP_SUM + FN_SUM), 3) if TP_SUM + FN_SUM != 0 else 0.
	SPE_SUM = round(TN_SUM / (TN_SUM + FP_SUM), 3) if TN_SUM + FP_SUM != 0 else 0.
	PPR_SUM = round(TP_SUM / (TP_SUM + FP_SUM), 3) if TP_SUM + FP_SUM != 0 else 0.
	F1_SUM = round(2 * TP_SUM / (2 * TP_SUM + FP_SUM + FN_SUM), 3) if TP_SUM + FP_SUM != 0 else 0.
	ACC_MEAN = round(ACC_Tol/len(classes), 3)
	SEN_MEAN = round(SEN_Tol/len(classes), 3)
	SPE_MEAN = round(SPE_Tol/len(classes), 3)
	PPR_MEAN = round(PPR_Tol/len(classes), 3)
	F1_MEAN = round(F1_Tol/len(classes), 3)
	table.add_row(['SUM', ACC_SUM, SEN_SUM, SPE_SUM, PPR_SUM, F1_SUM])
	table.add_row(['MEAN', ACC_MEAN, SEN_MEAN, SPE_MEAN, PPR_MEAN, F1_MEAN])
	print(table)
	plt.imshow(cm_nor, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar(fraction=0.046, pad=0.05)
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)
	thresh = cm_nor.max() / 2.
	length = len(classes)
	if length == 5:
		fontsize = 13
	elif length == 17:
		fontsize = 6
	for i, j in itertools.product(range(cm_nor.shape[0]), range(cm_nor.shape[1])):
		plt.text(j, i, format(cm_nor[i, j], '.2f'), horizontalalignment="center",
				 color="white" if cm_nor[i, j] > thresh else "black", fontsize=fontsize)
	plt.tight_layout()
	plt.ylabel('Predicted Labels', fontsize=fontsize)
	plt.xlabel('True Labels', fontsize=fontsize)
	plt.title('Normalized confusion matrix, with Acc=%.2f' % (100*acc), fontsize=13)

def confusion_matrix(preds, labels, conf_matrix):
	for p, t in zip(preds, labels):
		conf_matrix[p, t] += 1

	return conf_matrix


def plot_cfm(model, test_loader, classes, mode, cnt, clt, seed, best_test_acc, index):
	length = len(classes)
	conf_matrix = torch.zeros(length, length)
	acc_val = 0
	model.eval()
	total = 0
	correct = 0
	for data in test_loader:
		images, labels = data
		total += len(labels)
		out = model(images)
		prediction = torch.max(out, 1)[1]
		conf_matrix = confusion_matrix(
			prediction, labels=labels, conf_matrix=conf_matrix)
		correct += (prediction == labels).sum().item()
	acc_val = 100 * correct / total
	print(acc_val)
	attack_types = classes
	plt.tight_layout()
	plt.figure(figsize=(6, 6))
	plt.rcParams['font.sans-serif'] = ['Times New Roman']
	plot_confusion_matrix(
		conf_matrix.numpy(), classes=attack_types, cmap=plt.cm.Blues, normalize=True, index = index)

	plt.savefig(fname=f"./image_for_{len(classes)}/%s,Confusion-matrix,%d,%d,%d,%.4f.pdf" % (mode, cnt, clt, seed, best_test_acc), format="pdf", bbox_inches='tight')


def plot_loss_acc(acc_list, loss_list, mode, cnt, clt):
	fig = plt.figure()
	x = np.arange(1, len(acc_list) + 1)
	a1 = fig.add_axes([0, 0, 1, 1])
	a1.plot(x, acc_list, 'tab:blue', label='acc')
	a1.set_ylabel('acc')
	a2 = a1.twinx()
	a2.plot(x, loss_list, 'tab:orange', label='loss')
	a2.set_ylabel('loss')
	plt.title('acc & loss')
	a1.set_xlabel('Epoch')
	a1.legend()
	a2.legend()
	plt.savefig(fname="../image/%s,Acc-Loss,%d,%d.pdf" % (mode, cnt, clt), format="pdf", bbox_inches='tight')
	plt.show()

