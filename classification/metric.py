import torch
from torchmetrics.classification import BinaryAUROC


def partial_auc(y_true, y_pred, min_tpr=0.80):
    v_gt = abs(y_true - 1)
    v_pred = 1.0 - y_pred
    max_fpr = abs(1 - min_tpr)
    partial_auc_scaled = BinaryAUROC(max_fpr=max_fpr)(v_pred, v_gt)
    partial_auc = 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)
    return partial_auc


def f1_score(y_true, y_pred, smooth=1e-4):
    y_pred = torch.argmax(y_pred, axis=1, keepdim=True)
    TP = torch.sum(y_true * y_pred)
    FN = torch.sum(y_true * (1 - y_pred))
    FP = torch.sum((1 - y_true) * y_pred)
    precision = (TP + smooth) / (TP + FP + smooth)
    recall = (TP + smooth) / (TP + FN + smooth)
    return 2 * (precision * recall) / (precision + recall)


def accuracy(y_true, y_pred):
    y_pred = torch.argmax(y_pred, axis=1, keepdim=True)
    acc = y_pred == y_true
    return acc.sum() / y_true.size(0)
