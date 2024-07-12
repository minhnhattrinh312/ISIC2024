import torch

def accuracy(y_true, y_pred):
    y_pred = torch.argmax(y_pred, axis=1, keepdim=True)
    acc = y_pred == y_true
    return acc.sum() / y_true.size(0)


def f1_score(y_true, y_pred, smooth=1e-4):
    y_pred = torch.argmax(y_pred, axis=1, keepdim=True)
    TP = torch.sum(y_true * y_pred)
    FN = torch.sum(y_true * (1 - y_pred))
    FP = torch.sum((1 - y_true) * y_pred)
    precision = (TP + smooth) / (TP + FP + smooth)
    recall = (TP + smooth) / (TP + FN + smooth)
    return 2 * (precision * recall) / (precision + recall)
