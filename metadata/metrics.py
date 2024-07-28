import torch
from sklearn.metrics import r2_score
import torch.nn.functional as F
import numpy as np

#write loss for binary classification
def binary_cross_entropy(y_true, y_pred):
    y_true = y_true.view(-1, 1).float()
    y_pred = torch.sigmoid(y_pred)
    return F.binary_cross_entropy(y_pred, y_true)

