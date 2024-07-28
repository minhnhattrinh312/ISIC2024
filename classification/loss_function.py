import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, device, class_weight, num_classes, gamma=2):
        """
        class weight should be a list.
        """
        super().__init__()
        self.device = device
        self.class_weight = torch.tensor(class_weight, device=device)
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, y_true, y_pred):
        yTrueOnehot = torch.zeros(y_true.size(0), self.num_classes, device=self.device)
        yTrueOnehot = torch.scatter(yTrueOnehot, 1, y_true, 1)
        y_pred = torch.clamp(y_pred, min=1e-2, max=1 - 1e-2)

        focal = -yTrueOnehot * (1 - 1e-2 - y_pred) ** self.gamma * torch.log(y_pred + 1e-2) * self.class_weight
        active = yTrueOnehot * (1 - 1e-2 - y_pred) * self.class_weight
        bce = -yTrueOnehot * torch.log(y_pred + 1e-2) * self.class_weight
        loss = torch.sum(focal) + torch.sum(active) + torch.sum(bce)
        return loss / (torch.sum(self.class_weight) * y_true.size(0))


# BCE loss
class BCELoss(nn.Module):

    def __init__(self, device, class_weight, num_classes):
        """
        class weight should be a list.
        """
        super().__init__()
        self.device = device
        self.class_weight = torch.tensor(class_weight, device=device)
        self.num_classes = num_classes

    def forward(self, y_true, y_pred):
        yTrueOnehot = torch.zeros(y_true.size(0), self.num_classes, device=self.device)
        yTrueOnehot = torch.scatter(yTrueOnehot, 1, y_true, 1)
        y_pred = torch.clamp(y_pred, min=1e-2, max=1 - 1e-2)

        bce = -yTrueOnehot * torch.log(y_pred + 1e-2) * self.class_weight
        loss = torch.sum(bce)
        return loss / (torch.sum(self.class_weight) * y_true.size(0))
