from distutils.command.config import config
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
import numpy as np
import pytorch_lightning as pl
import timm.optim
from classification.loss_fucntion import FocalLoss
import torch.optim as optim
from classification.metric import accuracy, f1_score
from classification.utils import *
from classification.dataset import get_transform


class Classifier(pl.LightningModule):

    def __init__(self, model, class_weight, num_classes, learning_rate, factor_lr, patience_lr):
        super().__init__()
        self.model = model
        self.class_weight = class_weight
        self.num_classes = num_classes
        self.learning_rate = learning_rate

        # torch 2.3 => compile to make faster
        self.model = torch.compile(self.model, mode="reduce-overhead")

        self.class_weight = class_weight
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.factor_lr = factor_lr
        self.patience_lr = patience_lr
        ################ augmentation ############################
        self.train_transform, self.val_transform = get_transform()
        self.test_metric = []
        self.validation_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def on_after_batch_transfer(self, batch, dataloader_idx):
        if self.trainer.training:
            with torch.no_grad():
                batch = self.transform(*batch)
        return batch

    def _step(self, batch):
        image, y_true = batch
        y_pred = self.model(image)
        loss_focal = FocalLoss(self.device, self.class_weight, self.num_classes)(y_true, y_pred)
        acc, f1 = accuracy(y_true, y_pred), f1_score(y_true, y_pred)
        return loss_focal, acc, f1

    def training_step(self, batch, batch_idx):
        loss, acc, f1 = self._step(batch)
        metrics = {"loss": loss, "train_acc": acc, "train_f1": f1}
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc, f1 = self._step(batch)
        metrics = {"test_loss": loss, "test_acc": acc, "test_f1": f1}
        self.log_dict(metrics, prog_bar=True)
        return metrics

    def test_step(self, batch, batch_idx):
        loss, acc, f1 = self._step(batch)
        metrics = {"test_loss": loss, "test_acc": acc, "test_f1": f1}
        self.log_dict(metrics, prog_bar=True)
        return metrics

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self.model(x)
        return y_hat.cpu().numpy()

    def configure_optimizers(self):
        optimizer = timm.optim.Nadam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=self.factor_lr, patience=self.patience_lr, verbose=True
        )
        lr_schedulers = {
            "scheduler": scheduler,
            "monitor": "test_f1",
            "strict": False,
        }

        return [optimizer], lr_schedulers

    def lr_scheduler_step(self, scheduler, metric):
        if self.current_epoch < 100:
            return
        else:
            super().lr_scheduler_step(scheduler, metric)
