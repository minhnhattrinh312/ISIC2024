import numpy as np
import torch
import lightning
import timm.optim
from classification.loss_function import *
from classification.metric import partial_auc, recall, accuracy
from classification.utils import *
from classification.dataset import get_transform
import kornia.augmentation as K
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Classifier(lightning.LightningModule):

    def __init__(self, model, class_weight, num_classes, learning_rate, factor_lr, patience_lr):
        super().__init__()
        self.model = model
        self.class_weight = class_weight
        self.num_classes = num_classes
        self.learning_rate = learning_rate

        # torch 2.3 => compile to make faster
        self.model = torch.compile(self.model)

        self.class_weight = class_weight
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.factor_lr = factor_lr
        self.patience_lr = patience_lr
        self.validation_step_outputs = []

        ################ augmentation ############################
        self.transform = get_transform()
        self.normalize = K.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x):
        return self.model(self.normalize(x))

    def on_after_batch_transfer(self, batch, dataloader_idx):
        image, y_true = batch
        if self.trainer.training:
            with torch.no_grad():
                image = self.normalize(self.transform(image))
        return image, y_true

    def training_step(self, batch, batch_idx):
        image, y_true = batch
        y_pred = self.model(image)
        loss_focal = FocalLoss(self.device, self.class_weight, self.num_classes)(y_true, y_pred)
        # loss_focal = BCELoss(self.device, self.class_weight, self.num_classes)(y_true, y_pred)
        recall_train = recall(y_true, y_pred)
        metrics = {"loss": loss_focal, "recall_train": recall_train}
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True)
        return loss_focal

    def validation_step(self, batch, batch_idx):
        image, y_true = batch
        y_pred = self.model(self.normalize(image))
        # loss_focal = FocalLoss(self.device, self.class_weight, self.num_classes)(y_true, y_pred)
        # metrics = {"batch_val_loss": loss_focal}
        # y_pred get the probability of the positive class
        # copy y_pred to detach it from the graph
        output_dict = {
            "batch_y_true": y_true,
            "batch_y_pred": y_pred,
        }
        self.validation_step_outputs.append(output_dict)
        # self.log_dict(metrics, prog_bar=True)
        # return metrics

    def on_validation_epoch_end(self):
        # get batches y_true and y_pred from self.validation_step_outputs
        # calculate the partial_auc by using the batches y_true and y_pred
        # log the partial_auc
        y_true_epoch = torch.cat([x["batch_y_true"] for x in self.validation_step_outputs], dim=0)
        y_pred_epoch = torch.cat([x["batch_y_pred"] for x in self.validation_step_outputs], dim=0)
        partial_auc_val = partial_auc(y_true_epoch, y_pred_epoch[:, 1:2])
        recall_val = recall(y_true_epoch, y_pred_epoch)
        metrics = {"val_partial_auc": partial_auc_val, "recall_val": recall_val}
        self.log_dict(metrics, prog_bar=True)

        return metrics

    def configure_optimizers(self):
        optimizer = timm.optim.Nadam(self.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=self.factor_lr, patience=self.patience_lr)
        lr_schedulers = {
            "scheduler": scheduler,
            "monitor": "val_partial_auc",
            "strict": False,
        }

        return [optimizer], lr_schedulers

    # def lr_scheduler_step(self, scheduler, metric):
    #     if self.current_epoch < 30:
    #         return
    #     else:
    #         super().lr_scheduler_step(scheduler, metric)
