import torch
import torchmetrics
import torch.nn.functional as F
import pytorch_lightning as pl

from typing import Any


class CatDogOutput(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.classify_layer = torch.nn.Linear(input_size, 1)
        self.bbox_layer = torch.nn.Linear(input_size, 4)

    def forward(self, x):
        UUUU = self.classify_layer(x)
        classifier_pred = F.sigmoid(UUUU)
        bbox_pred = self.bbox_layer(x)
        return classifier_pred, bbox_pred


class CatDogClassifier(pl.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.AUROC = torchmetrics.AUROC(num_classes=2)
        self.batch_size = 32  # TODO(cgiudice): either move this to a config file or receive it as parameter

    def get_default_optimizer_params(self):
        return {"lr": 0.02}

    def forward(self, x):
        """
        Función para ser llamada al predecir (sin computar gradientes)
        :param x: Imagenes
        :return: clase, (xmin, ymin, xmax, ymax)
        """
        with torch.no_grad():  # TODO(cgiudice): think whether or not this is necessary and where do we want it
            pred_class, pred_bbox = self.model(x)
            return pred_class, pred_bbox

    def training_step(self, batch, batch_idx):
        img, target, bbox = batch
        pred_target, pred_bbox = self.model(img)

        # adds one dimension to target, just the way torch likes it
        target = target.unsqueeze(1).type('torch.FloatTensor')

        classification_loss = F.binary_cross_entropy(pred_target, target)
        bbox_loss = F.mse_loss(pred_bbox, bbox)

        self.log('train_auroc', self.AUROC(pred_target, target.type(torch.int32)), on_step=True, on_epoch=False)
        self.log("train_classification_loss", classification_loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log("train_bbox_loss_loss", bbox_loss, prog_bar=True, on_step=True, on_epoch=False)

        return classification_loss + self.hparams.bbox_alpha * bbox_loss

    def validation_step(self, val_batch, batch_idx):
        img, target, bbox = val_batch
        pred_target, pred_bbox = self.model(img)

        # adds one dimension to target, just the way torch likes it
        target = target.unsqueeze(1).type('torch.FloatTensor')

        classification_loss = F.binary_cross_entropy(pred_target, target)
        bbox_loss = F.mse_loss(pred_bbox, bbox)

        self.log('train_auroc', self.AUROC(pred_target, target.type(torch.int32)), on_step=True, on_epoch=False)
        self.log("train_classification_loss", classification_loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log("train_bbox_loss_loss", bbox_loss, prog_bar=True, on_step=True, on_epoch=False)
