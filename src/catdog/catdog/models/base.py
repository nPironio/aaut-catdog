from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F


class CatDogOutput(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.classify_layer = torch.nn.Linear(input_size, 1)
        self.bbox_layer = torch.nn.Linear(input_size, 4)

    def forward(self, x):
        return F.sigmoid(self.classify_layer(x)), self.bbox_layer(x)


class CatDogClassifier(pl.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def get_default_optimizer_params(self):
        return {"lr": 0.02}

    def forward(self, x):
        """
        Función para ser llamada al predecir (sin computar gradientes)
        :param x: Imagenes
        :return: clase, (xmin, ymin, xmax, ymax)
        """
        with torch.no_grad():
            pred_class, pred_bbox = self.model(x)
            return pred_class, pred_bbox

    def training_step(self, batch, batch_idx):
        img, target, bbox = batch
        pred_target, pred_bbox = self.model(img, target)

        classification_loss = F.binary_cross_entropy(pred_target, target)
        bbox_loss = F.mse_loss(pred_bbox, bbox)

        return classification_loss + self.hparams.bbox_alpha * bbox_loss
