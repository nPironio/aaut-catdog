import matplotlib.pyplot as plt
import numpy as np
import torch
import torchmetrics
import torch.nn.functional as F
import pytorch_lightning as pl

from typing import Any

from catdog.utils.image import plot_image_bbox


class CatDogOutput(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.classify_layer = torch.nn.Linear(input_size, 1)
        self.bbox_layer = torch.nn.Linear(input_size, 4)

    def forward(self, x):
        classifier_logits, bbox_logits = self.classify_layer(x), self.bbox_layer(x)
        classifier_pred, bbox_pred = F.sigmoid(classifier_logits), F.sigmoid(bbox_logits)
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
        with torch.no_grad():
            x = self.preprocess_img(x)
            pred_class, pred_bbox = self.model(x)
            return pred_class, pred_bbox

    def preprocess_img(self, img):
        return img

    def shared_step(self, batch):
        img, target, bbox = batch
        model_input = self.preprocess_img(img)
        pred_target, pred_bbox = self.model(model_input)
        return target, pred_target, bbox, pred_bbox

    def training_step(self, batch, batch_idx):
        target, pred_target, bbox, pred_bbox = self.shared_step(batch)

        # adds one dimension to target, just the way torch likes it
        target = target.unsqueeze(1).type('torch.FloatTensor')

        classification_loss = F.binary_cross_entropy(pred_target, target)
        bbox_loss = F.mse_loss(pred_bbox, bbox)

        self.log('train_auroc', self.AUROC(pred_target, target.type(torch.int32)), on_step=True, on_epoch=False)
        self.log("train_classification_loss", classification_loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log("train_bbox_loss_loss", bbox_loss, prog_bar=True, on_step=True, on_epoch=False)

        return classification_loss + self.hparams.bbox_alpha * bbox_loss

    def validation_step(self, val_batch, batch_idx):
        target, pred_target, bbox, pred_bbox = self.shared_step(val_batch)

        # adds one dimension to target, just the way torch likes it
        target = target.unsqueeze(1).type('torch.FloatTensor')

        classification_loss = F.binary_cross_entropy(pred_target, target)
        bbox_loss = F.mse_loss(pred_bbox, bbox)
        loss = classification_loss + self.hparams.bbox_alpha * bbox_loss

        self.log('validation_auroc', self.AUROC(pred_target, target.type(torch.int32)), on_step=True, on_epoch=False)
        self.log("validation_classification_loss", classification_loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log("validation_bbox_loss_loss", bbox_loss, prog_bar=True, on_step=True, on_epoch=False)

        imgs = val_batch[0]
        sel_idx = np.random.randint(len(imgs))
        return {"img": imgs[sel_idx], "pred_target": pred_target[sel_idx], "pred_bbox": pred_bbox[sel_idx], "loss": loss}


    def threshold(self, y):
        return y > 0.5

    def validation_epoch_end(self, outputs):
        val_loss = np.mean([x["loss"] for x in outputs])
        imgs = [x["img"] for x in outputs]
        pred_bboxs = [x["pred_bbox"] for x in outputs]
        pred_targets = [x["pred_target"] for x in outputs]


        # Plot images with pred and bbox
        idxs = np.random.choice(len(imgs), size=min(3, len(imgs)), replace=False)
        figs = []
        for i, idx in enumerate(idxs):
            fig = plt.figure(figsize=(8, 8))
            plot_image_bbox(imgs[idx].permute((1, 2, 0)), "cat" if self.threshold(pred_targets[idx])[0] else "dog",
                            *pred_bboxs[idx], ax=plt.gca())
            plt.tight_layout()
            figs.append(fig)
        self.logger.experiment.add_figure(tag=f"predictions sample", figure=figs, global_step=self.global_step)

        self.log("val_loss", value=val_loss)
