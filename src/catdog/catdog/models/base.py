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
        classifier_pred, bbox_pred = torch.sigmoid(classifier_logits), torch.sigmoid(bbox_logits)
        return classifier_pred, bbox_pred


class CatDogClassifier(pl.LightningModule):
    def __init__(self, optimizer_params=None, bbox_alpha=1, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.AUROC = torchmetrics.AUROC()  # num_classes=2 was not meant for binary problems, but multiclass problems
        self.Precision = torchmetrics.Precision()
        self.Recall = torchmetrics.Recall()
        self.batch_size = 32  # TODO(cgiudice): either move this to a config file or receive it as parameter

        optimizer_params = self.get_default_optimizer_params() if not optimizer_params else optimizer_params
        self.save_hyperparameters("optimizer_params", "bbox_alpha")

    def forward_pass(self, img):
        raise NotImplementedError

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
            pred_class, pred_bbox = self.forward_pass(x)
            return pred_class, pred_bbox

    def preprocess_img(self, img):
        return img

    def _shared_step(self, batch):
        img, target, bbox = batch
        model_input = self.preprocess_img(img)
        pred_target, pred_bbox = self.forward_pass(model_input)
        # adds one dimension to target, just the way torch likes it
        target = target.unsqueeze(1)

        classification_loss = F.binary_cross_entropy(pred_target, target)
        bbox_loss = F.mse_loss(pred_bbox, bbox)
        loss = classification_loss + self.hparams.bbox_alpha * bbox_loss

        return target.int(), pred_target, bbox, pred_bbox, classification_loss, bbox_loss, loss

    def training_step(self, batch, batch_idx):
        target, pred_target, bbox, pred_bbox, classification_loss, bbox_loss, loss = self._shared_step(batch)

        self.log_metrics(bbox_loss, classification_loss, loss, pred_target, target, stage="train")

        return loss

    def validation_step(self, batch, batch_idx):
        target, pred_target, bbox, pred_bbox, classification_loss, bbox_loss, loss = self._shared_step(batch)
        imgs = batch[0]
        sel_idx = np.random.randint(len(imgs))
        return {"img": imgs[sel_idx], "sel_idx": sel_idx, "target": target,
                "pred_target": pred_target, "pred_bbox": pred_bbox,
                "classification_loss": classification_loss, "bbox_loss": bbox_loss, "loss": loss}

    def test_step(self, batch, batch_idx):
        target, pred_target, bbox, pred_bbox, classification_loss, bbox_loss, loss = self._shared_step(batch)
        imgs = batch[0]
        sel_idx = np.random.randint(len(imgs))
        return {"img": imgs[sel_idx], "sel_idx": sel_idx, "target": target,
                "pred_target": pred_target, "pred_bbox": pred_bbox,
                "classification_loss": classification_loss, "bbox_loss": bbox_loss, "loss": loss}

    def threshold(self, y):
        return (y > 0.5).int()

    def validation_epoch_end(self, outputs):
        self._shared_epoch_end(outputs, 'val')

    def test_epoch_end(self, outputs):
        self._shared_epoch_end(outputs, 'test')

    def _shared_epoch_end(self, outputs, stage):
        classification_loss = np.mean([x["classification_loss"].item() for x in outputs])  # item gives us the scalar in a one element tensor
        bbox_loss = np.mean([x["bbox_loss"].item() for x in outputs])
        loss = np.mean([x["loss"].item() for x in outputs])
        imgs = [x["img"] for x in outputs]
        sel_idx = np.array([x["sel_idx"] for x in outputs])
        target = torch.cat([x["target"] for x in outputs]).flatten()
        pred_bboxs = [x["pred_bbox"] for x in outputs]
        pred_target = [x["pred_target"] for x in outputs]

        # Plot images with pred and bbox
        idxs = np.random.choice(len(imgs), size=min(3, len(imgs)), replace=False)
        figs = []
        for i, idx in enumerate(idxs):
            fig = plt.figure(figsize=(8, 8))
            plot_image_bbox(imgs[idx].permute((1, 2, 0)).detach().cpu(),  # make sure tensor is not in GPU
                            "cat" if self.threshold(pred_target[idx][sel_idx[idx]])[0] else "dog",
                            *pred_bboxs[idx][sel_idx[idx]], ax=plt.gca())
            plt.tight_layout()
            figs.append(fig)
        self.logger.experiment.add_figure(tag=f"predictions sample", figure=figs, global_step=self.global_step)

        self.log_metrics(bbox_loss, classification_loss, loss, torch.cat(pred_target).flatten(), target, stage=stage)
        self.log(f"{stage}_loss", value=loss)

    def log_metrics(self, bbox_loss, classification_loss, loss, pred_target, target, stage=""):
            thresholded_pred = self.threshold(pred_target)
            # Losses
            self.logger.experiment.add_scalars("loss/total_loss", {stage: loss}, global_step=self.global_step)
            self.logger.experiment.add_scalars("loss/classification_loss", {stage: classification_loss},
                                               global_step=self.global_step)
            self.logger.experiment.add_scalars("loss/bbox_loss", {stage: bbox_loss}, global_step=self.global_step)

            # Metrics
            self.logger.experiment.add_scalars("metrics/precision", {stage: self.Precision(thresholded_pred, target)},
                                               global_step=self.global_step)
            self.logger.experiment.add_scalars("metrics/recall", {stage: self.Recall(thresholded_pred, target)},
                                               global_step=self.global_step)
            self.logger.experiment.add_scalars("metrics/AUROC", {stage: self.AUROC(pred_target, target)},
                                               global_step=self.global_step)
