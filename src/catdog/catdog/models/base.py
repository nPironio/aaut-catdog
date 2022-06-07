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
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.AUROC = torchmetrics.AUROC()  # num_classes=2 was not meant for binary problems, but multiclass problems
        self.Precision = torchmetrics.Precision()
        self.Recall = torchmetrics.Recall()
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

    def _shared_step(self, batch):
        img, target, bbox = batch
        model_input = self.preprocess_img(img)
        pred_target, pred_bbox = self.model(model_input)
        
        # adds one dimension to target, just the way torch likes it
        target = target.unsqueeze(1)
        
        classification_loss = F.binary_cross_entropy(pred_target, target)
        bbox_loss = F.mse_loss(pred_bbox, bbox)
        loss = classification_loss + self.hparams.bbox_alpha * bbox_loss
        
        # TODO(cgiudice): maybe use self.threshold?
        target = target.type(torch.int32)  # it is onwards assumed that only integer operations will be performed with the target
        
        return target, pred_target, bbox, pred_bbox, classification_loss, bbox_loss, loss

    def training_step(self, batch, batch_idx):
        target, pred_target, bbox, pred_bbox, classification_loss, bbox_loss, loss  = self._shared_step(batch)

        self.log('train_auroc', self.AUROC(pred_target, target), on_step=True, on_epoch=False)
        self.log("train_classification_loss", classification_loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log("train_bbox_loss_loss", bbox_loss, prog_bar=True, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        target, pred_target, bbox, pred_bbox, classification_loss, bbox_loss, loss  = self._shared_step(batch)

        self.log('val_precision', self.Precision(pred_target, target), on_step=False, on_epoch=True)    
        self.log('vali_recall', self.Recall(pred_target, target), on_step=False, on_epoch=True)    
        self.log('val_auroc', self.AUROC(pred_target, target), on_step=True, on_epoch=False)
        self.log("val_classification_loss", classification_loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log("val_bbox_loss_loss", bbox_loss, prog_bar=True, on_step=True, on_epoch=False)

        imgs = batch[0]
        sel_idx = np.random.randint(len(imgs))
        return {"img": imgs[sel_idx], "pred_target": pred_target[sel_idx], "pred_bbox": pred_bbox[sel_idx], "loss": loss}
    
    def test_step(self, batch, batch_idx):
        target, pred_target, bbox, pred_bbox, classification_loss, bbox_loss, loss  = self._shared_step(batch)

        self.log('test_precision', self.Precision(pred_target, target), on_step=False, on_epoch=True)    
        self.log('test_recall', self.Recall(pred_target, target), on_step=False, on_epoch=True)    
        self.log('test_auroc', self.AUROC(pred_target, target), on_step=False, on_epoch=True)    
        self.log("test_classification_loss", classification_loss, prog_bar=True, on_step=False, on_epoch=True)    
        self.log("test_bbox_loss", bbox_loss, prog_bar=True, on_step=False, on_epoch=True)    

        imgs = batch[0]
        sel_idx = np.random.randint(len(imgs))
        
        return {"img": imgs[sel_idx], "pred_target": pred_target[sel_idx], "pred_bbox": pred_bbox[sel_idx], "loss": loss}

    def threshold(self, y):
        return y > 0.5

    def validation_epoch_end(self, outputs):
        self._shared_epoch_end(outputs, 'val')

    def test_epoch_end(self, outputs):
        self._shared_epoch_end(outputs, 'test')

    def _shared_epoch_end(self, outputs, stage):
        loss = np.mean([x["loss"].item() for x in outputs])  # item gives us the scalar in a one element tensor
        imgs = [x["img"] for x in outputs]
        pred_bboxs = [x["pred_bbox"] for x in outputs]
        pred_targets = [x["pred_target"] for x in outputs]

        # Plot images with pred and bbox
        idxs = np.random.choice(len(imgs), size=min(3, len(imgs)), replace=False)
        figs = []
        for i, idx in enumerate(idxs):
            fig = plt.figure(figsize=(8, 8))
            plot_image_bbox(imgs[idx].permute((1, 2, 0)).detach().cpu(),  # make sure tensor is not in GPU
                            "cat" if self.threshold(pred_targets[idx])[0] else "dog",
                            *pred_bboxs[idx], ax=plt.gca())
            plt.tight_layout()
            figs.append(fig)
        self.logger.experiment.add_figure(tag=f"predictions sample", figure=figs, global_step=self.global_step)

        self.log(f"{stage}_loss", value=loss)