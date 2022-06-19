from typing import Any

from catdog.models.base import CatDogClassifier, CatDogOutput
from torch.optim import Adam
import torchvision

class TransferLearningClassifier(CatDogClassifier):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        self.resnet = torchvision.models.resnet18(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.resnet.fc = CatDogOutput(512)

    def configure_optimizers(self):
        params = self.hparams.get("optimizer_params", None)
        params = self.get_default_optimizer_params() if not params else params
        # Only optimize the parameters for the last layer
        return Adam(self.resnet.fc.parameters(), **params)

    def forward_pass(self, img):
        return self.resnet(img)
