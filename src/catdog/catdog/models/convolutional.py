import torch.nn
from .base import CatDogClassifier, CatDogOutput
from catdog.utils.image import appropiate_padding

class ConvolutionalClassifier(CatDogClassifier):
    def __init__(self, input_shape, optimizer_params=None, bbox_alpha=1):

        super().__init__()
        in_channels, current_height, current_width = input_shape
        conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=15, kernel_size=(3, 3),
                                padding=appropiate_padding((current_height, current_width), (3,3)))

        conv2 = torch.nn.Conv2d(in_channels=15, out_channels=15, kernel_size=(3, 3),
                                padding=appropiate_padding((current_height, current_width), (3,3)))
        current_height //= 2
        current_width //= 2
        pool = torch.nn.AdaptiveMaxPool2d((current_height, current_width))

        self.model = torch.nn.Sequential(conv1, torch.nn.ReLU(), conv2, torch.nn.ReLU(), pool,
                                         torch.nn.Flatten(start_dim=1), CatDogOutput(15*current_height*current_width))

        self.save_hyperparameters("optimizer_params", "bbox_alpha")



    def configure_optimizers(self):
        params = self.hparams["optimizer_params"]
        params = self.get_default_optimizer_params() if not params else params
        return torch.optim.Adam(self.parameters(), **params)
