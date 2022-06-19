import torch.nn
from .base import CatDogClassifier, CatDogOutput
from catdog.utils.image import appropiate_padding


class ConvolutionalClassifier(CatDogClassifier):
    def __init__(self, input_shape, *args, **kwargs):
        super().__init__(*args, **kwargs)

        in_channels, current_height, current_width = input_shape
        conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(3, 3),
                                # padding=appropiate_padding((current_height, current_width), (3,3))
                                )
        conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3),
                                # padding=appropiate_padding((current_height, current_width), (3,3))
                                )

        conv3 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3),
                                # padding=appropiate_padding((current_height, current_width), (3,3))
                                )

        self.model = torch.nn.Sequential(conv1, torch.nn.ReLU(), torch.nn.MaxPool2d((2, 2)),
                                         conv2, torch.nn.ReLU(), torch.nn.MaxPool2d((2, 2)),
                                         conv3, torch.nn.ReLU(), torch.nn.MaxPool2d((2, 2)),
                                         torch.nn.Flatten(start_dim=1),
                                         CatDogOutput(36992)
                                         )

    def configure_optimizers(self):
        params = self.hparams["optimizer_params"]
        params = self.get_default_optimizer_params() if not params else params
        return torch.optim.Adam(self.parameters(), **params)

    def forward_pass(self, img):
        return self.model(img)