from collections import OrderedDict
import torch.nn
from .base import CatDogClassifier, CatDogOutput
from catdog.utils.image import appropiate_padding


class ConvolutionalClassifier(CatDogClassifier):
    def __init__(self, input_shape, dropout_p=0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        in_channels, current_height, current_width = input_shape
        conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(3, 3),
                                # padding=appropiate_padding((current_height, current_width), (3,3))
                                )
        if not dropout_p:
            regularization_1 = ('batch_norm_1', torch.nn.BatchNorm2d(32))
        else:
            regularization_1 = ('dropout_1', torch.nn.Dropout2d(dropout_p))
        
        conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3),
                                # padding=appropiate_padding((current_height, current_width), (3,3))
                                )
        if not dropout_p:
            regularization_2 = ('batch_norm_2', torch.nn.BatchNorm2d(64))
        else:
            regularization_2 = ('dropout_2', torch.nn.Dropout2d(dropout_p))

        conv3 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3),
                                # padding=appropiate_padding((current_height, current_width), (3,3))
                                )
        if not dropout_p:
            regularization_3 = ('batch_norm_3', torch.nn.BatchNorm2d(128))
        else:
            regularization_3 = ('dropout_3', torch.nn.Dropout2d(dropout_p))

        layers = OrderedDict([
            ('conv1', conv1), ('relu1', torch.nn.ReLU()), 
                ('pool1',torch.nn.MaxPool2d((2, 2))), regularization_1,
            ('conv2', conv2), ('relu2', torch.nn.ReLU()), 
                ('pool2',torch.nn.MaxPool2d((2, 2))), regularization_2,
            ('conv3', conv3), ('relu3', torch.nn.ReLU()), 
                ('pool3',torch.nn.MaxPool2d((2, 2))), regularization_3,
            ('flatten', torch.nn.Flatten(start_dim=1)),
            ('output', CatDogOutput(460800))
        ])
        self.model = torch.nn.Sequential(layers)

    def configure_optimizers(self):
        params = self.hparams["optimizer_params"]
        params = self.get_default_optimizer_params() if not params else params
        return torch.optim.Adam(self.parameters(), **params)

    def forward_pass(self, img):
        output = self.model(img)
        # import ipdb; ipdb.set_trace()
        return output