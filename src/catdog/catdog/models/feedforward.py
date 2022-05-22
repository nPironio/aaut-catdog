import torch.nn
from .base import CatDogClassifier, CatDogOutput


class MLPClassifier(CatDogClassifier):
    def __init__(self, input_size, activation=torch.nn.ReLU, hidden_sizes=(),
                 optimizer_params=None, bbox_alpha=1):

        super().__init__()
        layers = []
        if hidden_sizes:
            sizes = [input_size, *hidden_sizes]
            for in_size, out_size in zip(sizes, sizes[1:]):
                layers.append(torch.nn.Linear(in_size, out_size))
                layers.append(torch.nn.BatchNorm1d(out_size))
                layers.append(activation())

            last_layer_input_size = hidden_sizes[-1]
        else:
            last_layer_input_size = input_size

        self.model = torch.nn.Sequential(*layers, CatDogOutput(last_layer_input_size))
        self.save_hyperparameters("optimizer_params", "bbox_alpha")

    def forward(self, x):
        flattened_x = x.reshape((self.batch_size,-1))  # TODO(cgiudice): move this out so self.model(x) also uses reshape
        return super().forward(flattened_x)

    def configure_optimizers(self):
        params = self.hparams["optimizer_params"]
        params = self.get_default_optimizer_params() if not params else params
        return torch.optim.Adam(self.parameters(), **params)
