import torch.nn
from .base import CatDogClassifier, CatDogOutput


class MLPClassifier(CatDogClassifier):
    def __init__(self, input_size, activation=torch.nn.ReLU, hidden_sizes=(), dropout_p=0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dropout = torch.nn.Dropout(p=dropout_p)

        layers = []
        if hidden_sizes:
            sizes = [input_size, *hidden_sizes]
            for in_size, out_size in zip(sizes, sizes[1:]):
                layers.append(torch.nn.Linear(in_size, out_size))
                if not dropout_p:  # assuming batchnorm does not work well with dropout
                    layers.append(torch.nn.BatchNorm1d(out_size))
                layers.append(activation())
                if dropout_p:
                    layers.append(self.dropout)

            last_layer_input_size = hidden_sizes[-1]
        else:
            last_layer_input_size = input_size

        self.model = torch.nn.Sequential(*layers, CatDogOutput(last_layer_input_size))

    def preprocess_img(self, img):
        return img.flatten(start_dim=1, end_dim=-1) #won't work if it's just one instance

    # def forward(self, x):
    #     # flattened_x = x.reshape((self.batch_size,-1))  # TODO(cgiudice): move this out so self.model(x) also uses reshape
    #     return super().forward(flattened_x)

    def configure_optimizers(self):
        params = self.hparams["optimizer_params"]
        params = self.get_default_optimizer_params() if not params else params
        return torch.optim.Adam(self.parameters(), **params)

    def forward_pass(self, img):
        return self.model(img)