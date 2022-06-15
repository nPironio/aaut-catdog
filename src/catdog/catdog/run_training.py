import torch
import pandas as pd
import numpy as np

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from definitions import DATA_PATH
from catdog.dataset import CatDogDataset
from catdog.models.feedforward import MLPClassifier


usr = "char"
batch_size = 32

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

seed = 142
torch.manual_seed(seed)
np.random.seed(seed)


# TODO(cgiudice): hacer configurable con hydra
# Tengo GPU pero no cuda compatible, me rompe cuando intento correr con GPU
if usr == "nico":
    device = "cpu"
    num_workers = 4

if usr == "char":
    num_workers = 0


if __name__ == "__main__":

    # load data
    train_set_pd = pd.read_csv(DATA_PATH + 'train.csv')
    train_set_torch = CatDogDataset(train_set_pd)
    print(f"Loaded train set with length {len(train_set_torch)}")
    test_set_pd = pd.read_csv(DATA_PATH + 'test.csv')
    test_set_torch = CatDogDataset(test_set_pd)
    print(f"Loaded test set with length {len(test_set_torch)}")

    # define data loaders
    train_loader = torch.utils.data.DataLoader(train_set_torch, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_set_torch, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # define model
    model_kwargs = {'input_size': 500*500*3, 'hidden_sizes': (50, 10)}
    model = MLPClassifier(**model_kwargs)
    model.to(device)

    # define trainer
    seed_everything(142, workers=True)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='./lightning_checkpoints/',
        filename='test_model_training',
        mode='min',
        save_top_k=3
    )
    logger = TensorBoardLogger("tb_logs", name="MLP_classifier")
    trainer = Trainer(gpus=int(device == 'cuda'),  # hacky way to say 0 or 1
                      max_epochs=20,
                      logger=logger,
                      callbacks=[EarlyStopping(monitor="val_loss", patience=15, mode='min'),
                                 checkpoint_callback])

    # train and eval model
    trainer.fit(model, train_loader, test_loader)  # here we are using test set as validation set
    trainer.test(ckpt_path="best", dataloaders=test_loader)
