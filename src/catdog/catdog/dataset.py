from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class CatDogDataset(Dataset):
    def __init__(self, cat_dog_df, transforms=None, img_path="../data/images/"):
        self.files = (img_path + cat_dog_df["file"]).values
        self.width = cat_dog_df["width"].values
        self.height = cat_dog_df["height"].values
        self.target = np.where(cat_dog_df["class"].values == "cat", 1, 0)
        self.bbox = cat_dog_df[["xmin", "ymin", "xmax", "ymax"]].values

        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = np.asarray(Image.open(self.files[idx]).convert("RGB"))
        bbox = self.bbox[idx]
        if self.transforms is not None:
            img, bbox = self.transforms(img, bbox)

        return img, self.target[idx], bbox
