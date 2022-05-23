import numpy as np

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as fn


class CatDogDataset(Dataset):
    def __init__(self, cat_dog_df, transforms=None, img_output_size=(500,500), img_path="../data/images/"):
        self.files = (img_path + cat_dog_df["file"]).values
        self.width = cat_dog_df["width"].values
        self.height = cat_dog_df["height"].values
        self.target = np.where(cat_dog_df["class"].values == "cat", 1, 0)
        self.bbox = cat_dog_df[["xmin", "ymin", "xmax", "ymax"]].values.astype(np.float32)
        self.resizer = fn.Resize(img_output_size)

        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        resized_img = self.resizer(Image.open(self.files[idx]).convert("RGB"))
        np_img = np.asarray(resized_img)
        bbox = self.bbox[idx]
        if self.transforms is not None:
            np_img, bbox = self.transforms(np_img, bbox)

        torch_img = fn.functional.to_tensor(np_img)
        return torch_img, self.target[idx], bbox
