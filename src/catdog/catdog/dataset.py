import numpy as np
import torchvision.transforms as fn

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from definitions import IMG_PATH



class CatDogDataset(Dataset):
    def __init__(self, cat_dog_df, transformations=None, feature_scaling=255, img_output_size=(500, 500)):
        self.files = (IMG_PATH + cat_dog_df["file"]).values
        self.width = cat_dog_df["width"].values
        self.height = cat_dog_df["height"].values
        self.target = np.where(cat_dog_df["class"].values == "cat", 1, 0).astype(np.float32)
        self.bbox = cat_dog_df[["xmin", "ymin", "xmax", "ymax"]].values.astype(np.float32)
        self.resizer = transforms.Resize(img_output_size)

        self.to_tensor = transforms.ToTensor()
        self.normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.scaling = feature_scaling


        self.transformations = transformations


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        resized_img = self.resizer(self.to_tensor(Image.open(self.files[idx]).convert("RGB")))
        img = self.normalizer(resized_img)

        bbox = self.bbox[idx]
        if self.transformations is not None:
            img, bbox = self.transformations(img, bbox)

        return img, self.target[idx], bbox
