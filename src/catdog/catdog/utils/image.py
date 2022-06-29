import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torchvision.transforms as T


def appropiate_padding(input_shape, kernel_shape):
    def n_pad(x, k):
        return k - (x % k)

    return [n_pad(n, k) for n, k in zip(input_shape, kernel_shape)]


def plot_image_bbox(img, category, xmin, ymin, xmax, ymax, ax=None):
    ax = plt.gca() if not ax else ax
    xmin *= img.shape[0]
    xmax *= img.shape[0]
    ymin *= img.shape[1]
    ymax *= img.shape[1]
    ax.imshow(img.type(torch.uint8))
    ax.add_patch(
        patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1.5, edgecolor='r', facecolor='none'))
    ax.annotate(category, (xmax + 5, ymax + 5), color="r", size=15)


class BBoxIdentityWrapper:
    """
    Provides a wrapper for image transforms that don't modify the bounding boxes.
    Use case example:
        >> wrappedColorJitter = BBoxIdentityWrapper(torchvision.transforms.ColorJitter(brightness=0, contrast=0,
                                                                                    saturation=0, hue=0))
        >> transformed_img, bbox = wrappedColorJitter(img, bbox)
    """
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img, bbox):
        return self.transform(img), bbox

class RandomHorizontalFlipBBox:
    def __init__(self, p=0.5):
        self.p = p
        self.flipper = T.RandomHorizontalFlip(1)

    def __call__(self, img, bbox: np.ndarray):
        flip = np.random.uniform(size=len(img)) < self.p
        img[flip] = self.flipper(img[flip])
        bbox[flip] = [1, 0, 1, 0] - bbox[flip][:, [2, 1, 0, 3]] * [1, -1, 1, -1]
        return img, bbox


class RandomVerticalFlipBBox:
    def __init__(self, p=0.5):
        self.p = p
        self.flipper = T.RandomVerticalFlip(1)

    def __call__(self, img, bbox: np.ndarray):
        flip = np.random.uniform(size=len(img)) < self.p
        img[flip] = self.flipper(img[flip])
        bbox[flip] = [0, 1, 0, 1] - bbox[flip][:, [0, 3, 2, 1]] * [-1, 1, -1, 1]
        return img, bbox
