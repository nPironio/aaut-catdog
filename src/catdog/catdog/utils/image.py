import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches


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
    ax.imshow(img)
    ax.add_patch(
        patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1.5, edgecolor='r', facecolor='none'))
    ax.annotate(category, (xmax + 5, ymax + 5), color="r", size=15)