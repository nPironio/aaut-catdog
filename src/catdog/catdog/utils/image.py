import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches


def plot_image_bbox(img_path, category, xmin, ymin, xmax, ymax):
    img = mpimg.imread(img_path)
    plt.imshow(img)
    plt.gca().add_patch(
        patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1.5, edgecolor='r', facecolor='none'))
    plt.gca().annotate(category, (xmax + 5, ymax + 5), color="r", size=15)
    plt.show()
