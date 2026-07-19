from directory_tree import DisplayTree
from matplotlib import pyplot as plt
from fastai.vision.core import show_image, show_titled_image

def print_data_folder_structure(root_dir, max_depth=1):
    """Print the folder structure of the dataset directory."""
    config_tree = {
        "dirPath": root_dir,
        "onlyDirs": False,
        "maxDepth": max_depth,
        "sortBy": 1,  # Sort by type (files first, then folders)
    }
    DisplayTree(**config_tree)

def plot_img(img, label=None, info=None, ax=None):

    def add_info_text(ax, info):
        ax.text(
            0.5, -0.1, info, transform=ax.transAxes, ha="center", va="top", fontsize=10
        )
        ax.xaxis.set_label_position("top")

    # using show_image from fastai to handle different image types
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    if label:
        title = f"Label: {label}"
        show_titled_image((img, title), ax=ax)
    else:
        show_image(img, ax=ax)

    if info:
        # Add info as text below the image
        add_info_text(ax, info)

    if ax is None:
        plt.show()