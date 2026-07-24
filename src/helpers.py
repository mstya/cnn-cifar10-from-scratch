import numpy as np
import matplotlib.ticker as mticker
from directory_tree import DisplayTree
from matplotlib import pyplot as plt
from fastai.vision.core import show_image, show_titled_image
from torchvision import transforms


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

def get_grid(num_rows, num_cols, figsize=(16, 8)):
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    if num_rows == 1:
        axes = [axes]  # Ensure axes is iterable
    elif num_cols == 1:
        axes = [[ax] for ax in axes]  # Ensure 2D list
    return fig, axes

def visual_exploration(dataset, num_rows=2, num_cols=4):
    """Visual exploration of the dataset by displaying random samples in a grid."""
    # Calculate total number of samples to display
    total_samples = num_rows * num_cols

    # Randomly select indices from the dataset without replacement
    # This ensures we get a diverse sample of the dataset
    indices = np.random.choice(len(dataset), total_samples, replace=False)

    # Create a grid of subplots with appropriate figure size
    # Each subplot gets (3 x 4) inches per image for good visibility
    fig, axes = get_grid(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 4))

    # Iterate through each axis and corresponding random index
    for ax, idx in zip(axes.flatten(), indices):
        # Load image and label from dataset at the random index
        image, label = dataset[idx]

        # Get human-readable description for the label
        description = dataset.get_label_description(idx)

        # Create a combined label string with both number and description
        label = f"{label} - {description}"

        # Create info string showing index and image dimensions
        info = f"Index: {idx} Size: {image.size}"

        # Plot the image on the current axis with label and info
        plot_img(image, label=label, info=info, ax=ax)

    # Display the complete grid of images
    plt.show()

class Denormalize:
    def __init__(self, mean, std):

        new_mean = [-m / s for m, s in zip(mean, std)]
        new_std = [1 / s for s in std]

        self.denormalize = transforms.Normalize(mean=new_mean, std=new_std)

    def __call__(self, tensor):
        return self.denormalize(tensor)


def plot_training_metrics(metrics):
    """
    Plots the training and validation metrics from a model training process.

    This function generates two side-by-side plots:
    1. Training Loss vs. Validation Loss.
    2. Validation Accuracy.

    Args:
        metrics (list): A list or tuple containing three lists:
                        [train_losses, val_losses, val_accuracies].
    """
    # Unpack the metrics into their respective lists
    train_losses, val_losses, val_accuracies = metrics

    # Determine the number of epochs from the length of the training losses list
    num_epochs = len(train_losses)
    # Create a 1-indexed range of epoch numbers for the x-axis
    epochs = range(1, num_epochs + 1)

    # Create a figure and a set of subplots with 1 row and 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Configure the first subplot for training and validation loss ---
    # Select the first subplot
    ax1 = axes[0]
    # Plot training loss data
    ax1.plot(epochs, train_losses, color='#085c75', linewidth=2.5, marker='o', markersize=5, label='Training Loss')
    # Plot validation loss data
    ax1.plot(epochs, val_losses, color='#fa5f64', linewidth=2.5, marker='o', markersize=5, label='Validation Loss')
    # Set the title and axis labels for the loss plot
    ax1.set_title('Training & Validation Loss', fontsize=14)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    # Display the legend
    ax1.legend()
    # Add a grid for better readability
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- Configure the second subplot for validation accuracy ---
    # Select the second subplot
    ax2 = axes[1]
    # Plot validation accuracy data
    ax2.plot(epochs, val_accuracies, color='#fa5f64', linewidth=2.5, marker='o', markersize=5,
             label='Validation Accuracy')
    # Set the title and axis labels for the accuracy plot
    ax2.set_title('Validation Accuracy', fontsize=14)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    # Display the legend
    ax2.legend()
    # Add a grid for better readability
    ax2.grid(True, linestyle='--', alpha=0.6)

    # --- Apply dynamic and consistent styling to both subplots ---
    # Calculate a suitable interval for the x-axis ticks to avoid clutter
    x_interval = (num_epochs - 1) // 10 + 1

    # Loop through each subplot to apply common axis settings
    for ax in axes:
        # Set the y-axis to start at 0 and the x-axis to span the epochs
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=1, right=num_epochs)

        # Set the major tick locator for the x-axis using the dynamic interval
        ax.xaxis.set_major_locator(mticker.MultipleLocator(x_interval))
        # Set the font size for the tick labels on both axes
        ax.tick_params(axis='both', which='major', labelsize=10)

    # Adjust subplot parameters for a tight layout
    plt.tight_layout()
    # Display the plots
    plt.show()