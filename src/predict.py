import json
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

# Mapping index -> class name, matching the `target` values produced in
# notebooks/01_eda.ipynb (see data/processed/cifar10_metadata.csv:
# df[['label', 'target']].drop_duplicates().sort_values('target')).
CIFAR10_CLASSES = [
    "frog",
    "truck",
    "deer",
    "automobile",
    "bird",
    "horse",
    "ship",
    "cat",
    "dog",
    "airplane",
]

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def load_data_stats(config_path=_CONFIG_DIR / "data_stats.json"):
    """Load the mean/std used to normalize training images."""
    with open(config_path, encoding="utf-8") as file:
        stats = json.load(file)
    return stats["mean"], stats["std"]


def build_predict_transform(mean, std, image_size=32):
    """Build the same preprocessing pipeline used for validation images."""
    return Compose([
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize(mean=mean, std=std),
    ])


def predict(image, model, class_names=CIFAR10_CLASSES, transform=None, device=None):
    """Predict the class of a single image using a trained model.

    Args:
        image: a PIL.Image.Image, or a path (str/Path) to an image file.
        model: a trained torch.nn.Module (e.g. SimpleCNNConv3) already loaded
            with trained weights.
        class_names: list mapping class index -> class name. Defaults to
            CIFAR10_CLASSES.
        transform: optional preprocessing transform. Defaults to the
            resize/normalize pipeline built from configs/data_stats.json.
        device: torch.device to run inference on. Defaults to the model's
            current device.

    Returns:
        str: the predicted class name.
    """
    if isinstance(image, (str, Path)):
        with Image.open(image) as img:
            image = img.convert("RGB")
    elif isinstance(image, Image.Image):
        image = image.convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    if transform is None:
        mean, std = load_data_stats()
        transform = build_predict_transform(mean, std)

    if device is None:
        device = next(model.parameters()).device

    input_tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        predicted_index = outputs.argmax(dim=1).item()

    return class_names[predicted_index]
