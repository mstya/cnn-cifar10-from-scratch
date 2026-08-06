"""Train the CIFAR-10 CNN end to end, from raw Kaggle data to a saved checkpoint.

Usage:
    python train.py
    python train.py --config configs/base_config.json --epochs 5

Prerequisites:
    - Kaggle credentials configured locally (e.g. ~/.kaggle/kaggle.json) so
      `kagglehub` can download the `cifar-10` competition data non-interactively.

Mirrors the pipeline built up in notebooks/01_eda.ipynb and
notebooks/02_model.ipynb, but as a single reproducible script so a fresh
checkout can be trained with one command.
"""
import argparse
import json
import os
import random
from os import listdir
from os.path import isfile, join
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: never try to pop up a GUI window

import kagglehub
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset

from src.dataset import TmpCifarDataset, define_transformations
from src.experiment import prepare_results_dir, save_classes, save_config, save_model, save_results
from src.metrics import plot_training_metrics
from src.model import SimpleCNN, SimpleCNNConv3
from src.training import device, training_loop

REPO_ROOT = Path(__file__).resolve().parent

MODEL_REGISTRY = {
    "SimpleCNN": SimpleCNN,
    "SimpleCNNConv3": SimpleCNNConv3,
    "SimpleCNN3conv": SimpleCNNConv3,  # alias used by configs/base_config.json
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "base_config.json"),
                         help="Path to the training config JSON.")
    parser.add_argument("--epochs", type=int, default=None,
                         help="Override the number of epochs from the config.")
    parser.add_argument("--results-dir", default=str(REPO_ROOT / "experiments"),
                         help="Directory experiments are written to.")
    return parser.parse_args()


def build_metadata(dataset_path):
    """Reproduce notebooks/01_eda.ipynb's metadata table: one row per training
    image, with a `target` column mapping each label to a stable class index
    (order of first appearance in trainLabels.csv, which is id-ascending)."""
    labels_path = os.path.join(dataset_path, "trainLabels.csv")
    train_path = os.path.join(dataset_path, "train")

    labels_df = pd.read_csv(labels_path)

    files = [(int(Path(join(train_path, f)).stem), f) for f in listdir(train_path) if isfile(join(train_path, f))]
    files_df = pd.DataFrame(files, columns=["id", "file_name"])

    cifar_pd = pd.merge(labels_df, files_df, on="id", how="inner")

    all_labels = cifar_pd["label"].drop_duplicates().reset_index()
    all_labels["target"] = all_labels.index
    all_labels = all_labels.drop(columns=["index"])

    cifar_pd = pd.merge(cifar_pd, all_labels, on="label", how="left")

    class_names = all_labels.sort_values("target")["label"].tolist()

    return cifar_pd, class_names, train_path


def compute_data_stats(dataset, seed, train_split):
    """Reproduce notebooks/01_eda.ipynb's per-channel mean/std computation
    over the training split, and the train/val split indices."""
    from torch.utils.data import random_split
    from torchvision.transforms import ToTensor

    val_split = 1 - train_split
    stats_dataset = TmpCifarDataset(dataset.train_lib_dir, dataset.df, transform=ToTensor())
    split = random_split(stats_dataset, [train_split, val_split], generator=torch.Generator().manual_seed(seed))
    loader = DataLoader(split[0], batch_size=64, shuffle=False)

    channel_sum = torch.zeros(3)
    channel_squared_sum = torch.zeros(3)
    pixel_count = torch.zeros(3)
    for images, _ in loader:
        channel_sum += images.sum(dim=(0, 2, 3))
        channel_squared_sum += (images ** 2).sum(dim=(0, 2, 3))
        pixel_count += images.shape[0] * images.shape[2] * images.shape[3]

    mean = (channel_sum / pixel_count).tolist()
    std = ((channel_squared_sum / pixel_count - torch.tensor(mean) ** 2).sqrt()).tolist()

    split_indices = {"train_indices": split[0].indices, "val_indices": split[1].indices}

    return mean, std, split_indices


def load_or_compute_data_prep(cifar_pd, train_path, seed, train_split):
    """Reuse the committed configs/data_stats.json and configs/data_split.json
    if present (that's the norm for a repo checkout); compute and persist them
    otherwise (e.g. the very first run against a fresh dataset)."""
    stats_path = REPO_ROOT / "configs" / "data_stats.json"
    split_path = REPO_ROOT / "configs" / "data_split.json"

    if stats_path.exists() and split_path.exists():
        with open(stats_path, encoding="utf-8") as file:
            stats = json.load(file)
        with open(split_path, encoding="utf-8") as file:
            split = json.load(file)
        return stats["mean"], stats["std"], split

    print("configs/data_stats.json or configs/data_split.json missing — computing them now...")
    raw_dataset = TmpCifarDataset(train_path, cifar_pd)
    mean, std, split = compute_data_stats(raw_dataset, seed, train_split)

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as file:
        json.dump({"mean": mean, "std": std}, file)
    with open(split_path, "w", encoding="utf-8") as file:
        json.dump(split, file)

    return mean, std, split


def main():
    args = parse_args()

    with open(args.config, encoding="utf-8") as file:
        config_json = json.load(file)

    seed = int(config_json["seed"])
    random.seed(seed)
    torch.manual_seed(seed)

    batch_size = int(config_json["data"]["batch_size"])
    train_split = float(config_json["data"]["train_split"])
    augmentation = config_json["data"]["augmentation"]

    model_name = config_json["model"]["name"]
    dropout = config_json["model"]["dropout"]

    learning_rate = float(config_json["training"]["learning_rate"])
    weight_decay = float(config_json["training"]["weight_decay"])
    scheduler_config = config_json["training"]["scheduler"]
    epochs = args.epochs if args.epochs is not None else int(config_json["training"]["epochs"])

    print("--- Downloading / locating CIFAR-10 dataset via kagglehub ---")
    dataset_path = kagglehub.competition_download("cifar-10")

    print("--- Building training metadata ---")
    cifar_pd, class_names, train_path = build_metadata(dataset_path)

    metadata_path = REPO_ROOT / "data" / "processed" / "cifar10_metadata.csv"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    cifar_pd.to_csv(metadata_path, index=False)
    print(f"Classes ({len(class_names)}): {class_names}")

    mean, std, split = load_or_compute_data_prep(cifar_pd, train_path, seed, train_split)

    train_transform, val_transform = define_transformations(augmentation, mean, std)

    train_dataset = TmpCifarDataset(train_path, cifar_pd, transform=train_transform)
    val_dataset = TmpCifarDataset(train_path, cifar_pd, transform=val_transform)
    train_subset = Subset(train_dataset, split["train_indices"])
    val_subset = Subset(val_dataset, split["val_indices"])

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    model_class = MODEL_REGISTRY.get(model_name, SimpleCNNConv3)
    model = model_class(num_classes=len(class_names), dropout=dropout)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=scheduler_config["mode"],
        factor=scheduler_config["factor"],
        patience=scheduler_config["patience"],
        min_lr=scheduler_config["min_lr"],
    )

    model, metrics = training_loop(
        model, train_loader, val_loader, loss_function, optimizer, epochs, device, scheduler,
    )

    print("--- Saving experiment artifacts ---")
    results_dir, exp_number, exp_path = prepare_results_dir(args.results_dir)

    save_config(exp_path, config_json)
    save_classes(exp_path, class_names)
    save_model(exp_path, model)
    save_results(results_dir, exp_path, exp_number, learning_rate, metrics, model.__class__.__name__, epochs,
                 batch_size, optimizer.__class__.__name__, weight_decay, seed, str(augmentation), dropout,
                 scheduler_config)
    plot_training_metrics(metrics, exp_path)

    print(f"\nDone. Model and artifacts saved to {exp_path}/")
    print(f"To serve this checkpoint, set \"model_experiment\": {exp_number} in configs/serving_config.json,")
    print("then run `make serve` (or `uvicorn src.api:app --reload` without Docker).")
    print(f"(Or try it immediately without touching the config: MODEL_EXPERIMENT_DIR={exp_path} uvicorn src.api:app --reload)")


if __name__ == "__main__":
    main()
