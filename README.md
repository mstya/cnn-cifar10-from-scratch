# CNN for CIFAR-10 from Scratch

An educational project for CIFAR-10 image classification using a CNN implemented in PyTorch.

## Project structure

```text
configs/       # configurations, dataset statistics, and train/validation split
data/processed # metadata CIFAR-10
notebooks/     # EDA and training runs
src/           # dataset, model, training loop, and metrics
experiments/   # run artifacts and summary.csv
```

## Data and evaluation protocol

- CIFAR-10: 50,000 `32x32` images across 10 balanced classes.
- Train/validation split: 85% / 15%; indices are stored in `configs/data_split.json`.
- Normalization is calculated from the training subset only:
  - mean: `[0.4916, 0.4826, 0.4470]`
  - std: `[0.2469, 0.2434, 0.2616]`
- Primary metric: `Best Val Accuracy`.
- Final configurations were evaluated with seeds 1, 2, and 3 while keeping the split fixed.

## Current model

`SimpleCNN` consists of two convolution blocks:

```text
Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d
Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d
Flatten -> Linear(4096, 512) -> ReLU -> Dropout -> Linear(512, 10)
```

## Key experiments

| Stage | Change | Result |
| --- | --- | ---: |
| Initial baseline | Flip + Rotation, 10 epochs, dropout 0.6 | 71.15% for seed 1 |
| Dropout tuning | Best `dropout=0.2` | 74.33% after 10 epochs |
| Augmentation | `RandomCrop(32, padding=4) + RandomHorizontalFlip(0.5)` instead of rotation | 80.11% mean across 3 seeds after 50 epochs |
| Longer training | Same configuration, 100 epochs | 81.88% mean across 3 seeds |
| Scheduler | `ReduceLROnPlateau` for 100 epochs | 83.50% mean across 3 seeds |
| Current baseline | Scheduler for 150 epochs | **83.88%** mean across 3 seeds |

## Best current configuration

```text
Model:           SimpleCNN
Batch size:      64
Augmentation:    RandomCrop(32, padding=4) + RandomHorizontalFlip(p=0.5)
Dropout:         0.2
Optimizer:       Adam
Learning rate:   0.0005
Weight decay:    0.0004
Scheduler:       ReduceLROnPlateau(mode="min", factor=0.5, patience=4, min_lr=1e-6)
Epochs:          150
```

Results across different seeds:

| Seed | Best Val Accuracy | Best Epoch |
| ---: | ---: | ---: |
| 1 | 83.75% | 139 |
| 2 | 83.72% | 139 |
| 3 | 84.16% | 140 |
| **Mean** | **83.88%** | — |

The complete run history and parameters are stored in `experiments/summary.csv`. Each experiment also contains its own `config.json`, `metrics.csv`, `results.png` plot, and `model.pt` checkpoint.

## Next steps

- Add early stopping.
- Evaluate per-class accuracy, a confusion matrix, and examples of incorrect predictions.
- Compare this model with a deeper CNN architecture.
