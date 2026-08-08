# CNN for CIFAR-10 from Scratch

An educational project for CIFAR-10 image classification using a CNN implemented in PyTorch.

## Quick start

```bash
pip install -r requirements.txt

# 1. Train (needs Kaggle credentials, e.g. ~/.kaggle/kaggle.json — see "Training" below)
make train                  # same as: python -m src.train

# 2. Serve the checkpoint configured in configs/serving_config.json as an API
make serve                  # same as: docker compose up --build

# 3. Classify an image
curl -X POST http://localhost:8000/predict -F "file=@path/to/image.png"
```

See [Training](#training) and [Serving the API](#serving-the-api) below for details
(quick smoke tests, running the API without Docker, switching which checkpoint is served, etc).

## Project structure

```text
configs/       # configurations, dataset statistics, train/validation split, and
               # serving_config.json (which experiment the API/Docker/notebook serve)
data/processed # metadata CIFAR-10
notebooks/     # EDA, training, analysis, and inference runs
src/           # dataset, model, training loop, metrics, prediction, API, and train.py
experiments/   # run artifacts and summary.csv
Dockerfile     # packages src/api.py + a trained checkpoint into a servable image
Makefile       # `make train` / `make serve`
```

## Training

Prerequisite: Kaggle credentials configured locally (e.g. `~/.kaggle/kaggle.json`)
so `kagglehub` can download the `cifar-10` competition data non-interactively.

```bash
pip install -r requirements.txt
make train                            # or: python -m src.train
python -m src.train --epochs 5        # quick smoke test
```

This downloads the dataset (cached by `kagglehub` after the first run), rebuilds
`data/processed/cifar10_metadata.csv`, trains the model, and writes a new
`experiments/<N>/` directory containing `model.pt`, `config.json`,
`classes.json`, `metrics.csv`, and `results.png` — the same artifacts produced
by `notebooks/02_model.ipynb`, but from a single non-interactive command.

### Early stopping

`src/training.py`'s `training_loop` always keeps the checkpoint from whichever
epoch had the best validation accuracy so far (that part isn't new). Early
stopping adds one more thing on top: it **ends training early** once
validation accuracy hasn't improved for a run of consecutive epochs, instead
of always running the full epoch budget.

Concretely, each epoch:
1. If `val_accuracy` beats the best seen so far (by more than `min_delta`),
   that checkpoint is saved and a counter resets to 0.
2. Otherwise the counter increments.
3. Once the counter reaches `patience`, training stops (the loop `break`s) —
   the returned model is still the best checkpoint from step 1, not the last
   epoch run.

It's configured under `training.early_stopping` in `configs/base_config.json`:

```json
"early_stopping": { "patience": 20, "min_delta": 0.0 }
```

- **`patience`** — how many epochs in a row without improvement to tolerate
  before stopping. Set to `null`/omit the whole `early_stopping` block to
  disable it (the original always-run-`epochs`-epochs behavior). Deliberately
  set higher than the LR scheduler's own `patience` (4) above it, so
  `ReduceLROnPlateau` gets a few chances to lower the learning rate — which
  often lets validation accuracy recover — before early stopping gives up.
- **`min_delta`** — minimum improvement to count as "improved" (guards
  against stopping the counter's reset on noise-level ties).

Override per run without touching the config: `--early-stopping-patience N`
(`--early-stopping-patience 0` disables it for that run).

**Why it's worth having:** Exp #93 (the current best checkpoint) hit its best
validation accuracy (88.13%) at epoch 91 of 150, then spent the remaining 59
epochs oscillating between 87.6–88.13% without ever beating it. A
`patience=20` run would have stopped around epoch 111, saving roughly a
quarter of that run's training time for the same result.

## Serving the API

`src/api.py` is a FastAPI app exposing `POST /predict` (multipart image upload
→ predicted label + per-class probabilities) and `GET /health`. It's packaged
into a Docker image with the trained checkpoint baked in at build time — no
GPU/Kaggle credentials needed to run it, only to train.

**Which checkpoint gets served is controlled by one file:
[`configs/serving_config.json`](configs/serving_config.json)** (currently `{"model_experiment": 93}`,
the best of the three seeds — see [Best result](#best-result--simplecnnconv3)).
It's the single source of truth read by `src/api.py`, the `Dockerfile` (via the
`Makefile`/`docker-compose.yml`), and `notebooks/04_inference.ipynb` — point it
at a new experiment (e.g. the one `src/train.py` just produced) and every one of
those picks it up automatically. Make sure the checkpoint you point it at actually
loads against the current `src/model.py` first (see the caveat in that section).

```bash
make serve
# ...equivalent to:
MODEL_EXPERIMENT=$(python3 -c "import json; print(json.load(open('configs/serving_config.json'))['model_experiment'])") \
  docker compose up --build
```

Then:

```bash
curl -X POST http://localhost:8000/predict -F "file=@path/to/image.png"
```

Running it without Docker (e.g. during development, with `requirements.txt`
already installed) works the same way, and also defaults to
`configs/serving_config.json`:

```bash
uvicorn src.api:app --reload
# ...or override without touching the config:
MODEL_EXPERIMENT_DIR=experiments/<N> uvicorn src.api:app --reload
```

(`requirements-api.txt` is the slim, Docker-only dependency set used by the
`Dockerfile` — it omits training-only packages and torch/torchvision, which
the `Dockerfile` installs separately from the CPU-only wheel index.)

## Data and evaluation protocol

- CIFAR-10: 50,000 `32x32` images across 10 balanced classes.
- Train/validation split: 85% / 15%; indices are stored in `configs/data_split.json`.
- Normalization is calculated from the training subset only:
  - mean: `[0.4916, 0.4826, 0.4470]`
  - std: `[0.2469, 0.2434, 0.2616]`
- Primary metric: `Best Val Accuracy`.
- Final configurations were evaluated with seeds 1, 2, and 3 while keeping the split fixed.

## Model architecture

Two architectures have been trained, both in `src/model.py`:

**`SimpleCNN`** — the initial baseline, two convolution blocks:

```text
Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d
Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d
Flatten -> Linear(4096, 512) -> ReLU -> Dropout -> Linear(512, 10)
```

**`SimpleCNNConv3`** (config name `SimpleCNN3conv`) — **current / deployed model**, adds a
third, deeper convolution block:

```text
Conv2d(3, 32)   -> BatchNorm2d -> ReLU -> MaxPool2d
Conv2d(32, 64)  -> BatchNorm2d -> ReLU -> MaxPool2d
Conv2d(64, 128) -> BatchNorm2d -> ReLU -> MaxPool2d(stride=1)
Flatten -> Linear(6272, 512) -> ReLU -> Dropout(0.15x)
        -> Linear(512, 256)  -> ReLU -> Dropout(0.1x)
        -> Linear(256, 10)
```

## Key experiments

| Stage | Change | Result |
| --- | --- | ---: |
| Initial baseline (`SimpleCNN`) | Flip + Rotation, 10 epochs, dropout 0.6 | 71.15% for seed 1 |
| Dropout tuning | Best `dropout=0.2` | 74.33% after 10 epochs |
| Augmentation | `RandomCrop(32, padding=4) + RandomHorizontalFlip(0.5)` instead of rotation | 80.11% mean across 3 seeds after 50 epochs |
| Longer training | Same configuration, 100 epochs | 81.88% mean across 3 seeds |
| Scheduler | `ReduceLROnPlateau` for 100 epochs | 83.50% mean across 3 seeds |
| `SimpleCNN` baseline | Scheduler for 150 epochs | 83.88% mean across 3 seeds |
| Deeper architecture (`SimpleCNNConv3`) | Added a 3rd conv block, re-tuned dropout/weight decay | **88.02%** mean across 3 seeds |

## Best result — `SimpleCNNConv3`

```text
Model:           SimpleCNNConv3
Batch size:      64
Augmentation:    RandomCrop(32, padding=4) + RandomHorizontalFlip(p=0.5)
Dropout:         0.2
Optimizer:       Adam
Learning rate:   0.0005
Weight decay:    0.0004
Scheduler:       ReduceLROnPlateau(mode="min", factor=0.5, patience=4, min_lr=1e-6)
Epochs:          150
```

Results across different seeds (all verified to load against the current `src/model.py`):

| Seed | Experiment | Best Val Accuracy | Best Epoch |
| ---: | :--- | ---: | ---: |
| 1 | Exp #92 | 88.11% | 90 |
| 2 | **Exp #93 (currently served, best single run)** | **88.13%** | 91 |
| 3 | Exp #90 | 87.81% | 87 |
| **Mean** | | **88.02%** | — |

That's a **+4.14 pt** improvement in best-val-accuracy over the `SimpleCNN` baseline (83.88% mean).
`configs/serving_config.json` points at **experiment 93**, the best of the three above.

**Note on older runs:** `experiments/summary.csv` also records earlier `SimpleCNNConv3` attempts
(Exp #65/#66/#68/#78) scoring similarly (up to 88.19%), but the classifier head's shape changed in
a later commit, so those checkpoints' `model.pt` no longer load against the current `src/model.py`
(`load_state_dict` raises a key-mismatch error) — the table above only includes checkpoints that
were verified to still load.

The complete run history and parameters are stored in `experiments/summary.csv`. Each experiment also contains its own `config.json`, `metrics.csv`, `results.png` plot, and `model.pt` checkpoint — but check it actually loads with the current `src/model.py` before relying on an old one (see note above).

## Next steps

- [DONE] Add early stopping (see [Early stopping](#early-stopping)).
- [DONE] Evaluate per-class accuracy, a confusion matrix, and examples of incorrect predictions.
- [DONE] Compare this model with a deeper CNN architecture (`SimpleCNNConv3`).
- [DONE] Re-run seeds 1 and 2 at 150 epochs against the current architecture (Exp #92/#93) to get a real multi-seed mean.
- Tag each `experiments/<N>/config.json` with a model/architecture version so future architecture
  changes don't silently orphan old checkpoints (see note above).
- [DONE] Point `configs/serving_config.json` at the best of the three seeds (experiment 93).
- Retrain seed 3 to see if it can beat experiment 93's 88.13%.
