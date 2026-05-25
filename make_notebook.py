import json

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "colab": {"provenance": []}
    },
    "cells": []
}

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}

def code(src):
    lines = src.strip().split("\n")
    source = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}

cells = []

# ── Title ──
cells.append(md("# Drill Core Image Classification — Data Pipeline\n**Google Colab** | Run cells top to bottom"))

# ── Cell 1 ──
cells.append(md("## Cell 1 — Install Dependencies\nRun once. If Colab prompts a runtime restart, do it, then skip this cell."))
cells.append(code("""!pip install timm huggingface_hub --quiet"""))

# ── Cell 2 ──
cells.append(md("## Cell 2 — Imports"))
cells.append(code("""import os
import random
import warnings
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from collections import Counter

warnings.filterwarnings('ignore')
print(f"PyTorch version: {torch.__version__}")"""))

# ── Cell 3 ──
cells.append(md("## Cell 3 — Config\nAll hyperparameters and paths live here. Change things **only** in this cell."))
cells.append(code("""class Config:
    # --- Paths (Colab) ---
    # After downloading + unzipping DCID.zip, DCID-7 lives here:
    DATA_ROOT   = '/content/DCID/DCID-512-7'
    TRAIN_DIR   = os.path.join(DATA_ROOT, 'train')
    TEST_DIR    = os.path.join(DATA_ROOT, 'test')
    # Checkpoints saved to Drive so they survive session resets
    CKPT_DIR    = '/content/drive/MyDrive/drill_core_project/checkpoints'

    # --- Data ---
    IMG_SIZE    = 224       # Standard ImageNet input size
    NUM_CLASSES = 7
    VAL_SPLIT   = 0.1      # 10% of training data -> validation
    SEED        = 42

    # --- Training (used in next notebook) ---
    BATCH_SIZE  = 32
    NUM_WORKERS = 2
    LR          = 1e-3
    EPOCHS      = 20
    BACKBONE    = 'efficientnet_b0'

cfg = Config()
print("Config loaded.")"""))

# ── Cell 4 ──
cells.append(md("## Cell 4 — Reproducibility + Device"))
cells.append(code("""def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(cfg.SEED)

device = torch.device(
    'cuda'  if torch.cuda.is_available() else
    'mps'   if torch.backends.mps.is_available() else
    'cpu'
)
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"GPU:   {torch.cuda.get_device_name(0)}")
    print(f"VRAM:  {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")"""))

# ── Cell 5 ──
cells.append(md("## Cell 5 — Mount Google Drive\nSaves checkpoints so your work survives Colab session resets."))
cells.append(code("""from google.colab import drive
drive.mount('/content/drive')
os.makedirs(cfg.CKPT_DIR, exist_ok=True)
print(f"Checkpoint directory ready: {cfg.CKPT_DIR}")"""))

# ── Cell 6 ──
cells.append(md("## Cell 6 — Download DCID Dataset\n**Source:** [168sir/drill-core-image-dataset](https://huggingface.co/datasets/168sir/drill-core-image-dataset) · License: CC BY-NC 4.0\n\n⏱ ~5–10 min on Colab free tier. Run **once per session** (data doesn't persist across resets).\n\nAfter unzipping you'll have:\n```\n/content/DCID/\n    DCID-512-7/        ← we use this (7 classes, 512×512)\n        train/<class>/  ← 4,000 images per class\n        test/<class>/   ← 1,000 images per class\n    DCID-512-35/       (35-class version — not used now)\n    noise-512-7/       (noisy variants — not used now)\n    noise-512-35/\n```"))
cells.append(code("""from huggingface_hub import hf_hub_download
import zipfile

print("Downloading DCID.zip (~3.82 GB) — please wait...")
zip_path = hf_hub_download(
    repo_id   = "168sir/drill-core-image-dataset",
    filename  = "DCID.zip",
    repo_type = "dataset",
    local_dir = "/content"
)
print(f"Downloaded to: {zip_path}")

print("Extracting...")
with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall("/content/DCID")
print("Done! Dataset ready at /content/DCID/")"""))

# ── Cell 7 ──
cells.append(md("## Cell 7 — Verify Dataset Structure"))
cells.append(code("""def verify_dataset(train_dir, test_dir):
    assert os.path.isdir(train_dir), f"Train dir not found: {train_dir}"
    assert os.path.isdir(test_dir),  f"Test dir not found:  {test_dir}"
    train_classes = sorted(os.listdir(train_dir))
    test_classes  = sorted(os.listdir(test_dir))
    assert train_classes == test_classes, "Train/test classes don't match!"
    print(f"\\n✅ Dataset verified — {len(train_classes)} classes:")
    for cls in train_classes:
        n_train = len(os.listdir(os.path.join(train_dir, cls)))
        n_test  = len(os.listdir(os.path.join(test_dir,  cls)))
        print(f"  {cls:35s}  train: {n_train:5d}  test: {n_test:5d}")

verify_dataset(cfg.TRAIN_DIR, cfg.TEST_DIR)"""))

# ── Cell 8 ──
cells.append(md("## Cell 8 — Transforms\n- **Training:** augmentation (random crop, flips, rotation, color jitter) to simulate real-world variation in drill core photos\n- **Val / Test:** only resize + normalize — deterministic, no randomness"))
cells.append(code("""# ImageNet normalization stats — used because backbone was pretrained on ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),              # Slightly larger than target
    transforms.RandomCrop(cfg.IMG_SIZE),        # Random 224x224 crop
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

eval_transforms = transforms.Compose([
    transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

print("Transforms defined.")"""))

# ── Cell 9 ──
cells.append(md("## Cell 9 — Dataset + Train/Val Split\n`SubsetWithTransform` is a small wrapper that lets the train and val subsets have **different transforms** even though they share the same base `ImageFolder`. Without it, both would get the same augmentations."))
cells.append(code("""class SubsetWithTransform(torch.utils.data.Dataset):
    \"\"\"Applies a per-subset transform to a torch Subset.\"\"\"
    def __init__(self, subset, transform):
        self.subset    = subset
        self.transform = transform

    def __getitem__(self, idx):
        img, label = self.subset[idx]   # PIL image at this point
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.subset)


# Load raw images (transform=None -> PIL, needed for clean splitting)
full_train_raw = datasets.ImageFolder(root=cfg.TRAIN_DIR, transform=None)
test_dataset   = datasets.ImageFolder(root=cfg.TEST_DIR,  transform=eval_transforms)

CLASS_NAMES = full_train_raw.classes
print(f"Train images: {len(full_train_raw)}")
print(f"Test images:  {len(test_dataset)}")
print(f"Classes ({len(CLASS_NAMES)}): {CLASS_NAMES}")

# 90% train, 10% val
val_size   = int(cfg.VAL_SPLIT * len(full_train_raw))
train_size = len(full_train_raw) - val_size

train_subset, val_subset = random_split(
    full_train_raw,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(cfg.SEED)
)

train_dataset = SubsetWithTransform(train_subset, train_transforms)
val_dataset   = SubsetWithTransform(val_subset,   eval_transforms)

print(f"\\nSplit -> Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")"""))

# ── Cell 10 ──
cells.append(md("## Cell 10 — DataLoaders\n`pin_memory=True` keeps batches in page-locked CPU memory for faster CPU→GPU transfer on Colab."))
cells.append(code("""train_loader = DataLoader(
    train_dataset, batch_size=cfg.BATCH_SIZE,
    shuffle=True,  num_workers=cfg.NUM_WORKERS, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=cfg.BATCH_SIZE,
    shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=cfg.BATCH_SIZE,
    shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True
)

print(f"Batches -> Train: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)}")

# Batch shape sanity check
imgs, labels = next(iter(train_loader))
print(f"Batch shape: {imgs.shape}   (batch x channels x H x W)")
print(f"Labels:      {labels.tolist()}")"""))

# ── Cell 11 ──
cells.append(md("## Cell 11 — Visualize Sample Images\nAlways check augmented images look sensible before training."))
cells.append(code("""def denormalize(tensor):
    img = tensor.numpy().transpose(1, 2, 0)
    img = np.array(IMAGENET_STD) * img + np.array(IMAGENET_MEAN)
    return np.clip(img, 0, 1)

seen = {}
for img, label in train_dataset:
    label = label if isinstance(label, int) else label.item()
    if label not in seen:
        seen[label] = img
    if len(seen) == len(CLASS_NAMES):
        break

n_cols = 4
n_rows = (len(CLASS_NAMES) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
axes = axes.flatten()

for i, (label, img) in enumerate(sorted(seen.items())):
    axes[i].imshow(denormalize(img))
    axes[i].set_title(CLASS_NAMES[label], fontsize=12, fontweight='bold')
    axes[i].axis('off')
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.suptitle('One sample per class — Training set (with augmentation)', fontsize=14)
plt.tight_layout()
plt.savefig('sample_images.png', dpi=120, bbox_inches='tight')
plt.show()
print("Saved: sample_images.png")"""))

# ── Cell 12 ──
cells.append(md("## Cell 12 — Visualize Class Distribution\nVerify the dataset is balanced across train, val, and test."))
cells.append(code("""def get_labels(ds):
    if hasattr(ds, 'targets'):
        return ds.targets
    inner = ds.subset
    return [inner.dataset.targets[i] for i in inner.indices]

train_labels = get_labels(train_dataset)
val_labels   = get_labels(val_dataset)
test_labels  = test_dataset.targets

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
palette = sns.color_palette("Blues_d", len(CLASS_NAMES))

for ax, labels, title in zip(axes,
                              [train_labels, val_labels, test_labels],
                              ['Train', 'Val', 'Test']):
    counts = Counter(labels)
    bars = ax.bar(
        [CLASS_NAMES[i] for i in sorted(counts)],
        [counts[i]      for i in sorted(counts)],
        color=palette
    )
    ax.bar_label(bars, fontsize=9)
    ax.set_title(f'{title} Class Distribution  (n={sum(counts.values())})',
                 fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylabel('Image count')
    ax.set_ylim(0, max(counts.values()) * 1.15)
    sns.despine(ax=ax)

plt.tight_layout()
plt.savefig('class_distribution.png', dpi=120, bbox_inches='tight')
plt.show()
print("Saved: class_distribution.png")
print("\\n✅ Data pipeline complete — ready to build the model.")"""))

nb["cells"] = cells

with open("/sessions/intelligent-hopeful-hawking/mnt/outputs/drill_core_pipeline.ipynb", "w") as f:
    json.dump(nb, f, indent=2)

print("Notebook written successfully.")
