# Drill Core Classification — Project Context

**Student:** Artur Khvan (akhvan@depaul.edu)
**Last updated:** 2026-05-24

---

## Overview

Two simultaneous class projects sharing the same dataset and classification backbone. Build the pipeline once, extend it twice.

| | Deep Learning Project | ML Project |
|---|---|---|
| **Class** | Deep Learning | Machine Learning |
| **Partner** | Solo | Luan Dinh (team) |
| **Due** | May 25, 2026 ⚠️ URGENT | June 6, 2026 |
| **Unique deliverable** | Interval Priority Score | Synthetic 3D Visualization |

---

## Dataset: DCID-7

- **HuggingFace repo:** `168sir/drill-core-image-dataset`
- **File:** `DCID.zip` (~3.82 GB), extracts to `DCID/DCID-512-7/`
- **Classes:** 7 lithology types
- **Images:** 35,000 total — 4,000 train + 1,000 test per class
- **Resolution:** 512×512 px (resized to 224×224 in pipeline)
- **License:** CC BY-NC 4.0
- **Val split:** No official val set — carved 10% from training data with seed 42

---

## Architecture Decisions

| Decision | Choice | Reason |
|---|---|---|
| Compute | Google Colab T4 (15 GB VRAM) | MacBook M2 only has 8 GB unified RAM |
| Backbone | EfficientNet-B0 | More parameter-efficient than ResNet50, trains faster at 224×224 |
| Transfer learning | ImageNet pretrained → freeze base → fine-tune head | Standard approach for small domain-specific datasets |
| Val strategy | `random_split` 90/10 from train, seed=42 | DCID only provides train/test splits |
| Input size | 224×224 | Standard for ImageNet pretrained models |
| Normalization | mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] | ImageNet statistics |

---

## Files Produced

| File | Location | Status |
|---|---|---|
| `drill_core_pipeline.ipynb` | outputs folder | ✅ Complete (Cells 1–12) |

---

## Notebook Structure (`drill_core_pipeline.ipynb`)

### Cell 1 — Install dependencies
```python
!pip install timm huggingface_hub --quiet
```

### Cell 2 — Imports
```python
import os, random, zipfile
import numpy as np
import torch, torchvision
import timm
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
```

### Cell 3 — Config
```python
class Config:
    DATA_ROOT   = '/content/DCID/DCID-512-7'
    TRAIN_DIR   = os.path.join(DATA_ROOT, 'train')
    TEST_DIR    = os.path.join(DATA_ROOT, 'test')
    CKPT_DIR    = '/content/drive/MyDrive/drill_core_project/checkpoints'
    IMG_SIZE    = 224
    NUM_CLASSES = 7
    VAL_SPLIT   = 0.1
    SEED        = 42
    BATCH_SIZE  = 32
    NUM_WORKERS = 2
    LR          = 1e-3
    EPOCHS      = 20
    BACKBONE    = 'efficientnet_b0'
```

> **If Google Drive mount fails:** Change `CKPT_DIR = '/content/checkpoints'` and replace Cell 5 with `os.makedirs(cfg.CKPT_DIR, exist_ok=True)`.

### Cell 4 — Seed + device
```python
def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

set_seed(Config.SEED)
device = (
    torch.device('cuda') if torch.cuda.is_available()
    else torch.device('mps') if torch.backends.mps.is_available()
    else torch.device('cpu')
)
print(f"Using device: {device}")
```

### Cell 5 — Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```
> If this errors with credential propagation, re-run the cell (auth popup sometimes hidden), enable third-party cookies in Chrome, or use the `/content/checkpoints` workaround above.

### Cell 6 — Download dataset
```python
from huggingface_hub import hf_hub_download
import zipfile

zip_path = hf_hub_download(
    repo_id="168sir/drill-core-image-dataset",
    filename="DCID.zip",
    repo_type="dataset",
    local_dir="/content"
)
with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall("/content/DCID")
print("Dataset extracted.")
```

### Cell 7 — Verify dataset
```python
def verify_dataset():
    for split in ['train', 'test']:
        split_dir = os.path.join(Config.DATA_ROOT, split)
        classes = sorted(os.listdir(split_dir))
        print(f"\n{split.upper()} — {len(classes)} classes:")
        for cls in classes:
            n = len(os.listdir(os.path.join(split_dir, cls)))
            print(f"  {cls}: {n} images")

verify_dataset()
```

### Cell 8 — Transforms
```python
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])
eval_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])
```

### Cell 9 — SubsetWithTransform + train/val split
```python
class SubsetWithTransform(torch.utils.data.Dataset):
    """Wraps a Subset so each split can have its own transform."""
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label
    def __len__(self):
        return len(self.subset)

# Load raw dataset (no transform applied here — SubsetWithTransform handles it)
raw_ds = datasets.ImageFolder(Config.TRAIN_DIR)
val_size = int(len(raw_ds) * Config.VAL_SPLIT)
train_size = len(raw_ds) - val_size
generator = torch.Generator().manual_seed(Config.SEED)
train_sub, val_sub = random_split(raw_ds, [train_size, val_size], generator=generator)

train_ds = SubsetWithTransform(train_sub, train_transforms)
val_ds   = SubsetWithTransform(val_sub,   eval_transforms)
test_ds  = datasets.ImageFolder(Config.TEST_DIR, transform=eval_transforms)

class_names = raw_ds.classes
print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
print(f"Classes: {class_names}")
```

### Cell 10 — DataLoaders
```python
cfg = Config()
train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                          num_workers=cfg.NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE, shuffle=False,
                          num_workers=cfg.NUM_WORKERS, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=cfg.BATCH_SIZE, shuffle=False,
                          num_workers=cfg.NUM_WORKERS, pin_memory=True)
```

### Cell 11 — Sample image visualization (one per class, denormalized)

### Cell 12 — Class distribution bar charts (train / val / test)

---

## Pending Tasks

### Task 2 — Model definition + training loop
- `timm.create_model('efficientnet_b0', pretrained=True, num_classes=7)`
- Freeze backbone layers, attach classification head
- Training loop with validation, loss/accuracy tracking, cosine LR scheduler
- Save best checkpoint to `cfg.CKPT_DIR`

### Task 3 — Evaluation
- Accuracy, macro-F1, per-class precision/recall
- Confusion matrix heatmap (seaborn)
- Correct/incorrect sample visualizations

### Task 4 — Interval Priority Score (DL project only)
Rule-based scoring on top of inference output:
- **Inputs:** softmax confidence vector, predicted class, Shannon entropy
- **Logic:** Low confidence + high entropy → high priority flag
- **Output:** priority score per image, ranked table, flag statistics
- **Optional:** Grad-CAM visualizations for top-priority intervals

### Task 5 — Synthetic 3D Visualization (ML project only)
- Extract feature embeddings from EfficientNet-B0 penultimate layer
- Dimensionality reduction: PCA → UMAP → 3D coordinates
- Reversed-pyramid coordinate assignment per class
- Gaussian concentration field: `C(p) = Σ sᵢ exp(-||p - xᵢ||² / 2σ²)`
- 3D scatter/surface plot coloured by lithology class

---

## Shared Evaluation Metrics (both projects)
- Overall accuracy
- Macro-F1
- Per-class precision / recall / F1
- Confusion matrix heatmap

---

## Known Issues / Gotchas

1. **Google Drive mount in Colab** — can fail with `MessageError: credential propagation was unsuccessful`. Fix: re-run cell, enable third-party cookies in Chrome, or skip Drive and use `/content/checkpoints`.
2. **SubsetWithTransform is required** — `random_split` subsets share the underlying dataset's transform. Without the wrapper, both train and val get the same augmentation.
3. **Dataset download size** — DCID.zip is ~3.82 GB. Colab download takes several minutes; don't interrupt.
4. **HuggingFace xet storage** — use `hf_hub_download` (not `wget`) to reliably pull the zip.

---

## Key References
- Dataset: https://huggingface.co/datasets/168sir/drill-core-image-dataset
- timm docs: https://timm.fast.ai/
- EfficientNet paper: https://arxiv.org/abs/1905.11946
