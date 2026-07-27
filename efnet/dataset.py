"""
efnet/dataset.py
================
Dataset classes for EFNet training and evaluation.

Training dataset  : FolderFaceDataset — works with any root/label/img.jpg tree.
                    Handles both numeric folders (CASIA-WebFace: 0/, 1/, …)
                    and named folders (Alice/, Bob/, …) automatically.
Evaluation dataset: LFWVerificationDataset — Kaggle CSV format.
Enrollment dataset: EnrollmentDataset — few-shot reference images.

Normalisation: pixel values in [-1, 1] (mean=0.5, std=0.5 per channel).
This matches the convention used by ArcFace / InsightFace baselines.

Training augmentations:
  • RandomHorizontalFlip (p=0.5)
  • ColorJitter — lighting robustness
  • RandomGrayscale  (p=0.1)
  • RandomErasing    (p=0.2, small patches) — occlusion robustness
"""

import csv
from pathlib import Path
from typing import Optional, Tuple, List, Dict

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_train_transform() -> T.Compose:
    """Augmented transform for pre-training."""
    return T.Compose([
        T.Resize((112, 112)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.03),
        T.RandomGrayscale(p=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        T.RandomErasing(p=0.2, scale=(0.02, 0.10), ratio=(0.3, 3.0), value=0),
    ])


def get_val_transform() -> T.Compose:
    """Deterministic transform for evaluation and inference."""
    return T.Compose([
        T.Resize((112, 112)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


# ---------------------------------------------------------------------------
# 1. FolderFaceDataset  (CASIA-WebFace, custom datasets, etc.)
# ---------------------------------------------------------------------------

class FolderFaceDataset(Dataset):
    """
    Load face images from a directory tree where each sub-folder = one identity.

    Supports two naming conventions automatically:

    A) Numeric folders (CASIA-WebFace style):
            root/0/1.jpg  root/0/2.jpg  ...
            root/1/1.jpg  ...
            root/10571/...
       Folders are sorted by integer value, and the folder name IS the class
       index.  This ensures label 0 → folder "0", label 1 → folder "1", etc.
       (Alphabetical sort would produce 0,1,10,100,… which is wrong.)

    B) Named folders (custom / organisational datasets):
            root/Alice/img001.jpg ...
            root/Bob/img001.jpg   ...
       Folders are sorted alphabetically; class indices are assigned 0,1,2,…

    Detection: if ALL folder names are valid integers → numeric mode.
               Otherwise → named mode.

    Args:
        root       : Root directory containing per-identity sub-directories.
        transform  : torchvision transform. Defaults to get_train_transform().
        extensions : Accepted image file extensions.
    """

    IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    def __init__(self, root: str,
                 transform: Optional[T.Compose] = None,
                 extensions: Optional[set] = None):
        self.root      = Path(root)
        self.transform = transform or get_train_transform()
        exts           = extensions or self.IMG_EXTS

        # Collect all subdirectories
        subdirs = [d for d in self.root.iterdir()
                   if d.is_dir() and not d.name.startswith('.')]
        if not subdirs:
            raise RuntimeError(f"No sub-directories found under '{root}'.")

        # Decide sort mode
        numeric_mode = all(d.name.isdigit() for d in subdirs)

        if numeric_mode:
            # Sort by integer value; folder name = class label directly
            subdirs.sort(key=lambda d: int(d.name))
            self.classes      = [d.name for d in subdirs]
            self.class_to_idx = {d.name: int(d.name) for d in subdirs}
        else:
            # Sort alphabetically; assign sequential indices
            subdirs.sort(key=lambda d: d.name)
            self.classes      = [d.name for d in subdirs]
            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # Build flat (path, label) list
        self.samples: List[Tuple[Path, int]] = []
        for d in subdirs:
            label = self.class_to_idx[d.name]
            for f in d.iterdir():
                if f.suffix.lower() in exts:
                    self.samples.append((f, label))

        if not self.samples:
            raise RuntimeError(
                f"No images found under '{root}'. "
                "Check extensions and folder structure."
            )

        mode_str = "numeric-folder" if numeric_mode else "named-folder"
        print(f"[FolderFaceDataset] {mode_str} mode | "
              f"{len(self.classes):,} identities | "
              f"{len(self.samples):,} images | root='{root}'")

    @property
    def num_classes(self) -> int:
        """Max label + 1  (correct for both numeric and named modes)."""
        return max(self.class_to_idx.values()) + 1

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


# ---------------------------------------------------------------------------
# 2. LFW Verification dataset  (Kaggle CSV format)
# ---------------------------------------------------------------------------

class LFWVerificationDataset(Dataset):
    """
    LFW verification dataset — Kaggle jessicali9530/lfw-dataset format.

    Image path structure (after unzip):
        lfw_root/lfw-deepfunneled/lfw-deepfunneled/<name>/<name>_XXXX.jpg

    Pair files (all in lfw_root/):
        matchpairsDevTest.csv    — positive pairs  (same person)
        mismatchpairsDevTest.csv — negative pairs  (different people)

    matchpairs columns    : name, imagenum1, imagenum2
    mismatchpairs columns : name1, imagenum1, name2, imagenum2

    Args:
        lfw_root  : Directory that contains the CSVs AND the lfw-deepfunneled
                    sub-folder (i.e. /content/data/lfw).
        split     : 'DevTest' (default) or 'DevTrain'.
        transform : torchvision transform.
    """

    def __init__(self, lfw_root: str,
                 split: str = 'DevTest',
                 transform: Optional[T.Compose] = None,
                 # legacy kwargs — silently accepted for compatibility
                 pairs_file: str = '',
                 pairs_csv: bool = True):
        self.transform = transform or get_val_transform()
        self.pairs: List[Tuple[str, str, int]] = []

        root = Path(lfw_root)

        # Resolve image root — handles single or double nesting
        for candidate in [
            root / 'lfw-deepfunneled' / 'lfw-deepfunneled',
            root / 'lfw-deepfunneled',
            root,
        ]:
            if candidate.is_dir():
                self.img_root = candidate
                break
        else:
            raise RuntimeError(f"Cannot find image directory under '{lfw_root}'")

        self._load_csv(root, split)
        print(f"[LFW] {len(self.pairs)} pairs | split={split} | "
              f"images from '{self.img_root}'")

    def _img_path(self, name: str, num) -> str:
        return str(self.img_root / name / f"{name}_{int(num):04d}.jpg")

    def _load_csv(self, root: Path, split: str):
        """
        Load positive and negative pairs from the Kaggle LFW CSV files.

        matchpairsDevTest.csv columns  : name, imagenum1, imagenum2
        mismatchpairsDevTest.csv cols  : name, imagenum1, name, imagenum2
            ↑ TWO columns both named 'name' — DictReader would silently drop
            the second one, making name2 always empty → all negatives skipped
            → 100% fake accuracy. Fix: read mismatch CSV POSITIONALLY.

        pairs.csv columns              : name, imagenum1, imagenum2, (empty)
            This file contains only positive pairs and is used as a fallback
            when matchpairs{split}.csv is missing. It has a trailing empty
            column which DictReader exposes as key '' — we strip it.
        """
        n_pos = n_neg = 0

        # ── Positive pairs ────────────────────────────────────────────────
        # Prefer split-specific file; fall back to pairs.csv (all positives)
        for fname in [f'matchpairs{split}.csv', 'pairs.csv']:
            p = root / fname
            if not p.exists():
                continue
            with open(p, newline='') as f:
                reader = csv.reader(f)
                header = [h.strip() for h in next(reader)]
                # columns: name, imagenum1, imagenum2[, optional empty]
                for row in reader:
                    if len(row) < 3:
                        continue
                    name, n1, n2 = row[0].strip(), row[1].strip(), row[2].strip()
                    if not (name and n1 and n2):
                        continue
                    p1 = self._img_path(name, n1)
                    p2 = self._img_path(name, n2)
                    if Path(p1).exists() and Path(p2).exists():
                        self.pairs.append((p1, p2, 1))
                        n_pos += 1
            break   # stop after first found file

        # ── Negative pairs — read POSITIONALLY ───────────────────────────
        # Columns: name, imagenum1, name, imagenum2   (two 'name' columns!)
        # Positional indices: 0=name1, 1=n1, 2=name2, 3=n2
        neg = root / f'mismatchpairs{split}.csv'
        if neg.exists():
            with open(neg, newline='') as f:
                reader = csv.reader(f)
                next(reader)   # skip header
                for row in reader:
                    if len(row) < 4:
                        continue
                    name1 = row[0].strip()
                    n1    = row[1].strip()
                    name2 = row[2].strip()
                    n2    = row[3].strip()
                    if not (name1 and n1 and name2 and n2):
                        continue
                    p1 = self._img_path(name1, n1)
                    p2 = self._img_path(name2, n2)
                    if Path(p1).exists() and Path(p2).exists():
                        self.pairs.append((p1, p2, 0))
                        n_neg += 1

        if not self.pairs:
            raise RuntimeError(
                f"No valid pairs found in '{root}' for split='{split}'.\n"
                f"  Expected: matchpairs{split}.csv + mismatchpairs{split}.csv\n"
                "  Check that image files exist under lfw-deepfunneled/."
            )

        # Warn if one side is empty — symptom of the path or CSV problem
        if n_pos == 0:
            print(f"  [LFW Warn] No positive pairs loaded — check matchpairs{split}.csv")
        if n_neg == 0:
            print(f"  [LFW Warn] No negative pairs loaded — check mismatchpairs{split}.csv")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        p1, p2, same = self.pairs[idx]
        img1 = self.transform(Image.open(p1).convert('RGB'))
        img2 = self.transform(Image.open(p2).convert('RGB'))
        return img1, img2, same


# ---------------------------------------------------------------------------
# 3. EnrollmentDataset  (few-shot reference images for inference)
# ---------------------------------------------------------------------------

class EnrollmentDataset(Dataset):
    """
    Load reference images for few-shot enrollment.

        root/
            Alice/  ref1.jpg  ref2.jpg  ref3.jpg
            Bob/    ref1.jpg  ref2.jpg

    Returns (image_tensor, identity_name, image_path).
    """

    IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    def __init__(self, root: str, transform: Optional[T.Compose] = None):
        self.root      = Path(root)
        self.transform = transform or get_val_transform()
        self.samples: List[Tuple[Path, str]] = []

        for d in sorted(self.root.iterdir()):
            if not d.is_dir():
                continue
            for f in sorted(d.iterdir()):
                if f.suffix.lower() in self.IMG_EXTS:
                    self.samples.append((f, d.name))

        identities = sorted({s[1] for s in self.samples})
        print(f"[EnrollmentDataset] {len(identities)} identities | "
              f"{len(self.samples)} reference images | root='{root}'")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, name = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, name, str(path)
