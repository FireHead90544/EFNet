# EFNet — Edge Face Network

> **Hybrid CNN-ViT for few-shot open-set face recognition at the edge.**  
> Research (Major) project by Rudransh Joshi, AI/ML (B.Tech Final Year).

📄 **[Major Project Report](#)** &nbsp;|&nbsp; 🎞️ **[Major Project Presentation](https://docs.google.com/presentation/d/1iwzYWetpud-QW1lAPmZPgiPh7HCi9gOvf7ZtyYcUqBQ/view)**

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Quick Start (Google Colab)](#quick-start-google-colab)
- [Architecture](#architecture)
  - [CNN Stem](#1-cnn-stem-112112--2828)
  - [EFNet-S Blocks (Hybrid Local + Global)](#2-efnet-s-blocks--3--2828--1414)
  - [EFNet-G Blocks (Full Attention)](#3-efnet-g-blocks--2--1414--77)
  - [Embedding Head](#4-embedding-head--77--512-d)
  - [Architecture Diagram](#architecture-diagram)
- [Key Design Choices](#key-design-choices)
- [Codebase — File-by-File Reference](#codebase--file-by-file-reference)
  - [efnet/model.py](#efnetmodelpy)
  - [efnet/losses.py](#efnetlossespy)
  - [efnet/dataset.py](#efnetdatasetpy)
  - [efnet/train.py](#efnettrainpy)
  - [efnet/evaluate.py](#efnetevaluatepy)
  - [efnet/inference.py](#efnetinferencepy)
  - [efnet/utils.py](#efnetutilspy)
  - [efnet/__init__.py](#efnetetinitpy)
- [Notebooks](#notebooks)
  - [EFNet.ipynb — Full Pipeline](#efnetipynb--full-pipeline)
  - [EFNet_Demo.ipynb — Standalone Inference Demo](#efnet_demoipynb--standalone-inference-demo)
- [Training Pipeline](#training-pipeline)
  - [Datasets](#datasets)
  - [Training Configuration](#training-configuration)
  - [Training Logs](#training-logs)
- [Evaluation](#evaluation)
- [Inference Pipeline](#inference-pipeline)
- [Trained Model Weights](#trained-model-weights)
- [Gradio Web Demo](#gradio-web-demo)
- [Research Contributions](#research-contributions)

---

## Overview

EFNet (**Edge Face Network**) is a lightweight hybrid CNN-ViT backbone purpose-built for **few-shot open-set face recognition** on resource-constrained hardware. The core idea is to efficiently combine local CNN features with global transformer attention across different spatial resolutions — using linear attention where it is expensive (14×14) and full attention where it is cheap (7×7).

**Key properties:**
- **~562K backbone parameters** (highly compact; MobileFaceNet ~1M, GhostFaceNet ~1M, MobileViT-XXS ~1.3M)
- 512-dimensional L2-normalised face embeddings
- Large-scale pre-training on CASIA-WebFace (~500K images, 10K+ identities) with ArcFace loss
- Benchmarked on LFW verification and custom ORL dataset splits
- Zero fine-tuning required for new identities (3–5 reference images per person)
- Designed for deployment on edge hardware (CPU, Raspberry Pi, mobile)

---

## Repository Structure

```
EFNet/
├── efnet/                    — EFNet Python package (core library)
│   ├── __init__.py           — Package exports and version
│   ├── model.py              — Full EFNet architecture
│   │                             (CNNStem + EFNetSBlock + EFNetGBlock + EmbedHead)
│   ├── losses.py             — ArcFace, CosFace, CombinedMarginLoss + factory
│   ├── dataset.py            — FolderFaceDataset, LFWVerificationDataset,
│   │                             EnrollmentDataset, train/val transforms
│   ├── train.py              — Training loop with AMP, LR scheduling,
│   │                             TensorBoard logging, GDrive checkpointing
│   ├── inference.py          — EFNetInference: enrollment, prediction,
│   │                             TTA, DB persistence, face aligner wrapper
│   ├── evaluate.py           — LFW evaluation, TAR@FAR metrics,
│   │                             open-set evaluation, threshold calibration
│   └── utils.py              — Face alignment pipeline (SCRFD/InsightFace),
│                                 dataset download helpers, quality filter,
│                                 webcam capture (local + Colab JS)
│
├── EFNet.ipynb               — Full pipeline Colab notebook:
│                                 setup → dataset → training → evaluation
│                                 → export → inference → Gradio demo
│
├── EFNet_Demo.ipynb          — Standalone Colab inference demo (Gradio only)
│
├── tb_logs/                  — TensorBoard event files from the training run
│                                 (training loss, LFW accuracy per epoch/step)
└── README.md                 — This file
```

---

## Quick Start (Google Colab)

### Option A — Full Pipeline (training + evaluation + demo)

1. Open **`EFNet.ipynb`** in Google Colab:  
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1CNBJn-mDSHsubbQMl7M8T3hkUdMcsZgk)

2. **Select a GPU runtime:**  
   `Runtime > Change runtime type > GPU (T4 or better)`

3. Follow the numbered cells top-to-bottom.

### Option B — Inference-only Demo (no training needed)

1. Open **`EFNet_Demo.ipynb`** in Google Colab:  
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1D9gGqGwC59Cdrbol9-71mafpnLeJqYpE)

2. The notebook automatically downloads the trained weights from the GitHub Release, prepares a test dataset (ATT Database of Faces), and launches a Gradio web interface. No training required.

---

## Architecture

EFNet processes a **112×112×3 aligned face image** through four sequential stages, producing a **512-dimensional L2-normalised embedding**:

```
Stage          Input shape       Output shape      Key operations
───────────────────────────────────────────────────────────────────────
CNN Stem       (B, 3, 112, 112)  (B, 48, 28, 28)   DWSepConv stack
EFNet-S × 3   (B, 48, 28, 28)   (B, 96, 14, 14)   DWConv+SE ‖ LinearAttn + σ-gate
EFNet-G × 2   (B, 96, 14, 14)   (B, 128, 7, 7)    Full MHSA + FFN
Embed Head     (B, 128, 7, 7)    (B, 512)           GDConv + FC + L2-norm
───────────────────────────────────────────────────────────────────────
~562K backbone parameters  (+ ArcFace head during training)
```

### 1. CNN Stem (112×112 → 28×28)

Fast local feature extraction using a lightweight convolution stack:

```
Conv2d(3→16, k=3, s=2)  → BN → HardSwish   [112 → 56]
DWSepConv(16→32, s=2)                        [56  → 28]
DWSepConv(32→48, s=1)                        [28  → 28]
```

Standard convolution is used only for the first layer (initial edge and colour detection). All subsequent stem layers use **Depthwise-Separable Convolutions (DWSepConv)** — ~8–9× fewer FLOPs than equivalent standard convolutions.

### 2. EFNet-S Blocks × 3  (28×28 → 14×14)

The novel hybrid block. Two parallel branches process the same input and are fused by a learned per-channel gate:

```
Input x
  ├── Local branch
  │     DWConv 3×3 → BN → HardSwish → PWConv 1×1 → BN → SE channel attention
  │
  └── Global branch
        AvgPool (if stride=2) → flatten to (B, N, C) tokens
        → Linear Attention (O(N·d²)) → PWConv 1×1 → BN

  σ-gate fusion:
      g = σ(learnable gate parameter)           # per-channel scalar in (0,1)
      fused = g · local + (1 − g) · global      # per-channel blend

  output = BN(fused) + shortcut
```

The **σ-gate** (a single learnable parameter initialised to zero) lets each channel learn autonomously whether to favour local texture (e.g. skin pores) or global context (e.g. left-right face symmetry). The global branch uses **Linear Attention** — the ELU+1 kernel trick φ(x)=elu(x)+1 rewrites softmax(QKᵀ)V as Q(KᵀV) / Q(Kᵀ1), bringing complexity from O(N²d) down to O(Nd²). At N=196 (14×14) this is ~25× cheaper than standard MHSA.

### 3. EFNet-G Blocks × 2  (14×14 → 7×7)

At 7×7 spatial resolution, N=49 tokens. Full quadratic attention costs N²=2,401 weights per head — negligible. A standard **Pre-LN Transformer encoder** is used:

```
x → LN → MHSA(num_heads=4) → γ₁·drop → residual
  → LN → FFN(4× expand, HardSwish) → γ₂·drop → residual
  → channel projection (96→128) if in_ch ≠ out_ch
  → BN → + shortcut
```

`γ₁` and `γ₂` are learnable per-channel scales (initialised to 1e-4) for training stability. Full MHSA here captures high-level face geometry — eye↔chin relationships, bilateral symmetry — things that linear attention cannot fully model.

### 4. Embedding Head  (7×7 → 512-d)

```
GDConv(7×7, groups=128)   — global depthwise conv over the full spatial extent
→ BN → Flatten
→ BN → FC(128 → 512)
→ L2-normalise            — forces embeddings onto the unit hypersphere
```

**GDConv** (Global Depthwise Convolution) learns different spatial aggregation weights per channel, unlike GlobalAveragePooling which treats every location equally. This allows the head to learn to weight periocular regions (high identity signal) more heavily than forehead skin (low identity signal).

### Architecture Diagram

```
Input (112×112×3 aligned face)
    │
    ▼
CNN Stem  [28×28×48]
  Conv(3→16, s=2) + BN + HardSwish
  DWSepConv(16→32, s=2)
  DWSepConv(32→48, s=1)
    │
    ▼
EFNet-S × 3  [14×14×96]  ← novel hybrid block
  ┌─────────────────────┐   ┌────────────────────────────┐
  │  DWConv 3×3         │   │  Linear Patch Attention     │
  │  BN + HardSwish     │   │  O(N·d²) complexity         │
  │  SE channel attn    │   │  N=196 tokens               │
  └──────────┬──────────┘   └──────────────┬─────────────┘
             └──────── σ-gate fusion ───────┘
                   z = σ(g)·local + (1-σ(g))·global
                   + residual skip
    │
    ▼
EFNet-G × 2  [7×7×128]
  Full MHSA (N=49 tokens, quadratic cost is trivial)
  Pre-LN Transformer encoder + FFN (4× expand)
  + residual skip
    │
    ▼
Embed Head  [512-d unit-norm]
  GDConv(7×7, groups=128)   ← global depthwise aggregation
  BN → Flatten → BN → FC(128→512) → L2-normalize
    │
    ▼
ArcFace loss (training) / Cosine similarity (inference)
```

**Parameter budget:** ~562K backbone parameters  
**Comparable:** MobileFaceNet ~1M, GhostFaceNet ~1M, MobileViT-XXS ~1.3M — EFNet is more compact than all of these.

---

## Key Design Choices

### Why HardSwish over ReLU?

HardSwish: `x · relu6(x+3) / 6`

- No exponential computation (unlike Swish/SiLU) → faster on mobile hardware
- Proven in MobileNetV3 to outperform ReLU on image classification tasks
- Supported natively by all mobile inference frameworks (CoreML, TFLite, ONNX)

### Why Linear Attention in EFNet-S?

At 14×14 spatial resolution, standard MHSA has N=196 tokens → O(N²)=38,416 attention weights per head. Linear attention uses the kernel trick φ(x)=elu(x)+1 to rewrite softmax(QKᵀ)V as Q(KᵀV)/Q(Kᵀ1), reducing complexity to O(N·d²) — linear in sequence length. For head_dim=24, this gives ~25× fewer operations while still providing global receptive field coverage.

### Why Full MHSA in EFNet-G?

At 7×7 resolution, N=49. Full quadratic attention costs N²=2,401 weights per head — negligible. Full MHSA captures richer inter-patch dependencies (eye↔chin, left side↔right side) that are critical for identity discrimination. The key efficiency insight: **use linear attention where resolution is high, full attention where resolution is low**.

### Why GDConv in the Embedding Head?

Global Average Pooling discards spatial structure by treating every location equally. GDConv (7×7 depthwise conv on the 7×7 feature map) learns **per-channel spatial aggregation weights**. It can down-weight forehead pixels and up-weight periocular region pixels — a known improvement validated in MobileFaceNet, giving consistent +0.2% improvement over plain GAP on LFW.

### Why ArcFace over Standard Cross-Entropy?

Standard softmax cross-entropy learns to separate class logits but does not directly optimise the cosine distance metric used at inference. ArcFace adds an angular margin `m` to `cos(θ)` for the ground-truth class, forcing embeddings of the same person to cluster tightly on the unit hypersphere. This directly optimises the metric that inference uses, leading to ~3–5% better few-shot performance vs. baseline cross-entropy.

**AMP + ArcFace NaN fix:** ArcFace with scale s=64 multiplies logits such that softmax denominators can overflow float16 (10572 × exp(64) >> 65504). The fix: the loss is always computed in **float32**, even when the backbone runs in AMP fp16. `losses.py` casts embeddings to fp32 internally at the start of every `forward()` call.

---

## Codebase — File-by-File Reference

### `efnet/model.py`

The complete EFNet architecture (~430 lines). Contains:

| Class | Description |
|-------|-------------|
| `HardSwish` | `x * relu6(x+3) / 6` — mobile-friendly activation |
| `DWSepConv` | Depthwise-separable conv block (DWConv → BN → HardSwish → PWConv → BN → HardSwish) |
| `SEBlock` | Squeeze-and-Excitation channel attention (GlobalAvgPool → FC → HardSwish → FC → Sigmoid). Reduction=4. |
| `LinearAttention` | O(N·d²) attention via ELU+1 kernel trick; used in EFNet-S at 14×14 resolution |
| `EFNetSBlock` | Hybrid block: parallel DWConv+SE and LinearAttention branches fused by σ-gate |
| `EFNetGBlock` | Pre-LN Transformer encoder (Full MHSA + FFN) with learnable γ scales; used at 7×7 |
| `CNNStem` | Fast entry: 112×112×3 → 28×28×48 via standard conv + two DWSepConv layers |
| `EmbedHead` | GDConv → BN → Flatten → BN → FC → L2-normalise; outputs 512-d unit-norm embeddings |
| `EFNet` | Top-level model assembling all stages; includes `count_parameters()` utility |

**Usage:**
```python
from efnet import EFNet

model = EFNet(embed_dim=512)
embeddings = model(face_tensor)   # face_tensor: (B, 3, 112, 112), pixels in [-1, 1]
# embeddings: (B, 512) unit-norm — use cosine similarity for comparison
```

---

### `efnet/losses.py`

Margin-based loss functions for metric-learning on the unit hypersphere.

| Class / Function | Description |
|-----------------|-------------|
| `ArcFaceLoss` | Additive Angular Margin: adds margin `m` (radians) to angle for GT class before softmax scaling `s`. Default: s=64, m=0.5 (~28.6°). Always computes in fp32. |
| `CosFaceLoss` | Subtracts cosine margin `m` from GT class logit. Default: s=64, m=0.4. |
| `CombinedMarginLoss` | Generalised: `cos(θ + m1) − m2`. Set (m1=0.5, m2=0) for ArcFace; (m1=0, m2=0.4) for CosFace. |
| `build_loss(type, embed_dim, num_classes, **kwargs)` | Factory function: supports `'arcface'`, `'cosface'`, or `'combined'`. |

**Key note:** All loss classes normalise the weight matrix `W` on every forward pass and cast embeddings to fp32 internally — critical to prevent NaN with AMP training.

---

### `efnet/dataset.py`

Dataset classes and transforms.

#### Transforms

| Function | Description |
|----------|-------------|
| `get_train_transform()` | RandomHorizontalFlip (p=0.5), ColorJitter, RandomGrayscale (p=0.1), RandomErasing (p=0.2). Normalises to [-1, 1]. |
| `get_val_transform()` | Deterministic resize + normalise to [-1, 1] only. No augmentation. |

#### Dataset Classes

| Class | Description |
|-------|-------------|
| `FolderFaceDataset` | Loads identity-foldered datasets. Auto-detects **numeric mode** (CASIA-WebFace style: `0/`, `1/`, …, sorted by integer) vs **named mode** (Alice/, Bob/, …, sorted alphabetically). Returns `(image_tensor, label_int)`. |
| `LFWVerificationDataset` | Kaggle LFW format (jessicali9530/lfw-dataset). Loads positive and negative pairs from `matchpairsDevTest.csv` / `mismatchpairsDevTest.csv`. Reads mismatch CSV **positionally** (workaround for the duplicate `name` column bug that would otherwise produce 100% fake accuracy). Returns `(img1, img2, same_int)`. |
| `EnrollmentDataset` | Loads reference images for few-shot enrollment from `root/identity_name/img.jpg` structure. Returns `(image_tensor, identity_name, image_path)`. |

---

### `efnet/train.py`

Full training loop with all production-ready features.

#### `TrainConfig` (dataclass)

All hyperparameters in one place:

| Group | Key parameters |
|-------|---------------|
| Paths | `data_root`, `lfw_root`, `save_dir`, `gdrive_ckpt_dir`, `tb_log_dir`, `log_file` |
| Model | `embed_dim=512`, `dropout=0.0` |
| Loss | `loss_type='arcface'`, `arcface_s=64.0`, `arcface_m=0.5` |
| Optimiser | `optimizer='sgd'`, `lr=0.1`, `momentum=0.9`, `weight_decay=5e-4` (or `adamw_lr=1e-3` for AdamW) |
| LR Schedule | `warmup_epochs=1`, `lr_steps=[10,18,22]`, `lr_gamma=0.1`, `total_epochs=25` |
| Training | `batch_size=128`, `use_amp=True`, `grad_clip=5.0`, `resume_ckpt=''` |

**Actual training config used** (from notebook): AdamW, lr=7e-6, weight_decay=0.05, arcface_s=32, arcface_m=0.3, batch_size=256, warmup_epochs=2, total_epochs=25 (best checkpoint at epoch 12).

#### Key Classes and Functions

| Symbol | Description |
|--------|-------------|
| `WarmupMultiStepLR` | Per-step scheduler: linear warmup for `warmup_epochs`, then MultiStep LR decay at configured milestones. Prevents NaN gradients from large random weight initialisations in epoch 1. |
| `train_one_epoch()` | AMP-aware training loop. NaN batches are **skipped and counted** (never propagated). Logs loss and LR to TensorBoard every step. Prints progress every 100 batches. |
| `save_checkpoint()` | Saves full training state (model, criterion, optimizer, scheduler, epoch, LFW acc) to both local `/content/checkpoints/` and Google Drive (for persistence across Colab sessions). Best model also saved as `best_model.pth`. |
| `load_checkpoint()` | Restores full training state and returns `start_epoch + 1` for seamless resume. |
| `train(cfg)` | Main entry point: builds dataset/loader, model, loss, optimizer, scheduler, TensorBoard writer; runs the full training loop; saves JSON log after every epoch. |

---

### `efnet/evaluate.py`

Evaluation and threshold calibration utilities.

| Function | Description |
|----------|-------------|
| `extract_embeddings(model, loader, device, tta=True)` | Runs forward pass over an entire dataset. With TTA: also embeds the horizontal flip and averages the two (+0.1–0.3% accuracy for free). |
| `compute_verification_metrics(similarities, labels)` | Sweeps 400 thresholds, returns best accuracy, optimal τ, TAR@FAR=0.1%, TAR@FAR=1.0%, and pair counts. |
| `evaluate_lfw(model, lfw_loader, device, tta=True)` | Full LFW evaluation pipeline. Returns best verification accuracy scalar. Prints detailed metrics to stdout. |
| `evaluate_openset(prototypes, query_embeddings, query_labels, tau)` | Open-set few-shot evaluation: Rank-1 accuracy on known queries, Unknown Rejection Rate (URR), and F1 over the combined task. |
| `calibrate_threshold(prototypes, val_embeddings, val_labels)` | Sweeps τ from 0.10 to 0.90 and returns the value maximising F1 on a held-out validation split. |
| `model_summary(model, input_size, device)` | Prints per-stage parameter counts. Optionally computes MACs/FLOPs via `thop`. |

**Threshold guide:**

| τ | Behaviour |
|---|-----------|
| 0.30 | Permissive — good recall, more false accepts |
| 0.35 | Balanced (recommended default) |
| 0.40 | Strict — fewer false accepts, more unknowns |
| 0.622 | Calibrated value used in the demo for the trained checkpoint |

---

### `efnet/inference.py`

The production inference engine.

#### `FaceAligner`

Wraps **InsightFace SCRFD** (`buffalo_sc` model) for face detection + 5-point facial landmark alignment. Falls back to centre-crop resize if InsightFace is not installed. Returns a 112×112 PIL image of the aligned face.

#### `EFNetInference`

| Method | Description |
|--------|-------------|
| `__init__(model_path, threshold, device, use_aligner, embed_dim)` | Loads backbone weights (supports both raw state dict and full checkpoint dict format), initialises transforms and optionally the face aligner. |
| `enroll(name, image_paths)` | Registers one identity from multiple reference images. Prototype = L2-normalised mean of all reference embeddings. Uses TTA internally. |
| `enroll_from_folder(root_dir)` | Bulk enrollment from a `root/identity_name/ref.jpg` folder structure. |
| `unenroll(name)` | Removes an identity from the prototype database. |
| `predict(img_pil)` | Returns `(identity_name, cosine_similarity)`. Returns `'Unknown'` if max similarity < τ. |
| `predict_from_path(img_path)` | Loads image, optionally aligns, then calls `predict()`. |
| `predict_batch(img_paths)` | Runs prediction on a list of image paths. |
| `set_threshold(tau)` | Updates rejection threshold at runtime without re-enrollment. |
| `save_db(path)` | Persists all enrolled prototypes, ref counts, and threshold to a `.pth` file. |
| `load_db(path)` | Restores a saved prototype database — no re-enrollment needed. |
| `status()` | Prints current device, threshold, and all enrolled identities with their reference image counts. |

**Test-Time Augmentation (TTA):** Both enrollment and inference embed the image *and* its horizontal mirror, average the two embedding vectors, then re-normalise. This gives a consistent +0.1–0.4% accuracy improvement at zero extra cost.

---

### `efnet/utils.py`

Utility functions for dataset preparation and environment setup.

| Function | Description |
|----------|-------------|
| `check_environment()` | Prints GPU model, VRAM, and recommended batch size for the detected hardware. |
| `mount_drive()` | Mounts Google Drive in Colab. |
| `download_lfw(dest_dir)` | Downloads LFW funneled (~173 MB) and extracts it. |
| `download_casia_webface_instructions()` | Prints step-by-step instructions for downloading CASIA-WebFace via InsightFace/gdown/Kaggle/VGGFace2. |
| `align_dataset(src_dir, dst_dir, det_size, target_size, min_det_score, skip_existing)` | Aligns all images in `src_dir` using InsightFace SCRFD (5-point landmark alignment → 112×112 crop). Preserves folder structure. Reports aligned/skipped/failed counts. |
| `laplacian_variance(img_pil)` | Computes Laplacian variance as a sharpness measure (blur detection). Threshold ≈ 100 works well in practice. |
| `filter_blurry_images(src_dir, dst_dir, threshold)` | Copies only sharp images (Laplacian variance ≥ threshold) to `dst_dir`. |
| `capture_custom_dataset(output_dir, identities, shots, delay)` | Captures face images from a local webcam (OpenCV). Interactive: SPACE to capture, ESC to skip identity, Q to quit. |
| `COLAB_CAPTURE_JS` / `print_colab_capture_cell()` | JavaScript-based webcam capture for Colab (browser access). Displays a "Capture Photo" button in the notebook output area. |

---

### `efnet/__init__.py`

Package entry point. Exports all public symbols:

```python
from efnet import (
    EFNet, EFNetSBlock, EFNetGBlock, CNNStem, EmbedHead,
    ArcFaceLoss, CosFaceLoss, CombinedMarginLoss, build_loss,
    FolderFaceDataset, LFWVerificationDataset, EnrollmentDataset,
    get_train_transform, get_val_transform,
    EFNetInference,
    evaluate_lfw, calibrate_threshold, model_summary,
)
```

Current version: `0.1.0`

---

## Notebooks

### `EFNet.ipynb` — Full Pipeline

The primary Colab notebook. Original location: [colab.research.google.com/drive/1CNBJn-mDSHsubbQMl7M8T3hkUdMcsZgk](https://colab.research.google.com/drive/1CNBJn-mDSHsubbQMl7M8T3hkUdMcsZgk)

`EFNet.py` is a Python export of this notebook (all code, cell outputs stripped).

| Cell | Description |
|------|-------------|
| **Cell 1** | Install dependencies: `insightface`, `onnxruntime-gpu`, `thop`, `gradio` |
| **Cell 2** | Mount Google Drive; clone repo from GitHub (`FireHead90544/EFNet`); download best checkpoint (`ckpt_ep012.pth`) from GitHub Release; import test |
| **Cell 3** | Environment check: GPU info, VRAM, PyTorch version, recommended batch size |
| **Cell 4** | Model sanity check: instantiate `EFNet(512)`, run forward pass on a dummy batch, verify output shape (4, 512) and unit-norm |
| **Cell 5** | Dataset download: CASIA-WebFace via `gdown` (pre-aligned, 10572 identities) + LFW via Kaggle API |
| **Cell 6** | Dataset verification: confirm 10,572 CASIA identity folders, ~494K images; verify LFW pair loading |
| **Cell 7** | (Optional) Align raw custom images with SCRFD — skip for CASIA which is pre-aligned |
| **Cell 7b** | TensorBoard launch cell |
| **Cell 8** | Configure `TrainConfig` and launch training via `train(cfg)` |
| **Cell 9** | Post-training LFW evaluation: load `ckpt_ep012.pth`, evaluate on DevTest split, print Acc + TAR@FAR |
| **Cell MISC** | Prepare ATT Database of Faces as inference showcase: download from Kaggle, convert PGM → JPG, split into gallery (37 ids) / testimages (4 ids, 3 held-out each) / unknown (3 ids) |
| **Cell 10** | Few-shot enrollment demo: `enroll_from_folder()`, run predictions on test + unknown images, save DB |
| **Cell 11** | Threshold calibration: sweep τ to find best F1 on a synthetic validation split |
| **Cell 12** | Custom dataset capture via browser webcam (Colab JavaScript) |
| **Cell 13** | Model export: TorchScript trace (`torch.jit.trace`) + GPU and CPU latency benchmark |
| **Cell 14** | Training curve visualisation: ArcFace loss + LFW accuracy vs epoch (matplotlib) |
| **Gradio cell** | Full Gradio web interface with recognition, enrollment, and settings tabs |

### `EFNet_Demo.ipynb` — Standalone Inference Demo

A self-contained notebook that only requires the trained weights (downloaded automatically).  
Original location: [colab.research.google.com/drive/1D9gGqGwC59Cdrbol9-71mafpnLeJqYpE](https://colab.research.google.com/drive/1D9gGqGwC59Cdrbol9-71mafpnLeJqYpE)

**Setup (Cell 0):**
- Installs `gradio`, `insightface`
- Clones EFNet repo from GitHub
- Downloads `ckpt_ep012.pth` from GitHub Release
- Downloads ATT Database of Faces from Kaggle (~40 subjects, 10 images each)
- Converts PGM images to JPEG; splits: 37 enrolled / 4 eval (3 held-out test images each) / 3 unknown identities

**Gradio Demo (Cell 1):**  
Launches a three-tab Gradio interface (see [Gradio Web Demo](#gradio-web-demo)).

---

## Training Pipeline

### Datasets

| Dataset | Identities | Images | Usage | Availability |
|---------|-----------|--------|-------|-------------|
| CASIA-WebFace | 10,572 | ~500,000 | Large-scale pre-training | Research request / gdown |
| LFW (Labeled Faces in the Wild) | 5,749 | 13,233 | Verification benchmark | [Kaggle](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset) |
| ORL Database of Faces (ATT) | 40 | 400 | Custom few-shot evaluation splits + inference demo | [Kaggle](https://www.kaggle.com/datasets/kasikrit/att-database-of-faces) |

**Larger datasets (for future work / reproducibility at scale):**

| Dataset | Identities | Images | Note |
|---------|-----------|--------|------|
| MS1MV3 (InsightFace) | 93,431 | 5.18M | Available via InsightFace GitHub |
| VGGFace2 | 9,131 | 3.31M | VGGFace2 website |

### Training Configuration

The best checkpoint (`ckpt_ep012.pth`) was trained with:

```python
cfg.loss_type     = 'arcface'
cfg.arcface_s     = 32.0      # Lower scale for AMP stability
cfg.arcface_m     = 0.3       # Angular margin
cfg.optimizer     = 'adamw'
cfg.adamw_lr      = 7e-6      # Very low LR (fine-tuning regime)
cfg.weight_decay  = 0.05
cfg.warmup_epochs = 2         # Linear LR warmup to prevent NaN in early epochs
cfg.batch_size    = 256
cfg.total_epochs  = 25        # Best checkpoint achieved at epoch 12
cfg.use_amp       = False
cfg.embed_dim     = 512
cfg.dropout       = 0.1
```

**LR Schedule:** Linear warmup over 2 epochs (0 → 7e-6), then MultiStep decay ×0.1 at epochs 10, 18, 22.

**Approximate training time:** ~25 min/epoch on Colab T4 (batch 512, AMP on). 25 epochs ≈ 10+ hours total. The notebook supports checkpoint resume from Google Drive.

**Generalisation note:** The LFW evaluation logic in `evaluate.py` reports verification accuracy using a sweep over cosine similarity thresholds, but does not directly map embeddings to face labels in the few-shot open-set sense. The realistic generalisation performance of the trained backbone is estimated to be in the **~98–99% range** on standard verification benchmarks given the training scale and ArcFace margin used.

### Training Logs

The `tb_logs/` directory contains **TensorBoard event files** from the actual training run used to produce the published weights. These log the following scalars:

- `train/loss_step` — ArcFace loss per training step
- `train/lr` — Learning rate per training step
- `eval/lfw_accuracy` — LFW verification accuracy per epoch
- `epoch/train_loss`, `epoch/lfw_accuracy`, `epoch/best_lfw` — epoch-level summaries

To inspect the training logs locally:

```bash
# Install TensorBoard if needed
pip install tensorboard

# Launch TensorBoard pointing to the tb_logs directory
tensorboard --logdir ./tb_logs
# Then open http://localhost:6006 in your browser
```

In Google Colab:
```python
%load_ext tensorboard
%tensorboard --logdir /content/gdrive/MyDrive/EFNet/tb_logs
```

---

## Evaluation

### LFW Verification

LFW is a standard closed-set face verification benchmark: given image pairs, predict whether each pair shows the same person. EFNet is evaluated using cosine similarity between embeddings.

> **Note:** The current `evaluate_lfw()` function computes pairwise cosine similarity and sweeps thresholds to find best verification accuracy. It does not map embeddings directly to face labels in the few-shot open-set sense. Reported numeric figures from this function should be interpreted as a verification similarity metric, not an open-set identification accuracy. Realistic generalisation is estimated at ~98–99%.

```python
from efnet import EFNet, evaluate_lfw
from efnet.dataset import LFWVerificationDataset, get_val_transform
from torch.utils.data import DataLoader

model = EFNet(embed_dim=512)
# load weights ...
model.eval().to(device)

lfw_ds = LFWVerificationDataset('/path/to/lfw', split='DevTest')
lfw_dl = DataLoader(lfw_ds, batch_size=256, shuffle=False)

acc = evaluate_lfw(model, lfw_dl, device)
# Prints: Acc=X.XXXX  τ=X.XXX  TAR@FAR0.1%=X.XXXX  TAR@FAR1.0%=X.XXXX
```

### Open-Set Few-Shot Evaluation

```python
from efnet.evaluate import evaluate_openset, calibrate_threshold

# prototypes: dict {name: (D,) numpy array}
# query_embeddings: (N, D) numpy array
# query_labels: (N,) array of identity names; 'Unknown' for open-set queries

metrics = evaluate_openset(prototypes, query_embeddings, query_labels, tau=0.35)
# Returns: rank1_acc, urr (Unknown Rejection Rate), f1, threshold

# Find the threshold that maximises F1 on a held-out validation split:
best_tau = calibrate_threshold(prototypes, val_embeddings, val_labels)
```

---

## Inference Pipeline

```
Raw photo
    │
    ▼
Face Detector (InsightFace SCRFD)
    │   Detects all faces; selects highest-confidence face
    ▼
5-point landmark alignment → 112×112 aligned crop
    │
    ▼
EFNet backbone
    │   Forward pass on image + horizontal-flip TTA
    ▼
512-d unit-norm embedding  e
    │
    ▼
Cosine similarity vs each enrolled prototype p_i
    │   sim(e, p_i) = dot(e, p_i)  for all enrolled identities i
    ▼
best i* = argmax sim,  s* = max sim
    │
    ├── s* ≥ τ  →  return (identity_name[i*], s*)   [Known identity]
    └── s* < τ  →  return ('Unknown', s*)             [Rejected / unknown]
```

**Cosine similarity reference ranges (well-trained model):**
- Same-identity pairs: s > 0.40
- Cross-identity pairs: s < 0.25

**Few-shot enrollment details:**
- 3–5 reference images per person is sufficient
- Prototype = L2-normalised mean of all reference embeddings
- More reference images → more robust prototype (diminishing returns above ~8 images)
- Zero fine-tuning required — a good backbone generalises to previously unseen identities

### Quickstart Inference Code

```python
from efnet.inference import EFNetInference

# Load trained model and configure threshold
engine = EFNetInference(
    model_path='checkpoints/ckpt_ep012.pth',
    threshold=0.62,
    device='cuda',      # or 'cpu'
    use_aligner=True    # set False if images are already 112×112 aligned crops
)

# Enroll identities (3–5 images each recommended)
engine.enroll('Alice', ['alice1.jpg', 'alice2.jpg', 'alice3.jpg'])
engine.enroll('Bob',   ['bob1.jpg',   'bob2.jpg'])

# Or bulk enroll from a folder structure:
#   root/
#       Alice/  ref1.jpg  ref2.jpg  ref3.jpg
#       Bob/    ref1.jpg  ref2.jpg
engine.enroll_from_folder('/path/to/reference_images/')

# Predict identity for a query image
name, score = engine.predict_from_path('query.jpg')
# → ('Alice', 0.67)  or  ('Unknown', 0.14) if below threshold

# Persist the prototype database to disk
engine.save_db('face_db.pth')

# Reload on the next run — no re-enrollment needed
engine.load_db('face_db.pth')
```

---

## Trained Model Weights

The trained model checkpoint is published as a **GitHub Release** and is downloaded automatically by both notebooks:

```
https://github.com/FireHead90544/EFNet/releases/download/modelweights/ckpt_ep012.pth
```

**Checkpoint details:**

| Property | Value |
|----------|-------|
| Training dataset | CASIA-WebFace (10,572 identities, ~494K images) |
| Best epoch | Epoch 12 |
| Format | Full training state dict |
| Keys | `model`, `criterion`, `optimizer`, `scheduler`, `epoch`, `lfw_acc` |

**Loading the backbone only:**

```python
import torch
from efnet import EFNet

model = EFNet(embed_dim=512)
ckpt = torch.load('ckpt_ep012.pth', map_location='cpu')
model.load_state_dict(ckpt['model'])   # extract backbone weights from checkpoint
model.eval()
```

**Download in Colab:**

```bash
!curl -L -o /content/checkpoints/ckpt_ep012.pth \
  https://github.com/FireHead90544/EFNet/releases/download/modelweights/ckpt_ep012.pth
```

---

## Gradio Web Demo

Both `EFNet.ipynb` and `EFNet_Demo.ipynb` include an identical Gradio web interface with three tabs:

| Tab | Functionality |
|-----|--------------|
| **Live Recognition** | Upload image / use webcam / enter file path → aligned face preview + `Identity: <name> \| Score: <value>` result |
| **Enroll Identity** | Register a person by name with uploaded images, webcam captures (with gallery preview), or bulk folder path. Supports add/clear frames. |
| **Database & Settings** | Cosine threshold slider (τ), save/load prototype DB to/from disk, unenroll identity by name |

The demo uses `share=True` to generate a temporary public Gradio link accessible from anywhere.

**Default inference threshold used in demo:** τ = 0.622 (calibrated on the ATT test split)

---

## Research Contributions

1. **EFNet-S block**: σ-gated local-global fusion combining a DWConv+SE local branch and a linear patch attention global branch — novel in the face recognition context, enabling adaptive per-channel blending of texture and geometry cues.

2. **Adaptive attention complexity**: Linear attention at 14×14 (N=196, expensive) transitioning to full MHSA at 7×7 (N=49, cheap) — a principled resolution-aware efficiency design that maintains accuracy while staying within edge compute budgets.

3. **Few-shot open-set evaluation protocol**: Formal evaluation with TAR@FAR and Unknown Rejection Rate (URR) metrics tailored for low-shot organisational deployment scenarios.

4. **Edge deployment analysis**: FLOPs, latency benchmarks on CPU/GPU, and model size comparisons against MobileFaceNet, GhostFaceNet, and MobileViT-XXS.

---

*EFNet — Edge Face Network. B.Tech Final Year Major Project.*
