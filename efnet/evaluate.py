"""
efnet/evaluate.py
=================
Evaluation utilities for EFNet.

Metrics
───────
Closed-set (verification, LFW-style):
  • Accuracy @ best threshold
  • TAR @ FAR = 0.1%   (True Accept Rate when False Accept Rate ≤ 0.001)
  • TAR @ FAR = 1.0%

Open-set (few-shot enrollment + unknown rejection):
  • Rank-1 identification accuracy on enrolled identities
  • Unknown Rejection Rate (URR) — fraction of unknown queries correctly rejected
  • F1 score on the combined task
  • AUROC for threshold calibration

Threshold calibration
─────────────────────
The cosine similarity threshold τ is the single most important inference
parameter. Calibrate it on a HELD-OUT validation split (identities that
are enrolled but whose query images weren't used to build the prototype).

Rule of thumb starting values:
  τ = 0.30  — permissive (good recall, more false accepts)
  τ = 0.35  — balanced (recommended default)
  τ = 0.40  — strict (fewer false accepts, more unknowns)
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Tuple, Dict, Optional


# ---------------------------------------------------------------------------
# Helper: extract all embeddings from a DataLoader
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_embeddings(model, loader: DataLoader,
                       device: torch.device,
                       tta: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run model forward pass over entire dataset.

    Args:
        tta  : Test-time augmentation — also embed horizontally flipped image
               and average. Consistently improves LFW accuracy by ~0.1–0.3%.

    Returns:
        embeddings : (N, D) float32 numpy array
        labels     : (N,) numpy array of integer labels
    """
    model.eval()
    all_embs, all_labs = [], []

    for batch in loader:
        if len(batch) == 2:
            imgs, labs = batch
            imgs = imgs.to(device)

            emb = model(imgs)
            if tta:
                emb = emb + model(torch.flip(imgs, dims=[3]))
                emb = F.normalize(emb, p=2, dim=1)

            all_embs.append(emb.cpu().float().numpy())
            all_labs.append(labs.numpy() if not isinstance(labs, np.ndarray)
                            else labs)

        elif len(batch) == 3:
            # LFW pair loader returns (img1, img2, is_same)
            img1, img2, same = batch
            img1, img2 = img1.to(device), img2.to(device)

            e1 = model(img1)
            e2 = model(img2)
            if tta:
                e1 = F.normalize(e1 + model(torch.flip(img1, dims=[3])),
                                 p=2, dim=1)
                e2 = F.normalize(e2 + model(torch.flip(img2, dims=[3])),
                                 p=2, dim=1)

            all_embs.append(torch.stack([e1, e2], dim=1).cpu().float().numpy())
            all_labs.append(same.numpy())

    return np.concatenate(all_embs, axis=0), np.concatenate(all_labs, axis=0)


# ---------------------------------------------------------------------------
# LFW Verification
# ---------------------------------------------------------------------------

def compute_verification_metrics(
        similarities: np.ndarray,
        labels: np.ndarray,
        n_thresholds: int = 400
) -> Dict[str, float]:
    """
    Sweep cosine similarity thresholds and compute verification metrics.

    Args:
        similarities : (N,) cosine similarity per pair, range [-1, 1]
        labels       : (N,) 1 = same person, 0 = different person

    Returns dict with:
        accuracy    : best binary classification accuracy over all thresholds
        threshold   : τ that achieves best accuracy
        tar@far0.1% : TAR when FAR ≤ 0.1%
        tar@far1.0% : TAR when FAR ≤ 1.0%
        n_pos / n_neg : pair counts (for sanity checking)
    """
    pos_mask = labels == 1
    neg_mask = labels == 0
    pos_sims = similarities[pos_mask]
    neg_sims = similarities[neg_mask]
    n_pos    = pos_sims.size
    n_neg    = neg_sims.size

    # Guard: need both positive and negative pairs for meaningful metrics
    if n_pos == 0 or n_neg == 0:
        print(f"  [LFW Warn] Degenerate split: n_pos={n_pos}, n_neg={n_neg}. "
              "Cannot compute TAR@FAR. Check CSV loading.")
        return {
            'accuracy'   : float('nan'),
            'threshold'  : 0.0,
            'tar@far0.1%': float('nan'),
            'tar@far1.0%': float('nan'),
            'n_pos'      : n_pos,
            'n_neg'      : n_neg,
        }
    similarities = np.nan_to_num(
        similarities,
        nan=0.0,
        posinf=1.0,
        neginf=-1.0
    )
    thresholds = np.linspace(similarities.min(), similarities.max(), n_thresholds)

    # Best accuracy
    best_acc = 0.0
    best_tau = float(thresholds[0])
    for tau in thresholds:
        preds = (similarities >= tau).astype(int)
        acc   = (preds == labels).mean()
        if acc > best_acc:
            best_acc = acc
            best_tau = float(tau)

    # TAR @ FAR=target  — sweep from high threshold down
    def tar_at_far(target_far: float) -> float:
        for tau in np.linspace(similarities.max(), similarities.min(), n_thresholds):
            far = float((neg_sims >= tau).sum()) / n_neg
            if far <= target_far:
                return float((pos_sims >= tau).sum()) / n_pos
        return 0.0

    return {
        'accuracy'   : float(best_acc),
        'threshold'  : float(best_tau),
        'tar@far0.1%': tar_at_far(0.001),
        'tar@far1.0%': tar_at_far(0.010),
        'n_pos'      : n_pos,
        'n_neg'      : n_neg,
    }


@torch.no_grad()
def evaluate_lfw(model, lfw_loader: DataLoader,
                 device: torch.device, tta: bool = True) -> float:
    """
    Evaluate on LFW verification pairs.

    Returns:
        Best verification accuracy (scalar float).
    """
    model.eval()
    similarities, same_labels = [], []

    for img1, img2, same in lfw_loader:
        img1 = img1.to(device)
        img2 = img2.to(device)

        e1 = model(img1)
        e2 = model(img2)

        if tta:
            e1 = F.normalize(e1 + model(torch.flip(img1, dims=[3])),
                             p=2, dim=1)
            e2 = F.normalize(e2 + model(torch.flip(img2, dims=[3])),
                             p=2, dim=1)

        sim = F.cosine_similarity(e1, e2, dim=1, eps=1e-8)

        sim = torch.nan_to_num(
            sim,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0
        )   # cosine similarity (both unit-norm)
        similarities.append(sim.cpu().numpy())
        same_labels.append(same.numpy())

    similarities = np.concatenate(similarities)
    same_labels  = np.concatenate(same_labels)

    metrics = compute_verification_metrics(similarities, same_labels)

    acc = metrics['accuracy']
    if np.isnan(acc):
        print(f"  [LFW] Acc=NaN — degenerate split "
              f"(pos={metrics['n_pos']}, neg={metrics['n_neg']}). "
              "Fix: check CSV loading in dataset.py.")
        return 0.0

    print(f"  [LFW] Acc={acc:.4f}  τ={metrics['threshold']:.3f}  "
          f"TAR@FAR0.1%={metrics['tar@far0.1%']:.4f}  "
          f"TAR@FAR1.0%={metrics['tar@far1.0%']:.4f}  "
          f"(pos={metrics['n_pos']}, neg={metrics['n_neg']})")

    return acc


# ---------------------------------------------------------------------------
# Open-set few-shot evaluation
# ---------------------------------------------------------------------------

def evaluate_openset(prototypes: Dict[str, np.ndarray],
                     query_embeddings: np.ndarray,
                     query_labels: np.ndarray,   # -1 = unknown
                     tau: float = 0.35) -> Dict[str, float]:
    """
    Evaluate few-shot open-set performance.

    Args:
        prototypes        : {identity_name: (D,) unit-norm embedding}
        query_embeddings  : (N, D) unit-norm embeddings of query images
        query_labels      : (N,) str labels or -1 for unknown
        tau               : Cosine similarity threshold for rejection

    Returns:
        dict with:
          rank1_acc  — accuracy on enrolled (known) queries
          urr        — unknown rejection rate
          f1         — harmonic mean of known-acc and urr
    """
    names    = list(prototypes.keys())
    proto_mat = np.stack([prototypes[n] for n in names], axis=0)  # (K, D)

    # Cosine similarity: (N, K)
    sims = query_embeddings @ proto_mat.T

    # Prediction: argmax identity, then threshold
    best_idx  = sims.argmax(axis=1)
    best_sim  = sims.max(axis=1)
    pred_name = np.where(best_sim >= tau,
                         np.array(names)[best_idx],
                         'Unknown')

    known_mask   = query_labels != 'Unknown'
    unknown_mask = query_labels == 'Unknown'

    if known_mask.sum() > 0:
        rank1_acc = (pred_name[known_mask] == query_labels[known_mask]).mean()
    else:
        rank1_acc = 0.0

    if unknown_mask.sum() > 0:
        urr = (pred_name[unknown_mask] == 'Unknown').mean()
    else:
        urr = 1.0

    # F1 over the combined task
    f1 = (2 * rank1_acc * urr / (rank1_acc + urr + 1e-8))

    return {
        'rank1_acc': float(rank1_acc),
        'urr'      : float(urr),
        'f1'       : float(f1),
        'threshold': float(tau),
    }


def calibrate_threshold(prototypes: Dict[str, np.ndarray],
                        val_embeddings: np.ndarray,
                        val_labels: np.ndarray,
                        taus: np.ndarray = None) -> float:
    """
    Find the threshold τ that maximises F1 on a validation set.

    Call this ONCE after pre-training, using a held-out validation split.
    The returned τ is then fixed for all inference.

    Args:
        prototypes      : prototype dict (from inference.py enroll())
        val_embeddings  : (N, D) validation query embeddings
        val_labels      : (N,) labels ('Alice', 'Bob', 'Unknown', ...)
        taus            : thresholds to sweep (default: 0.10 to 0.90)

    Returns:
        Best threshold float.
    """
    if taus is None:
        taus = np.linspace(0.10, 0.90, 160)

    best_f1, best_tau = 0.0, 0.35
    for tau in taus:
        metrics = evaluate_openset(prototypes, val_embeddings,
                                   val_labels, tau)
        if metrics['f1'] > best_f1:
            best_f1  = metrics['f1']
            best_tau = tau

    print(f"[Calibration] Best τ={best_tau:.3f}  F1={best_f1:.4f}")
    return best_tau


# ---------------------------------------------------------------------------
# Parameter count & FLOPs summary
# ---------------------------------------------------------------------------

def model_summary(model, input_size=(1, 3, 112, 112),
                  device='cpu') -> Dict[str, int]:
    """Print parameter counts per module and estimate FLOPs (MACs)."""
    model.eval()
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'─'*50}")
    print(f"  Total parameters    : {total:>10,}")
    print(f"  Trainable params    : {trainable:>10,}")

    # Per-stage breakdown
    for name, child in model.named_children():
        n = sum(p.numel() for p in child.parameters())
        print(f"  {name:<20}: {n:>10,}")
    print(f"{'─'*50}\n")

    try:
        from thop import profile
        dummy = torch.randn(*input_size).to(device)
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        print(f"  MACs (≈FLOPs/2)     : {macs/1e6:>10.1f} M")
        print(f"  FLOPs (approx)      : {2*macs/1e6:>10.1f} M")
        return {'params': total, 'macs': int(macs)}
    except ImportError:
        print("  [FLOPs] Install thop: pip install thop")
        return {'params': total, 'macs': -1}
