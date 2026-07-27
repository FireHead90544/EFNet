"""
efnet/inference.py
==================
Few-shot open-set inference engine for EFNet.

Enrollment
──────────
For each known identity, run 3–5 reference images through the frozen EFNet
backbone → average their L2-normalised embeddings → store as the prototype.
This is the "few-shot learning" step — zero fine-tuning required.

Prediction
──────────
For a query face image:
  1. Forward pass through EFNet → embedding e (unit-norm)
  2. Compute cosine similarity: sim(e, p_i) for every enrolled prototype p_i
  3. Find best match:  i* = argmax sim(e, p_i),  s* = sim(e, p_i*)
  4. Decision:
       s* ≥ τ  →  return (identity_name[i*], s*)
       s* < τ  →  return ('Unknown', s*)

Cosine similarity ranges [-1, 1]; for unit-norm embeddings it equals dot
product. For well-trained ArcFace models, same-identity pairs typically
score s > 0.40, and different-identity pairs score s < 0.25 (with a gap).

Test-Time Augmentation (TTA)
─────────────────────────────
For both enrollment and inference, we embed the image AND its horizontal
mirror, then average the two embeddings before L2-normalisation. This
consistently improves accuracy by 0.1–0.4 percentage points for free.

Threshold guidance
──────────────────
Use evaluate.py::calibrate_threshold() on a held-out validation set.
Starting point: τ = 0.35.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

from .model   import EFNet
from .dataset import get_val_transform


# ---------------------------------------------------------------------------
# Face detector wrapper (SCRFD via InsightFace, fallback to no-op)
# ---------------------------------------------------------------------------

class FaceAligner:
    """
    Wraps InsightFace SCRFD for face detection + 5-point alignment.
    Falls back to a centre crop if InsightFace is not installed.

    Usage:
        aligner = FaceAligner()
        aligned = aligner(raw_pil_image)   # returns 112×112 PIL image or None
    """

    def __init__(self, det_size: Tuple[int, int] = (320, 320)):
        self._available = False
        try:
            import insightface
            from insightface.app import FaceAnalysis
            self._app = FaceAnalysis(name='buffalo_sc',
                                     providers=['CPUExecutionProvider'])
            self._app.prepare(ctx_id=0, det_size=det_size)
            self._available = True
            print("[FaceAligner] InsightFace SCRFD loaded successfully.")
        except ImportError:
            print("[FaceAligner] InsightFace not found. "
                  "Falling back to centre-crop mode.\n"
                  "  Install with: pip install insightface onnxruntime")

    def __call__(self, img_pil: Image.Image) -> Optional[Image.Image]:
        """
        Returns a 112×112 aligned face crop, or None if no face detected.
        """
        if not self._available:
            return img_pil.resize((112, 112), Image.BILINEAR)

        img_rgb = np.array(img_pil.convert('RGB'))
        faces   = self._app.get(img_rgb)
        if not faces:
            return None

        # Pick highest-confidence face
        face = max(faces, key=lambda f: f.det_score)

        # InsightFace returns aligned 112×112 crop as numpy (RGB)
        aligned = face.normed_embedding  # this is the embedding, not the image
        # For the actual image crop, use face.embedding_type != normed

        # Proper crop using landmarks
        from insightface.utils.face_align import norm_crop
        aligned_img = norm_crop(img_rgb, face.kps)
        return Image.fromarray(aligned_img)


# ---------------------------------------------------------------------------
# EFNetInference — the main inference engine
# ---------------------------------------------------------------------------

class EFNetInference:
    """
    Few-shot open-set face recognition inference engine.

    Quick start
    ───────────
    engine = EFNetInference('checkpoints/best_model.pth')
    engine.enroll('Alice', ['alice1.jpg', 'alice2.jpg', 'alice3.jpg'])
    engine.enroll('Bob',   ['bob1.jpg', 'bob2.jpg'])

    name, score = engine.predict_from_path('query.jpg')
    # name = 'Alice' or 'Bob' or 'Unknown'
    # score = cosine similarity (0.0–1.0)

    engine.save_db('face_db.pth')    # persist enrolled identities
    engine.load_db('face_db.pth')    # reload on next run
    """

    def __init__(self, model_path: str,
                 threshold: float = 0.35,
                 device: Optional[str] = None,
                 use_aligner: bool = False,
                 embed_dim: int = 512):
        """
        Args:
            model_path  : Path to saved EFNet backbone state dict
                          (saved with torch.save(model.state_dict(), path))
            threshold   : Cosine similarity threshold τ. Queries below this
                          are classified as 'Unknown'.
            device      : 'cuda', 'cpu', or None (auto-detect)
            use_aligner : Whether to run SCRFD face detection + alignment.
                          Set True for raw photos; False for pre-aligned crops.
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device    = torch.device(device)
        self.threshold = threshold

        # Load model
        self.model = EFNet(embed_dim=embed_dim)
        state_dict = torch.load(model_path, map_location=self.device)
        # Handle both raw state-dict and checkpoint dict formats
        if 'model' in state_dict:
            state_dict = state_dict['model']
        self.model.load_state_dict(state_dict)
        self.model.eval().to(self.device)

        self.transform  = get_val_transform()
        self.prototypes: Dict[str, torch.Tensor] = {}   # name → (D,) unit-norm
        self.ref_counts: Dict[str, int]          = {}   # name → num ref images
        self.aligner    = FaceAligner() if use_aligner else None

        print(f"[EFNetInference] Model loaded from '{model_path}' "
              f"on {self.device}. τ={threshold}")

    # ── Embedding extraction ───────────────────────────────────────────────

    @torch.no_grad()
    def _embed(self, img_pil: Image.Image) -> torch.Tensor:
        """
        Get unit-norm embedding for a single 112×112 PIL image.
        TTA: embed + embed(flip) → average → normalise.
        Returns: (D,) tensor on self.device
        """
        img_pil = img_pil.convert('RGB')
        x       = self.transform(img_pil).unsqueeze(0).to(self.device)
        x_flip  = torch.flip(x, dims=[3])

        emb = self.model(x) + self.model(x_flip)   # (1, D)
        return F.normalize(emb, p=2, dim=1).squeeze(0)

    def _load_and_align(self, img_path: str) -> Optional[Image.Image]:
        """Load image, optionally run face detection + alignment."""
        img = Image.open(img_path).convert('RGB')
        if self.aligner is not None:
            img = self.aligner(img)
        return img

    # ── Enrollment ─────────────────────────────────────────────────────────

    def enroll(self, name: str,
               image_paths: List[str],
               verbose: bool = True) -> torch.Tensor:
        """
        Register an identity from multiple reference images.

        The prototype is the mean of all reference embeddings, re-normalised.
        More reference images → more robust prototype (diminishing returns
        above ~8 images).

        Args:
            name         : Display name for this identity (e.g., 'Alice Smith')
            image_paths  : List of paths to reference face images.
                           Should be 112×112 aligned crops, or raw photos
                           if use_aligner=True.
            verbose      : Print enrollment confirmation.

        Returns:
            prototype embedding (D,) tensor.
        """
        embeddings = []
        failed     = []
        for path in image_paths:
            try:
                img = self._load_and_align(path)
                if img is None:
                    failed.append(path)
                    continue
                embeddings.append(self._embed(img))
            except Exception as e:
                failed.append(path)
                if verbose:
                    print(f"  [Enroll] Warning: failed to load '{path}': {e}")

        if len(embeddings) == 0:
            raise RuntimeError(f"No valid embeddings for '{name}'. "
                               f"Check image paths and alignment.")

        prototype             = torch.stack(embeddings).mean(0)
        prototype             = F.normalize(prototype, p=2, dim=0)
        self.prototypes[name] = prototype
        self.ref_counts[name] = len(embeddings)

        if verbose:
            print(f"[Enroll] '{name}' registered from "
                  f"{len(embeddings)}/{len(image_paths)} images."
                  + (f" ({len(failed)} failed)" if failed else ""))
        return prototype

    def enroll_from_folder(self, root_dir: str, verbose: bool = True):
        """
        Bulk enroll all identities from a folder structure:
            root_dir/
                Alice/
                    ref1.jpg  ref2.jpg  ref3.jpg
                Bob/
                    ref1.jpg  ref2.jpg
        """
        root = Path(root_dir)
        for identity_dir in sorted(root.iterdir()):
            if not identity_dir.is_dir():
                continue
            imgs = sorted([
                str(f) for f in identity_dir.iterdir()
                if f.suffix.lower() in ('.jpg', '.jpeg', '.png')
            ])
            if imgs:
                self.enroll(identity_dir.name, imgs, verbose=verbose)
        print(f"[Enroll] Total enrolled: {len(self.prototypes)} identities.")

    def unenroll(self, name: str):
        """Remove a registered identity."""
        self.prototypes.pop(name, None)
        self.ref_counts.pop(name, None)
        print(f"[Enroll] '{name}' removed from database.")

    # ── Prediction ─────────────────────────────────────────────────────────

    def predict(self, img_pil: Image.Image,
                return_all: bool = False
                ) -> Tuple[str, float]:
        """
        Predict identity for a pre-aligned PIL face image.

        Args:
            img_pil    : 112×112 PIL image (or raw photo if use_aligner=True)
            return_all : If True, returns full similarity scores as third value.

        Returns:
            (identity_name, cosine_similarity)
            identity_name = 'Unknown' if max similarity < threshold.
        """
        if len(self.prototypes) == 0:
            return 'Unknown', 0.0

        emb = self._embed(img_pil)

        # Compute cosine similarity to every prototype
        sims = {
            name: torch.dot(emb, proto).item()
            for name, proto in self.prototypes.items()
        }
        best_name = max(sims, key=sims.get)
        best_sim  = sims[best_name]

        if best_sim < self.threshold:
            result = ('Unknown', best_sim)
        else:
            result = (best_name, best_sim)

        if return_all:
            return result + (sims,)
        return result

    def predict_from_path(self, img_path: str,
                          return_all: bool = False) -> Tuple[str, float]:
        """Predict from an image file path."""
        img = self._load_and_align(img_path)
        if img is None:
            return 'Unknown (no face detected)', 0.0
        return self.predict(img, return_all=return_all)

    def predict_batch(self, img_paths: List[str]) -> List[Tuple[str, float]]:
        """Run prediction on a list of image paths."""
        return [self.predict_from_path(p) for p in img_paths]

    # ── Threshold management ───────────────────────────────────────────────

    def set_threshold(self, tau: float):
        """Update the rejection threshold at runtime."""
        self.threshold = tau
        print(f"[EFNetInference] Threshold updated to τ={tau:.3f}")

    # ── Database persistence ────────────────────────────────────────────────

    def save_db(self, path: str):
        """
        Persist the enrolled prototype database to disk.
        Load with load_db() on next run — no re-enrollment needed.
        """
        data = {
            'prototypes' : {k: v.cpu() for k, v in self.prototypes.items()},
            'ref_counts' : self.ref_counts,
            'threshold'  : self.threshold,
        }
        torch.save(data, path)
        print(f"[DB] Saved {len(self.prototypes)} identities → '{path}'")

    def load_db(self, path: str):
        """Load a previously saved prototype database."""
        data             = torch.load(path, map_location=self.device)
        self.prototypes  = {k: v.to(self.device)
                            for k, v in data['prototypes'].items()}
        self.ref_counts  = data.get('ref_counts', {})
        self.threshold   = data.get('threshold', self.threshold)
        print(f"[DB] Loaded {len(self.prototypes)} identities from '{path}'. "
              f"τ={self.threshold:.3f}")

    # ── Status / debug ──────────────────────────────────────────────────────

    def status(self):
        """Print current enrollment status."""
        print(f"\n{'─'*50}")
        print(f"  EFNetInference status")
        print(f"  Device    : {self.device}")
        print(f"  Threshold : τ = {self.threshold:.3f}")
        print(f"  Enrolled  : {len(self.prototypes)} identities")
        for name, count in self.ref_counts.items():
            print(f"    • {name:<30} ({count} ref images)")
        print(f"{'─'*50}\n")
