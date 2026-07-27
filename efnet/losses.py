"""
efnet/losses.py
===============
Loss functions for EFNet training.

ArcFace (Additive Angular Margin Loss)
    The state-of-the-art loss for face recognition.
    Adds a fixed angular margin m to the angle between the embedding and
    its class weight vector before softmax scaling.

    Why ArcFace over regular cross-entropy?
    ─────────────────────────────────────────
    Regular softmax treats face-recognition as classification and learns
    separable features. But at inference you need METRIC properties —
    embeddings of the same person must be close, different people far apart.
    ArcFace imposes a geodesic (angular) margin on the hypersphere, directly
    optimising for the cosine similarity metric used at inference.

    Key hyper-parameters:
        s = scale (temperature inverse) — default 64.0
            Concentrates gradient signal; too low → slow convergence,
            too high → unstable. 64 is the community standard.
        m = angular margin (radians) — default 0.5 (~28.6°)
            Larger m = tighter clusters, harder training. 0.5 is standard.
            For few-shot deployment increase to 0.6 for tighter embeddings.

    Forward receives PRE-NORMALISED embeddings (unit norm, which EFNet's
    head already ensures) and integer class labels.

CosFace (LMCL)
    Alternative margin: subtracts m from cos(θ) in cosine space.
    Slightly simpler, comparable performance to ArcFace in practice.
    Provided as a drop-in alternative.

CombinedMarginLoss
    Generalised margin: ArcFace + CosFace margin simultaneously.
    Formula: cos(θ + m1) - m2   (set m1=0.5, m2=0 for ArcFace;
                                   m1=0, m2=0.4 for CosFace)
    Useful for fine-grained ablations in the research paper.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# ArcFace Loss
# ---------------------------------------------------------------------------

class ArcFaceLoss(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss.

    Paper: Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face
           Recognition", CVPR 2019.

    AMP / float16 note
    ──────────────────
    ArcFace multiplies logits by scale s=64. With 10572 classes the softmax
    denominator sums 10572 * exp(64) — values that overflow float16 (max
    ~65504) instantly, producing NaN. The fix: ALWAYS compute the loss in
    float32, regardless of whether the backbone runs in AMP/fp16. Embeddings
    are cast to fp32 at the start of forward(); the backbone weights stay fp16
    and the AMP GradScaler still handles the scale correctly.

    Args:
        embed_dim   : Dimension of L2-normalised embeddings (512).
        num_classes : Total number of training identities.
        s           : Logit scale. Default 64.0.
        m           : Angular margin in radians. Default 0.5 (~28.6°).
        easy_margin : If True, removes the piecewise boundary condition.
    """

    def __init__(self, embed_dim: int, num_classes: int,
                 s: float = 64.0, m: float = 0.5,
                 easy_margin: bool = False):
        super().__init__()
        self.s           = s
        self.m           = m
        self.easy_margin = easy_margin

        self.weight  = nn.Parameter(torch.FloatTensor(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th    = math.cos(math.pi - m)
        self.mm    = math.sin(math.pi - m) * m

    def forward(self, embeddings: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        # ── Always compute in float32 (prevents AMP fp16 overflow → NaN) ──
        embeddings = embeddings.float()
        W          = F.normalize(self.weight.float(), p=2, dim=1)

        cos_theta   = (embeddings @ W.T).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        sin_theta   = torch.sqrt((1.0 - cos_theta ** 2).clamp(min=1e-10))
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m

        if self.easy_margin:
            cos_theta_m = torch.where(cos_theta > 0, cos_theta_m, cos_theta)
        else:
            cos_theta_m = torch.where(cos_theta > self.th,
                                      cos_theta_m, cos_theta - self.mm)

        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)
        logits  = (one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta) * self.s

        return F.cross_entropy(logits, labels.long())


# ---------------------------------------------------------------------------
# CosFace Loss  (alternative / ablation baseline)
# ---------------------------------------------------------------------------

class CosFaceLoss(nn.Module):
    """
    CosFace: Large Margin Cosine Loss (LMCL).

    Paper: Wang et al., "CosFace: Large Margin Cosine Loss for Deep Face
           Recognition", CVPR 2018.

    Margin subtracted in cosine space: logit = s · (cos θ − m)
    for the ground-truth class; s · cos θ for all others.

    Args:
        embed_dim   : Embedding dimension.
        num_classes : Number of training identities.
        s           : Scale. Default 64.0.
        m           : Cosine margin. Default 0.4.
    """

    def __init__(self, embed_dim: int, num_classes: int,
                 s: float = 64.0, m: float = 0.4):
        super().__init__()
        self.s      = s
        self.m      = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        embeddings = embeddings.float()
        W         = F.normalize(self.weight.float(), p=2, dim=1)
        cos_theta = (embeddings @ W.T).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        one_hot   = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)
        logits    = (cos_theta - one_hot * self.m) * self.s
        return F.cross_entropy(logits, labels.long())


# ---------------------------------------------------------------------------
# Combined Margin Loss  (ArcFace + CosFace simultaneously)
# ---------------------------------------------------------------------------

class CombinedMarginLoss(nn.Module):
    """
    General margin loss combining angular (m1) and cosine (m2) margins.
    Formula for GT class: s · (cos(θ + m1) − m2)

    Set (m1=0.5, m2=0.0) for pure ArcFace.
    Set (m1=0.0, m2=0.4) for pure CosFace.
    Set (m1=0.5, m2=0.1) for a combined variant tested in literature.
    """

    def __init__(self, embed_dim: int, num_classes: int,
                 s: float = 64.0, m1: float = 0.5, m2: float = 0.0):
        super().__init__()
        self.s      = s
        self.m1     = m1
        self.m2     = m2
        self.cos_m1 = math.cos(m1)
        self.sin_m1 = math.sin(m1)
        self.th     = math.cos(math.pi - m1)
        self.mm     = math.sin(math.pi - m1) * m1
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        embeddings  = embeddings.float()
        W           = F.normalize(self.weight.float(), p=2, dim=1)
        cos_theta   = (embeddings @ W.T).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        sin_theta   = torch.sqrt((1.0 - cos_theta ** 2).clamp(min=1e-10))
        cos_theta_m = cos_theta * self.cos_m1 - sin_theta * self.sin_m1
        cos_theta_m = torch.where(cos_theta > self.th,
                                  cos_theta_m - self.m2,
                                  cos_theta - self.mm - self.m2)
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)
        logits  = (one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta) * self.s
        return F.cross_entropy(logits, labels.long())


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def build_loss(loss_type: str, embed_dim: int, num_classes: int,
               **kwargs) -> nn.Module:
    """
    Factory function for loss selection.

    Usage:
        criterion = build_loss('arcface', 512, 10572, s=64, m=0.5)
    """
    lut = {
        'arcface': ArcFaceLoss,
        'cosface': CosFaceLoss,
        'combined': CombinedMarginLoss,
    }
    if loss_type not in lut:
        raise ValueError(f"Unknown loss '{loss_type}'. Choose from: {list(lut)}")
    return lut[loss_type](embed_dim, num_classes, **kwargs)
