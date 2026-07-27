"""
efnet/model.py
==============
EFNet — Edge Face Network
Full architecture: CNN Stem → EFNet-S × 3 → EFNet-G × 2 → Embed Head

Author: Rudransh Joshi
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Activation
# ---------------------------------------------------------------------------

class HardSwish(nn.Module):
    """HardSwish: x * relu6(x+3) / 6. Faster than Swish; mobilenet-proven."""
    def forward(self, x):
        return x * F.relu6(x + 3.0) / 6.0


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class DWSepConv(nn.Module):
    """
    Depthwise-separable convolution block.
    DWConv(k=3) → BN → HardSwish → PWConv(1×1) → BN → HardSwish
    ~8-9× fewer FLOPs than a standard Conv with same in/out channels.
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride,
                            padding=1, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = HardSwish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.dw(x)))
        x = self.act(self.bn2(self.pw(x)))
        return x


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation channel attention.
    GlobalAvgPool → FC → HardSwish → FC → Sigmoid → channel-wise scale.
    Teaches the network WHICH channels matter for a given spatial context.
    Reduction=4 keeps param count low.
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1  = nn.Linear(channels, mid, bias=False)
        self.act  = HardSwish()
        self.fc2  = nn.Linear(mid, channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        s = self.pool(x).view(B, C)
        s = self.act(self.fc1(s))
        s = torch.sigmoid(self.fc2(s)).view(B, C, 1, 1)
        return x * s


class LinearAttention(nn.Module):
    """
    Linear-complexity attention via kernel trick.

    Standard MHSA is O(N²·d). Here we use the ELU+1 kernel φ(x) = elu(x)+1
    to rewrite softmax(QKᵀ)V as Q·(Kᵀ·V) / Q·(Kᵀ·1), reducing complexity
    to O(N·d²) — linear in sequence length.

    For N=196 (14×14) this is ~25× cheaper than full attention.

    Structure:
        LayerNorm → QKV projection → split heads → φ(Q), φ(K) →
        context = Kᵀ·V  (d×d)  →  out = Q·context / normalizer
    """
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.heads    = num_heads
        self.head_dim = dim // num_heads
        self.norm     = nn.LayerNorm(dim)
        self.qkv      = nn.Linear(dim, dim * 3, bias=False)
        self.proj     = nn.Linear(dim, dim, bias=False)

    @staticmethod
    def _kernel(x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1.0          # guarantees > 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)
        B, N, C = x.shape
        h, d = self.heads, self.head_dim

        x = self.norm(x)
        qkv = self.qkv(x).reshape(B, N, 3, h, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # each: (B, h, N, d)

        q = self._kernel(q)
        k = self._kernel(k)

        # Context matrix: Kᵀ·V  →  (B, h, d, d)
        kv = torch.einsum('bhnd,bhne->bhde', k, v)
        # Attended output: Q·(Kᵀ·V)  →  (B, h, N, d)
        out = torch.einsum('bhnd,bhde->bhne', q, kv)

        # Normalizer: Q·(Kᵀ·1)  →  (B, h, N, 1)
        k_sum  = k.sum(dim=2)                                   # (B, h, d)
        denom  = torch.einsum('bhnd,bhd->bhn', q, k_sum
                              ).clamp(min=1e-6).unsqueeze(-1)
        out    = (out / denom)                                   # (B, h, N, d)

        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


# ---------------------------------------------------------------------------
# EFNet-S Block  (medium resolution: 28×28 or 14×14)
# ---------------------------------------------------------------------------

class EFNetSBlock(nn.Module):
    """
    Hybrid local + global block.

    Two parallel branches:
      Local  : DWConv 3×3 → BN → HardSwish → SE  (captures texture, edges,
               fine landmarks — things that need neighbouring pixels)
      Global : LayerNorm → LinearAttention         (captures inter-region
               relationships — eye↔eyebrow, left↔right symmetry, etc.)

    The branches are fused by a learned per-channel σ-gate:
        z = σ(g) · local + (1 − σ(g)) · global
    where g is a 1×1 conv that observes the concatenated branch features.
    This lets each channel learn whether to trust local or global context.

    A residual skip wraps the whole block. Stride=2 in the first EFNet-S
    block downsamples 28×28 → 14×14 and expands channels 48→96.
    """

    def __init__(self, in_ch: int, out_ch: int,
                 stride: int = 1, num_heads: int = 4):
        super().__init__()
        self.stride = stride

        # ---- Local branch ----
        self.loc_dw   = nn.Conv2d(in_ch, in_ch, 3, stride=stride,
                                  padding=1, groups=in_ch, bias=False)
        self.loc_bn   = nn.BatchNorm2d(in_ch)
        self.loc_act  = HardSwish()
        self.loc_pw   = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.loc_pwbn = nn.BatchNorm2d(out_ch)
        self.loc_se   = SEBlock(out_ch)

        # ---- Global branch ----
        self.glob_pool = nn.AvgPool2d(2, 2) if stride == 2 else nn.Identity()
        self.glob_attn = LinearAttention(in_ch, num_heads)
        self.glob_pw   = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.glob_pwbn = nn.BatchNorm2d(out_ch)

        # ---- σ-gate (per-channel scalar, initialised at 0 → σ(0)=0.5) ----
        self.gate = nn.Parameter(torch.zeros(1, out_ch, 1, 1))

        # ---- Output norm ----
        self.out_bn = nn.BatchNorm2d(out_ch)

        # ---- Shortcut ----
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        sc = self.shortcut(x)

        # Local path
        loc = self.loc_dw(x)
        loc = self.loc_act(self.loc_bn(loc))
        loc = self.loc_pwbn(self.loc_pw(loc))
        loc = self.loc_se(loc)

        # Global path — downsample first if stride=2
        xg = self.glob_pool(x)                          # (B, C, H', W')
        Hg, Wg = xg.shape[2], xg.shape[3]
        seq  = xg.permute(0, 2, 3, 1).reshape(B, Hg * Wg, C)   # (B,N,C)
        seq  = self.glob_attn(seq)
        xg   = seq.reshape(B, Hg, Wg, C).permute(0, 3, 1, 2)   # (B,C,H',W')
        glob = self.glob_pwbn(self.glob_pw(xg))

        # σ-gated fusion
        g      = torch.sigmoid(self.gate)               # (1, out_ch, 1, 1)
        fused  = g * loc + (1.0 - g) * glob

        return self.out_bn(fused) + sc


# ---------------------------------------------------------------------------
# EFNet-G Block  (low resolution: 7×7 = 49 tokens → full MHSA is trivial)
# ---------------------------------------------------------------------------

class EFNetGBlock(nn.Module):
    """
    Global attention block using standard Pre-LN Transformer encoder layer.

    At 7×7 spatial resolution, N=49 tokens. Full MHSA cost ∝ N²=2401 —
    negligible. We use full quadratic attention here because:
      • We can afford it at this resolution
      • Full MHSA captures richer inter-patch dependencies than linear approx
      • This is where high-level face geometry (overall shape, symmetry) lives

    Structure per block:
        x → LN → MHSA → x (residual)
          → LN → FFN(4× expand) → x (residual)
        → channel-project to out_ch (if needed)
        → BN → + shortcut
    """

    def __init__(self, in_ch: int, out_ch: int,
                 stride: int = 1, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()

        self.pool = nn.AvgPool2d(2, 2) if stride == 2 else nn.Identity()

        # Transformer encoder layer (Pre-LN)
        self.norm1 = nn.LayerNorm(in_ch)
        self.attn  = nn.MultiheadAttention(in_ch, num_heads,
                                            batch_first=True, bias=False,
                                            dropout=dropout)
        self.norm2 = nn.LayerNorm(in_ch)
        self.ffn   = nn.Sequential(
            nn.Linear(in_ch, in_ch * 4, bias=False),
            HardSwish(),
            nn.Dropout(dropout),
            nn.Linear(in_ch * 4, in_ch, bias=False),
        )
        self.drop = nn.Dropout(dropout)

        self.gamma1 = nn.Parameter(1e-4 * torch.ones(in_ch))
        self.gamma2 = nn.Parameter(1e-4 * torch.ones(in_ch))

        # Channel expansion (if in_ch ≠ out_ch)
        self.ch_proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        ) if in_ch != out_ch else nn.Identity()

        self.out_bn = nn.BatchNorm2d(out_ch)

        # Shortcut
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        sc = self.shortcut(x)

        # Spatial downsampling if stride=2
        x = self.pool(x)
        Hg, Wg = x.shape[2], x.shape[3]

        # Flatten to sequence (B, N, C)
        seq = x.permute(0, 2, 3, 1).reshape(B, Hg * Wg, C)

        # MHSA with Pre-LN
        n  = self.norm1(seq)
        a, _ = self.attn(n, n, n)
        seq = seq + self.gamma1 * self.drop(a)

        # FFN with Pre-LN
        seq = seq + self.gamma2 * self.drop(self.ffn(self.norm2(seq)))

        # Reshape back to spatial (B, C, H', W')
        out = seq.reshape(B, Hg, Wg, C).permute(0, 3, 1, 2)
        out = self.ch_proj(out)

        return self.out_bn(out) + sc


# ---------------------------------------------------------------------------
# Sub-modules: Stem and Embedding Head
# ---------------------------------------------------------------------------

class CNNStem(nn.Module):
    """
    Fast local feature extractor: 112×112×3 → 28×28×48.

    Conv(3→16, s=2)          : 112→56  — standard conv for initial edge detection
    DWSepConv(16→32, s=2)    : 56→28   — depthwise-sep to save params
    DWSepConv(32→48, s=1)    : 28→28   — deepen without spatial change
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            HardSwish(),
            DWSepConv(16, 32, stride=2),
            DWSepConv(32, 48, stride=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EmbedHead(nn.Module):
    """
    GDConv → BN → Flatten → BN → FC → BN → L2-normalise.

    GDConv (Global Depthwise Convolution):
      A 7×7 depthwise conv on the 7×7 feature map aggregates the entire
      spatial extent channel-wise. Unlike a plain global average pool,
      GDConv learns different spatial aggregation patterns per channel —
      critical because identity cues are not uniformly distributed
      (eyes carry more identity signal than forehead skin, for instance).

    L2 normalisation forces all embeddings onto a unit hypersphere,
    making cosine similarity = dot product and enabling clean ArcFace training.
    """
    def __init__(self, in_ch: int = 128, embed_dim: int = 512):
        super().__init__()
        self.gdconv  = nn.Conv2d(in_ch, in_ch, 7, groups=in_ch, bias=False)
        self.bn1     = nn.BatchNorm2d(in_ch)
        self.flatten = nn.Flatten()
        self.bn2     = nn.BatchNorm1d(in_ch)
        self.fc      = nn.Linear(in_ch, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn1(self.gdconv(x))   # (B, C, 1, 1)
        x = self.bn2(self.flatten(x))  # (B, C)
        x = self.fc(x)       # (B, embed_dim)
        return F.normalize(x, p=2, dim=1)


# ---------------------------------------------------------------------------
# Full EFNet model
# ---------------------------------------------------------------------------

class EFNet(nn.Module):
    """
    EFNet — Edge Face Network
    =========================
    Hybrid CNN-ViT for few-shot open-set face recognition.

    Stage           Input shape      Output shape      Key operation
    ─────────────────────────────────────────────────────────────────
    CNN Stem        (B,3,112,112)   (B,48,28,28)      DWSep conv
    EFNet-S × 3     (B,48,28,28)   (B,96,14,14)      DWConv+SE ‖ LinearAttn
    EFNet-G × 2     (B,96,14,14)   (B,128,7,7)       Full MHSA
    Embed Head      (B,128,7,7)    (B,512)            GDConv + FC + L2 norm
    ─────────────────────────────────────────────────────────────────
    ~800K–1.2M backbone parameters  (+ ArcFace head during training)

    Usage
    ─────
    model = EFNet(embed_dim=512)
    embeddings = model(face_tensor)   # face_tensor: (B, 3, 112, 112), [-1, 1]
    # embeddings: (B, 512) unit-norm vectors; use cosine similarity for comparison
    """

    def __init__(self, embed_dim: int = 512, dropout: float = 0.0):
        super().__init__()

        self.stem = CNNStem()

        self.efnet_s = nn.Sequential(
            EFNetSBlock(48,  96,  stride=2, num_heads=4),   # 28→14, 48→96
            EFNetSBlock(96,  96,  stride=1, num_heads=4),   # 14×14, 96ch
            EFNetSBlock(96,  96,  stride=1, num_heads=4),   # 14×14, 96ch
        )

        self.efnet_g = nn.Sequential(
            EFNetGBlock(96,  128, stride=2, num_heads=4, dropout=dropout),  # 14→7, 96→128
            EFNetGBlock(128, 128, stride=1, num_heads=4, dropout=dropout),  # 7×7, 128ch
        )

        self.head = EmbedHead(in_ch=128, embed_dim=embed_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)       # (B, 48, 28, 28)
        x = self.efnet_s(x)    # (B, 96, 14, 14)
        x = self.efnet_g(x)    # (B, 128,  7,  7)
        return self.head(x)    # (B, 512) unit-norm

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    model = EFNet(embed_dim=512)
    x     = torch.randn(4, 3, 112, 112)
    out   = model(x)
    print(f"Input  : {x.shape}")
    print(f"Output : {out.shape}")
    print(f"L2 norms (should all be ≈1): {out.norm(dim=1)}")
    print(f"Total trainable params: {model.count_parameters():,}")
