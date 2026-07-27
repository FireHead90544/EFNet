"""
efnet — Edge Face Network
=========================
Hybrid CNN-ViT for few-shot open-set face recognition at the edge.
"""

from .model     import EFNet, EFNetSBlock, EFNetGBlock, CNNStem, EmbedHead
from .losses    import ArcFaceLoss, CosFaceLoss, CombinedMarginLoss, build_loss
from .dataset   import (FolderFaceDataset,
                        LFWVerificationDataset, EnrollmentDataset,
                        get_train_transform, get_val_transform)
from .inference import EFNetInference
from .evaluate  import evaluate_lfw, calibrate_threshold, model_summary

__all__ = [
    'EFNet', 'EFNetSBlock', 'EFNetGBlock', 'CNNStem', 'EmbedHead',
    'ArcFaceLoss', 'CosFaceLoss', 'CombinedMarginLoss', 'build_loss',
    'FolderFaceDataset', 'LFWVerificationDataset', 'EnrollmentDataset',
    'get_train_transform', 'get_val_transform',
    'EFNetInference',
    'evaluate_lfw', 'calibrate_threshold', 'model_summary',
]

__version__ = '0.1.0'
