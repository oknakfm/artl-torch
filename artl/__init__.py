"""
artl-torch
==========
Outlier-robust neural network training via the
Augmented and Regularized Trimmed Loss (ARTL).

    Okuno and Yagishita (2026+)
    "Outlier-robust neural network training: variation regularization
     meets trimmed loss to prevent functional breakdown"
"""

from .trainer  import ARTLTrainer
from .datasets import make_checkered, make_stripe, DATASETS

__all__ = [
    "ARTLTrainer",
    "make_checkered",
    "make_stripe",
    "DATASETS",
]
__version__ = "0.2.0"
