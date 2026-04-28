"""Shared utilities."""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

# ImageNet normalisation constants (used for DINOv2 preprocessing)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resolve_device(device_str: str) -> torch.device:
    """Resolve 'auto' to the best available device (cuda > mps > cpu)."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def tensor_to_uint8_rgb(tensor: Tensor) -> np.ndarray:
    """Reverse ImageNet normalisation and return (H, W, 3) uint8 RGB array."""
    img = tensor.cpu().float().permute(1, 2, 0).numpy()   # (H, W, 3) float
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return (img * 255).clip(0, 255).astype(np.uint8)
