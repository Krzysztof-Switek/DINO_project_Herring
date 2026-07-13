"""Shared utilities."""
from __future__ import annotations

import os
import random

import numpy as np
import torch
from torch import Tensor

# ImageNet normalisation constants (used for DINOv2 preprocessing)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    """Seed every RNG (python, numpy, torch, CUDA) for reproducible training.

    Call this ONCE at the start of a training entry point, BEFORE the model is
    instantiated (so head weight init is fixed) and before any DataLoader is
    iterated (so shuffling is fixed). Without it two identical runs land on
    different checkpoints purely by chance — the exact effect diagnosed for the
    10.07 vs 12.07 runs (0.79 vs 0.85 MAE from the same recipe/split).

    ``deterministic=True`` also pins cuDNN and requests deterministic torch
    kernels (``warn_only`` so an op lacking a deterministic implementation warns
    instead of crashing the run). Same-machine run-to-run reproducibility is the
    goal; bitwise identity across different GPUs / CUDA versions is not guaranteed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Required for deterministic CUDA matmul once use_deterministic_algorithms is on.
    # Must be set before the CUDA context is created (i.e. before model.to(cuda)).
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            # Older torch or an unsupported build — cuDNN flags above still apply.
            pass


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` — seed each worker deterministically.

    Derives the worker seed from torch's per-worker initial seed so multi-worker
    shuffling/augmentation is reproducible. Harmless no-op benefit at num_workers=0.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader_generator(seed: int = 42) -> torch.Generator:
    """Return a ``torch.Generator`` seeded for reproducible DataLoader shuffling."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


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
