"""
Utility to enforce reproducible random seeds across notebooks.
Import and call `set_seed(SEED)` at the top of every notebook.
"""

import os
import random

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def set_seed(seed: int = 42, deterministic: bool = False):
    """
    Set random seeds for Python, NumPy, and PyTorch (if available).

    Args:
        seed (int): Seed value to use everywhere.
        deterministic (bool): If True, forces deterministic PyTorch behavior
                              (may reduce performance).
    """
    # Python
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    return seed

def sprinkles(img, size, perc, style='black', channels=3):
    """
    img: torch.Tensor of shape (C, H, W), values in [0, 1]
    """

    # Clone instead of copy (Torch-safe)
    x = img.clone()

    C, H, W = x.shape

    number_of_pixels_to_change = int(perc * H * W)
    number_of_sprinkles = max(1, number_of_pixels_to_change // (size * size))

    mask = torch.zeros((1, H, W), dtype=torch.float32)

    for _ in range(number_of_sprinkles):
        y = torch.randint(0, H - size, (1,)).item()
        x0 = torch.randint(0, W - size, (1,)).item()

        if style == 'black':
            x[:, y:y+size, x0:x0+size] = 0.0
        elif style == 'white':
            x[:, y:y+size, x0:x0+size] = 1.0

        mask[:, y:y+size, x0:x0+size] = 1.0

    return x, img, mask
