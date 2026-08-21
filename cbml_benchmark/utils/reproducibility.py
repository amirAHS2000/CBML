import os
import random

import numpy as np
import torch
from torch.utils.data import get_worker_info


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, PyTorch and CUDA RNGs for reproducible experiments."""
    seed = int(seed)

    # Must be set before CUDA/cuBLAS kernels are selected.
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # warn_only avoids crashing on an operation that has no deterministic
        # implementation on the installed PyTorch/CUDA stack.
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    """Seed DataLoader worker-side Python/NumPy RNGs from PyTorch's worker seed."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
