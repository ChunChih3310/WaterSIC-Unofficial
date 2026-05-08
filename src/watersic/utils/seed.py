from __future__ import annotations

import random

import numpy as np
import torch


def seed_host(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def seed_cuda(seed: int) -> None:
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_everything(seed: int) -> None:
    seed_host(seed)
    seed_cuda(seed)
