
from __future__ import annotations
import torch, random, numpy as np

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def device_from_str(s: str) -> torch.device:
    if s == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
