
from __future__ import annotations
import torch
from torch.utils.data import Dataset

class TextFileDataset(Dataset):
    """
    Expects a text file and a tokenizer. Creates overlapping sequences of fixed length.
    """
    def __init__(self, path: str, tokenizer, block_size: int):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        ids = tokenizer.encode(text, add_special_tokens=False)  # we'll add BOS/EOS in batches if desired
        self.data = torch.tensor(ids, dtype=torch.long)
        self.block_size = block_size
        self.tokenizer = tokenizer

    def __len__(self):
        return max(0, len(self.data) - self.block_size - 1)

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + 1 + self.block_size]
        return x, y
