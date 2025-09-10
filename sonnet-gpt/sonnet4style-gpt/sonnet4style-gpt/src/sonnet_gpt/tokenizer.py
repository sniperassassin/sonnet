
from __future__ import annotations
from dataclasses import dataclass
from typing import List

# Simple byte-level tokenizer (UTF-8 bytes + special tokens)
@dataclass
class ByteTokenizer:
    bos_token_id: int = 256
    eos_token_id: int = 257

    @property
    def vocab_size(self) -> int:
        return 258  # 0..255 bytes + BOS + EOS

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        b = text.encode("utf-8")
        ids = list(b)
        if add_special_tokens:
            return [self.bos_token_id] + ids + [self.eos_token_id]
        return ids

    def decode(self, ids: List[int]) -> str:
        # drop special tokens if present
        filtered = [i for i in ids if i < 256]
        return bytes(filtered).decode("utf-8", errors="ignore")
