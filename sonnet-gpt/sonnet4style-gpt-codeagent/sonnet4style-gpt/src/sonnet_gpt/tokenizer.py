
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Union
import os

# Optional import: sentencepiece for BPE/Unigram
try:
    import sentencepiece as spm
except ImportError:
    spm = None

@dataclass
class TokenizerConfig:
    type: str = "byte"  # "byte" or "bpe"
    bos_token_id: int = 256
    eos_token_id: int = 257
    sp_model_path: Optional[str] = None  # path to trained BPE model

class Tokenizer:
    def __init__(self, cfg: TokenizerConfig):
        self.cfg = cfg
        if cfg.type == "bpe":
            if spm is None:
                raise ImportError("Install sentencepiece for BPE tokenizer: pip install sentencepiece")
            if cfg.sp_model_path is None or not os.path.exists(cfg.sp_model_path):
                raise ValueError(f"BPE tokenizer model not found at {cfg.sp_model_path}")
            self.sp = spm.SentencePieceProcessor(model_file=cfg.sp_model_path)
            self.vocab_size = self.sp.get_piece_size()
            self.bos_token_id = self.sp.piece_to_id("<BOS>") if self.sp.piece_to_id("<BOS>") >= 0 else self.vocab_size
            self.eos_token_id = self.sp.piece_to_id("<EOS>") if self.sp.piece_to_id("<EOS>") >= 0 else self.vocab_size + 1
        else:
            # byte-level
            self.vocab_size = 258  # 0..255 bytes + BOS + EOS
            self.bos_token_id = cfg.bos_token_id
            self.eos_token_id = cfg.eos_token_id
            self.sp = None

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        if self.cfg.type == "bpe":
            ids = self.sp.encode(text, out_type=int)
        else:
            ids = list(text.encode("utf-8"))
        if add_special_tokens:
            return [self.bos_token_id] + ids + [self.eos_token_id]
        return ids

    def decode(self, ids: List[int]) -> str:
        if self.cfg.type == "bpe":
            # remove BOS/EOS if present
            ids = [i for i in ids if i != self.bos_token_id and i != self.eos_token_id]
            return self.sp.decode(ids)
        else:
            filtered = [i for i in ids if i < 256]
            return bytes(filtered).decode("utf-8", errors="ignore")
