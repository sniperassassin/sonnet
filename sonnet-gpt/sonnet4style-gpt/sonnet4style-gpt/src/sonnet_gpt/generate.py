
from __future__ import annotations
import torch, os, json
from .tokenizer import ByteTokenizer
from .model import GPT
from .utils import device_from_str

def load_model_from_checkpoint(ckpt_dir: str):
    with open(os.path.join(ckpt_dir, "config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)
    device = device_from_str(cfg["training"]["device"])
    model = GPT(
        vocab_size=cfg["model"]["vocab_size"],
        max_seq_len=cfg["model"]["max_seq_len"],
        n_layers=cfg["model"]["n_layers"],
        n_heads=cfg["model"]["n_heads"],
        d_model=cfg["model"]["d_model"],
        d_ff=cfg["model"]["d_ff"],
        dropout=cfg["model"]["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, "model.pt"), map_location=device))
    model.eval()
    tok = ByteTokenizer(bos_token_id=cfg["tokenizer"]["bos_token_id"], eos_token_id=cfg["tokenizer"]["eos_token_id"])
    return model, tok, cfg

def generate_text(ckpt_dir: str, prompt: str, max_new_tokens: int = 128):
    model, tok, cfg = load_model_from_checkpoint(ckpt_dir)
    device = next(model.parameters()).device
    ids = tok.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(
        x,
        max_new_tokens=max_new_tokens,
        temperature=cfg["generate"]["temperature"],
        top_k=cfg["generate"]["top_k"],
        top_p=cfg["generate"]["top_p"],
        repetition_penalty=cfg["generate"]["repetition_penalty"]
    )
    return tok.decode(out[0].tolist())
