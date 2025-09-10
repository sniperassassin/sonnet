
from __future__ import annotations
import os, math
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .model import GPT
from .tokenizer import ByteTokenizer
from .data import TextFileDataset
from .utils import set_seed, device_from_str

def train_loop(cfg):
    set_seed(cfg.training["seed"])
    device = device_from_str(cfg.training["device"])

    # tokenizer
    tok = ByteTokenizer(bos_token_id=cfg.tokenizer["bos_token_id"], eos_token_id=cfg.tokenizer["eos_token_id"])

    # data
    ds = TextFileDataset(cfg.data["dataset_path"], tok, cfg.model["max_seq_len"])
    dl = DataLoader(ds, batch_size=cfg.training["micro_batch_size"], shuffle=True, drop_last=True, num_workers=cfg.data.get("num_workers", 0))

    # model
    model = GPT(
        vocab_size=cfg.model["vocab_size"],
        max_seq_len=cfg.model["max_seq_len"],
        n_layers=cfg.model["n_layers"],
        n_heads=cfg.model["n_heads"],
        d_model=cfg.model["d_model"],
        d_ff=cfg.model["d_ff"],
        dropout=cfg.model["dropout"],
    ).to(device)

    # optimizer
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optimizer["lr"],
        betas=tuple(cfg.optimizer["betas"]),
        weight_decay=cfg.optimizer["weight_decay"],
    )

    # scheduler (simple cosine or none)
    if cfg.scheduler["type"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.training["max_steps"])
    else:
        scheduler = None

    model.train()
    grad_accum = cfg.training["batch_size"] // cfg.training["micro_batch_size"]
    step = 0
    pbar = tqdm(total=cfg.training["max_steps"], desc="training")

    while step < cfg.training["max_steps"]:
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            logits, loss = model(xb, yb)
            (loss / grad_accum).backward()

            if (step + 1) % grad_accum == 0:
                optim.step()
                optim.zero_grad(set_to_none=True)
                if scheduler:
                    scheduler.step()

            if (step + 1) % cfg.training["eval_interval"] == 0:
                pbar.set_postfix(loss=float(loss.item()), lr=optim.param_groups[0]["lr"])

            pbar.update(1)
            step += 1
            if step >= cfg.training["max_steps"]:
                break

    pbar.close()
    # save
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), os.path.join("checkpoints", "model.pt"))
    with open(os.path.join("checkpoints", "config.json"), "w", encoding="utf-8") as f:
        import json; json.dump(cfg.raw, f, indent=2)
    return os.path.join("checkpoints", "model.pt")
