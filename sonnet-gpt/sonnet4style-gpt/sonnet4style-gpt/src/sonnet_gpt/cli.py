
from __future__ import annotations
import argparse, json, os
from .config import GPTConfig
from .train import train_loop
from .generate import generate_text

def main():
    parser = argparse.ArgumentParser(description="Sonnet4Style-GPT CLI")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("train")
    gen = sub.add_parser("generate")
    gen.add_argument("--prompt", type=str, required=True)
    gen.add_argument("--ckpt_dir", type=str, default="checkpoints")
    gen.add_argument("--max_new_tokens", type=int, default=None)

    args = parser.parse_args()
    cfg = GPTConfig.from_json(args.config)

    if args.cmd == "train":
        ckpt = train_loop(cfg)
        print(f"Saved checkpoint to {ckpt}")
    elif args.cmd == "generate":
        mnt = args.max_new_tokens or cfg.generate["max_new_tokens"]
        text = generate_text(args.ckpt_dir, args.prompt, max_new_tokens=mnt)
        print(text)

if __name__ == "__main__":
    main()
