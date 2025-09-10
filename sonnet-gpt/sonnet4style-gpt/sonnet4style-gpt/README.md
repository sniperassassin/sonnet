# Sonnet4Style-GPT

A minimal, production-ready scaffold for a GPT-style decoder-only language model with
JSON-driven hyperparameters. Includes:

- **Config**: Global JSON config validated against a JSON Schema.
- **Model**: Simple GPT block (Multi-Head Self-Attention + MLP) in PyTorch.
- **Tokenizer**: Byte-level tokenizer (256 base tokens + special tokens).
- **CLI**: Train / Generate / Evaluate via `python -m sonnet_gpt.cli --config <config.json>`.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# try tiny config + sample data
python -m sonnet_gpt.cli --config configs/tiny.json train
python -m sonnet_gpt.cli --config configs/tiny.json generate --prompt "To be, or not to be"
```

## Config

- See `configs/base.json` for a well-documented template.
- All configs are validated with `sonnet_gpt/schema.json`.

## Notes

This is a learning scaffold (not a state-of-the-art implementation). For serious training,
add mixed precision, gradient checkpointing, dataset streaming, and better tokenization.
