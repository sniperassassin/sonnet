
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict
import json, os
from jsonschema import Draft202012Validator
from importlib import resources
from .tokenizer import TokenizerConfig

@dataclass
class GPTConfig:
    raw: Dict[str, Any]

    @classmethod
    def from_json(cls, path: str) -> "GPTConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # load schema bundled with package
        with resources.files("sonnet_gpt").joinpath("schema.json").open("r", encoding="utf-8") as sf:
            schema = json.load(sf)
        Draft202012Validator(schema).validate(data)
        return cls(data)

    @property
    def model(self) -> Dict[str, Any]: return self.raw["model"]
    @property
    def tokenizer_cfg(self) -> TokenizerConfig:
        t = self.raw["tokenizer"]
        return TokenizerConfig(type=t.get("type","byte"),
                               bos_token_id=t.get("bos_token_id",256),
                               eos_token_id=t.get("eos_token_id",257),
                               sp_model_path=t.get("sp_model_path"))
    @property
    def data(self) -> Dict[str, Any]: return self.raw["data"]
    @property
    def training(self) -> Dict[str, Any]: return self.raw["training"]
    @property
    def optimizer(self) -> Dict[str, Any]: return self.raw["optimizer"]
    @property
    def scheduler(self) -> Dict[str, Any]: return self.raw["scheduler"]
    @property
    def generate(self) -> Dict[str, Any]: return self.raw["generate"]
