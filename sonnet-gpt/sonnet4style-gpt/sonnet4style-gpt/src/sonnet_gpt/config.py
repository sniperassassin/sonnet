
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List
import json, os
from jsonschema import validate, Draft202012Validator
from importlib import resources

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

    # convenience getters
    @property
    def model(self) -> Dict[str, Any]: return self.raw["model"]
    @property
    def tokenizer(self) -> Dict[str, Any]: return self.raw["tokenizer"]
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
