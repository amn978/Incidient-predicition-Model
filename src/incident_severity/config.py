"""Config loading + small helpers shared across the package."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


def load_config(path: str | os.PathLike | None = None) -> dict:
    """Load YAML config and resolve all paths against the project root.

    Order of precedence: explicit `path` > env var ISP_CONFIG > repo default.
    """
    path = Path(path or os.environ.get("ISP_CONFIG") or DEFAULT_CONFIG).resolve()
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    root = path.parent.parent
    cfg["_root"] = str(root)
    cfg["paths"] = {k: str(root / v) for k, v in cfg["paths"].items()}
    return cfg


def class_names(cfg: dict) -> list[str]:
    return list(cfg["classes"])


def critical_idx(cfg: dict) -> int:
    return class_names(cfg).index(cfg["success_metrics"]["critical_class"])


def low_idx(cfg: dict) -> list[int]:
    names = class_names(cfg)
    return [names.index(c) for c in cfg["success_metrics"]["low_classes"]]
