"""Configuration loading."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path = "config.yaml") -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    base = path.resolve().parent
    for key in ("data_dir", "output_dir"):
        p = Path(cfg[key])
        cfg[key] = str(p if p.is_absolute() else base / p)
    for section, key in (("awards", "csv_path"), ("prices", "csv_dir")):
        p = Path(cfg[section][key])
        cfg[section][key] = str(p if p.is_absolute() else base / p)
    Path(cfg["data_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    return cfg
