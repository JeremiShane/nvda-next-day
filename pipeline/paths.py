"""Repo-relative paths for the NVDA feature-selection pipeline.

DATA_DIR defaults to <repo>/data. Set NVDA_DATA_DIR to override it
(absolute path, or a path relative to the repo root). Spec tables and
window definitions stay in SPEC_DIR and are not written by the pipeline.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_DIR = REPO_ROOT / "spec"


def _resolve_data_dir() -> Path:
    override = os.environ.get("NVDA_DATA_DIR")
    if not override:
        return (REPO_ROOT / "data").resolve()
    path = Path(override).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


DATA_DIR = _resolve_data_dir()
RAW_DIR = DATA_DIR / "raw"
