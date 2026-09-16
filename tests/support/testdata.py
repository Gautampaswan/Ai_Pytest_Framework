"""Load JSON/YAML fixtures from tests/testdata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from tests.config.settings import settings


def testdata_path(name: str) -> Path:
    path = settings.TESTDATA_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing test data file: {path}")
    return path


def load_json(name: str) -> Any:
    return json.loads(testdata_path(name).read_text(encoding="utf-8"))


def load_yaml(name: str) -> Any:
    payload = yaml.safe_load(testdata_path(name).read_text(encoding="utf-8"))
    return payload if payload is not None else {}
