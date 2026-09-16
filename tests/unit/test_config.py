"""Configuration contract tests for the Ollama model setting."""

from __future__ import annotations

from pathlib import Path

import pytest
from dotenv import dotenv_values

import lib.ai as ai_mod


pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TestModelConfig:
    def test_env_sample_declares_llama_model(self):
        values = dotenv_values(PROJECT_ROOT / ".env.sample")

        assert values.get("MODEL") == "llama3.2"

    def test_runtime_model_is_non_empty_string(self):
        assert isinstance(ai_mod.MODEL, str)
        assert ai_mod.MODEL.strip()

    def test_ollama_client_targets_local_runtime(self):
        base_url = str(ai_mod.client.base_url)

        assert "11434" in base_url
