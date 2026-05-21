from pathlib import Path

import pytest

from nanochat.transformers_backend import resolve_hf_model_path


def test_hub_repo_id_unchanged():
    assert resolve_hf_model_path("Qwen/Qwen2.5-0.5B") == "Qwen/Qwen2.5-0.5B"


def test_missing_local_path_raises():
    with pytest.raises(FileNotFoundError):
        resolve_hf_model_path("D:/hf_models/does-not-exist-xyz")


def test_existing_local_path(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text("{}")
    resolved = resolve_hf_model_path(str(tmp_path))
    assert Path(resolved).is_dir()
