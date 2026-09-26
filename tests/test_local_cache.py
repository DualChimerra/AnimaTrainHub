from __future__ import annotations

import os
from pathlib import Path

import pytest

from studio.infrastructure import local_cache


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "environ", dict(os.environ))
    os.environ.pop(local_cache.OPT_OUT_ENV, None)
    for var in local_cache._CACHE_ENV_DIRS:
        os.environ.pop(var, None)


def test_apply_points_every_cache_into_the_repo(
    tmp_path: Path, clean_env: None  # noqa: ARG001
) -> None:
    applied = local_cache.apply(tmp_path)
    assert set(applied) == set(local_cache._CACHE_ENV_DIRS)
    for var, value in applied.items():
        assert os.environ[var] == value
        assert Path(value).is_relative_to(tmp_path / local_cache.CACHE_DIR_NAME)


def test_pip_cache_is_the_one_we_precreate(
    tmp_path: Path, clean_env: None  # noqa: ARG001
) -> None:
    local_cache.apply(tmp_path)
    root = tmp_path / local_cache.CACHE_DIR_NAME
    assert (root / "pip").is_dir()
    assert not (root / "huggingface").exists()


def test_never_overrides_an_existing_value(
    tmp_path: Path, clean_env: None, monkeypatch: pytest.MonkeyPatch  # noqa: ARG001
) -> None:
    monkeypatch.setenv("HF_HOME", "/somewhere/shared/hf")
    applied = local_cache.apply(tmp_path)
    assert "HF_HOME" not in applied
    assert os.environ["HF_HOME"] == "/somewhere/shared/hf"
    assert "PIP_CACHE_DIR" in applied


def test_opt_out_disables_everything(
    tmp_path: Path, clean_env: None, monkeypatch: pytest.MonkeyPatch  # noqa: ARG001
) -> None:
    monkeypatch.setenv(local_cache.OPT_OUT_ENV, "1")
    assert local_cache.apply(tmp_path) == {}
    assert "PIP_CACHE_DIR" not in os.environ
    assert not (tmp_path / local_cache.CACHE_DIR_NAME).exists()


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_opt_out_accepted_spellings(
    tmp_path: Path, clean_env: None, monkeypatch: pytest.MonkeyPatch, value: str  # noqa: ARG001
) -> None:
    monkeypatch.setenv(local_cache.OPT_OUT_ENV, value)
    assert local_cache.apply(tmp_path) == {}


def test_apply_is_idempotent(tmp_path: Path, clean_env: None) -> None:  # noqa: ARG001
    first = local_cache.apply(tmp_path)
    assert first
    assert local_cache.apply(tmp_path) == {}


def test_apply_can_target_a_separate_env_dict(
    tmp_path: Path, clean_env: None  # noqa: ARG001
) -> None:
    env: dict[str, str] = {}
    applied = local_cache.apply(tmp_path, env=env)
    assert applied
    assert env["HF_HOME"] == applied["HF_HOME"]
    assert "HF_HOME" not in os.environ


def test_hf_home_is_the_only_huggingface_variable(clean_env: None) -> None:  # noqa: ARG001
    assert "HF_HOME" in local_cache._CACHE_ENV_DIRS
    assert "HUGGINGFACE_HUB_CACHE" not in local_cache._CACHE_ENV_DIRS
    assert "TRANSFORMERS_CACHE" not in local_cache._CACHE_ENV_DIRS


def test_apply_survives_a_patched_os_name(
    tmp_path: Path, clean_env: None, monkeypatch: pytest.MonkeyPatch  # noqa: ARG001
) -> None:
    monkeypatch.setattr(os, "name", "nt")
    applied = local_cache.apply(tmp_path)
    assert applied
    assert applied["HF_HOME"].endswith("huggingface")


def test_describe_reports_current_values(
    tmp_path: Path, clean_env: None  # noqa: ARG001
) -> None:
    assert set(local_cache.describe(tmp_path)) == set(local_cache._CACHE_ENV_DIRS)
    assert all(v == "" for v in local_cache.describe(tmp_path).values())
    local_cache.apply(tmp_path)
    assert all(v for v in local_cache.describe(tmp_path).values())
