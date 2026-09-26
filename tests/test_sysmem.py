from __future__ import annotations

import pytest

from training import sysmem


def test_trim_working_set_returns_bool():
    assert sysmem.trim_working_set() in (True, False)


def test_available_ram_bytes_positive():
    avail = sysmem.available_ram_bytes()
    assert avail is None or avail > 0


def _write_weight(tmp_path, size=1024):
    p = tmp_path / "w.safetensors"
    p.write_bytes(b"\0" * size)
    return p


def test_budget_disabled_skips(tmp_path, monkeypatch):
    monkeypatch.setattr(sysmem, "available_ram_bytes", lambda: 0)
    sysmem.check_load_budget(
        False, weight_paths=[_write_weight(tmp_path)], stage="test",
    )


def test_budget_raises_when_ram_below_file_size(tmp_path, monkeypatch):
    weight = _write_weight(tmp_path)
    monkeypatch.setattr(sysmem, "_file_bytes", lambda paths: 13 * 1024**3)
    monkeypatch.setattr(sysmem, "available_ram_bytes", lambda: 10 * 1024**3)
    with pytest.raises(RuntimeError, match="available system RAM"):
        sysmem.check_load_budget(True, weight_paths=[weight], stage="model load")


def test_budget_raises_when_gpu_free_below_need(tmp_path, monkeypatch):
    weight = _write_weight(tmp_path)
    monkeypatch.setattr(sysmem, "_file_bytes", lambda paths: 13 * 1024**3)
    monkeypatch.setattr(sysmem, "available_ram_bytes", lambda: 64 * 1024**3)
    monkeypatch.setattr(sysmem, "gpu_free_bytes_global", lambda: 7 * 1024**3)
    with pytest.raises(RuntimeError, match="free GPU VRAM"):
        sysmem.check_load_budget(True, weight_paths=[weight], stage="model load")


def test_budget_passes_with_headroom(tmp_path, monkeypatch):
    weight = _write_weight(tmp_path)
    monkeypatch.setattr(sysmem, "_file_bytes", lambda paths: 13 * 1024**3)
    monkeypatch.setattr(sysmem, "available_ram_bytes", lambda: 64 * 1024**3)
    monkeypatch.setattr(sysmem, "gpu_free_bytes_global", lambda: 30 * 1024**3)
    sysmem.check_load_budget(True, weight_paths=[weight], stage="model load")


def test_budget_empty_paths_and_missing_files_are_noop(tmp_path, monkeypatch):
    monkeypatch.setattr(sysmem, "available_ram_bytes", lambda: 0)
    sysmem.check_load_budget(True, weight_paths=[], stage="test")
    sysmem.check_load_budget(
        True, weight_paths=[tmp_path / "missing.safetensors"], stage="test",
    )


def test_budget_query_failure_is_permissive(tmp_path, monkeypatch):
    monkeypatch.setattr(sysmem, "available_ram_bytes", lambda: None)
    sysmem.check_load_budget(
        True, weight_paths=[_write_weight(tmp_path)], stage="model load",
    )


def test_budget_sums_directory_contents(tmp_path):
    d = tmp_path / "te"
    d.mkdir()
    (d / "a.safetensors").write_bytes(b"\0" * 100)
    (d / "b.safetensors").write_bytes(b"\0" * 200)
    assert sysmem._file_bytes([d]) == 300


def test_guard_enabled_from_env_defaults_on(monkeypatch):
    monkeypatch.delenv("LORA_RAM_GUARD", raising=False)
    assert sysmem.guard_enabled_from_env() is True


@pytest.mark.parametrize("value", ["0", "false", "off", "no", " OFF "])
def test_guard_enabled_from_env_off_values(monkeypatch, value):
    monkeypatch.setenv("LORA_RAM_GUARD", value)
    assert sysmem.guard_enabled_from_env() is False


def test_guard_enabled_from_env_on_values(monkeypatch):
    monkeypatch.setenv("LORA_RAM_GUARD", "1")
    assert sysmem.guard_enabled_from_env() is True


def test_budget_error_uses_caller_settings_hint(tmp_path, monkeypatch):
    weight = _write_weight(tmp_path)
    monkeypatch.setattr(sysmem, "_file_bytes", lambda paths: 13 * 1024**3)
    monkeypatch.setattr(sysmem, "available_ram_bytes", lambda: 10 * 1024**3)
    with pytest.raises(RuntimeError, match="Settings -> Training -> Training parameters"):
        sysmem.check_load_budget(
            True, weight_paths=[weight], stage="training model load",
            settings_hint="Settings -> Training -> Training parameters",
        )
    with pytest.raises(RuntimeError, match="Settings -> VRAM strategy"):
        sysmem.check_load_budget(True, weight_paths=[weight], stage="model load")
