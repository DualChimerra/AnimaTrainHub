from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
for _p in (_REPO, _REPO / "runtime"):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

import anima_daemon  # noqa: E402

_cell = anima_daemon._cell_lora_configs

_PATHS = ["a.safetensors", "b.safetensors"]
_SCALES = [0.8, 0.6]


def test_bf16_scale_axis_returns_none() -> None:
    out = _cell(
        {"axis": "lora_scale", "values": []}, None, 0.3, None,
        _PATHS, _SCALES, fp8_scale_axes=False,
    )
    assert out is None


def test_non_lora_axes_return_none() -> None:
    out = _cell(
        {"axis": "steps"}, {"axis": "cfg_scale"}, 20, 3.5,
        _PATHS, _SCALES, fp8_scale_axes=True,
    )
    assert out is None


def test_fp8_x_scale_sets_all_entries() -> None:
    out = _cell(
        {"axis": "lora_scale"}, {"axis": "steps"}, 0.3, 20,
        _PATHS, _SCALES, fp8_scale_axes=True,
    )
    assert out == [
        {"path": "a.safetensors", "scale": 0.3},
        {"path": "b.safetensors", "scale": 0.3},
    ]


def test_fp8_y_scale_sets_all_entries() -> None:
    out = _cell(
        {"axis": "steps"}, {"axis": "lora_scale"}, 20, 0.9,
        _PATHS, _SCALES, fp8_scale_axes=True,
    )
    assert all(lc["scale"] == 0.9 for lc in out)
    assert [lc["path"] for lc in out] == _PATHS


def test_fp8_both_scale_axes_y_wins() -> None:
    out = _cell(
        {"axis": "lora_scale"}, {"axis": "lora_scale"}, 0.3, 0.9,
        _PATHS, _SCALES, fp8_scale_axes=True,
    )
    assert all(lc["scale"] == 0.9 for lc in out)


def test_fp8_scale_with_ckpt_axis_combined() -> None:
    out = _cell(
        {"axis": "lora_scale"},
        {"axis": "lora_ckpt", "lora_index": 1},
        0.5, "c.safetensors",
        _PATHS, _SCALES, fp8_scale_axes=True,
    )
    assert out == [
        {"path": "a.safetensors", "scale": 0.5},
        {"path": "c.safetensors", "scale": 0.5},
    ]


def test_ckpt_axis_keeps_base_scales() -> None:
    out = _cell(
        {"axis": "lora_ckpt", "lora_index": 0}, None, "c.safetensors", None,
        _PATHS, _SCALES, fp8_scale_axes=False,
    )
    assert out == [
        {"path": "c.safetensors", "scale": 0.8},
        {"path": "b.safetensors", "scale": 0.6},
    ]


def test_single_axis_y_none() -> None:
    out = _cell(
        {"axis": "lora_scale"}, None, 0.4, None,
        _PATHS, _SCALES, fp8_scale_axes=True,
    )
    assert all(lc["scale"] == 0.4 for lc in out)
