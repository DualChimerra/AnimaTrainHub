"""Unit tests for anima_generate.py's XY matrix loop.

Does not run real inference (mocks `_T.sample_image` to return a fake PIL); only
verifies:
  - Traversal order: (yi, xi) nested loop; y=None degenerates to a single row
  - File naming xy_x{xi:02d}_y{yi:02d}_s{seed}.png
  - update_monitor receives sample_path + xy metadata
  - lora_scale axis: multiplier is reset before each cell + updated per axis value
  - cfg.seed=0: all cells share the same random seed (only overridden per-cell when
    axis=seed)
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Let `import runtime.anima_generate` find anima_train (sys.path manipulation at script top)
_REPO = Path(__file__).resolve().parent.parent
for _p in (_REPO, _REPO / "runtime"):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)


@pytest.fixture
def gen_module(monkeypatch):
    """import runtime.anima_generate, swapping external dependencies for mocks."""
    # Top-level import will import anima_train + inference_core; use a minimal stub to avoid real deps
    at = sys.modules.get("anima_train")
    if at is None or not hasattr(at, "sample_image"):
        at = types.ModuleType("anima_train")
        at.sample_image = lambda *a, **k: None  # replaced via monkeypatch in tests
        sys.modules["anima_train"] = at
    if "studio.services.inference_core" not in sys.modules:
        ic = types.ModuleType("studio.services.inference_core")

        class _LoRASpec:
            def __init__(self, path: str, scale: float = 1.0):
                self.path = path
                self.scale = scale

        ic.LoRASpec = _LoRASpec
        ic.apply_loras = lambda *a, **k: []
        sys.modules["studio.services.inference_core"] = ic

    import importlib

    if "runtime.anima_generate" in sys.modules:
        del sys.modules["runtime.anima_generate"]
    if "anima_generate" in sys.modules:
        del sys.modules["anima_generate"]
    mod = importlib.import_module("anima_generate")
    return mod


def _make_fake_img(tmp: Path):
    """Fake PIL.Image: save(path) just writes an empty file (the test only checks
    whether it's written to disk + the filename)."""
    fake = MagicMock()
    fake.save = lambda p: Path(p).write_bytes(b"")
    return fake


def _mock_sample_image(records: list, fake_img):
    """Stand-in for family.sample_image: records each call's args + returns a fake image.

    Since P4-4, the CLI dispatches via family.sample_image(model, vae, text, prompt, **kw),
    with prompt as the 4th positional arg."""
    def _stub(*args, **kwargs):
        records.append({
            "steps": kwargs.get("steps"),
            "cfg_scale": kwargs.get("cfg_scale"),
            "sampler_name": kwargs.get("sampler_name"),
            "prompt": args[3] if len(args) > 3 else kwargs.get("prompt"),
        })
        return fake_img
    return _stub


def _stub_family(sample_fn):
    """Duck-typed stand-in for ModelFamily: carries only sample_image."""
    fam = types.SimpleNamespace()
    fam.sample_image = sample_fn
    return fam


# ---------------------------------------------------------------------------
# _set_lora_multiplier
# ---------------------------------------------------------------------------


def test_set_lora_multiplier_updates_network_and_per_lora(gen_module) -> None:
    """network.multiplier + every lora.multiplier are all set to the given value."""
    fake_lora_a = MagicMock()
    fake_lora_a.multiplier = 1.0
    fake_lora_b = MagicMock()
    fake_lora_b.multiplier = 1.0
    fake_network = MagicMock()
    fake_network.multiplier = 1.0
    fake_network.loras = [fake_lora_a, fake_lora_b]
    fake_adapter = MagicMock()
    fake_adapter.network = fake_network

    gen_module._set_lora_multiplier(fake_adapter, 0.5)

    assert fake_network.multiplier == 0.5
    assert fake_lora_a.multiplier == 0.5
    assert fake_lora_b.multiplier == 0.5


def test_set_lora_multiplier_handles_no_network(gen_module) -> None:
    """No error when adapter.network=None."""
    fake_adapter = MagicMock()
    fake_adapter.network = None
    gen_module._set_lora_multiplier(fake_adapter, 0.5)  # noop


# ---------------------------------------------------------------------------
# _apply_axis
# ---------------------------------------------------------------------------


def test_apply_axis_steps(gen_module) -> None:
    s, c, sd, sm = gen_module._apply_axis(
        {"axis": "steps"}, 30,
        cur_steps=25, cur_cfg_scale=4.0, cur_seed=42, cur_sampler="er_sde",
        base_specs=[], adapters=[],
    )
    assert s == 30
    assert (c, sd, sm) == (4.0, 42, "er_sde")


def test_apply_axis_cfg_scale_and_sampler(gen_module) -> None:
    s, c, sd, sm = gen_module._apply_axis(
        {"axis": "cfg_scale"}, 7.5,
        cur_steps=25, cur_cfg_scale=4.0, cur_seed=42, cur_sampler="er_sde",
        base_specs=[], adapters=[],
    )
    assert c == 7.5
    s, c, sd, sm = gen_module._apply_axis(
        {"axis": "sampler_name"}, "euler_a",
        cur_steps=s, cur_cfg_scale=c, cur_seed=sd, cur_sampler=sm,
        base_specs=[], adapters=[],
    )
    assert sm == "euler_a"


def test_apply_axis_lora_scale_mutates_adapter(gen_module) -> None:
    fake_network = MagicMock()
    fake_network.multiplier = 1.0
    fake_network.loras = []
    fake_adapter = MagicMock()
    fake_adapter.network = fake_network

    spec = MagicMock()
    spec.scale = 1.0
    gen_module._apply_axis(
        {"axis": "lora_scale", "lora_index": 0}, 0.7,
        cur_steps=25, cur_cfg_scale=4.0, cur_seed=42, cur_sampler="er_sde",
        base_specs=[spec], adapters=[fake_adapter],
    )
    assert fake_network.multiplier == 0.7


# ---------------------------------------------------------------------------
# _run_xy_matrix -- traversal order + filenames + monitor
# ---------------------------------------------------------------------------


def test_run_xy_matrix_x_only_no_y(gen_module, tmp_path, monkeypatch) -> None:
    """y=None degenerates to 1xN (a single row)."""
    fake_img = _make_fake_img(tmp_path)
    records: list[dict] = []
    family = _stub_family(_mock_sample_image(records, fake_img))

    monitor_calls: list[dict] = []
    def fake_monitor(**kw):
        monitor_calls.append(kw)

    gen_module._run_xy_matrix(
        xy_matrix={"x": {"axis": "steps", "values": [20, 25, 30]}, "y": None},
        base_specs=[], adapters=[],
        prompt="test prompt",
        negative_prompt="",
        base_seed=42,
        base_steps=25, base_cfg_scale=4.0, base_sampler="er_sde",
        scheduler="simple",
        height=1024, width=1024,
        family=family, model=None, vae=None, text=None,
        device="cpu", dtype=None,
        output_dir=tmp_path,
        update_monitor=fake_monitor,
    )

    # 3 cells (X has 3 values, Y=None)
    assert len(records) == 3
    # steps increase per X value
    assert [r["steps"] for r in records] == [20, 25, 30]
    # File naming: yi=00 fixed, xi from 00 to 02, seed=42
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == [
        "xy_x00_y00_s42.png",
        "xy_x01_y00_s42.png",
        "xy_x02_y00_s42.png",
    ]
    # monitor receives xy={xi,yi,xv,yv} metadata
    assert len(monitor_calls) == 3
    assert monitor_calls[0]["xy"] == {"xi": 0, "yi": 0, "xv": 20, "yv": None}
    assert monitor_calls[2]["xy"] == {"xi": 2, "yi": 0, "xv": 30, "yv": None}


def test_run_xy_matrix_2d_traversal_order(gen_module, tmp_path, monkeypatch) -> None:
    """A 2x3 grid traversed as (yi outer, xi inner) = 6 cells."""
    fake_img = _make_fake_img(tmp_path)
    records: list[dict] = []
    family = _stub_family(_mock_sample_image(records, fake_img))

    gen_module._run_xy_matrix(
        xy_matrix={
            "x": {"axis": "steps", "values": [10, 20, 30]},
            "y": {"axis": "cfg_scale", "values": [3.0, 5.0]},
        },
        base_specs=[], adapters=[],
        prompt="p", negative_prompt="",
        base_seed=7,
        base_steps=25, base_cfg_scale=4.0, base_sampler="er_sde",
        scheduler="simple",
        height=512, width=512,
        family=family, model=None, vae=None, text=None,
        device="cpu", dtype=None,
        output_dir=tmp_path,
        update_monitor=None,
    )

    assert len(records) == 6
    # Order: all x for y0(3.0) finish first, then y1(5.0)
    expected = [
        (10, 3.0), (20, 3.0), (30, 3.0),
        (10, 5.0), (20, 5.0), (30, 5.0),
    ]
    actual = [(r["steps"], r["cfg_scale"]) for r in records]
    assert actual == expected


def test_run_xy_matrix_lora_scale_resets_each_cell(gen_module, tmp_path, monkeypatch) -> None:
    """multiplier is reset to base_scale before each cell, then changed per axis value."""
    fake_img = _make_fake_img(tmp_path)
    family = _stub_family(lambda *a, **k: fake_img)

    fake_network = MagicMock()
    fake_network.loras = []
    fake_network.multiplier = 1.0
    fake_adapter = MagicMock()
    fake_adapter.network = fake_network

    multiplier_history: list[float] = []
    def _track_multiplier(value):
        multiplier_history.append(float(value))
    type(fake_network).multiplier = property(
        lambda self: multiplier_history[-1] if multiplier_history else 1.0,
        lambda self, v: _track_multiplier(v),
    )

    spec = MagicMock()
    spec.scale = 0.6  # base scale

    gen_module._run_xy_matrix(
        xy_matrix={
            "x": {"axis": "lora_scale", "values": [0.3, 0.9], "lora_index": 0},
            "y": None,
        },
        base_specs=[spec], adapters=[fake_adapter],
        prompt="p", negative_prompt="",
        base_seed=42,
        base_steps=25, base_cfg_scale=4.0, base_sampler="er_sde",
        scheduler="simple",
        height=512, width=512,
        family=family, model=None, vae=None, text=None,
        device="cpu", dtype=None,
        output_dir=tmp_path,
        update_monitor=None,
    )

    # Each cell: reset to base 0.6 first -> then changed to the axis value
    # Cell 0 (xv=0.3): 0.6 -> 0.3
    # Cell 1 (xv=0.9): 0.6 -> 0.9
    assert 0.6 in multiplier_history       # base reset at least once
    assert 0.3 in multiplier_history
    assert 0.9 in multiplier_history
    # Reset must happen before the value change -- 0.6 appears at least 2 times (once per cell)
    assert multiplier_history.count(0.6) >= 2


def test_run_xy_matrix_seed_axis_overrides_base(gen_module, tmp_path, monkeypatch) -> None:
    """When axis=seed, the cell filename uses the axis value rather than base_seed."""
    fake_img = _make_fake_img(tmp_path)
    family = _stub_family(lambda *a, **k: fake_img)

    gen_module._run_xy_matrix(
        xy_matrix={"x": {"axis": "seed", "values": [100, 200, 300]}, "y": None},
        base_specs=[], adapters=[],
        prompt="p", negative_prompt="",
        base_seed=42,  # base is 42 but axis=seed should override it
        base_steps=25, base_cfg_scale=4.0, base_sampler="er_sde",
        scheduler="simple",
        height=512, width=512,
        family=family, model=None, vae=None, text=None,
        device="cpu", dtype=None,
        output_dir=tmp_path,
        update_monitor=None,
    )

    files = sorted(p.name for p in tmp_path.iterdir())
    # Filenames contain the axis value, not base_seed=42
    assert files == [
        "xy_x00_y00_s100.png",
        "xy_x01_y00_s200.png",
        "xy_x02_y00_s300.png",
    ]


def test_run_xy_matrix_zero_seed_randomizes_once(gen_module, tmp_path, monkeypatch) -> None:
    """base_seed=0 -> randomized once, then shared by all cells."""
    fake_img = _make_fake_img(tmp_path)
    family = _stub_family(lambda *a, **k: fake_img)
    # Pin random.randint's return value to make asserting easier
    monkeypatch.setattr(gen_module.random, "randint", lambda a, b: 12345)

    gen_module._run_xy_matrix(
        xy_matrix={"x": {"axis": "steps", "values": [20, 25]}, "y": None},
        base_specs=[], adapters=[],
        prompt="p", negative_prompt="",
        base_seed=0,  # triggers random
        base_steps=25, base_cfg_scale=4.0, base_sampler="er_sde",
        scheduler="simple",
        height=512, width=512,
        family=family, model=None, vae=None, text=None,
        device="cpu", dtype=None,
        output_dir=tmp_path,
        update_monitor=None,
    )

    files = sorted(p.name for p in tmp_path.iterdir())
    # All cells use the same post-randomization seed 12345
    assert files == ["xy_x00_y00_s12345.png", "xy_x01_y00_s12345.png"]


def test_run_xy_matrix_skips_failing_cell(gen_module, tmp_path, monkeypatch) -> None:
    """A single cell failing does not affect other cells (fault tolerance)."""
    fake_img = _make_fake_img(tmp_path)
    call_count = {"n": 0}

    def flaky_sample_image(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("simulated CUDA OOM")
        return fake_img

    family = _stub_family(flaky_sample_image)

    gen_module._run_xy_matrix(
        xy_matrix={"x": {"axis": "steps", "values": [10, 20, 30]}, "y": None},
        base_specs=[], adapters=[],
        prompt="p", negative_prompt="",
        base_seed=42,
        base_steps=25, base_cfg_scale=4.0, base_sampler="er_sde",
        scheduler="simple",
        height=512, width=512,
        family=family, model=None, vae=None, text=None,
        device="cpu", dtype=None,
        output_dir=tmp_path,
        update_monitor=None,
    )

    # Only cell 0 + cell 2 are written to disk (cell 1 raised)
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == ["xy_x00_y00_s42.png", "xy_x02_y00_s42.png"]
