from __future__ import annotations

import json
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from safetensors.torch import save_file

from studio.services.inference.core import LoRASpec, apply_loras, read_lora_meta


@contextmanager
def _patched_adapter(factory: object) -> Iterator[None]:
    fake_mod = types.ModuleType("utils.lycoris_adapter")
    fake_mod.AnimaLycorisAdapter = factory  # type: ignore[attr-defined]
    with patch.dict(sys.modules, {"utils.lycoris_adapter": fake_mod}):
        yield


def _write_lora_safetensors(
    path: Path,
    *,
    rank: int,
    alpha: float,
    algo: str,
    factor: int,
    weight_decompose: bool = False,
    rs_lora: bool = False,
) -> None:
    sd = {"lora_unet_dummy.lokr_w1": torch.zeros(2, 2)}
    meta = {
        "ss_network_dim": str(rank),
        "ss_network_alpha": str(alpha),
        "ss_network_module": "lycoris.kohya",
        "ss_network_args": json.dumps({
            "algo": algo,
            "factor": factor,
            "preset": "anima_full",
            "weight_decompose": weight_decompose,
            "rs_lora": rs_lora,
        }),
    }
    save_file(sd, str(path), metadata=meta)


def test_read_lora_meta_from_ss_network_dim_alpha(tmp_path: Path) -> None:
    p = tmp_path / "lora.safetensors"
    _write_lora_safetensors(p, rank=64, alpha=32.0, algo="lokr", factor=16)

    meta = read_lora_meta(str(p))
    assert meta.rank == 64
    assert meta.alpha == 32.0
    assert meta.algo == "lokr"
    assert meta.factor == 16


def test_read_lora_meta_unusual_dim(tmp_path: Path) -> None:
    p = tmp_path / "lora_dim8.safetensors"
    _write_lora_safetensors(p, rank=8, alpha=4.0, algo="lokr", factor=8)

    meta = read_lora_meta(str(p))
    assert meta.rank == 8
    assert meta.alpha == 4.0


def test_read_lora_meta_missing_metadata(tmp_path: Path) -> None:
    p = tmp_path / "no_meta.safetensors"
    save_file({"x": torch.zeros(1)}, str(p))

    meta = read_lora_meta(str(p))
    assert meta.rank == 32
    assert meta.algo == "lokr"
    assert meta.factor == 8


def test_read_lora_meta_invalid_fields(tmp_path: Path) -> None:
    p = tmp_path / "bad.safetensors"
    save_file({"x": torch.zeros(1)}, str(p), metadata={
        "ss_network_dim": "not_a_number",
        "ss_network_alpha": "also_bad",
        "ss_network_args": "not_json{",
    })
    meta = read_lora_meta(str(p))
    assert meta.rank == 32
    assert meta.alpha == 32.0  # alpha fallback to rank
    assert meta.algo == "lokr"




def test_read_lora_meta_model_family_grandfather(tmp_path: Path) -> None:
    marked = tmp_path / "k2.safetensors"
    save_file({"x": torch.zeros(1)}, str(marked), metadata={
        "ss_network_args": json.dumps({"algo": "lokr", "model_family": "krea2"}),
    })
    assert read_lora_meta(str(marked)).model_family == "krea2"

    legacy = tmp_path / "legacy.safetensors"
    _write_lora_safetensors(legacy, rank=32, alpha=16.0, algo="lokr", factor=8)
    assert read_lora_meta(str(legacy)).model_family == "anima"


def test_apply_loras_rejects_cross_family(tmp_path: Path) -> None:
    import pytest

    from studio.services.inference.core import LoRASpec, apply_loras

    p = tmp_path / "k2_style.safetensors"
    save_file({"x": torch.zeros(1)}, str(p), metadata={
        "ss_network_args": json.dumps({"algo": "lokr", "model_family": "krea2"}),
    })
    with pytest.raises(ValueError, match="krea2"):
        apply_loras(object(), [LoRASpec(path=str(p), scale=1.0)],
                    "cpu", torch.float32, family_id="anima")


def test_read_lora_meta_dora_and_rs_lora(tmp_path: Path) -> None:
    p = tmp_path / "dora_rs.safetensors"
    _write_lora_safetensors(
        p, rank=64, alpha=8.0, algo="lokr", factor=4,
        weight_decompose=True, rs_lora=True,
    )
    meta = read_lora_meta(str(p))
    assert meta.weight_decompose is True
    assert meta.rs_lora is True


def test_read_lora_meta_defaults_without_dora_rs(tmp_path: Path) -> None:
    p = tmp_path / "plain.safetensors"
    _write_lora_safetensors(p, rank=16, alpha=8.0, algo="lokr", factor=8)
    meta = read_lora_meta(str(p))
    assert meta.weight_decompose is False
    assert meta.rs_lora is False


def test_apply_loras_propagates_dora_and_rs_lora(tmp_path: Path) -> None:
    p = tmp_path / "dora.safetensors"
    _write_lora_safetensors(
        p, rank=64, alpha=8.0, algo="lokr", factor=4,
        weight_decompose=True, rs_lora=True,
    )

    created: list[MagicMock] = []

    def _fake_adapter(*args: object, **kwargs: object) -> MagicMock:
        m = MagicMock()
        m.init_kwargs = dict(kwargs)
        m.network = MagicMock()
        m.network.loras = []
        m.load_state_dict.return_value = MagicMock(missing_keys=[], unexpected_keys=[])
        created.append(m)
        return m

    model = MagicMock()
    with _patched_adapter(_fake_adapter):
        apply_loras(model, [LoRASpec(path=str(p), scale=1.0)], device="cpu", dtype=torch.float32)

    assert created[0].init_kwargs["weight_decompose"] is True
    assert created[0].init_kwargs["rs_lora"] is True


def test_apply_loras_each_lora_injects_separately(tmp_path: Path) -> None:
    p1 = tmp_path / "a.safetensors"
    p2 = tmp_path / "b.safetensors"
    _write_lora_safetensors(p1, rank=16, alpha=8.0, algo="lokr", factor=8)
    _write_lora_safetensors(p2, rank=8, alpha=4.0, algo="lokr", factor=8)

    created: list[MagicMock] = []

    def _fake_adapter(*args: object, **kwargs: object) -> MagicMock:
        m = MagicMock()
        m.init_kwargs = dict(kwargs)
        m.network = MagicMock()
        m.network.loras = []
        m.load_state_dict.return_value = MagicMock(missing_keys=[], unexpected_keys=[])
        created.append(m)
        return m

    model = MagicMock()

    with _patched_adapter(_fake_adapter):
        adapters = apply_loras(
            model,
            [LoRASpec(path=str(p1), scale=1.0), LoRASpec(path=str(p2), scale=0.5)],
            device="cpu",
            dtype=torch.float32,
        )

    assert len(adapters) == 2
    for a in adapters:
        a.inject.assert_called_once_with(model)
    assert created[0].init_kwargs["rank"] == 16
    assert created[0].init_kwargs["alpha"] == 8.0
    assert created[1].init_kwargs["rank"] == 8
    assert created[1].init_kwargs["alpha"] == 4.0
    assert created[0].network.multiplier == 1.0
    assert created[1].network.multiplier == 0.5
    created[0].network.to.assert_called_with(device="cpu", dtype=torch.float32)
    created[1].network.to.assert_called_with(device="cpu", dtype=torch.float32)


@pytest.mark.parametrize("algo", ["lora", "loha"])
def test_apply_loras_uses_fp32_for_lora_and_loha_algos(tmp_path: Path, algo: str) -> None:
    """Comfy parity dtype handling is per LycorisNetwork, not only LoKr.

    LoRA and LoHa are represented by the same AnimaLycorisAdapter with different
    metadata `algo` values, so fp32 network/tensor loading must apply to them too.
    """
    p = tmp_path / f"{algo}.safetensors"
    _write_lora_safetensors(p, rank=8, alpha=4.0, algo=algo, factor=8)

    created: list[MagicMock] = []
    loaded_dtypes: list[torch.dtype] = []

    def _fake_adapter(*args: object, **kwargs: object) -> MagicMock:
        m = MagicMock()
        m.init_kwargs = dict(kwargs)
        m.network = MagicMock()
        m.network.loras = []

        def _load(sd, *_args, **_kwargs):
            loaded_dtypes.extend(t.dtype for t in sd.values())
            return MagicMock(missing_keys=[], unexpected_keys=[])

        m.load_state_dict.side_effect = _load
        created.append(m)
        return m

    model = MagicMock()
    with _patched_adapter(_fake_adapter):
        apply_loras(model, [LoRASpec(path=str(p), scale=1.0)], device="cpu", dtype=torch.float32)

    assert created[0].init_kwargs["algo"] == algo
    created[0].network.to.assert_called_once_with(device="cpu", dtype=torch.float32)
    assert loaded_dtypes
    assert all(dtype == torch.float32 for dtype in loaded_dtypes)


def test_apply_loras_skips_missing_path(tmp_path: Path) -> None:
    p_real = tmp_path / "real.safetensors"
    _write_lora_safetensors(p_real, rank=16, alpha=8.0, algo="lokr", factor=8)
    p_fake = tmp_path / "nonexistent.safetensors"

    def _fake_adapter(*args: object, **kwargs: object) -> MagicMock:
        m = MagicMock()
        m.network = MagicMock()
        m.network.loras = []
        m.load_state_dict.return_value = MagicMock(missing_keys=[], unexpected_keys=[])
        return m

    model = MagicMock()
    with _patched_adapter(_fake_adapter):
        adapters = apply_loras(
            model,
            [LoRASpec(path=str(p_fake)), LoRASpec(path=str(p_real))],
            device="cpu",
            dtype=torch.float32,
        )

    assert len(adapters) == 1


def test_apply_loras_empty_specs() -> None:
    model = MagicMock()
    assert apply_loras(model, [], device="cpu", dtype=torch.float32) == []


def test_model_cache_hot_reloads_same_topology_lora_ckpt(tmp_path: Path) -> None:
    p1 = tmp_path / "a.safetensors"
    p2 = tmp_path / "b.safetensors"
    _write_lora_safetensors(p1, rank=16, alpha=8.0, algo="lokr", factor=8)
    _write_lora_safetensors(p2, rank=16, alpha=8.0, algo="lokr", factor=8)

    created: list[MagicMock] = []
    loaded_dtypes: list[torch.dtype] = []

    def _fake_adapter(*args: object, **kwargs: object) -> MagicMock:
        m = MagicMock()
        m.network = MagicMock()
        m.network.loras = []

        def _load(sd, *_args, **_kwargs):
            loaded_dtypes.extend(t.dtype for t in sd.values())
            return MagicMock(missing_keys=[], unexpected_keys=[])

        m.load_state_dict.side_effect = _load
        created.append(m)
        return m

    from runtime.anima_daemon import ModelCache

    cache = ModelCache()
    cache.model = MagicMock()
    cache.device = "cpu"
    cache.dtype = torch.bfloat16

    with _patched_adapter(_fake_adapter):
        first = cache.apply_loras([{"path": str(p1), "scale": 1.0}])
        second = cache.apply_loras([{"path": str(p2), "scale": 0.5}])

    assert first is second
    assert len(created) == 1
    created[0].detach.assert_not_called()
    assert created[0].inject.call_count == 1
    assert created[0].load_state_dict.call_count == 2
    assert created[0].network.multiplier == 0.5
    assert cache.last_lora_specs == [LoRASpec(path=str(p2), scale=0.5)]
    assert loaded_dtypes
    assert all(dtype == torch.float32 for dtype in loaded_dtypes)


def test_model_cache_moves_offloaded_model_before_injecting_lora(tmp_path: Path, monkeypatch) -> None:
    """VAE decode offloads the base model to CPU; adding LoRA next must move it back first."""
    p = tmp_path / "a.safetensors"
    _write_lora_safetensors(p, rank=16, alpha=8.0, algo="lokr", factor=8)

    from runtime import anima_daemon as mod

    events: list[str] = []

    class FakeModel:
        def to(self, device=None, **_kwargs):
            events.append(f"model.to:{device}")
            return self

        def eval(self):
            events.append("model.eval")
            return self

    class FakeAdapter:
        def __init__(self) -> None:
            self.network = MagicMock()

        def load_state_dict(self, *_args, **_kwargs):
            return MagicMock(missing_keys=[], unexpected_keys=[])

    def fake_apply_loras(model, specs, device, dtype, family_id="anima"):
        events.append("apply_loras")
        assert "model.to:cuda" in events
        assert dtype == torch.float32
        return [FakeAdapter()]

    cache = mod.ModelCache()
    cache.model = FakeModel()
    cache.qwen_model = None
    cache.device = "cuda"
    cache.dtype = torch.bfloat16

    monkeypatch.setattr(mod, "apply_loras", fake_apply_loras)

    adapters = cache.apply_loras([{"path": str(p), "scale": 1.0}])

    assert len(adapters) == 1
    assert events[:2] == ["model.to:cuda", "apply_loras"]


def test_model_cache_reinjects_when_lora_topology_changes(tmp_path: Path) -> None:
    p1 = tmp_path / "rank16.safetensors"
    p2 = tmp_path / "rank8.safetensors"
    _write_lora_safetensors(p1, rank=16, alpha=8.0, algo="lokr", factor=8)
    _write_lora_safetensors(p2, rank=8, alpha=4.0, algo="lokr", factor=8)

    created: list[MagicMock] = []

    def _fake_adapter(*args: object, **kwargs: object) -> MagicMock:
        m = MagicMock()
        m.network = MagicMock()
        m.network.loras = []
        m.detach.return_value = True
        m.load_state_dict.return_value = MagicMock(missing_keys=[], unexpected_keys=[])
        created.append(m)
        return m

    from runtime.anima_daemon import ModelCache

    cache = ModelCache()
    cache.model = MagicMock()
    cache.device = "cpu"
    cache.dtype = torch.float32

    with _patched_adapter(_fake_adapter):
        cache.apply_loras([{"path": str(p1), "scale": 1.0}])
        cache.apply_loras([{"path": str(p2), "scale": 1.0}])

    assert len(created) == 2
    created[0].detach.assert_called_once()
    created[1].inject.assert_called_once_with(cache.model)


# ---------------------------------------------------------------------------
# generate tempdir helpers
# ---------------------------------------------------------------------------


def test_generate_tempdir_path() -> None:
    import tempfile
    from studio.services.inference.core import (
        GENERATE_TEMP_PREFIX,
        generate_tempdir,
    )
    d = generate_tempdir(42)
    assert d.parent == Path(tempfile.gettempdir())
    assert d.name == f"{GENERATE_TEMP_PREFIX}42"


def test_cleanup_generate_tempdir_removes_dir() -> None:
    from studio.services.inference.core import (
        cleanup_generate_tempdir,
        generate_tempdir,
    )
    d = generate_tempdir(99999)
    d.mkdir(parents=True, exist_ok=True)
    (d / "img.png").write_bytes(b"\x89PNG")
    assert d.exists()

    cleanup_generate_tempdir(99999)
    assert not d.exists()


def test_cleanup_generate_tempdir_noop_when_missing() -> None:
    from studio.services.inference.core import (
        cleanup_generate_tempdir,
        generate_tempdir,
    )
    d = generate_tempdir(88888)
    if d.exists():
        import shutil
        shutil.rmtree(d)
    cleanup_generate_tempdir(88888)


def test_cleanup_stale_generate_tempdirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import tempfile
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    from studio.services.inference.core import (
        GENERATE_TEMP_PREFIX,
        cleanup_stale_generate_tempdirs,
    )

    leak1 = tmp_path / f"{GENERATE_TEMP_PREFIX}111"
    leak2 = tmp_path / f"{GENERATE_TEMP_PREFIX}222"
    keep = tmp_path / "unrelated_dir"
    leak1.mkdir()
    (leak1 / "x.png").write_bytes(b"\x89")
    leak2.mkdir()
    keep.mkdir()
    (keep / "important.txt").write_text("dont touch")

    cleanup_stale_generate_tempdirs()

    assert not leak1.exists()
    assert not leak2.exists()
    assert keep.exists()
    assert (keep / "important.txt").exists()


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def _peft_sd(layers: dict[str, int], *, with_alpha: bool = False, seed: int = 7) -> dict:
    torch.manual_seed(seed)
    sd: dict = {}
    for layer, rank in layers.items():
        sd[f"diffusion_model.{layer}.lora_A.weight"] = torch.randn(rank, 8, dtype=torch.float16)
        sd[f"diffusion_model.{layer}.lora_B.weight"] = torch.randn(8, rank, dtype=torch.float16)
        if with_alpha:
            sd[f"diffusion_model.{layer}.alpha"] = torch.tensor(float(rank) / 2)
    return sd


def test_normalize_peft_lora_sd_converts_keys_and_infers_rank():
    from studio.services.inference.core import _normalize_peft_lora_sd

    sd = _peft_sd({"blocks.0.q": 4, "blocks.1.k": 2})
    normalized, max_rank, reg_dims = _normalize_peft_lora_sd(sd)

    assert max_rank == 4
    assert reg_dims == {"lora_unet_blocks_1_k": 2}
    assert set(normalized) == {
        "lora_unet_blocks_0_q.lora_down.weight",
        "lora_unet_blocks_0_q.lora_up.weight",
        "lora_unet_blocks_0_q.alpha",
        "lora_unet_blocks_1_k.lora_down.weight",
        "lora_unet_blocks_1_k.lora_up.weight",
        "lora_unet_blocks_1_k.alpha",
    }
    assert float(normalized["lora_unet_blocks_0_q.alpha"]) == 4.0
    assert float(normalized["lora_unet_blocks_1_k.alpha"]) == 2.0
    assert normalized["lora_unet_blocks_0_q.lora_down.weight"].shape == (4, 8)
    assert normalized["lora_unet_blocks_0_q.lora_up.weight"].shape == (8, 4)


def test_normalize_peft_lora_sd_passthrough_and_rejects():
    import pytest

    from studio.services.inference.core import _normalize_peft_lora_sd

    kohya = {"lora_unet_blocks_0_q.lora_down.weight": torch.zeros(2, 4)}
    assert _normalize_peft_lora_sd(kohya) is None
    assert _normalize_peft_lora_sd({}) is None

    with_alpha = _peft_sd({"blocks.0.q": 2}, with_alpha=True)
    normalized, _, _ = _normalize_peft_lora_sd(with_alpha)
    assert float(normalized["lora_unet_blocks_0_q.alpha"]) == 1.0

    dora = _peft_sd({"blocks.0.q": 2})
    dora["diffusion_model.blocks.0.q.dora_scale"] = torch.ones(8, 1)
    with pytest.raises(ValueError, match="DoRA"):
        _normalize_peft_lora_sd(dora)

    with pytest.raises(ValueError, match="Unrecognized"):
        _normalize_peft_lora_sd({"diffusion_model.blocks.0.q.mystery": torch.zeros(1)})


def test_apply_loras_bf16_model_accepts_peft_file(tmp_path: Path):
    import torch.nn as nn

    from studio.services.inference.core import LoRASpec, apply_loras

    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = nn.Linear(8, 8, bias=False)

    class _Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([_Block()])

        def forward(self, x):
            return self.blocks[0].q(x)

    torch.manual_seed(0)
    model = _Tiny()
    sd = _peft_sd({"blocks.0.q": 2})
    path = tmp_path / "civit_peft.safetensors"
    save_file(sd, str(path))

    x = torch.randn(3, 8)
    base = model(x).detach().clone()

    adapters = apply_loras(
        model, [LoRASpec(path=str(path), scale=0.7)], "cpu", torch.float32,
        family_id="krea2",
    )

    assert len(adapters) == 1
    out = model(x).detach()
    up = sd["diffusion_model.blocks.0.q.lora_B.weight"].float()
    down = sd["diffusion_model.blocks.0.q.lora_A.weight"].float()
    expected = base + 0.7 * (x @ down.T @ up.T)   # scale=alpha/rank=1.0
    assert torch.allclose(out, expected, atol=1e-5)
