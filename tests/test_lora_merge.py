from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from utils.lycoris_adapter import _apply_reg_dims_

REPO_ROOT = Path(__file__).resolve().parent.parent

_spec = importlib.util.spec_from_file_location("lora_merge", REPO_ROOT / "tools" / "lora_merge.py")
lora_merge = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("lora_merge", lora_merge)
_spec.loader.exec_module(lora_merge)


class _FakeNet:
    def __init__(self, loras: list) -> None:
        self.loras = loras


def _locon(name: str, org: nn.Module, dim: int) -> object:
    from lycoris.modules.locon import LoConModule

    return LoConModule(name, org, 1.0, lora_dim=dim, alpha=dim)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_reg_dims_reinits_locon_linear_submodules() -> None:
    mod = _locon("layer_x", nn.Linear(16, 12), dim=8)
    _apply_reg_dims_(_FakeNet([mod]), {"layer_x": 4})
    assert mod.lora_dim == 4
    assert mod.lora_down.weight.shape == (4, 16)
    assert mod.lora_up.weight.shape == (12, 4)
    assert torch.all(mod.lora_up.weight == 0)


def test_reg_dims_locon_no_match_untouched() -> None:
    mod = _locon("layer_x_extra", nn.Linear(16, 12), dim=8)
    _apply_reg_dims_(_FakeNet([mod]), {"layer_x": 4})
    assert mod.lora_dim == 8
    assert mod.lora_down.weight.shape == (8, 16)


def test_reg_dims_locon_conv_skipped() -> None:
    mod = _locon("conv_x", nn.Conv2d(4, 4, 3, padding=1), dim=8)
    before = tuple(mod.lora_down.weight.shape)
    _apply_reg_dims_(_FakeNet([mod]), {"conv_x": 4})
    assert mod.lora_dim == 8
    assert tuple(mod.lora_down.weight.shape) == before


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

L1, L2 = "lora_unet_blocks_0_attn_q", "lora_unet_blocks_1_mlp"


def _write_lokr(path: Path) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    sd = {}
    for layer, dim in ((L1, 3), (L2, 2)):
        sd[f"{layer}.lokr_w1"] = torch.randn(2, 2)
        sd[f"{layer}.lokr_w2_a"] = torch.randn(4, dim)
        sd[f"{layer}.lokr_w2_b"] = torch.randn(dim, 6)
        sd[f"{layer}.alpha"] = torch.tensor(dim * 0.5)  # scale = alpha/dim = 0.5
    from safetensors.torch import save_file

    save_file(sd, str(path), metadata={
        "ss_network_dim": "3",
        "ss_network_alpha": "1.5",
        "ss_network_module": "lycoris.kohya",
        "ss_network_args": json.dumps({"algo": "lokr", "factor": 2}),
    })
    return sd


def _write_plain(path: Path, extra: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
    torch.manual_seed(1)
    sd = {}
    for layer in (L1, L2):
        sd[f"{layer}.lora_down.weight"] = torch.randn(2, 12)
        sd[f"{layer}.lora_up.weight"] = torch.randn(8, 2)
        sd[f"{layer}.alpha"] = torch.tensor(1.0)  # scale = 0.5
    sd.update(extra or {})
    from safetensors.torch import save_file

    save_file(sd, str(path), metadata={
        "ss_network_dim": "2",
        "ss_network_alpha": "1.0",
        "ss_network_module": "lycoris.kohya",
        "ss_network_args": json.dumps({"algo": "lora"}),
    })
    return sd


def _dense_ref(lokr: dict, plain: dict, w_lokr: float, w_plain: float, layer: str) -> torch.Tensor:
    dim = lokr[f"{layer}.lokr_w2_a"].shape[1]
    kron = torch.kron(lokr[f"{layer}.lokr_w1"], lokr[f"{layer}.lokr_w2_a"] @ lokr[f"{layer}.lokr_w2_b"])
    ref = w_lokr * (float(lokr[f"{layer}.alpha"]) / dim) * kron
    rank = plain[f"{layer}.lora_down.weight"].shape[0]
    ref += w_plain * (float(plain[f"{layer}.alpha"]) / rank) * (
        plain[f"{layer}.lora_up.weight"] @ plain[f"{layer}.lora_down.weight"]
    )
    return ref


def test_merge_matches_dense_reference(tmp_path: Path) -> None:
    lokr_sd = _write_lokr(tmp_path / "style.safetensors")
    plain_sd = _write_plain(tmp_path / "slider.safetensors")
    out = tmp_path / "merged.safetensors"
    lora_merge.merge(
        [(tmp_path / "style.safetensors", 1.0), (tmp_path / "slider.safetensors", -5.0)],
        out, torch.float32, trim_energy=None, rank_cap=None,
    )
    got = lora_merge._load_layers(out)
    for layer in (L1, L2):
        t = got[layer]
        rank = t["lora_down.weight"].shape[0]
        assert float(t["alpha"]) == pytest.approx(float(rank))  # per-layer scale=1
        dw = t["lora_up.weight"] @ t["lora_down.weight"]
        ref = _dense_ref(lokr_sd, plain_sd, 1.0, -5.0, layer)
        assert torch.allclose(dw, ref, atol=1e-5), f"{layer} ΔW deviates from the reference value"
    assert got[L1]["lora_down.weight"].shape[0] == 8
    assert got[L2]["lora_down.weight"].shape[0] == 6


def test_merge_metadata_reg_dims_and_dim(tmp_path: Path) -> None:
    _write_lokr(tmp_path / "style.safetensors")
    _write_plain(tmp_path / "slider.safetensors")
    out = tmp_path / "merged.safetensors"
    lora_merge.merge(
        [(tmp_path / "style.safetensors", 1.0), (tmp_path / "slider.safetensors", -5.0)],
        out, torch.float32, trim_energy=None, rank_cap=None,
    )
    from safetensors import safe_open

    with safe_open(str(out), framework="pt", device="cpu") as f:
        meta = f.metadata()
    assert meta["ss_network_dim"] == "8"
    args = json.loads(meta["ss_network_args"])
    assert args["algo"] == "lora"
    assert args["lora_reg_dims"] == {L2: 6}
    sources = json.loads(meta["anima_merge_sources"])
    assert [s["weight"] for s in sources] == [1.0, -5.0]

    from studio.services.inference.core import read_lora_meta

    m = read_lora_meta(str(out))
    assert (m.rank, m.algo, m.lora_reg_dims) == (8, "lora", {L2: 6})


def test_merge_rank_cap_truncates(tmp_path: Path) -> None:
    lokr_sd = _write_lokr(tmp_path / "style.safetensors")
    plain_sd = _write_plain(tmp_path / "slider.safetensors")
    out = tmp_path / "merged.safetensors"
    lora_merge.merge(
        [(tmp_path / "style.safetensors", 1.0), (tmp_path / "slider.safetensors", -5.0)],
        out, torch.float32, trim_energy=None, rank_cap=4,
    )
    got = lora_merge._load_layers(out)
    for layer in (L1, L2):
        assert got[layer]["lora_down.weight"].shape[0] <= 4
        dw = got[layer]["lora_up.weight"] @ got[layer]["lora_down.weight"]
        ref = _dense_ref(lokr_sd, plain_sd, 1.0, -5.0, layer)
        assert (dw - ref).norm() < ref.norm()


def test_merge_rejects_dora(tmp_path: Path) -> None:
    _write_plain(tmp_path / "dora.safetensors", extra={f"{L1}.dora_scale": torch.ones(8, 1)})
    with pytest.raises(SystemExit, match="dora"):
        lora_merge.merge(
            [(tmp_path / "dora.safetensors", 1.0)],
            tmp_path / "out.safetensors", torch.float32, trim_energy=None, rank_cap=None,
        )
