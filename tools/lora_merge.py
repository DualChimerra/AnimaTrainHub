"""Merge multiple LoRAs by weight into a single exact plain LoRA (color-grading POC).

Background: on the Generate page, stacking multiple LoRAs is linear (see
studio/services/inference/core.py::apply_loras -- each LoRA is injected independently and their
deltas are summed at forward time), i.e.:

    ΔW_total = Σ_i  weight_i · (alpha_i / rank_i) · ΔW_i

Therefore, merging several LoRAs (including negative-weight color-grading slider LoRAs) into one
file can be done exactly, mathematically -- no SVD approximation needed:

  - Plain LoRA layers: ΔW = up @ down, so the weight can be baked directly into the factors and
    the results concatenated along the rank dimension.
  - LoKr layers (w2-decomposed form): using the Kronecker mixed-product identity
        kron(w1, w2a @ w2b) = kron(w1, w2a) @ kron(I, w2b)
    this expands losslessly into a product of two matrices of rank <= in_l·r, which can then be
    concatenated with the other sources.

Output is a kohya-format file with algo="lora" (lora_unet_* prefix + ss_* metadata),
alpha=rank (scale factor 1), so mounting it at weight 1.0 is equivalent to the original
multi-LoRA combination.

Usage example (style x1.0 + color-temperature x-5 + saturation x-5):

    python tools/lora_merge.py \
        --lora path/to/style.safetensors 1.0 \
        --lora path/to/temperature_v6.safetensors -5 \
        --lora path/to/saturation_v6.safetensors -5 \
        --out path/to/merged.safetensors --verify

Supported scope (POC): plain LoRA (Linear layers, no lora_mid), LoKr (w1 as a full matrix, or
decomposed as w1_a/w1_b + w2_a/w2_b). The DoRA (weight_decompose) path is non-linear, and the
LoKr dual-full-matrix form would expand to an excessively large rank -- both are rejected with
a hard error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from studio.services.inference.core import LoRAMeta, read_lora_meta  # noqa: E402


def _load_layers(path: Path) -> dict[str, dict[str, torch.Tensor]]:
    """Load all tensors grouped by layer name (everything before the first '.')."""
    from safetensors import safe_open

    layers: dict[str, dict[str, torch.Tensor]] = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for key in f.keys():
            layer, _, suffix = key.partition(".")
            layers.setdefault(layer, {})[suffix] = f.get_tensor(key)
    return layers


def _layer_factors(
    layer: str, tensors: dict[str, torch.Tensor], meta: LoRAMeta, path: Path
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Single-layer tensors -> (up (out×r), down (r×in), scale), fp32, user weight not applied yet.

    scale = alpha / rank, matching LyCORIS LoConModule / LokrModule's self.scale
    (rs_lora is already rejected at the entry point, so there's no √rank branch here).
    """
    if "dora_scale" in tensors:
        raise SystemExit(f"{path.name} layer {layer} has dora_scale (DoRA is non-linear), cannot merge linearly")

    if "lora_down.weight" in tensors:  # plain LoRA (LoCon)
        down = tensors["lora_down.weight"].float()
        up = tensors["lora_up.weight"].float()
        if "lora_mid.weight" in tensors or down.dim() != 2:
            raise SystemExit(f"{path.name} layer {layer} is not a Linear plain LoRA (conv/tucker not supported by this POC)")
        rank = down.shape[0]
        alpha = float(tensors["alpha"]) if "alpha" in tensors else float(rank)
        return up, down, alpha / rank

    if "lokr_w2_a" in tensors:  # LoKr, w2-decomposed form
        if "lokr_t2" in tensors:
            raise SystemExit(f"{path.name} layer {layer} is a LoKr tucker form, not supported by this POC")
        if "lokr_w1" in tensors:
            w1 = tensors["lokr_w1"].float()
        else:
            w1 = (tensors["lokr_w1_a"] @ tensors["lokr_w1_b"]).float()
        w2a = tensors["lokr_w2_a"].float()
        w2b = tensors["lokr_w2_b"].float()
        # kron(w1, w2a@w2b) = kron(w1, w2a) @ kron(I_{w1 col count}, w2b)
        up = torch.kron(w1, w2a)                                # (out, in_l·r)
        down = torch.kron(torch.eye(w1.shape[1]), w2b)          # (in_l·r, in)
        rank = w2a.shape[1]  # LyCORIS lora_dim; scale = alpha / lora_dim
        alpha = float(tensors["alpha"]) if "alpha" in tensors else float(rank)
        return up, down, alpha / rank

    if "lokr_w2" in tensors:
        raise SystemExit(
            f"{path.name} layer {layer} is a LoKr dual-full-matrix form; the exact expansion rank would be "
            "too large, not supported by this POC (would need an SVD path)"
        )
    raise SystemExit(f"{path.name} layer {layer} has an unrecognized tensor shape: {sorted(tensors)}")


def _trim_factors(
    up: torch.Tensor, down: torch.Tensor, energy: Optional[float], cap: Optional[int]
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """QR + core SVD truncate up@down to the smallest rank retaining `energy` fraction of the
    singular-value energy, then cap it by `cap`."""
    q1, r1 = torch.linalg.qr(up)          # up = q1 @ r1
    q2, r2 = torch.linalg.qr(down.T)      # down = r2.T @ q2.T
    u, s, vh = torch.linalg.svd(r1 @ r2.T)
    keep = s.shape[0]
    if energy is not None:
        cum = torch.cumsum(s * s, dim=0)
        keep = min(keep, int(torch.searchsorted(cum, energy * cum[-1]).item()) + 1)
    if cap is not None:
        keep = min(keep, cap)
    new_up = q1 @ (u[:, :keep] * s[:keep])
    new_down = (vh[:keep] @ q2.T)
    return new_up, new_down, keep


def merge(
    sources: list[tuple[Path, float]],
    out_path: Path,
    save_dtype: torch.dtype,
    trim_energy: Optional[float],
    rank_cap: Optional[int],
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Perform the merge and write to disk. Returns each layer's final (up, down) (fp32, reused by verify)."""
    metas = [read_lora_meta(str(p)) for p, _ in sources]
    for (p, _), m in zip(sources, metas):
        if m.weight_decompose:
            raise SystemExit(f"{p.name} was trained with weight_decompose (DoRA) enabled, which is non-linear and cannot be merged")
        if m.rs_lora:
            raise SystemExit(f"{p.name} was trained with rs_lora enabled; that scaling semantics is not covered by this POC")

    all_layers: list[dict[str, dict[str, torch.Tensor]]] = [_load_layers(p) for p, _ in sources]
    layer_names = sorted(set().union(*[set(d) for d in all_layers]))

    merged: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    max_rank = 0
    for layer in layer_names:
        ups, downs = [], []
        for (path, weight), layers, meta in zip(sources, all_layers, metas):
            if layer not in layers:
                continue
            up, down, scale = _layer_factors(layer, layers[layer], meta, path)
            coeff = weight * scale
            # Balanced baking: split |coeff| via square root between the two sides, sign goes to
            # up, which keeps half-precision storage error low
            s = abs(coeff) ** 0.5
            ups.append(up * (s if coeff >= 0 else -s))
            downs.append(down * s)
        up = torch.cat(ups, dim=1)
        down = torch.cat(downs, dim=0)
        if trim_energy is not None or rank_cap is not None:
            up, down, _ = _trim_factors(up, down, trim_energy, rank_cap)
        merged[layer] = (up, down)
        max_rank = max(max_rank, up.shape[1])

    # Each layer keeps its own rank (no zero-padding). Layers whose rank != max_rank are written
    # into lora_reg_dims (exact fullmatch pattern); on the studio loading side, _apply_reg_dims_
    # reconstructs the per-layer shape from it. External loaders like ComfyUI read the shape
    # directly from the tensors + per-layer alpha, so they're naturally compatible.
    state: dict[str, torch.Tensor] = {}
    reg_dims: dict[str, int] = {}
    for layer, (up, down) in merged.items():
        r = up.shape[1]
        if r != max_rank:
            reg_dims[layer] = r
        state[f"{layer}.lora_up.weight"] = up.to(save_dtype).contiguous()
        state[f"{layer}.lora_down.weight"] = down.to(save_dtype).contiguous()
        # On the studio build side, scale is always alpha_global/rank_global=1 (reg_dims doesn't
        # recompute scale); setting per-layer alpha=r makes "alpha/rank"-style loaders also land
        # on scale=1
        state[f"{layer}.alpha"] = torch.tensor(float(r))

    # metadata follows the same convention as AnimaLycorisAdapter.save() (utils/lycoris_adapter.py),
    # so read_lora_meta / the Generate page can reconstruct it as algo=lora, scale=1
    network_args = {
        "algo": "lora",
        "preset": "anima_full",
        "dropout": 0.0,
        "rank_dropout": 0.0,
        "module_dropout": 0.0,
        "weight_decompose": False,
        "rs_lora": False,
    }
    if reg_dims:
        network_args["lora_reg_dims"] = reg_dims
    provenance = [{"file": p.name, "weight": w} for p, w in sources]
    metadata = {
        "ss_network_dim": str(max_rank),
        "ss_network_alpha": str(float(max_rank)),
        "ss_network_module": "lycoris.kohya",
        "ss_network_args": json.dumps(network_args),
        "anima_merge_sources": json.dumps(provenance, ensure_ascii=False),
    }
    from safetensors.torch import save_file

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(out_path), metadata=metadata)
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"Wrote {out_path}  (rank={max_rank}, {len(merged)} layers, {size_mb:.0f} MB)")
    return merged


def verify(
    sources: list[tuple[Path, float]],
    out_path: Path,
    spectrum: bool,
) -> None:
    """Per-layer comparison: ΔW reconstructed from the written file vs. the weighted sum of ΔW
    from each source (fp32 reference)."""
    metas = [read_lora_meta(str(p)) for p, _ in sources]
    all_layers = [_load_layers(p) for p, _ in sources]
    out_meta = read_lora_meta(str(out_path))
    out_layers = _load_layers(out_path)

    worst = (0.0, "")
    errs: list[float] = []
    energy_ranks: list[int] = []
    for layer, tensors in out_layers.items():
        up = tensors["lora_up.weight"].float()
        down = tensors["lora_down.weight"].float()
        alpha = float(tensors["alpha"])
        got = (alpha / down.shape[0]) * (up @ down)

        ref = torch.zeros_like(got)
        for (path, weight), layers, meta in zip(sources, all_layers, metas):
            if layer not in layers:
                continue
            u, d, scale = _layer_factors(layer, layers[layer], meta, path)
            ref += (weight * scale) * (u @ d)

        denom = ref.norm().item() or 1.0
        rel = (got - ref).norm().item() / denom
        errs.append(rel)
        if rel > worst[0]:
            worst = (rel, layer)

        if spectrum:
            s = torch.linalg.svdvals(ref)
            cum = torch.cumsum(s * s, dim=0)
            energy_ranks.append(int(torch.searchsorted(cum, 0.999 * cum[-1]).item()) + 1)

    print(
        f"verify: {len(errs)} layers  relative Frobenius error "
        f"max={max(errs):.3e} (layer {worst[1]})  mean={sum(errs) / len(errs):.3e}  "
        f"(storage-dtype quantization noise only; should be <1e-6 when stored as fp32)"
    )
    if spectrum and energy_ranks:
        energy_ranks.sort()
        n = len(energy_ranks)
        print(
            f"spectrum analysis: rank needed to cover 99.9% energy  p50={energy_ranks[n // 2]}  "
            f"p90={energy_ranks[int(n * 0.9)]}  max={energy_ranks[-1]} / actual rank {out_meta.rank}"
        )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--lora",
        nargs=2,
        action="append",
        required=True,
        metavar=("PATH", "WEIGHT"),
        help="source LoRA and its weight, repeatable. Weight semantics match the Generate page's strength slider",
    )
    parser.add_argument("--out", required=True, help="output safetensors path")
    parser.add_argument(
        "--dtype", choices=["fp32", "fp16", "bf16"], default="fp16",
        help="storage dtype (default fp16; fp32 has no quantization loss but doubles the size)",
    )
    parser.add_argument(
        "--trim-energy", type=float, default=None, metavar="E",
        help="optional SVD rank truncation, retaining fraction E of the singular-value energy (e.g. 0.999). No truncation by default, fully exact",
    )
    parser.add_argument(
        "--rank-cap", type=int, default=None, metavar="N",
        help="optional per-layer rank cap (SVD truncation, trading precision for size). No cap by default",
    )
    parser.add_argument("--verify", action="store_true", help="numerically verify layer by layer after writing")
    parser.add_argument(
        "--spectrum", action="store_true",
        help="include singular-value spectrum analysis during verify (estimates future resize headroom, slower)",
    )
    args = parser.parse_args(argv)

    sources: list[tuple[Path, float]] = []
    for path_str, weight_str in args.lora:
        path = Path(path_str)
        if not path.exists():
            raise SystemExit(f"File does not exist: {path}")
        sources.append((path, float(weight_str)))

    save_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    for path, weight in sources:
        meta = read_lora_meta(str(path))
        print(f"  {path.name}  weight={weight}  algo={meta.algo} rank={meta.rank} alpha={meta.alpha}")

    merge(sources, Path(args.out), save_dtype, args.trim_energy, args.rank_cap)
    if args.verify or args.spectrum:
        verify(sources, Path(args.out), args.spectrum)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
