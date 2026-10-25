"""Noise gap profile + noise ladder for a style dataset, without training a LoRA.

Loads the base model from a normal training config (any family: anima / krea2 /
moss_image), exactly as training would see it -- same cached latents, same
captions, same forward -- but with no adapter injected. Then:

1. Gap profile: the base model's denoising error per t on the artist's images
   vs. on images it generated itself from the same captions. Bands where the
   artist is "harder" than the model's own output are where a style LoRA has
   to learn. The positive gap is turned into a suggested dual_peak config.
2. Noise ladder: each image noised to t0 and finished by the base model, for
   several t0. Shows at which noise level palette, shapes and pose get replaced
   by the model's defaults, plus pixel metrics per rung.

Usage (GPU; reads the training yaml, writes only into --out):

    python tools/noise_gap_probe.py --config path/to/train.yaml --out probe_out

The probe is a heuristic for choosing a sampler, not a measurement of a
trained LoRA. See runtime/noise_probe/ for the math.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import json
import logging
import math
import re
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (_REPO_ROOT, _REPO_ROOT / "runtime"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch  # noqa: E402

from dual_peak_sampling import DualPeakConfig  # noqa: E402
from noise_probe import (  # noqa: E402
    TRAIT_IDENTITY,
    bin_edges,
    compare_images,
    config_bin_masses,
    denoise_mse,
    effect_share,
    euler_denoise,
    fit_dual_peak,
    gap_profile,
    scheduled_velocity,
    seeded_noise,
    shifted_sigmas,
    trait_target,
    truncate_schedule,
)
from noise_probe.core import GapProfile  # noqa: E402

logger = logging.getLogger("noise_gap_probe")

DEFAULT_LADDER = (0.3, 0.45, 0.6, 0.75, 0.85, 0.92, 0.97)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", required=True, help="training config yaml (model paths, family, dataset)")
    p.add_argument("--out", default="", help="output directory (default: noise_probe_out/<time>)")
    p.add_argument("--max-images", type=int, default=12, help="dataset images used for the profile")
    p.add_argument("--bins", type=int, default=20, help="t bins for the profile")
    p.add_argument("--seeds", type=int, default=2, help="noise seeds per image and t")
    p.add_argument("--steps", type=int, default=0, help="full-schedule Euler steps (0 = sample_infer_steps, max 40)")
    p.add_argument("--cfg", type=float, default=None, help="CFG scale (default: sample_cfg_scale from the config)")
    p.add_argument("--shift", type=float, default=None, help="sampling schedule shift (default: the family's)")
    p.add_argument("--ladder-images", type=int, default=4)
    p.add_argument("--ladder-t", default=",".join(str(t) for t in DEFAULT_LADDER))
    p.add_argument("--anchor", type=float, default=0.3,
                   help="blend weight of the shipped dual_peak default in the fit target (0 = data only)")
    p.add_argument("--strip-trigger", action="store_true",
                   help="remove trigger_word from captions (default: captions exactly as training sees them)")
    p.add_argument("--skip-profile", action="store_true")
    p.add_argument("--skip-ladder", action="store_true")
    p.add_argument("--reuse-reference", action="store_true",
                   help="reuse reference latents from a previous run in --out")
    p.add_argument("--trait-weights", default="palette=1,shapes=1,layout=0.35",
                   help="how much each ladder trait pulls the fit (layout low = don't chase dataset poses)")
    p.add_argument("--lora", default="", help="trained LoRA: run the learned-profile + window test instead")
    p.add_argument("--lora-scale", type=float, default=1.0)
    p.add_argument("--windows", default="0.85-1,0.6-0.85,0.3-0.6,0-0.3",
                   help="t windows where the LoRA alone is switched on (--lora mode)")
    p.add_argument("--refit-only", action="store_true",
                   help="no GPU: refit from the CSVs of a previous run in --out (try other weights/anchor)")
    return p.parse_args(argv)


# --- model / data setup -------------------------------------------------------


def _load_config(config_path: Path):
    from training.bootstrap import apply_yaml_config, load_yaml_config

    args = apply_yaml_config(argparse.Namespace(), load_yaml_config(config_path))
    # The probe needs cached latents and one deterministic caption per image.
    overrides = dict(cache_latents=True, navit_packing=False, reg_data_dir="", batch_size=1,
                     num_workers=0, caption_dropout=0.0, shuffle_caption=False, tag_dropout=0.0,
                     masked_loss=False, edit_mode=False)
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _family_shift(family, args) -> float:
    """Inference schedule shift as a direct Mobius factor.

    The spec's shift_policy value is family-interpreted (Krea2 stores mu,
    MOSS has a config override), so the probe resolves it here, outside the
    shared training code.
    """
    fid = family.spec.family_id
    value = float(family.spec.sampling.shift_policy.shift)
    if fid == "krea2":
        return math.exp(value)
    if fid == "moss_image":
        return float(getattr(args, "moss_sample_shift", value) or value)
    return value


def build_stack(config_path: Path, work_dir: Path, needed_prompts_fn):
    """Load VAE / text / DiT and the cached dataset the way training does, minus the adapter."""
    from training.context import TrainingContext
    from training.families import resolve_family
    from training.phases import dataset as dataset_phase
    from training.phases import models as models_phase

    args = _load_config(config_path)
    ctx = TrainingContext(args=args)
    ctx.config_path = config_path.resolve()
    ctx.config_dir = ctx.config_path.parent
    ctx.family = resolve_family(args)
    ctx.device = "cuda" if torch.cuda.is_available() else "cpu"
    precision = str(getattr(args, "mixed_precision", "bf16"))
    ctx.dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision, torch.float32)
    ctx.vae_dtype = torch.float32 if ctx.dtype == torch.float16 else ctx.dtype
    work_dir.mkdir(parents=True, exist_ok=True)
    ctx.output_dir = ctx.sample_dir = work_dir
    if ctx.device == "cpu":
        logger.warning("CUDA unavailable: this will be extremely slow")

    models_phase._resolve_paths(ctx)
    deferred = models_phase._defer_dit_for_text_cache(ctx)
    if not deferred:
        models_phase._load_dit(ctx)
    models_phase._load_vae(ctx)
    models_phase._load_text(ctx)
    dataset_phase.run(ctx)
    if hasattr(ctx.dataset, "flip_augment"):
        ctx.dataset.flip_augment = False  # always read the unflipped latent

    images = unique_images(ctx)
    prompts = needed_prompts_fn(args, images)
    if ctx.family.spec.text.strategy == "cached_varlen":
        # Everything goes into the prompt bundle under work_dir: no sidecars
        # are written next to the dataset images.
        ctx.family.prepare_text_cache([], prompts, cache_entries=(), cache_root=work_dir,
                                      text=ctx.text_stack, device=ctx.device, dtype=ctx.dtype)
    else:
        ctx.family.prepare_text_cache([], [])
    if deferred:
        models_phase._load_dit(ctx)
    models_phase._setup_block_swap(ctx)
    ctx.model.eval()
    ctx.model.requires_grad_(False)
    return ctx, images


def unique_images(ctx) -> list[dict]:
    """One entry per image (repeats and resolution fan-out collapsed), in dataset order."""
    seen, out = set(), []
    for idx, sample in enumerate(ctx.dataset.samples):
        key = str(sample["image"])
        if key in seen:
            continue
        seen.add(key)
        item = ctx.dataset[idx]
        latent = item["latent"]
        if latent.dim() == 4:
            latent = latent.unsqueeze(0)
        out.append({"image": key, "caption": item["caption"], "latent": latent})
    return out


def strip_trigger(caption: str, trigger: str) -> str:
    if not trigger:
        return caption
    cleaned = re.sub(rf"\s*{re.escape(trigger)}\s*,?", " ", caption)
    return re.sub(r"\s{2,}", " ", cleaned).strip(" ,")


def evenly(items: list, count: int) -> list:
    if count >= len(items):
        return list(items)
    step = len(items) / count
    return [items[int(i * step)] for i in range(count)]


class Model:
    """velocity_fn + cached text conditions for the loaded stack."""

    def __init__(self, ctx):
        self.ctx = ctx
        self._conds: dict[str, object] = {}
        args = ctx.args
        self._enc = dict(comfy_encoding=bool(getattr(args, "caption_comfy_encoding", True)),
                         kv_trim=bool(getattr(args, "kv_trim", False)))

    def cond(self, caption: str):
        if caption not in self._conds:
            ctx = self.ctx
            self._conds[caption] = ctx.family.encode_text_for_batch(
                ctx.text_stack, ctx.model, [caption], ctx.device, ctx.dtype, **self._enc)
        return self._conds[caption]

    def velocity(self, x, t, cond):
        ctx = self.ctx
        autocast = (torch.autocast("cuda", dtype=ctx.dtype) if str(ctx.device).startswith("cuda")
                    else contextlib.nullcontext())
        with autocast:
            return ctx.family.forward_train(ctx.model, x.to(ctx.device, ctx.dtype),
                                            t.to(ctx.device), cond)

    def roundtrip(self, latent) -> torch.Tensor:
        """decode -> encode, so a sampled latent carries the VAE encoder's statistics."""
        vae = self.ctx.vae
        with torch.no_grad():
            pixels = vae.decode(latent.to(self.ctx.device)).clamp(-1, 1)
            return vae.encode(pixels).to(latent.dtype).cpu()

    def decode(self, latent) -> torch.Tensor:
        """Latent [1,C,1,h,w] -> RGB [3,H,W] float in [0,1] on CPU."""
        with torch.no_grad():
            pixels = self.ctx.vae.decode(latent.to(self.ctx.device)).squeeze(2)
        return ((pixels[0].float().clamp(-1, 1) + 1) / 2).cpu()


def save_png(rgb: torch.Tensor, path: Path) -> None:
    from PIL import Image

    arr = (rgb.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype("uint8")
    Image.fromarray(arr).save(path)


# --- profile -----------------------------------------------------------------


def run_profile(model: Model, images: list[dict], opts, out: Path, sampling: dict) -> GapProfile:
    ref_path = out / "reference_latents.pt"
    reference: list[torch.Tensor] = []
    if opts.reuse_reference and ref_path.exists():
        saved = torch.load(ref_path, map_location="cpu")
        if [e["caption"] for e in images] == saved["captions"]:
            reference = saved["latents"]
            logger.info("reusing %d reference latents", len(reference))
    if not reference:
        ref_dir = out / "reference"
        ref_dir.mkdir(parents=True, exist_ok=True)
        uncond = model.cond(sampling["negative"])
        sigmas = shifted_sigmas(sampling["steps"], sampling["shift"])
        for k, entry in enumerate(images):
            started = time.perf_counter()
            noise = seeded_noise(entry["latent"].to(model.ctx.device, model.ctx.dtype), 50_000 + k)
            latent = euler_denoise(model.velocity, noise, sigmas, model.cond(entry["caption"]),
                                   uncond=uncond, cfg_scale=sampling["cfg"]).cpu()
            reference.append(latent)
            save_png(model.decode(latent), ref_dir / f"ref_{k:02d}.png")
            logger.info("reference %d/%d generated (%.1fs)", k + 1, len(images), time.perf_counter() - started)
        torch.save({"captions": [e["caption"] for e in images], "latents": reference}, ref_path)

    device, dtype = model.ctx.device, model.ctx.dtype
    artist = [(e["latent"].to(device, dtype), model.cond(e["caption"])) for e in images]
    # Sampler output never went through the VAE encoder, while the artist's
    # latents did; without the round trip the low-noise end compares encoder
    # statistics instead of style.
    refs = [(model.roundtrip(r).to(device, dtype), model.cond(e["caption"])) for r, e in zip(reference, images)]
    return gap_profile(
        model.velocity, artist, refs, edges=bin_edges(opts.bins), seeds=range(opts.seeds),
        progress=lambda label, i, n: logger.info("profile %s %d/%d", label, i, n),
    )


def write_profile(profile: GapProfile, out: Path) -> None:
    edges = profile.edges
    target = profile.target_masses()
    with open(out / "profile.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t_lo", "t_hi", "mse_artist", "mse_reference", "relative_gap", "gap_target_mass"])
        for i in range(len(edges) - 1):
            w.writerow([f"{edges[i]:.3f}", f"{edges[i + 1]:.3f}", f"{profile.mse_artist[i]:.6f}",
                        f"{profile.mse_reference[i]:.6f}", f"{profile.relative_gap[i]:.4f}", f"{target[i]:.4f}"])
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        c = profile.centers.numpy()
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        a1.plot(c, profile.mse_artist.numpy(), "o-", label="artist")
        a1.plot(c, profile.mse_reference.numpy(), "s--", label="base model's own images (VAE round trip)")
        a1.set_yscale("log")
        a1.set_ylabel("denoising MSE")
        a1.legend()
        a1.set_title("Where the base model departs from the artist")
        gap = profile.relative_gap.numpy()
        a2.bar(c, gap, width=float(edges[1] - edges[0]) * 0.9,
               color=["#c0504d" if g > 0 else "#9bbb59" for g in gap], alpha=0.6)
        a2.set_ylabel("artist / reference - 1")
        a2.set_xlabel("t (1 = noise)")
        fig.tight_layout()
        fig.savefig(out / "profile.png", dpi=130)
        plt.close(fig)
    except Exception:  # noqa: BLE001 -- plots are a convenience
        logger.exception("profile plot failed")


def read_profile(out: Path) -> GapProfile | None:
    path = out / "profile.csv"
    if not path.exists():
        return None
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    edges = torch.tensor([float(rows[0]["t_lo"])] + [float(r["t_hi"]) for r in rows], dtype=torch.float64)
    return GapProfile(edges=edges,
                      mse_artist=torch.tensor([float(r["mse_artist"]) for r in rows], dtype=torch.float64),
                      mse_reference=torch.tensor([float(r["mse_reference"]) for r in rows], dtype=torch.float64))


def parse_trait_weights(text: str) -> dict[str, float]:
    weights = {}
    for part in text.split(","):
        if part.strip():
            name, value = part.split("=")
            if name.strip() not in TRAIT_IDENTITY:
                raise ValueError(f"unknown trait {name!r}; known: {sorted(TRAIT_IDENTITY)}")
            weights[name.strip()] = float(value)
    return weights


def write_fit(out: Path, opts, profile: GapProfile | None, summary: list[dict] | None) -> dict:
    """Fit dual_peak to the ladder's trait target (or the gap target without a ladder)."""
    edges = bin_edges(opts.bins) if profile is None else profile.edges
    if summary:
        weights = parse_trait_weights(opts.trait_weights)
        target = trait_target([s["t0"] for s in summary],
                              {m: [s[m] for s in summary] for m in weights}, edges, weights)
        source = f"лестница ({opts.trait_weights})"
    elif profile is not None:
        target = profile.target_masses()
        source = "профиль разрыва"
    else:
        raise ValueError("nothing to fit: neither a ladder nor a profile")
    default_masses = config_bin_masses(DualPeakConfig(), edges)
    blended = (1 - opts.anchor) * target + opts.anchor * default_masses
    fitted, err = fit_dual_peak(blended, edges)
    fitted_masses = config_bin_masses(fitted, edges)

    lines = ["timestep_sampling: dual_peak"]
    lines += [f"dual_peak_{name}: {value}" for name, value in dataclasses.asdict(fitted).items()]
    (out / "suggested_dual_peak.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(out / "fit.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t_lo", "t_hi", "target_mass", "blended_mass", "default_mass", "suggested_mass"])
        for i in range(len(edges) - 1):
            w.writerow([f"{edges[i]:.3f}", f"{edges[i + 1]:.3f}", f"{target[i]:.4f}", f"{blended[i]:.4f}",
                        f"{default_masses[i]:.4f}", f"{fitted_masses[i]:.4f}"])
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        c = ((edges[:-1] + edges[1:]) / 2).numpy()
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(c, target.numpy(), "k:", label=f"target: {source}")
        ax.plot(c, default_masses.numpy(), "-", color="#4f81bd", label="current default")
        ax.plot(c, fitted_masses.numpy(), "-", color="#c0504d", linewidth=2, label="suggested")
        ax.set_xlabel("t (1 = noise)")
        ax.set_ylabel("probability per bin")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "fit.png", dpi=130)
        plt.close(fig)
    except Exception:  # noqa: BLE001
        logger.exception("fit plot failed")
    return {"fitted": fitted, "fit_error_l1": err, "source": source,
            "masses": fitted_masses, "default_masses": default_masses}


# --- ladder ------------------------------------------------------------------


def run_ladder(model: Model, images: list[dict], opts, out: Path, sampling: dict) -> list[dict]:
    from PIL import Image, ImageDraw

    ladder_t = [float(x) for x in opts.ladder_t.split(",") if x.strip()]
    columns = [*ladder_t, 1.0]
    ladder_dir = out / "ladder"
    ladder_dir.mkdir(parents=True, exist_ok=True)
    uncond = model.cond(sampling["negative"])
    full = shifted_sigmas(sampling["steps"], sampling["shift"])
    rows = []
    for k, entry in enumerate(evenly(images, opts.ladder_images)):
        x0 = entry["latent"].to(model.ctx.device, model.ctx.dtype)
        noise = seeded_noise(x0, 70_000 + k)
        cond = model.cond(entry["caption"])
        original = model.decode(x0)
        tiles = [("original", original)]
        for t0 in columns:
            x_t = ((1 - t0) * x0.float() + t0 * noise.float()).to(x0.dtype)
            result = euler_denoise(model.velocity, x_t, truncate_schedule(full, t0), cond,
                                   uncond=uncond, cfg_scale=sampling["cfg"])
            rgb = model.decode(result)
            tiles.append((f"t0={t0:g}", rgb))
            rows.append({"image": Path(entry["image"]).name, "t0": t0, **compare_images(original, rgb)})
        logger.info("ladder %d/%d done", k + 1, opts.ladder_images)

        thumb = 320
        h, w = original.shape[-2:]
        th = round(thumb * h / w)
        sheet = Image.new("RGB", (thumb * len(tiles), th + 24), "white")
        draw = ImageDraw.Draw(sheet)
        for i, (label, rgb) in enumerate(tiles):
            arr = (rgb.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype("uint8")
            sheet.paste(Image.fromarray(arr).resize((thumb, th)), (i * thumb, 24))
            draw.text((i * thumb + 6, 5), label, fill="black")
        sheet.save(ladder_dir / f"ladder_{k:02d}.png")

    with open(out / "ladder_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "t0", "palette", "shapes", "layout", "detail"])
        w.writeheader()
        w.writerows(rows)
    return rows


def ladder_summary(rows: list[dict]) -> list[dict]:
    by_t: dict[float, list[dict]] = {}
    for r in rows:
        by_t.setdefault(r["t0"], []).append(r)
    return [{"t0": t, **{m: sum(r[m] for r in rs) / len(rs) for m in ("palette", "shapes", "layout", "detail")}}
            for t, rs in sorted(by_t.items())]


def plot_ladder(summary: list[dict], out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        t = [s["t0"] for s in summary]
        fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
        for ax, (m, title) in zip(axes, (("palette", "palette distance (Lab)"),
                                         ("shapes", "edge-map correlation"),
                                         ("layout", "layout correlation"),
                                         ("detail", "high-freq energy ratio"))):
            ax.plot(t, [s[m] for s in summary], "o-")
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("t0")
        fig.tight_layout()
        fig.savefig(out / "ladder_metrics.png", dpi=130)
        plt.close(fig)
    except Exception:  # noqa: BLE001
        logger.exception("ladder plot failed")


# --- trained LoRA: where it learned, which window carries what -------------------


def _set_adapter_scale(adapters, scale: float) -> None:
    """Same multiplier path as anima_generate's XY matrix, for every injected adapter."""
    for adapter in adapters:
        if callable(getattr(adapter, "set_multiplier", None)):
            adapter.set_multiplier(scale)
            continue
        network = getattr(adapter, "network", None)
        if network is None:
            continue
        network.multiplier = float(scale)
        for lora in getattr(network, "loras", []):
            if hasattr(lora, "multiplier"):
                lora.multiplier = float(scale)


def parse_windows(text: str) -> list[tuple[float, float, bool]]:
    """``"0.85-1,!0.6-0.85"`` -> ``[(0.85, 1.0, False), (0.6, 0.85, True)]``.

    A leading ``!`` inverts the window: the LoRA runs everywhere except there,
    which shows what that band adds once the composition is already the LoRA's.
    """
    windows = []
    for part in text.split(","):
        part = part.strip()
        if part:
            inverted = part.startswith("!")
            lo, hi = (float(x) for x in part.lstrip("!").split("-"))
            if not 0.0 <= lo < hi <= 1.0:
                raise ValueError(f"bad window {part!r}: need 0 <= lo < hi <= 1")
            windows.append((lo, hi, inverted))
    return windows


def _window_scale(lo: float, hi: float, inverted: bool, scale: float):
    def scale_at(t: float) -> float:
        inside = lo <= t < hi or (hi == 1.0 and t >= lo)
        return scale if inside != inverted else 0.0
    return scale_at


def run_lora_probe(model: Model, images: list[dict], opts, out: Path, sampling: dict) -> dict:
    """Learned profile (error removed per t) + window generations for a trained LoRA."""
    from PIL import Image, ImageDraw

    from studio.services.inference.core import LoRASpec, apply_loras

    ctx = model.ctx
    adapters = apply_loras(ctx.model, [LoRASpec(path=str(opts.lora), scale=opts.lora_scale)], ctx.device,
                           torch.float32, family_id=ctx.family.spec.family_id, text_stack=ctx.text_stack)
    set_scale = lambda s: _set_adapter_scale(adapters, s)  # noqa: E731
    lora_dir = out / "lora"
    lora_dir.mkdir(parents=True, exist_ok=True)

    # 1. Learned profile on the LoRA's own dataset images (same noise with and without).
    edges = bin_edges(opts.bins)
    centers = ((edges[:-1] + edges[1:]) / 2).tolist()
    device, dtype = ctx.device, ctx.dtype
    mse = {0.0: torch.zeros(len(centers), dtype=torch.float64),
           opts.lora_scale: torch.zeros(len(centers), dtype=torch.float64)}
    for k, entry in enumerate([] if opts.skip_profile else images):
        x0, cond = entry["latent"].to(device, dtype), model.cond(entry["caption"])
        for scale in mse:
            set_scale(scale)
            mse[scale] += denoise_mse(model.velocity, x0, cond, centers, [s + 1000 * k for s in range(opts.seeds)])
        logger.info("lora profile %d/%d", k + 1, len(images))
    learned = None
    if not opts.skip_profile:
        base, tuned = mse[0.0] / len(images), mse[opts.lora_scale] / len(images)
        learned = 1 - tuned / base.clamp_min(1e-12)
        with open(lora_dir / "learned_profile.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t_lo", "t_hi", "mse_base", "mse_lora", "error_removed"])
            for i in range(len(centers)):
                w.writerow([f"{edges[i]:.3f}", f"{edges[i + 1]:.3f}", f"{base[i]:.6f}", f"{tuned[i]:.6f}",
                            f"{learned[i]:.4f}"])

    # 2. Window generations: LoRA on only inside [lo, hi) of t.
    windows = parse_windows(opts.windows)
    variants = [("none", lambda t: 0.0), ("full", lambda t: opts.lora_scale)]
    variants += [(f"{'!' if inv else ''}{lo:g}-{hi:g}", _window_scale(lo, hi, inv, opts.lora_scale))
                 for lo, hi, inv in windows]
    sigmas = shifted_sigmas(sampling["steps"], sampling["shift"])
    uncond = model.cond(sampling["negative"])
    rows = []
    for k, entry in enumerate(evenly(images, opts.ladder_images)):
        x0 = entry["latent"].to(device, dtype)
        noise = seeded_noise(x0, 90_000 + k)
        cond = model.cond(entry["caption"])
        dataset_rgb = model.decode(x0)
        results = {}
        for name, scale_at in variants:
            velocity = scheduled_velocity(model.velocity, set_scale, scale_at)
            results[name] = model.decode(euler_denoise(velocity, noise, sigmas, cond, uncond=uncond,
                                                       cfg_scale=sampling["cfg"]))
        none_vs_full = compare_images(results["full"], results["none"])
        for name, rgb in results.items():
            vs_full = compare_images(results["full"], rgb)
            row = {"image": Path(entry["image"]).name, "variant": name,
                   "pose_copy": compare_images(dataset_rgb, rgb)["layout"]}
            if name not in ("none", "full"):
                row.update({f"share_{m}": v for m, v in effect_share(vs_full, none_vs_full).items()})
            rows.append(row)
        logger.info("lora windows %d/%d", k + 1, opts.ladder_images)

        tiles = [("dataset", dataset_rgb), *results.items()]
        thumb = 300
        h, wd = dataset_rgb.shape[-2:]
        th = round(thumb * h / wd)
        sheet = Image.new("RGB", (thumb * len(tiles), th + 24), "white")
        draw = ImageDraw.Draw(sheet)
        for i, (label, rgb) in enumerate(tiles):
            arr = (rgb.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype("uint8")
            sheet.paste(Image.fromarray(arr).resize((thumb, th)), (i * thumb, 24))
            if label.startswith("!"):
                label = f"LoRA except {label[1:]}"
            elif label not in ("dataset", "none", "full"):
                label = f"LoRA {label}"
            draw.text((i * thumb + 6, 5), label, fill="black")
        sheet.save(lora_dir / f"windows_{k:02d}.png")
    set_scale(opts.lora_scale)

    fields = ["image", "variant", "pose_copy", "share_palette", "share_shapes", "share_layout"]
    with open(lora_dir / "windows.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    summary = []
    for name, _ in variants:
        sel = [r for r in rows if r["variant"] == name]
        item = {"variant": name, "pose_copy": sum(r["pose_copy"] for r in sel) / len(sel)}
        for m in ("palette", "shapes", "layout"):
            if f"share_{m}" in sel[0]:
                item[m] = sum(r[f"share_{m}"] for r in sel) / len(sel)
        summary.append(item)

    lines = [f"# LoRA probe: `{Path(opts.lora).name}`", "",
             f"Масштаб {opts.lora_scale}, изображений в профиле {len(images)}, "
             f"генераций на вариант {opts.ladder_images}, шаги {sampling['steps']}, CFG {sampling['cfg']}.", "",
             ]
    if learned is not None:
        lines += ["## Где LoRA научилась", "",
                  "Доля ошибки шумоподавления, которую LoRA убрала на своём датасете (с LoRA vs без, одинаковый шум).",
                  "", "| t | " + " | ".join(f"{c:.2f}" for c in centers[::2]) + " |",
                  "|---|" + "---|" * len(centers[::2]),
                  "| убрано | " + " | ".join(f"{float(v) * 100:+.1f}%" for v in learned[::2]) + " |", ""]
    lines += ["## LoRA на окнах t", "",
             "Доля эффекта полной LoRA (1 = как полная LoRA, 0 = как без LoRA). `a-b` — LoRA только на окне, "
             "`!a-b` — везде, кроме окна. «Копия позы» — корреляция композиции генерации с картинкой "
             "датасета для того же капшена.", "",
             "| вариант | палитра | формы | композиция | копия позы |", "|---|---|---|---|---|"]
    for s in summary:
        cells = [f"{s[m]:.2f}" if m in s else "—" for m in ("palette", "shapes", "layout")]
        lines.append(f"| {s['variant']} | {' | '.join(cells)} | {s['pose_copy']:.2f} |")
    lines += ["", "Сетки: `lora/windows_XX.png`.", ""]
    (lora_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return {"learned": learned, "summary": summary}


# --- report ------------------------------------------------------------------

_BANDS = ((0.0, 0.2), (0.2, 0.45), (0.45, 0.75), (0.75, 0.92), (0.92, 1.0))


def _band_masses(masses: torch.Tensor, edges: torch.Tensor) -> list[float]:
    centers = (edges[:-1] + edges[1:]) / 2
    return [float(masses[(centers >= lo) & (centers < hi)].sum()) for lo, hi in _BANDS]


def write_report(out: Path, meta: dict, fit: dict | None, profile: GapProfile | None,
                 summary: list[dict] | None) -> None:
    lines = ["# Noise gap probe", "",
             f"- Конфиг: `{meta['config']}`",
             f"- Семейство: `{meta['family']}`, изображений: {meta['images']}, "
             f"шаги {meta['steps']}, CFG {meta['cfg']}, shift {meta['shift']:.3g}", ""]
    if summary:
        lines += ["## Лестница зашумления", "",
                  "Изображение зашумлено до t0 и дорисовано базовой моделью. Чем дальше метрика "
                  "от исходника, тем больше эта черта решается выше t0.", "",
                  "| t0 | палитра ΔLab | формы (корр.) | композиция (корр.) | детали (отн.) |",
                  "|---|---|---|---|---|"]
        lines += [f"| {s['t0']:g} | {s['palette']:.2f} | {s['shapes']:.3f} | {s['layout']:.3f} | {s['detail']:.2f} |"
                  for s in summary]
        lines += ["", "t0=1 — генерация базовой моделью с нуля по тому же капшену. "
                  "Сетки: `ladder/ladder_XX.png`.", "", "![ladder](ladder_metrics.png)", ""]
    if fit is not None:
        edges = profile.edges if profile is not None else bin_edges(len(fit["masses"]))
        head = "| | " + " | ".join(f"{lo:g}–{hi:g}" for lo, hi in _BANDS) + " |"
        default_row = " | ".join(f"{m * 100:.1f}%" for m in _band_masses(fit["default_masses"], edges))
        fitted_row = " | ".join(f"{m * 100:.1f}%" for m in _band_masses(fit["masses"], edges))
        lines += ["## Предлагаемый dual_peak", "",
                  f"Цель: {fit['source']}, смешана с текущим дефолтом (anchor). "
                  f"Ошибка подгонки L1 {fit['fit_error_l1']:.3f}.", "",
                  head, "|---|" + "---|" * len(_BANDS),
                  f"| текущий дефолт | {default_row} |",
                  f"| предложение | {fitted_row} |",
                  "", "```yaml", (out / "suggested_dual_peak.yaml").read_text(encoding="utf-8").rstrip(), "```",
                  "", "![fit](fit.png)", ""]
    if profile is not None:
        gap = profile.relative_gap
        c = profile.centers
        lines += ["## Профиль разрыва (диагностика)", "",
                  "Ошибка шумоподавления базовой модели на артах / на её собственных картинках − 1. "
                  "На низком шуме сюда входит и непредсказуемая мелочь датасета (зерно, артефакты сжатия), "
                  "поэтому для подбора семплера основой служит лестница.", "",
                  "| t | " + " | ".join(f"{float(x):.2f}" for x in c[::2]) + " |",
                  "|---|" + "---|" * len(c[::2]),
                  "| разрыв | " + " | ".join(f"{float(g) * 100:+.0f}%" for g in gap[::2]) + " |",
                  "", "![profile](profile.png)", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def read_ladder(out: Path) -> list[dict] | None:
    path = out / "ladder_metrics.csv"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        rows = [{k: (v if k == "image" else float(v)) for k, v in r.items()} for r in csv.DictReader(f)]
    return ladder_summary(rows)


def main(argv=None) -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    opts = parse_args(argv)
    config_path = Path(opts.config)
    out = Path(opts.out or f"noise_probe_out/{time.strftime('%Y%m%d_%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.json"

    if opts.refit_only:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        profile, summary = read_profile(out), read_ladder(out)
        write_report(out, meta, write_fit(out, opts, profile, summary), profile, summary)
        logger.info("refit done: %s", out / "report.md")
        return

    started = time.perf_counter()

    def needed_prompts(args, images):
        trigger = str(getattr(args, "trigger_word", "") or "")
        if opts.strip_trigger:
            for e in images:
                e["caption"] = strip_trigger(e["caption"], trigger)
        return list(dict.fromkeys([*(e["caption"] for e in images),
                                   str(getattr(args, "sample_negative_prompt", "") or "")]))

    ctx, images = build_stack(config_path, out / "_work", needed_prompts)
    args = ctx.args
    images = evenly(images, opts.max_images)
    steps = opts.steps or min(40, int(getattr(args, "sample_infer_steps", 30) or 30))
    sampling = {
        "steps": steps,
        "cfg": float(opts.cfg if opts.cfg is not None else getattr(args, "sample_cfg_scale", 4.0)),
        "shift": float(opts.shift if opts.shift is not None else _family_shift(ctx.family, args)),
        "negative": str(getattr(args, "sample_negative_prompt", "") or ""),
    }
    logger.info("probe: family=%s images=%d steps=%d cfg=%.2f shift=%.3g",
                ctx.family.spec.family_id, len(images), steps, sampling["cfg"], sampling["shift"])
    model = Model(ctx)

    if opts.lora:
        run_lora_probe(model, images, opts, out, sampling)
        logger.info("done: %s", out / "lora" / "report.md")
        return

    # A skipped half keeps the previous run's results, so the halves can run separately.
    profile, summary = read_profile(out), read_ladder(out)
    if not opts.skip_profile:
        profile = run_profile(model, images, opts, out, sampling)
        write_profile(profile, out)
    if not opts.skip_ladder:
        summary = ladder_summary(run_ladder(model, images, opts, out, sampling))
        plot_ladder(summary, out)

    meta = {"config": str(config_path), "family": ctx.family.spec.family_id, "images": len(images),
            "steps": steps, "cfg": sampling["cfg"], "shift": sampling["shift"],
            "minutes": (time.perf_counter() - started) / 60}
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_report(out, meta, write_fit(out, opts, profile, summary), profile, summary)
    logger.info("done: %s", out / "report.md")


if __name__ == "__main__":
    main()
