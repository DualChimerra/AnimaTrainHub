"""CLI / interactive mode entry point: parse_args + prompt_for_args.

Extracted from the original runtime/anima_train.py L1963-2103 (ADR 0003 PR-A).
Called directly by test_anima_train_migration.py via parse_args / apply_yaml_config.

Public:
- parse_args -- goes through studio.argparse_bridge.build_parser, auto-generated from TrainingConfig
- prompt_for_args -- interactive mode, fills in missing fields
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional


def parse_args():
    """Auto-generates the parser from studio.schema.TrainingConfig; adds the
    CLI-only switches that are outside the schema (auto-install / interactive /
    no-live-curve / the deprecated --repeats and --reg-repeats).

    suppress_defaults=True (config pipeline cut 1 / R1): schema fields get no
    argparse default, so the parse result only contains keys the user passed
    explicitly. Full defaults are filled in uniformly by apply_yaml_config via
    TrainingConfig construction (migration / family default overlay /
    validation all happen in one place); CLI-only switches keep their normal
    defaults and are always present in the namespace.
    """
    from studio.infrastructure.argparse_bridge import build_parser
    from studio.schema import TrainingConfig

    p = build_parser(
        TrainingConfig, prog="anima_train",
        description="Anima LoRA Trainer v2", suppress_defaults=True,
    )
    # CLI-only switches outside the schema
    p.add_argument("--auto-install", action="store_true", help="Automatically install missing dependencies")
    p.add_argument("--interactive", action="store_true", help="Interactive mode, prompt for missing arguments")
    p.add_argument("--no-live-curve", action="store_true", help="Disable live loss curve refresh")
    # PP6.1 -- monitor state file path; defaults to output_dir/monitor_state.json if not given
    # Note: the old monitor server's --no-monitor / --monitor-host / --monitor-port /
    # --no-browser were removed along with the TrainingConfig fields (old yaml keys are silently dropped).
    p.add_argument(
        "--monitor-state-file",
        type=str,
        default=None,
        help="Training monitor state.json output path (defaults to output_dir/monitor_state.json)",
    )
    # Deprecated: per-image repeat now uses a folder name prefix instead (e.g. 5_concept)
    p.add_argument("--repeats", type=int, default=1, help=argparse.SUPPRESS)
    p.add_argument("--reg-repeats", type=int, default=1, help=argparse.SUPPRESS)
    return p.parse_args()


# ============================================================================
# Interactive mode helper functions
# ============================================================================

def _try_rich():
    try:
        from rich.prompt import Prompt, Confirm
        return Prompt, Confirm
    except Exception:
        return None, None


def _ask_str(label, default=""):
    Prompt, _ = _try_rich()
    if Prompt:
        return Prompt.ask(label, default=default) if default else Prompt.ask(label)
    raw = input(f"{label}{f' [{default}]' if default else ''}: ").strip()
    return raw or default


def _ask_bool(label, default=False):
    _, Confirm = _try_rich()
    if Confirm:
        return Confirm.ask(label, default=default)
    raw = input(f"{label} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes", "1", "true", "t")


def _ask_int(label, default):
    while True:
        raw = _ask_str(label, str(default))
        try:
            return int(raw)
        except ValueError:
            print("Please enter an integer.")


def _ask_float(label, default):
    while True:
        raw = _ask_str(label, str(default))
        try:
            return float(raw)
        except ValueError:
            print("Please enter a number.")


def _guess_default_paths():
    """Guess default model paths (used only when the user hasn't specified
    them explicitly in yaml/CLI).

    Root dir: prefers `secrets.models.root` (configured on the Studio
    settings page), otherwise `REPO_ROOT/models/` (matching the schema.py
    default and the `models/wd14/` already used by WD14).

    Transformer: the user may have several Anima versions installed
    (preview / preview2 / preview3-base / 1.0); find the first one that
    exists in ANIMA_VARIANTS order (latest first).
    """
    # Note: the original runtime/anima_train.py used Path(__file__).resolve().parent
    # to get runtime/; here cli.py lives under runtime/training/, so one extra
    # level up is needed to keep the same semantics.
    repo_root = Path(__file__).resolve().parent.parent
    # secrets may not be importable (the studio package is available when
    # running training straight from the CLI; this is a fallback for other cases)
    base: Optional[Path] = None
    transformer_path: str = ""
    try:
        from studio.services.models import find_anima_main, models_root
        base = models_root()
        existing = find_anima_main(base)
        if existing:
            transformer_path = str(existing)
    except Exception:
        base = repo_root / "models"
    if not base:
        base = repo_root / "models"
    if not transformer_path:
        # services unavailable / nothing downloaded -> give the latest version's
        # default filename as a hint so the user can fill in the path
        candidate = base / "diffusion_models" / "anima-base-v1.0.safetensors"
        transformer_path = str(candidate) if candidate.exists() else ""

    vae = base / "vae" / "qwen_image_vae.safetensors"
    qwen = base / "text_encoders"
    return {
        "transformer": transformer_path,
        "vae": str(vae) if vae.exists() else "",
        "qwen": str(qwen) if qwen.exists() else "",
    }


def prompt_for_args(args):
    """Interactively prompt for missing arguments."""
    defaults = _guess_default_paths()
    args.data_dir = args.data_dir or _ask_str("Dataset directory (images + .txt)", "")
    args.transformer_path = args.transformer_path or _ask_str("Transformer path (.safetensors)", defaults["transformer"])
    args.vae_path = args.vae_path or _ask_str("VAE path (.safetensors)", defaults["vae"])
    args.text_encoder_path = args.text_encoder_path or _ask_str("Qwen model directory", defaults["qwen"])
    args.output_dir = _ask_str("Output directory", args.output_dir)
    args.output_name = _ask_str("Output name", args.output_name)
    # resolution is now list[int]. A single value (including the default
    # [1024]) still gets asked interactively; if multiple resolutions are
    # already set (GUI / config driven) we leave it alone.
    _res = args.resolution
    if not isinstance(_res, (list, tuple)):
        args.resolution = [_ask_int("Resolution", int(_res))]
    elif len(_res) <= 1:
        args.resolution = [_ask_int("Resolution", int(_res[0]) if _res else 1024)]
    args.batch_size = _ask_int("Batch size", args.batch_size)
    args.grad_accum = _ask_int("Gradient accumulation", args.grad_accum)
    args.learning_rate = _ask_float("Learning rate", args.learning_rate)
    args.grad_checkpoint = _ask_bool("Enable gradient checkpointing?", args.grad_checkpoint)
    args.epochs = _ask_int("Epochs", args.epochs)
    args.max_steps = _ask_int("Max steps (0=unlimited)", args.max_steps)
    args.lora_rank = _ask_int("LoRA rank", args.lora_rank)
    args.lora_alpha = _ask_float("LoRA alpha", args.lora_alpha)
    args.loss_curve_steps = _ask_int("Loss curve steps (0=disabled)", args.loss_curve_steps)
    args.auto_install = _ask_bool("Automatically install missing dependencies?", args.auto_install)
    args.save_every_epoch = _ask_bool("Save every epoch?", args.save_every_epoch)
    args.mixed_precision = _ask_str("Mixed precision (bf16/fp32)", args.mixed_precision)
    return args
