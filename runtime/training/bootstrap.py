"""Startup-time utilities: dependency detection, YAML config loading, progress bar init.

Extracted from the original runtime/anima_train.py L60-180 (ADR 0003 PR-A).

Public functions:
- ensure_dependencies -- detect and optionally auto-install missing dependencies
- load_yaml_config / apply_yaml_config -- YAML config -> args merge
- init_progress -- Rich progress bar initialization
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def ensure_dependencies(auto_install: bool = False) -> None:
    """Detect and optionally auto-install missing dependencies."""
    required = {
        "numpy": "numpy",
        "PIL": "Pillow",
        "safetensors": "safetensors",
        "transformers": "transformers",
        "einops": "einops",
        "torchvision": "torchvision",
        "yaml": "pyyaml",
    }
    missing = []
    for module_name, pip_name in required.items():
        try:
            __import__(module_name)
        except Exception:
            missing.append(pip_name)
    if not missing:
        return
    missing_list = ", ".join(sorted(set(missing)))
    print(f"Missing dependencies: {missing_list}")
    if not auto_install:
        print(f"Install them with:\n  {sys.executable} -m pip install {missing_list}")
        raise SystemExit(1)
    cmd = [sys.executable, "-m", "pip", "install", *sorted(set(missing))]
    print("Installing missing dependencies...")
    try:
        subprocess.run(cmd, check=False)
    except Exception as exc:
        print(f"Auto-install failed: {exc}")
        raise SystemExit(1)
    still_missing = []
    for module_name, pip_name in required.items():
        try:
            __import__(module_name)
        except Exception:
            still_missing.append(pip_name)
    if still_missing:
        still_list = ", ".join(sorted(set(still_missing)))
        print(f"Still missing: {still_list}")
        raise SystemExit(1)


def load_yaml_config(config_path):
    """Load a YAML config file."""
    try:
        import yaml
    except ImportError:
        print("PyYAML not installed. Install with: pip install pyyaml")
        raise SystemExit(1)

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        config = {}

    return config


def apply_yaml_config(args, config):
    """Merge YAML with explicit CLI arguments and return new args, fully constructed via TrainingConfig.

    Config pipeline cut 1 (R1, docs/design/config-pipeline-refactor.md): the trainer and
    Studio now go through the same pydantic loading path -- field migration, the
    FAMILY_CONFIG_DEFAULTS per-family default overlay, and mutual-exclusion/capability
    validation all take effect from this single point, so this function no longer
    manually replays migrations (that was a relic of the old merge_yaml_into_namespace,
    which bypassed the validator).

    Explicit command-line arguments take priority over YAML: parse_args builds the
    parser with suppress_defaults, so args only contains explicitly-set keys --
    priority is an exact determination, not an approximation based on "value == default".

    On validation failure, errors are printed to stderr one by one followed by
    SystemExit(2) -- the same fail-fast pattern as the capability guard rails; the
    supervisor captures the tail of stderr as the task's error message.
    """
    from pydantic import ValidationError

    from studio.infrastructure.argparse_bridge import namespace_from_config
    from studio.schema import TrainingConfig

    try:
        return namespace_from_config(args, dict(config or {}), TrainingConfig)
    except ValidationError as exc:
        errors = exc.errors()
        print(f"Config validation failed ({len(errors)} issue(s)):", file=sys.stderr)
        for err in errors:
            loc = ".".join(str(p) for p in err["loc"]) or "config"
            print(f"  {loc}: {err['msg']}", file=sys.stderr)
        raise SystemExit(2) from exc


def init_progress(show_progress, total_steps):
    """Initialize the Rich progress bar.

    Returns `(progress, task_id, kind)`:
    - `(None, None, None)` when progress is disabled
    - `(Progress instance, task_id, "rich")` when Rich is available
    - `("plain", None, None)` when Rich is missing (main() falls back to plain-text progress)

    A non-tty (a pipe from a studio spawn) is forced down the log_every plain-text
    branch -- under a pipe, rich both spams the screen and swallows step lines
    (log_every is an elif), and a config baked with the old ``no_progress: false``
    default (the field is now hidden) used to result in zero step log lines in the
    task log. A bare-terminal CLI is unaffected.
    """
    if not show_progress:
        return None, None, None
    import sys as _sys

    if not (hasattr(_sys.stdout, "isatty") and _sys.stdout.isatty()):
        return None, None, None
    try:
        from rich.progress import (
            BarColumn, MofNCompleteColumn, Progress, TextColumn,
            TimeElapsedColumn, TimeRemainingColumn,
        )
        progress = Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("loss={task.fields[loss]:.4f}"),
            TextColumn("lr={task.fields[lr]:.2e}"),
            TextColumn("speed={task.fields[speed]:.2f} it/s"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            refresh_per_second=10,
        )
        task = progress.add_task("train", total=total_steps, loss=0.0, lr=0.0, speed=0.0)
        return progress, task, "rich"
    except Exception:
        return "plain", None, None
