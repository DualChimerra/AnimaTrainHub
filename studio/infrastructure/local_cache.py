"""Pulls third-party libraries' cache directories into the repo folder (new in this fork).

**Why this is needed**: everything this project owns is already in the repo -- `venv/`,
`models/`, `studio_data/`. But the libraries it depends on each keep their own caches, which
default to the user's home directory / system drive:

| Cache | Default location | Size |
|---|---|---|
| pip wheel cache | `%LOCALAPPDATA%\\pip\\Cache` / `~/.cache/pip` | **several GB** (CUDA torch wheels alone are 2-3GB) |
| HuggingFace hub | `~/.cache/huggingface` | several GB (eval models, tokenizers, download staging) |
| torch hub | `~/.cache/torch` | hundreds of MB |
| ModelScope | `~/.cache/modelscope` | several GB |
| npm | `~/.npm` / `%AppData%\\npm-cache` | hundreds of MB |
| triton | `~/.triton` | hundreds of MB |

Users who put the whole project on a dedicated SSD (the typical local-machine case for this
fork) find their system drive still gets eaten by several GB, and deleting the repo doesn't
take them with it. This module points these libraries' cache env vars at
`<repo>/.cache/<name>` as early as possible during process startup, so "the whole project =
one folder" actually holds true: copy it and you have everything, delete it and it's clean.

**Three rules**:

1. **Never override a value the user already set.** Someone who explicitly set `HF_HOME` is
   probably sharing a large cache across several projects; changing it here would silently
   change their behavior behind their back.
2. **Can be turned off entirely**: `ALS_SYSTEM_CACHES=1` -> this module is a no-op, everything
   goes back to the libraries' default locations (useful when you want multiple checkouts to
   share a pip cache).
3. **Standard library only**, and only touches environment variables: it has to work before
   the venv exists and before a single dependency is installed (`tools/launcher.py` calls it
   before the first `pip install` -- the pip cache is the biggest one of all, and setting it
   even one step too late wastes the whole point).
"""
from __future__ import annotations

import os
from pathlib import Path

#: When set to 1/true/yes/on, this module does nothing (caches go back to the libraries'
#: default locations).
OPT_OUT_ENV = "ALS_SYSTEM_CACHES"

#: Name of the cache root directory inside the repo. Already in .gitignore.
CACHE_DIR_NAME = ".cache"

#: Env var -> subdirectory name under `<cache_root>/`.
#:
#: `HF_HOME` alone covers both the hub cache and the transformers cache (in newer
#: huggingface_hub, `HUGGINGFACE_HUB_CACHE` / `TRANSFORMERS_CACHE` are both deprecated and
#: derived from it), so those two aren't set separately -- doing so would leave a pair of
#: conflicting configs behind whenever the library upgrades.
#:
#: `XDG_CACHE_HOME` is the Linux fallback: libraries without a dedicated var (matplotlib,
#: fontconfig, etc) read it. Harmless on Windows (the libraries that read it don't run there
#: anyway).
_CACHE_ENV_DIRS: dict[str, str] = {
    "PIP_CACHE_DIR": "pip",
    "HF_HOME": "huggingface",
    "TORCH_HOME": "torch",
    "MODELSCOPE_CACHE": "modelscope",
    "TRITON_CACHE_DIR": "triton",
    "WANDB_CACHE_DIR": "wandb",
    # npm reads the lowercase `npm_config_<key>` form; the frontend build's cache is also a
    # few hundred MB.
    "npm_config_cache": "npm",
    "XDG_CACHE_HOME": "xdg",
}


def opted_out() -> bool:
    return str(os.environ.get(OPT_OUT_ENV, "")).strip().lower() in {
        "1", "true", "yes", "on",
    }


def cache_root(repo_root: Path) -> Path:
    """For display purposes. **Don't** re-wrap this with `Path(repo_root)` here -- see apply()."""
    return repo_root / CACHE_DIR_NAME


def apply(repo_root: Path, *, env: dict[str, str] | None = None) -> dict[str, str]:
    """Points the cache env vars at `<repo_root>/.cache/`; returns the entries that were
    **actually written** this call.

    When `env` is omitted, modifies `os.environ` in place (inherited by the current process +
    any child processes it spawns after this); when a dict is passed, only that dict is
    modified (for callers that want to set it only for a child process, e.g. the supervisor).

    Path joining uses `os.path` rather than `pathlib`: the output is meant to be an env var
    string, no need for a Path; and `Path("...")` **picks its flavour based on `os.name`**, so
    any caller that monkeypatches `os.name` (the cli's npm-hint test does exactly this) would
    make this construct a `WindowsPath` on Linux and raise NotImplementedError outright. String
    concatenation has no such coupling.

    Directories are **not created here** -- each library creates its own, and pre-creating a
    pile of empty directories would just add noise to the repo root (and leave empty shells
    behind once a user turns on `ALS_SYSTEM_CACHES`). The one exception is pip: some versions
    won't auto-create their cache directory, see below.

    Returns an empty dict in two cases: the user turned it off entirely (`ALS_SYSTEM_CACHES`),
    or every entry was already explicitly set.
    """
    target = env if env is not None else os.environ
    if opted_out():
        return {}

    root = os.path.join(os.fspath(repo_root), CACHE_DIR_NAME)
    applied: dict[str, str] = {}
    for var, subdir in _CACHE_ENV_DIRS.items():
        # An existing value = an explicit choice by the user (or an outer launcher); leave it alone.
        if str(target.get(var, "")).strip():
            continue
        value = os.path.join(root, subdir)
        target[var] = value
        applied[var] = value

    # pip is the only one that needs pre-creating: some versions just silently give up on
    # caching when the cache directory doesn't exist, so "reinstalling the venv re-downloads
    # 2.5GB of torch" happens silently.
    if "PIP_CACHE_DIR" in applied:
        try:
            os.makedirs(applied["PIP_CACHE_DIR"], exist_ok=True)
        except OSError:
            # Read-only mount / insufficient permissions: let pip fall back to its own default
            # location instead of blocking startup.
            del target["PIP_CACHE_DIR"]
            del applied["PIP_CACHE_DIR"]

    return applied


def describe(repo_root: Path) -> dict[str, str]:
    """The currently effective cache locations (for diagnostics, doesn't modify any env vars).

    Values come from the actual `os.environ` values -- so a value the user set themselves, one
    this module set, and a library's default location after opting out (shown as an empty
    string) are all distinguishable at a glance.
    """
    return {var: str(os.environ.get(var, "")) for var in _CACHE_ENV_DIRS}
