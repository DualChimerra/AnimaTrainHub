"""Runtime mode (Colab / Local) -- detection, resolution, and the effective value.

This fork serves two kinds of users at once:

- **Colab / Kaggle and other cloud notebooks**: the process runs in a temporary container,
  the browser is on a different machine, and the port has to be exposed through the notebook's
  proxy; the disk can be reclaimed at any time, so `studio_data` is often pointed at a fast
  local disk via `ALS_STUDIO_DATA` and synced to Drive separately.
- **Local machine (Windows / Linux / macOS)**: the browser and the process share the same
  machine, so listening on 127.0.0.1 is enough, and the browser opens automatically on startup;
  the disk is persistent, and `studio_data/` sits right next to the repo.

The two have conflicting reasonable defaults (bind host, whether to open a browser, whether to
prompt for Drive sync, etc), and previously each had to remember this via its own CLI flag.
Here it's pulled out into an explicit, first-class setting:

    secrets.runtime.mode = "" | "local" | "colab"

`""` = the user hasn't chosen yet -- the frontend pops a picker once on app entry
(`RuntimeModePicker`), and once saved it won't ask again. The detection result (`detect()`) is
only used to **pre-select** an option, never to decide for the user, since detection always has
room for false positives (a self-hosted JupyterHub, local training running inside docker, etc).

The `ALS_RUNTIME_MODE` env var has the highest priority and is never persisted: a Colab
notebook's startup cell injects it so cloud users work out of the box without being asked;
local users don't set it and go through the secrets-stored choice instead.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal

Mode = Literal["local", "colab"]

MODES: tuple[str, ...] = ("local", "colab")

#: Stored value when nothing has been chosen. Deliberately not None -- keeps the pydantic
#: field a plain str, so the frontend can just check for emptiness.
UNSET = ""

ENV_OVERRIDE = "ALS_RUNTIME_MODE"


def normalize(value: object) -> str:
    """Normalizes any input value to `"local"` / `"colab"` / `""` (unknown values -> unselected)."""
    text = str(value or "").strip().lower()
    if text in ("kaggle", "cloud", "notebook"):
        # Similar cloud notebook environments are all folded into the colab bucket (they need
        # identical behavior).
        return "colab"
    if text in ("pc", "desktop", "localhost"):
        return "local"
    return text if text in MODES else UNSET


def detect_signals() -> dict[str, bool]:
    """Collects the signals used to identify a cloud notebook, for `detect()` and the
    `/api/runtime` diagnostic display.

    Each one only reads the environment / filesystem and avoids importing heavy modules
    (`google.colab` only exists on Colab; checked via sys.modules instead of a real import --
    a real import is a several-tens-of-milliseconds failure path off Colab, and some images
    print a warning for it).
    """
    env = os.environ
    return {
        # Variables injected by the Colab runtime itself, the most reliable signal.
        "colab_env": any(
            key in env
            for key in ("COLAB_RELEASE_TAG", "COLAB_GPU", "COLAB_JUPYTER_IP",
                        "COLAB_BACKEND_VERSION")
        ),
        # The colab module has already been imported (in a notebook, `from google.colab import
        # drive` is nearly always a required step).
        "colab_module": "google.colab" in sys.modules,
        # Colab's working-directory convention. Not enough on its own to decide (some people
        # create /content locally too), so detect() below requires it to combine with another
        # signal.
        "content_dir": Path("/content").is_dir(),
        # Kaggle notebook.
        "kaggle_env": any(
            key in env
            for key in ("KAGGLE_KERNEL_RUN_TYPE", "KAGGLE_URL_BASE",
                        "KAGGLE_DATA_PROXY_TOKEN")
        ),
    }


def detect() -> Mode:
    """Guesses whether the current environment is a cloud notebook or a local machine.

    Prefers guessing `local`: guessing colab by mistake gives a local user a 0.0.0.0 bind and
    "don't open a browser", which is more likely to leave someone stuck than the reverse; and
    the picker shows the detection result so the user can just change it.
    """
    signals = detect_signals()
    if signals["colab_env"] or signals["colab_module"] or signals["kaggle_env"]:
        return "colab"
    # `content_dir` is only included in signals for the user to self-check, not used as a
    # standalone deciding factor -- /content can exist locally too (a mount point, created by
    # someone else's script), so flipping to colab on this alone would false-positive too often.
    return "local"


def env_override() -> str:
    """The normalized value of `ALS_RUNTIME_MODE` (unset or invalid -> `""`)."""
    return normalize(os.environ.get(ENV_OVERRIDE))


def stored() -> str:
    """The mode the user chose, as stored in secrets (unselected -> `""`).

    secrets depends on the STUDIO_DATA path; the import is kept inside the function to avoid
    a paths -> secrets -> paths cycle within infrastructure.
    """
    try:
        from . import secrets as secrets_mod

        return normalize(secrets_mod.load().runtime.mode)
    except Exception:
        # A corrupted / not-yet-created secrets.json shouldn't block startup; treat it as
        # "never selected".
        return UNSET


def resolve() -> str:
    """The final effective mode: env override -> user selection -> `""` (unselected).

    Note this deliberately does **not** fall back to `detect()` -- "never selected" is a state
    the frontend needs to be able to see; if a detected value silently overrode it, the picker
    would never show up. Callers that need a usable value (e.g. `effective()`) decide their own
    fallback.
    """
    return env_override() or stored()


def effective() -> Mode:
    """Fallback resolution for callers that need "a value, right now" (CLI / backend defaults).

    Uses the detection result when unselected, so the CLI still behaves reasonably before the
    user has ever opened the UI.
    """
    resolved = resolve()
    if resolved in MODES:
        return resolved  # type: ignore[return-value]
    return detect()


def is_colab() -> bool:
    return effective() == "colab"


def describe() -> dict[str, object]:
    """Payload for `/api/runtime`: used both to render the frontend and to let users check why
    they were classified into a particular mode."""
    return {
        "mode": resolve(),
        "stored": stored(),
        "detected": detect(),
        "effective": effective(),
        "env_override": env_override(),
        "locked": bool(env_override()),
        "signals": detect_signals(),
        "modes": list(MODES),
    }
