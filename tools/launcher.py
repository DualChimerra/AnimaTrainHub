#!/usr/bin/env python
"""Local one-click launcher -- source for `AnimaLoraStudio.exe` (added in this fork).

Aimed at local users who "downloaded the repo and just want to double-click and go":
on Windows, `studio.bat` requires already knowing how to open a terminal, that PowerShell
needs `.\\`, and if Python isn't installed you just see one line of English error text
before the window vanishes. This launcher reimplements the same bootstrap in Python;
compiled into a single-file exe, double-clicking it works, and errors stay on screen in
plain language.

It **doesn't** bundle the whole app into the exe: torch / CUDA wheels are several GB,
which isn't realistic to stuff into PyInstaller and can't be matched to the user's GPU
anyway. The exe is just a bootstrapper (a few MB, pure stdlib) that does the same thing
`studio.bat` does:

    locate the repo -> create/reuse a venv -> install torch for the GPU -> install
    requirements -> start studio

For that reason this file **must use only the standard library**: it has to run before
the venv exists and before a single requirement is installed.

Usage (source form and exe form are equivalent):
    python tools/launcher.py                start in local mode
    python tools/launcher.py --mode colab   mode passed to studio (cloud usage generally
                                            goes straight through the notebook; this is
                                            mainly for testing)
    python tools/launcher.py --port 8800    passed through to `python -m studio run`
    python tools/launcher.py --reinstall    delete and recreate the venv (studio_data untouched)
    python tools/launcher.py --repo D:\\path manually specify the repo location

Unrecognized arguments are passed through unchanged to `python -m studio run`.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path
from typing import Any, NoReturn, Optional, Sequence

APP_NAME = "AnimaLora Studio"

#: Markers used to decide "is this directory the repo root". Both are required -- checking
#: only requirements.txt would match a bunch of unrelated Python projects.
REPO_MARKERS = ("requirements.txt", "studio/__init__.py")

#: cli.py's installer self-update protocol: cli exit code 42 = the launcher file itself was
#: changed and needs to reload itself. See docs/adr/0002-webui-self-update.md and the
#: matching branch in studio.bat.
INSTALLER_RELOAD_EXIT_CODE = 42

PYPI_MIRROR = "https://mirrors.cloud.tencent.com/pypi/simple/"

MIN_PYTHON = (3, 10)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
#
# **Every string this file prints must be pure ASCII**, the same discipline as studio.bat.
# The Windows console decodes using the system ANSI codepage (Russian cp866, Chinese cp936,
# Japanese cp932); a frozen exe writing a single `->`-style arrow glyph is an instant
# UnicodeEncodeError crash -- and the crash usually happens **in the error path itself**
# (die()'s hint line), so the user doesn't see "Python isn't installed" but a PyInstaller
# traceback instead. Comments and docstrings aren't bound by this rule (they're never
# printed).
#
# The block below is the second line of defense: if someone writes non-ASCII again in the
# future, degrade to `?` instead of letting the launcher crash. Deliberately using
# errors="replace" instead of forcing UTF-8 -- the latter would turn a non-UTF-8 console into
# a screen of garbage, harder to read than a few question marks.


def _make_output_crash_proof() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")  # type: ignore[union-attr]
        except (AttributeError, OSError, ValueError):
            # Redirected to an object that doesn't support reconfigure (a pipe wrapper, or
            # None under pythonw): shouldn't crash even without this second line of defense,
            # since the first one (pure ASCII) still holds.
            pass


_make_output_crash_proof()


def say(msg: str) -> None:
    print(f"[studio] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"[studio] WARNING: {msg}", file=sys.stderr, flush=True)


def die(msg: str, *hints: str) -> NoReturn:
    """Print an error + actionable hints, then hold the window open.

    In the double-click launch scenario, the console closes the instant the process exits,
    so the user never gets to read the error -- this is the entire cause of the "exe just
    flashes and disappears" class of bugs, so every error path pauses unconditionally.
    """
    print(f"\n[studio] ERROR: {msg}", file=sys.stderr, flush=True)
    for hint in hints:
        print(f"         -> {hint}", file=sys.stderr, flush=True)
    pause()
    raise SystemExit(1)


def pause() -> None:
    """Only wait for Enter when actually attached to a real terminal (don't hang in CI / pipes)."""
    if not sys.stdin or not sys.stdin.isatty():
        return
    try:
        input("\n[studio] Press Enter to close...")
    except (EOFError, KeyboardInterrupt):
        pass


# ---------------------------------------------------------------------------
# Locating the repo
# ---------------------------------------------------------------------------


def launcher_dir() -> Path:
    """Directory the exe / script itself lives in.

    PyInstaller onefile extracts its contents to a temp dir before running, and `__file__`
    points at that temp dir -- finding the repo must use `sys.executable` (the real exe
    location) instead.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def looks_like_repo(path: Path) -> bool:
    return all((path / marker).exists() for marker in REPO_MARKERS)


def find_repo(explicit: Optional[str]) -> Path:
    """Find the repo root in order: explicit path -> the exe's directory and its ancestors ->
    the current working directory.

    Walking up ancestors lets the exe live in a subdirectory like `tools/` or `dist/` and
    still work; the depth is capped at 4 -- any deeper and it's no longer a misplacement but
    genuinely the wrong location, which should error out instead of guessing further.
    """
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not looks_like_repo(path):
            die(
                f"{path} is not an AnimaLoraStudio checkout",
                f"expected to find {' and '.join(REPO_MARKERS)} there",
            )
        return path

    candidates: list[Path] = []
    start = launcher_dir()
    candidates.extend([start, *list(start.parents)[:4]])
    cwd = Path.cwd().resolve()
    candidates.extend([cwd, *list(cwd.parents)[:2]])

    for candidate in candidates:
        if looks_like_repo(candidate):
            return candidate

    die(
        "could not find the AnimaLoraStudio files next to this launcher",
        f"put {APP_NAME} in the folder that contains requirements.txt and studio/",
        "or pass --repo <path-to-the-folder>",
    )


# ---------------------------------------------------------------------------
# Cache directory
# ---------------------------------------------------------------------------


def load_local_cache(repo: Path) -> Optional[Any]:
    """Import `studio/infrastructure/local_cache.py` by path.

    Importing by path instead of duplicating the environment-variable list here: the list
    should have exactly one authoritative source, and copying it into two places will
    eventually drift. This works because at this point the repo has already been found, and
    that module is pure standard library (importable even before the venv exists).

    Returns None on failure -- the cache location is an optimization, not a correctness
    requirement, so if it can't be loaded, each library just starts up with its own default
    location. But it must warn: staying silent would make "thought it was redirected, but
    actually wasn't" impossible to notice.
    """
    module_path = repo / "studio" / "infrastructure" / "local_cache.py"
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("_als_local_cache", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as exc:  # noqa: BLE001
        warn(f"could not redirect caches into the project folder ({exc});"
             " they will use their default system locations")
        return None


def apply_local_caches(repo: Path) -> dict[str, str]:
    """Redirect third-party caches into `<repo>/.cache/`.

    **Must be called before the first `pip install`** -- pip's wheel cache is the largest
    of all the caches (a single CUDA torch wheel is 2-3GB); setting this even one step late
    means it's already landed on the system drive.
    """
    module = load_local_cache(repo)
    return dict(module.apply(repo)) if module is not None else {}


# ---------------------------------------------------------------------------
# venv
# ---------------------------------------------------------------------------



def venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def find_existing_venv(repo: Path) -> Optional[Path]:
    """`venv/` takes priority, `.venv/` as fallback -- matches the order used by studio.bat / studio.sh."""
    for name in ("venv", ".venv"):
        candidate = repo / name
        if venv_python(candidate).exists():
            return candidate
    return None


def bootstrap_python() -> list[str]:
    """Pick an interpreter to use for **creating** the venv.

    In a frozen exe, `sys.executable` is the exe itself -- it has no venv module and can't
    do `-m venv`, so a real Python must be found on the system instead. On Windows, `py -3`
    is preferred: many machines keep an old `python` on PATH for backward compatibility with
    older projects, while the py launcher picks the newest 3.x. When not frozen, just use
    the current interpreter directly.
    """
    if not getattr(sys, "frozen", False):
        return [sys.executable]

    if os.name == "nt" and shutil.which("py"):
        return ["py", "-3"]
    for name in ("python3", "python"):
        found = shutil.which(name)
        if found:
            return [found]
    die(
        "Python 3.10+ is not installed (or not on PATH)",
        "install it from https://www.python.org/downloads/",
        'tick "Add python.exe to PATH" in the installer',
    )


def check_version(python_argv: Sequence[str], *, label: str) -> None:
    """Below 3.10, only warn, don't block: some dependencies will fail to install, but the
    user might just want to run an already-installed environment, and hard-blocking would
    cut off their own way to fix it."""
    code = f"import sys;sys.exit(0 if sys.version_info>={MIN_PYTHON} else 1)"
    try:
        rc = subprocess.call([*python_argv, "-c", code])
    except OSError:
        return
    if rc != 0:
        want = ".".join(str(p) for p in MIN_PYTHON)
        warn(f"{label} is older than Python {want}; some dependencies may fail to install")


def create_venv(repo: Path) -> Path:
    """Create the venv. When frozen, spawns a subprocess using the system Python; otherwise uses the built-in venv module."""
    venv_dir = repo / "venv"
    say(f"no virtual environment found; creating {venv_dir} (first run takes a few minutes)")
    if getattr(sys, "frozen", False):
        argv = bootstrap_python()
        check_version(argv, label="python")
        rc = subprocess.call([*argv, "-m", "venv", str(venv_dir)])
        if rc != 0:
            die(
                "failed to create the virtual environment",
                "make sure Python 3.10+ is installed and you can write to this folder",
            )
    else:
        check_version([sys.executable], label=sys.executable)
        venv.EnvBuilder(with_pip=True).create(venv_dir)
    return venv_dir


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


def pip_install(py: Path, args: Sequence[str], *, what: str) -> bool:
    """Try the official PyPI first, fall back to the Tencent mirror on failure (direct connections from mainland China time out chronically). Returns whether it succeeded."""
    base = [str(py), "-m", "pip", "install", *args]
    if subprocess.call(base) == 0:
        return True
    say(f"{what}: pip failed, retrying via mirror...")
    return subprocess.call([*base, "-i", PYPI_MIRROR]) == 0


def install_torch(py: Path, repo: Path, forced_tag: Optional[str]) -> None:
    """Install the right torch for the GPU first, then let requirements.txt run.

    Order matters: requirements.txt only says `torch>=2.0.0`, so running it first would make
    pip pull the **CPU** wheel from PyPI, and every subsequent training run would silently
    run on CPU with no error at all. Installing the CUDA build from the official PyTorch
    index first satisfies the constraint, so pip won't replace it later.
    """
    if forced_tag:
        index = f"https://download.pytorch.org/whl/{forced_tag}"
        say(f"installing torch from {index} (--torch={forced_tag})")
    else:
        index = ""
        helper = repo / "tools" / "select_torch_index.py"
        if helper.exists():
            try:
                out = subprocess.run(
                    [str(py), str(helper)], capture_output=True, text=True, timeout=60
                )
                index = out.stdout.strip()
            except (OSError, subprocess.TimeoutExpired):
                index = ""
        if not index:
            say("no NVIDIA GPU detected; using the default PyTorch build from PyPI")
            return
        say(f"NVIDIA GPU detected; installing torch from {index}")

    if not pip_install(py, ["torch", "torchvision", "--index-url", index], what="torch"):
        warn("CUDA torch install failed; falling back to the PyPI default")
        warn("you can fix this later in Studio > Settings > PyTorch > Reinstall")


def requirements_marker(venv_dir: Path) -> Path:
    return venv_dir / ".studio-requirements.sha256"


def requirements_state(py: Path, repo: Path, marker: Path, *, update: bool = False) -> str:
    """Reuse `tools/check_requirements_changed.py` (content hash, not mtime)."""
    helper = repo / "tools" / "check_requirements_changed.py"
    if not helper.exists():
        return "stale" if not marker.exists() else "current"
    argv = [str(py), str(helper), "--marker", str(marker)]
    if update:
        argv.append("--update-marker")
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=60)
        return out.stdout.strip() or "stale"
    except (OSError, subprocess.TimeoutExpired):
        return "stale"


def ensure_deps(py: Path, repo: Path, venv_dir: Path, *, fresh: bool,
                forced_tag: Optional[str]) -> None:
    marker = requirements_marker(venv_dir)
    req = repo / "requirements.txt"

    if fresh:
        pip_install(py, ["--upgrade", "pip"], what="pip")
        install_torch(py, repo, forced_tag)
        if not req.exists():
            warn("requirements.txt not found, skipping dependency install")
            return
        say("installing Python dependencies (this takes a while on first run)...")
        if not pip_install(py, ["-r", str(req)], what="requirements"):
            die(
                "failed to install dependencies",
                "check your internet connection and run the launcher again",
                "or run it with --reinstall to rebuild the environment from scratch",
            )
        requirements_state(py, repo, marker, update=True)
        return

    if forced_tag:
        # The venv already exists but the user explicitly requested a different CUDA
        # version -- honor it, otherwise --torch would silently become a no-op from the
        # second run onward.
        install_torch(py, repo, forced_tag)

    if requirements_state(py, repo, marker) == "stale":
        say("requirements.txt changed since the last sync; installing new dependencies...")
        if pip_install(py, ["-r", str(req)], what="requirements"):
            requirements_state(py, repo, marker, update=True)
            say("dependency sync complete")
        else:
            warn("dependency sync failed; the existing environment still works "
                 "but may be missing new packages")
            warn("run the launcher with --reinstall if you hit import errors")


def reinstall_venv(repo: Path) -> None:
    venv_dir = find_existing_venv(repo)
    if not venv_dir:
        return
    say(f"--reinstall: {venv_dir} will be DELETED and rebuilt.")
    say("  - studio_data/ (your projects and LoRA weights) is NOT touched")
    say("  - pip packages you installed outside requirements.txt will be lost")
    if sys.stdin and sys.stdin.isatty():
        answer = input("Continue? [y/N] ").strip().lower()
        if not answer.startswith("y"):
            say("--reinstall aborted")
            raise SystemExit(0)
    say(f"removing {venv_dir}...")
    shutil.rmtree(venv_dir, ignore_errors=True)
    if venv_python(venv_dir).exists():
        die(f"could not remove {venv_dir}", "close any running Studio window and retry")


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


def run_studio(py: Path, repo: Path, mode: Optional[str], passthrough: Sequence[str]) -> int:
    """Start `python -m studio run` and honor the restart protocol.

    Mirrors studio.bat's outer loop: if `tmp/restart` still exists, start another round
    (the flag is written server-side by `/api/system/restart`); exit code 42 means the
    launcher file itself was updated, and since the exe form can't swap itself in place the
    way a POSIX `exec` would, the user is asked to restart it manually -- this path only
    happens once, right after a self-update, and is safer than silently continuing to run
    stale logic.
    """
    restart_flag = repo / "tmp" / "restart"
    env = dict(os.environ)
    # Mode is only pinned when the user explicitly passes --mode. Without it, leave it for
    # studio to resolve itself (falls back to whatever the user picked in the UI, via
    # detection) -- if the launcher unconditionally injected "local", the mode toggle in the
    # UI would forever show as "locked by an environment variable", trading one convenience
    # for breaking another feature.
    if mode:
        env["ALS_RUNTIME_MODE"] = mode
    # On a non-UTF-8 system locale (Japanese cp932, etc.) studio's non-ASCII output would
    # crash on print.
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    argv = [str(py), "-m", "studio", "run", *passthrough]
    while True:
        try:
            rc = subprocess.call(argv, cwd=str(repo), env=env)
        except KeyboardInterrupt:
            say("stopped (Ctrl+C)")
            return 130

        if not restart_flag.exists():
            return rc

        try:
            restart_flag.unlink()
        except OSError:
            pass

        if rc == INSTALLER_RELOAD_EXIT_CODE:
            say("the launcher itself was updated - please start it again")
            return 0
        say("restart requested, starting again...")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="AnimaLoraStudio",
        description=f"{APP_NAME} launcher - sets up the environment and starts the app.",
    )
    p.add_argument("--repo", metavar="PATH",
                   help="path to the AnimaLoraStudio folder (default: next to this launcher)")
    p.add_argument("--mode", choices=("local", "colab"), default=None,
                   help="pin the runtime mode instead of letting Studio ask/remember")
    p.add_argument("--reinstall", action="store_true",
                   help="delete venv/ and rebuild it (studio_data/ is kept)")
    p.add_argument("--torch", metavar="TAG", dest="torch_tag",
                   help="force a PyTorch CUDA build (cu128/cu126/cu124/cu118/cpu)")
    p.add_argument("--check", action="store_true",
                   help="report what the launcher found (folder, Python, venv, GPU) and exit")
    return p


def run_check(repo: Path) -> int:
    """A self-check report for "why won't this start".

    When helping someone remotely, "drag the exe into a terminal, add --check, and send me
    the output" beats ten rounds of back-and-forth questions -- whether Python is installed,
    whether the venv exists, whether the GPU is detected, all on one screen.
    """
    print()
    say(f"folder      : {repo}")
    say(f"launcher    : {'frozen exe' if getattr(sys, 'frozen', False) else 'python script'}")

    venv_dir = find_existing_venv(repo)
    if venv_dir:
        py = venv_python(venv_dir)
        try:
            out = subprocess.run([str(py), "-V"], capture_output=True, text=True, timeout=30)
            say(f"venv        : {venv_dir} ({out.stdout.strip() or out.stderr.strip()})")
        except (OSError, subprocess.TimeoutExpired):
            say(f"venv        : {venv_dir} (broken - cannot run {py.name})")
        state = requirements_state(py, repo, requirements_marker(venv_dir))
        say(f"dependencies: {state}")
    else:
        say("venv        : not created yet (first run will build it)")
        argv = bootstrap_python()
        try:
            out = subprocess.run([*argv, "-V"], capture_output=True, text=True, timeout=30)
            say(f"system python: {' '.join(argv)} ({out.stdout.strip() or out.stderr.strip()})")
        except (OSError, subprocess.TimeoutExpired):
            say(f"system python: {' '.join(argv)} (cannot run)")

    # "Where did it actually get installed" is the question this report is most often used to answer, so the cache location has to be in it.
    module = load_local_cache(repo)
    if module is not None:
        current = module.describe(repo)
        outside = {
            var: value for var, value in current.items()
            if not str(value).startswith(str(repo))
        }
        say(f"caches      : {len(current) - len(outside)}/{len(current)} "
            f"inside {repo / module.CACHE_DIR_NAME}")
        for var, value in outside.items():
            say(f"              {var}={value or '(library default, outside this folder)'}")

    smi = shutil.which("nvidia-smi")
    if not smi:
        say("gpu         : nvidia-smi not found (CPU-only, or drivers not installed)")
    else:
        try:
            out = subprocess.run(
                [smi, "--query-gpu=name,memory.total,driver_version",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=30,
            )
            say(f"gpu         : {out.stdout.strip() or 'nvidia-smi returned nothing'}")
        except (OSError, subprocess.TimeoutExpired):
            say("gpu         : nvidia-smi found but did not respond")

    print()
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args, passthrough = parser.parse_known_args(argv)

    print()
    say("+----------------------------------------------------------+")
    say(f"| {APP_NAME:<56} |")
    say("+----------------------------------------------------------+")
    repo = find_repo(args.repo)
    say(f"workspace : {repo}")
    # Before any pip call -- see apply_local_caches's docstring.
    cached = apply_local_caches(repo)
    if cached:
        say(f"cache     : {repo / '.cache'}")
        say("            ALS_SYSTEM_CACHES=1 restores system locations")

    if args.check:
        return run_check(repo)

    if args.reinstall:
        reinstall_venv(repo)

    venv_dir = find_existing_venv(repo)
    fresh = venv_dir is None
    if venv_dir is None:
        venv_dir = create_venv(repo)
    py = venv_python(venv_dir)
    if not py.exists():
        die(f"the virtual environment at {venv_dir} looks broken (no {py.name})",
            "run the launcher with --reinstall to rebuild it")
    if not fresh:
        check_version([str(py)], label=str(venv_dir))

    ensure_deps(py, repo, venv_dir, fresh=fresh, forced_tag=args.torch_tag)

    if args.mode == "colab":
        say("starting Studio in colab mode (bound to 0.0.0.0, no browser)")
    else:
        say("starting Studio - your browser will open automatically")
    rc = run_studio(py, repo, args.mode, passthrough)
    if rc not in (0, 130):
        print(f"\n[studio] Studio exited with code {rc}.", file=sys.stderr, flush=True)
        pause()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
