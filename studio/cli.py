"""Cross-platform launcher: replaces studio.bat, manages front/back-end processes in Python.

Subcommands:
    run    build the frontend (if missing) + start the backend (default)
    dev    front/back-end dev mode (Vite 5173 + uvicorn 8765 --reload, in parallel)
    build  build the frontend only
    test   run pytest then vitest

Entry points:
    python -m studio                       # same as run
    python -m studio dev
    python -m studio build
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = REPO_ROOT / "studio" / "web"
WEB_DIST = WEB_DIR / "dist"
NODE_MODULES = WEB_DIR / "node_modules"



# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def find_npm() -> Optional[str]:
    """On Windows prefer .cmd (CreateProcess can run it directly), fall back to .ps1; on Linux/Mac use the bare name.

    Don't put the bare ``npm`` name first in the Windows candidate list: the
    official Node.js installer drops ``npm`` (a bash script, for Git Bash)
    / ``npm.cmd`` / ``npm.ps1`` all in ``C:\\Program Files\\nodejs\\``, and
    ``shutil.which("npm")`` on Windows picks the bare-name one first, which
    makes subprocess fail immediately with WinError 193 (not a valid Win32 app).
    """
    candidates = ("npm.cmd", "npm.ps1", "npm") if os.name == "nt" else ("npm",)
    for candidate in candidates:
        path = shutil.which(candidate)
        if path:
            return path
    return None


def find_python() -> str:
    """Prefer the current interpreter (points at the right one automatically if a venv is active)."""
    return sys.executable


_NPM_MIRROR = "https://mirrors.cloud.tencent.com/npm/"
_PIP_MIRROR = "https://mirrors.cloud.tencent.com/pypi/simple/"


def _say(msg: str, level: str = "info") -> None:
    """Single entry point for CLI user-facing output (ADR-0009 PR-3 C4).

    Keeps the plain-print path (ADR-0009 round 2 §1.3 decision — the CLI's
    process lives ~5s, so writing to a log file has little value; users see
    `[studio] ...` in their terminal, which is cleaner than the logger's
    default format; capsys-based tests also favor plain UX).
    This wrapper gives us a single place to add verbose control / coloring
    later; right now it's equivalent to a prefixed print.

    level:
      - "info" / "success" -> stdout, `[studio] ` prefix
      - "warning" / "error" -> stderr, `[studio] ` prefix
    """
    file = sys.stderr if level in ("warning", "error") else sys.stdout
    # Note: don't use f"[studio] {msg}" here, or a bulk regex-replace of _say calls could mangle it.
    print("[studio] " + str(msg), file=file, flush=True)


def _npm_argv(npm: str, args: list[str]) -> list[str]:
    """Build the argv that subprocess can actually execute.

    ``.ps1`` can't be launched directly by ``CreateProcess``, so it must be
    wrapped in ``powershell.exe -File``; ``.cmd`` and the bare name (Linux)
    can just be used as-is.
    """
    if npm.lower().endswith(".ps1"):
        return ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", npm, *args]
    return [npm, *args]


def _npm_call(npm: str, args: list[str], cwd: str, timeout: int = 180) -> int:
    """Run an npm command; kill it and return 1 on timeout."""
    proc = subprocess.Popen(_npm_argv(npm, args), cwd=cwd)
    try:
        return proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        return 1


def _frontend_package_files_changed_since_install() -> bool:
    marker = NODE_MODULES / ".package-lock.json"
    if not marker.exists():
        return False
    try:
        marker_mtime = marker.stat().st_mtime
        for f in (WEB_DIR / "package.json", WEB_DIR / "package-lock.json"):
            if f.exists() and f.stat().st_mtime > marker_mtime:
                return True
    except OSError:
        return False
    return False


def npm_install_if_missing(npm: str) -> int:
    _bin = "eslint.cmd" if os.name == "nt" else "eslint"
    deps_complete = NODE_MODULES.exists() and (NODE_MODULES / ".bin" / _bin).exists()
    package_files_changed = deps_complete and _frontend_package_files_changed_since_install()
    if deps_complete and not package_files_changed:
        return 0
    try:
        rel = NODE_MODULES.relative_to(REPO_ROOT)
    except ValueError:
        rel = NODE_MODULES
    if package_files_changed:
        _say("studio/web/package.json or package-lock.json is newer than node_modules, running npm install...")
    else:
        _say(f"{rel} is missing or incomplete, running npm install (3 min timeout)...")
    rc = _npm_call(npm, ["install"], str(WEB_DIR), timeout=180)
    if rc != 0:
        _say(f"npm install failed or timed out, retrying with the China mirror ({_NPM_MIRROR})...")
        rc = subprocess.call(
            _npm_argv(npm, ["install", "--registry", _NPM_MIRROR]),
            cwd=str(WEB_DIR),
        )
    return rc


def _pip_install(args: list[str]) -> int:
    """Run pip install; retry with the Aliyun mirror on failure."""
    rc = subprocess.call([find_python(), "-m", "pip", "install"] + args)
    if rc != 0:
        _say(f"pip install failed, retrying with the China mirror ({_PIP_MIRROR})...")
        rc = subprocess.call(
            [find_python(), "-m", "pip", "install"] + args
            + ["-i", _PIP_MIRROR],
        )
    return rc


def _ensure_python_deps() -> int:
    """Check whether the key package (fastapi) is installed; if missing, install requirements.txt."""
    req = REPO_ROOT / "requirements.txt"
    if not req.exists():
        return 0
    try:
        import importlib.util
        if importlib.util.find_spec("fastapi") is not None:
            return 0
    except Exception:
        pass
    _say("fastapi is missing, reinstalling Python dependencies (requirements.txt)...")
    return _pip_install(["-r", str(req)])


def npm_build(npm: str) -> int:
    _say("Building the frontend (npm run build)...")
    return subprocess.call(_npm_argv(npm, ["run", "build"]), cwd=str(WEB_DIR))


# ---------------------------------------------------------------------------
# Subprocess coordination
# ---------------------------------------------------------------------------


class ProcGroup:
    """Manages several subprocesses at once; if any one exits or a signal arrives, kills them all."""

    def __init__(self) -> None:
        self.procs: list[tuple[str, subprocess.Popen]] = []
        self._stopping = False

    def spawn(
        self,
        label: str,
        cmd: list[str],
        cwd: Optional[Path] = None,
    ) -> subprocess.Popen:
        creationflags = 0
        preexec_fn = None
        if os.name == "nt":
            # CREATE_NEW_PROCESS_GROUP lets us send CTRL_BREAK_EVENT to the whole group
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        else:
            # On POSIX, put it in a new process group so killpg can be used to kill it
            preexec_fn = os.setsid  # type: ignore[assignment]
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            creationflags=creationflags,
            preexec_fn=preexec_fn,
        )
        _say(f"{label} pid={proc.pid}: {' '.join(cmd)}")
        self.procs.append((label, proc))
        return proc

    def wait_any(self) -> int:
        """Block until any process exits, and return that process's exit code."""
        while True:
            for label, p in self.procs:
                rc = p.poll()
                if rc is not None:
                    _say(f"{label} exited (rc={rc})")
                    return rc
            try:
                # give KeyboardInterrupt a chance to fire
                threading.Event().wait(0.5)
            except KeyboardInterrupt:
                return 130

    def stop_all(self, grace: float = 10.0) -> None:
        if self._stopping:
            return
        self._stopping = True
        for label, p in self.procs:
            if p.poll() is not None:
                continue
            _say(f"Stopping {label}...")
            try:
                if os.name == "nt":
                    p.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except Exception:
                pass
        for label, p in self.procs:
            try:
                p.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                _say(f"{label} did not exit in time, killing it")
                p.kill()


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------


def _print_npm_install_hint() -> None:
    """Print a platform-specific install hint when `find_npm()` returns None.

    Goes to stderr, alongside `[studio] Error: npm not found`; on a root
    environment the sudo prefix is dropped (root can install packages directly).
    """
    _say("Error: npm not found. Please install Node.js 18+", "error")
    if os.name == "nt":
        print(
            "  Windows: go to https://nodejs.org to download the installer, "
            "or run winget install OpenJS.NodeJS.LTS",
            file=sys.stderr,
        )
    else:
        sudo = "" if (hasattr(os, "getuid") and os.getuid() == 0) else "sudo "
        print(
            f"  Ubuntu/Debian: curl -fsSL https://deb.nodesource.com/setup_22.x "
            f"| {sudo}bash - && {sudo}apt-get install -y nodejs",
            file=sys.stderr,
        )
        print(
            "  Or use nvm (no root needed): "
            "curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh "
            "| bash && nvm install --lts",
            file=sys.stderr,
        )
    print("  Re-run this command after installing.", file=sys.stderr)


def cmd_build(_args: argparse.Namespace) -> int:
    npm = find_npm()
    if not npm:
        _print_npm_install_hint()
        return 2
    rc = npm_install_if_missing(npm)
    if rc != 0:
        return rc
    rc = npm_build(npm)
    if rc == 0:
        _write_build_marker()
    return rc


def _current_git_head() -> Optional[str]:
    """Current repo HEAD commit hash; not a git repo / no git command -> None."""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode == 0:
            return r.stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def _write_build_marker() -> None:
    """After a successful build, write HEAD to dist/.built-from. On the next
    cloud startup we can just compare HEAD to decide whether to rebuild,
    sidestepping the "git pull doesn't update mtime" pitfall."""
    head = _current_git_head()
    if not head:
        return
    try:
        (WEB_DIST / ".built-from").write_text(head, encoding="utf-8")
    except OSError:
        pass


def _spawn_browser_opener(url: str, *, delay: float = 1.0) -> None:
    """In the background, wait for the service to come up, then open url in the default browser; fail silently."""

    def _wait_and_open() -> None:
        deadline = time.monotonic() + 30.0
        time.sleep(delay)
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=1.5) as resp:
                    if 200 <= resp.status < 500:
                        break
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                time.sleep(0.5)
                continue
            except Exception:
                break
        try:
            webbrowser.open(url)
        except Exception:
            pass

    t = threading.Thread(target=_wait_and_open, name="studio-browser", daemon=True)
    t.start()


def _apply_pending_install() -> None:
    """At startup, handle pip-install requests (torch reinstall) that the server process couldn't complete itself.

    Must run before `_check_torch_cuda`: that function imports torch, after
    which the .pyd file is locked and can no longer be reinstalled.
    Doesn't raise on failure -- pending_install.apply_pending already prints
    its own errors, so the launcher can keep starting.
    """
    try:
        from studio.services.runtime import pending_install  # noqa: PLC0415
        pending_install.apply_pending()
    except Exception as exc:  # noqa: BLE001
        print(
            f"[studio] Warning: exception while processing a pending install request ({exc}), skipping",
            file=sys.stderr,
        )


def _try_enable_flash_attn() -> None:
    """At startup, check whether flash_attn is installed; if so, enable the cosmos / anima state machine.

    Silently skips if not installed (_check_torch_cuda already covers this,
    flash_attn is just a nice-to-have). Uses a dynamic import to avoid
    slowing down cli import time (loading cosmos_predict2_modeling triggers
    a torch import).
    """
    try:
        from studio.services.runtime import flash_attention as flash_attention_setup  # noqa: PLC0415
        if not flash_attention_setup.current_status()["installed"]:
            return
        from modeling.anima.cosmos_predict2_modeling import set_flash_attn_enabled  # noqa: PLC0415
        if set_flash_attn_enabled(True):
            _say("flash_attn enabled")
        else:
            # flash_attn is installed but set_flash_attn_enabled refused
            # (_FLASH_ATTN_AVAILABLE=False), usually meaning the import failed
            # (CUDA version mismatch, etc.); warn on stderr without more noise
            print(
                "[studio] Warning: flash_attn is installed but the model-layer import "
                "failed, falling back to SDPA",
                file=sys.stderr,
            )
    except Exception as exc:  # noqa: BLE001
        # A failure here must not fail Studio's startup; log a warning and continue
        print(
            f"[studio] Warning: exception while enabling flash_attn ({exc}), skipping the speedup",
            file=sys.stderr,
        )


def _report_lycoris_kernels() -> None:
    """Print LyCORIS 4's adapter-kernel selection without triggering JIT compilation.

    ``available_backends`` only detects whether a module can be imported;
    the actual shape/device selection is still done by LyCORIS on each call.
    We use ``preferred`` here rather than ``active`` to avoid describing a
    Triton kernel as already compiled when it hasn't run for the first time yet.
    """
    try:
        from importlib.metadata import PackageNotFoundError, version  # noqa: PLC0415

        lycoris_version = version("lycoris-lora")
    except PackageNotFoundError:
        return
    except Exception as exc:  # noqa: BLE001
        print(f"[studio] Failed to read LyCORIS status ({exc})", file=sys.stderr)
        return

    try:
        from lycoris.kernels import available_backends, resolve_backend  # noqa: PLC0415

        available = available_backends()
        preferred = resolve_backend()
    except Exception as exc:  # noqa: BLE001
        print(
            f"[studio] Warning: LyCORIS {lycoris_version} kernel backend init failed ({exc})",
            file=sys.stderr,
        )
        return

    triton_version = None
    for dist_name in ("triton-windows", "triton"):
        try:
            triton_version = version(dist_name)
            break
        except PackageNotFoundError:
            continue

    fused = ",".join(name for name in available if name in {"triton", "tilelang"}) or "none"
    triton_label = f" · Triton {triton_version}" if triton_version else ""
    _say(
        f"LoRA kernels: LyCORIS {lycoris_version} · preferred={preferred} "
        f"· fused={fused}{triton_label}"
    )


_WINDOWS_TRITON_FOR_TORCH: dict[tuple[int, int], tuple[int, int]] = {
    (2, 10): (3, 6),
    (2, 11): (3, 6),
    (2, 12): (3, 7),
    (2, 13): (3, 7),
    (2, 14): (3, 8),
}


def _major_minor(raw: str) -> Optional[tuple[int, int]]:
    match = re.match(r"^(\d+)\.(\d+)", raw)
    return (int(match.group(1)), int(match.group(2))) if match else None


def _ensure_windows_triton() -> None:
    """Match triton-windows to PyTorch before xformers imports Triton.

    PyTorch and Triton minor releases have an ABI/toolchain pairing.  A static
    requirement cannot express that dependency, so the launcher repairs a
    missing or mismatched Windows wheel from the official compatibility table.
    Unsupported/older Torch builds keep LyCORIS' compile/eager fallback.
    """
    if sys.platform != "win32":
        return

    try:
        import torch  # noqa: PLC0415

        torch_minor = _major_minor(torch.__version__)
    except Exception:  # noqa: BLE001
        return

    target = _WINDOWS_TRITON_FOR_TORCH.get(torch_minor)
    if target is None:
        if torch_minor is not None:
            print(
                f"[studio] Warning: no Windows Triton pin for torch {torch_minor[0]}.{torch_minor[1]} "
                "yet; LyCORIS will fall back automatically",
                file=sys.stderr,
            )
        return

    from importlib.metadata import PackageNotFoundError, version  # noqa: PLC0415

    try:
        installed_minor = _major_minor(version("triton-windows"))
    except PackageNotFoundError:
        installed_minor = None
    if installed_minor == target:
        return

    lower = f"{target[0]}.{target[1]}"
    upper = f"{target[0]}.{target[1] + 1}"
    requirement = f"triton-windows>={lower},<{upper}"
    installed = (
        f"{installed_minor[0]}.{installed_minor[1]}"
        if installed_minor is not None
        else "not installed"
    )
    _say(
        f"torch {torch_minor[0]}.{torch_minor[1]} requires Triton {lower} "
        f"(current: {installed}); installing compatible wheel..."
    )
    if _pip_install([requirement]) != 0:
        print(
            "[studio] Warning: Triton install failed; LyCORIS will use the compile/eager fallback",
            file=sys.stderr,
        )


def _check_torch_cuda() -> None:
    """Check at startup whether torch can use CUDA; CPU-only torch makes training / generation extremely slow.

    Four states:
    - CUDA available                     -> one-line OK
    - torch is a CPU-only build + has GPU -> big warning + reinstall command (most common mistake)
    - torch is a CPU-only build + no GPU  -> one-line info (user is genuinely on a CPU machine)
    - torch is a CUDA build but cuda unavailable -> warning (driver / WSL issue)

    `torch.version.cuda` is None on a CPU-only wheel, and something like "12.8"
    on a cu* wheel. Used here to distinguish a wrong install from a driver issue.
    """
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return  # handled earlier by _ensure_python_deps

    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:  # noqa: BLE001
            name = "?"
        _say(f"torch {torch.__version__} (GPU: {name})")
        return

    cuda_build = getattr(torch.version, "cuda", None)
    if cuda_build is None:
        # CPU-only wheel: check whether this machine actually has an NVIDIA GPU (wrong install)
        try:
            from studio.services.runtime import onnxruntime as onnxruntime_setup  # noqa: PLC0415
            has_gpu = bool(onnxruntime_setup.detect_cuda().get("available"))
        except Exception:  # noqa: BLE001
            has_gpu = False
        if has_gpu:
            print(
                f"[studio] Warning: an NVIDIA GPU was detected, but the installed PyTorch "
                f"is a CPU-only build ({torch.__version__}).\n"
                f"        Training / generation will run on the CPU, which is extremely slow "
                f"(often tens of seconds per step).\n"
                f"        Uninstall and reinstall the CUDA build:\n"
                f"          pip uninstall torch torchvision -y\n"
                f"          # pick the index matching your CUDA version, e.g. CUDA 12.8:\n"
                f"          pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128",
                file=sys.stderr,
            )
        else:
            print(
                f"[studio] torch {torch.__version__} (CPU-only build, no NVIDIA GPU detected)"
            )
        return

    # CUDA build but not usable at runtime: driver / WSL / container issue
    print(
        f"[studio] Warning: torch {torch.__version__} (CUDA {cuda_build} build), "
        f"but torch.cuda.is_available()=False.\n"
        f"        Possible causes: NVIDIA driver missing/outdated, or WSL missing CUDA support.",
        file=sys.stderr,
    )


def _check_onnxruntime() -> None:
    """Check onnxruntime status at startup (detect only, doesn't install anything).

    Mirrors xformers / flash-attention: silently skip if not installed (the
    Tagging page's WD14 / CLTagger picker shows a badge + setup button
    instead). If installed, print one status line; a CPU package with a GPU
    present warns and points the user at Settings to switch to the GPU build.
    """
    try:
        from studio.services.runtime import onnxruntime as onnxruntime_setup

        rt = onnxruntime_setup.current_runtime()
        if rt["installed"] is None:
            return

        installed = rt.get("installed") or "?"
        ver = rt.get("version") or "?"
        if rt.get("cuda_available"):
            _say(f"onnxruntime: {installed}=={ver} (CUDA EP available)")
            return

        cuda = onnxruntime_setup.detect_cuda()
        if cuda.get("available"):
            _say(
                f"NVIDIA GPU detected but onnxruntime only has the CPU EP (installed={installed}). "
                f"WD14 / CLTagger tagging will run on CPU (slower). You can reinstall the GPU build "
                f"from Settings -> ONNX Runtime.",
                "warning",
            )
        else:
            _say(f"onnxruntime: {installed}=={ver} (CPU only, no NVIDIA GPU detected)")

    except Exception as exc:  # noqa: BLE001
        _say(f"onnxruntime status check raised an exception (ignored): {exc}", "error")


WEB_SRC = WEB_DIR / "src"


def _web_dist_is_stale() -> bool:
    """Whether dist is behind src. Two checks run in parallel; either one saying stale triggers a rebuild.

    1) git HEAD comparison: the build writes HEAD to dist/.built-from, and we
       compare it against the current HEAD at startup. After a `git pull` on
       a cloud instance HEAD always changes, triggering a rebuild -- this is
       the fallback for "`git pull` doesn't update file mtimes" being
       unreliable on some git versions.
    2) mtime comparison: dist/index.html is older than the src/ tree or key
       files like package.json. This is the fallback for uncommitted local
       edits -- HEAD hasn't changed but files on disk are genuinely newer
       than dist, so it should still rebuild.

    mtime used to be a fallback only checked when HEAD matched, which meant
    local edits didn't show up in `studio run`. Running both in parallel
    doesn't change cloud `git pull` behavior, and local dev iteration no
    longer needs a commit after every change.
    """
    dist_index = WEB_DIST / "index.html"
    if not dist_index.exists():
        return True

    # Check 1: git HEAD comparison
    marker = WEB_DIST / ".built-from"
    head = _current_git_head()
    if head and marker.exists():
        try:
            built_from = marker.read_text(encoding="utf-8").strip()
            if built_from != head:
                return True
        except OSError:
            pass

    # Check 2: mtime comparison
    try:
        dist_mtime = dist_index.stat().st_mtime
        src_latest = max(
            (p.stat().st_mtime for p in WEB_SRC.rglob("*") if p.is_file()),
            default=0.0,
        )
        # also count changes to package.json / vite.config etc.
        for f in (WEB_DIR / "package.json", WEB_DIR / "vite.config.ts", WEB_DIR / "tsconfig.json"):
            if f.exists():
                src_latest = max(src_latest, f.stat().st_mtime)
        if src_latest > dist_mtime:
            return True
    except OSError:
        pass

    return False


_RESTART_FLAG = REPO_ROOT / "tmp" / "restart"

# PR-D -- installer self-check (ADR 0002). cmd_run snapshots the sha256 of
# these three files at entry; each time the server exits and a restart is
# requested, it recomputes them, and any change makes it return exit code 42
# so the wrapper (studio.sh / studio.bat) re-execs itself entirely. Why:
#
# - cli.py itself changed -> the old python process still has the old cli.py
#   loaded, so the next-iteration inner loop would keep running the old
#   logic; only having the wrapper re-run `python -m studio` picks up the
#   new cli.py.
# - studio.sh / studio.bat changed -> bash already has the loop body loaded
#   into memory, and cmd.exe may also cache its .bat parse; the shell
#   process itself must exec itself to pick up the new wrapper.
#
# Any change across the three files goes through the same protocol (simplest
# / most robust).
_INSTALLER_FILES: tuple[Path, ...] = (
    REPO_ROOT / "studio" / "cli.py",
    REPO_ROOT / "studio.sh",
    REPO_ROOT / "studio.bat",
)
_INSTALLER_RELOAD_EXIT_CODE = 42


def _installer_hashes() -> dict[str, Optional[str]]:
    """Snapshot the sha256 of the installer files. Missing file -> value is None
    (cross-platform: studio.bat doesn't exist on Linux, studio.sh doesn't
    exist on Windows; existence itself is part of the comparison)."""
    result: dict[str, Optional[str]] = {}
    for p in _INSTALLER_FILES:
        try:
            result[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        except OSError:
            result[p.name] = None
    return result


def _maybe_force_torch(args: argparse.Namespace) -> int:
    """When --torch <tag> is given, check whether the current install matches;
    if not, reinstall immediately (streamed output). Only called once at
    launcher startup; the restart mechanism loads the new torch afterward."""
    tag = getattr(args, 'torch', None)
    if not tag:
        return 0
    from studio.services.runtime import torch as torch_setup  # noqa: PLC0415
    current = torch_setup.detect_torch()
    current_build = current.get('cuda_build') or ('not installed' if not current.get('installed') else 'unknown')
    if current.get('installed') and current.get('cuda_build') == tag:
        _say(f"torch is already {tag}, skipping reinstall")
        return 0
    _say(f"--torch {tag} requested (current: {current_build}), reinstalling...")
    _say("Tip: press Ctrl+C to skip")
    try:
        res = torch_setup.reinstall(tag, stream=True)
        _say(f"torch reinstall complete: {res.get('version')} ({res.get('tag')})")
        return 0
    except KeyboardInterrupt:
        print("\n[studio] Interrupted by user, skipping torch reinstall", file=sys.stderr)
        return 0
    except RuntimeError as exc:
        _say(f"torch reinstall failed: {exc}", "error")
        return 1


def cmd_run(args: argparse.Namespace) -> int:
    """The `run` main loop.

    Inner loop: after each server exit, check the `tmp/restart` flag
    (written by the server side's `/api/system/restart`). If present, delete
    the flag, redo bootstrap, and restart the server; if absent, break out
    and exit normally.

    See `docs/adr/0002-webui-self-update.md` for the restart protocol. The
    outer shell wrapper (`studio.sh` / `studio.bat`) has the same loop as a
    fallback (for when cli.py exits abnormally but the flag is still there),
    and reacts to exit code 42 by re-execing itself (PR-D installer
    self-check: when cli.py / studio.sh / studio.bat itself was modified by
    an update, the wrapper + Python interpreter need to be reloaded from
    disk).

    The browser only opens once on a cold start; on restart it reuses the
    existing webui tab (the frontend polls `/api/health` and reconnects
    automatically) instead of popping a new window.
    """
    opened_browser = False
    # force-reinstall via --torch (only on the first pass, not repeated inside the restart loop)
    rc = _maybe_force_torch(args)
    if rc != 0:
        return rc
    # PR-D: snapshot the installer files' sha256 at startup; recompute after
    # the server exits, and exit code 42 makes the wrapper re-exec itself.
    startup_installer = _installer_hashes()
    while True:
        rc = _ensure_python_deps()
        if rc != 0:
            return rc

        if not args.no_build:
            if not WEB_DIST.exists():
                _say("studio/web/dist is missing, building the frontend first...")
                rc = cmd_build(args)
                if rc != 0:
                    return rc
            elif _web_dist_is_stale():
                _say("studio/web/dist is older than src (not rebuilt after git pull?), rebuilding the frontend...")
                rc = cmd_build(args)
                if rc != 0:
                    return rc
        if not getattr(args, 'skip_pending', False):
            _apply_pending_install()
        _check_torch_cuda()
        _ensure_windows_triton()
        _try_enable_flash_attn()
        _report_lycoris_kernels()
        _check_onnxruntime()
        url = f"http://{args.host}:{args.port}/"
        _say(f"Starting the backend -> {url}")
        if not args.no_browser and not opened_browser:
            _spawn_browser_opener(url)
            opened_browser = True
        try:
            rc = subprocess.call(
                [find_python(), "-m", "studio.server", "--host", args.host, "--port", str(args.port)]
            )
        except KeyboardInterrupt:
            # Terminal Ctrl+C: CTRL_C_EVENT is broadcast to the server
            # subprocess too (which does its own graceful shutdown); the
            # parent's blocking wait only raises KeyboardInterrupt after the
            # child has fully exited. This is a deliberate user stop, so no
            # traceback.
            _say("Stopped (Ctrl+C)")
            return 130

        if not _RESTART_FLAG.exists():
            return rc

        # PR-D: installer self-check. With the restart flag present, if
        # cli.py / studio.sh / studio.bat changed, **keep** the flag and
        # return 42 so the wrapper takes the exec-self path. Keeping the flag
        # matters -- the wrapper only re-execs when it sees (exit==42 && flag
        # exists); flag alone with exit!=42 is a normal restart.
        if _installer_hashes() != startup_installer:
            _say("Detected an update to a launcher file (cli.py / studio.sh / studio.bat), "
                  "exiting with code 42 so the wrapper reloads...")
            return _INSTALLER_RELOAD_EXIT_CODE

        # Restart requested: delete the flag and loop back to bootstrap again
        try:
            _RESTART_FLAG.unlink()
        except OSError:
            pass
        _say("Restart requested, restarting...")


def cmd_dev(args: argparse.Namespace) -> int:
    rc = _maybe_force_torch(args)
    if rc != 0:
        return rc
    rc = _ensure_python_deps()
    if rc != 0:
        return rc
    npm = find_npm()
    if not npm:
        _print_npm_install_hint()
        return 2
    rc = npm_install_if_missing(npm)
    if rc != 0:
        return rc
    if not getattr(args, 'skip_pending', False):
        _apply_pending_install()
    _check_torch_cuda()
    _ensure_windows_triton()
    _try_enable_flash_attn()
    _report_lycoris_kernels()
    _check_onnxruntime()

    pg = ProcGroup()
    try:
        pg.spawn("frontend", _npm_argv(npm, ["run", "dev", "--", "--port", str(args.fe_port)]), cwd=WEB_DIR)
        pg.spawn(
            "backend",
            [
                find_python(),
                "-m",
                "studio.server",
                "--host", args.host,
                "--port", str(args.port),
                "--reload",
            ],
        )
        frontend_url = f"http://127.0.0.1:{args.fe_port}/"
        print(
            f"[studio] frontend → {frontend_url}  "
            f"backend → http://{args.host}:{args.port}/"
        )
        if not args.no_browser:
            # dev mode opens the Vite port (so HMR works), not the backend port
            _spawn_browser_opener(frontend_url, delay=2.0)
        rc = pg.wait_any()
    finally:
        pg.stop_all()
    return rc


def cmd_test(_args: argparse.Namespace) -> int:
    """Run pytest + vitest. Any failure -> non-zero exit."""
    _say("pytest...")
    rc = subprocess.call([find_python(), "-m", "pytest", "tests/"], cwd=str(REPO_ROOT))
    if rc != 0:
        return rc
    npm = find_npm()
    if not npm:
        _say("Skipping vitest (npm not installed)")
        return 0
    if not NODE_MODULES.exists():
        _say("Skipping vitest (node_modules missing, run npm install first)")
        return 0
    _say("vitest...")
    return subprocess.call(_npm_argv(npm, ["run", "test"]), cwd=str(WEB_DIR))


# ---------------------------------------------------------------------------
# Runtime mode (Colab / Local)
# ---------------------------------------------------------------------------

# Deliberately not importing studio.infrastructure.runtime_mode here:
# build_parser() needs this tuple while constructing argparse, and
# infrastructure would drag in pydantic / paths too (a few hundred ms, plus
# it creates directories). The value set is just two literals, so duplicating
# it once is cheaper than adding that import edge; the actual resolution
# logic still lives only in runtime_mode (imported inside the function below).
_RUNTIME_MODES: tuple[str, ...] = ("local", "colab")


def _apply_runtime_mode_defaults(args: argparse.Namespace) -> None:
    """Fill in `--host` / `--no-browser` defaults based on the runtime mode (mutates args in place).

    - When `--mode` is passed explicitly, it's written to `ALS_RUNTIME_MODE`
      first, so the server subprocess (`python -m studio.server`, which
      inherits the environment) and the UI see the same mode.
    - `local`: 127.0.0.1 + auto-open browser -- the browser and process are
      on the same machine, so binding 0.0.0.0 would expose the training
      panel to the whole LAN, which shouldn't be the default.
    - `colab`: 0.0.0.0 + no browser -- the notebook's port proxy connects in
      from outside the container, and there's no browser to open inside the
      container anyway (webbrowser.open would fail silently or hang).

    An explicit user-provided value always wins: neither a non-None `--host`
    nor an already-set `--no-browser` gets overridden.
    """
    mode = getattr(args, "mode", None)
    if mode:
        os.environ["ALS_RUNTIME_MODE"] = mode

    from .infrastructure import runtime_mode  # noqa: PLC0415 -- see comment above

    effective = runtime_mode.effective()
    args.runtime_mode = effective

    if getattr(args, "host", None) is None:
        args.host = "0.0.0.0" if effective == "colab" else "127.0.0.1"
    if effective == "colab":
        args.no_browser = True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="studio", description="AnimaTrainHub launcher")
    sub = p.add_subparsers(dest="cmd")

    p_run = sub.add_parser("run", help="Build the frontend (if missing) + start the backend")
    # host default stays None; _apply_runtime_mode_defaults fills it in by
    # runtime mode: local -> 127.0.0.1 (local machine only), colab -> 0.0.0.0
    # (the notebook's proxy needs to reach it). An explicit --host always wins.
    p_run.add_argument("--host", default=None,
                       help="Bind address (default by runtime mode: local=127.0.0.1 / colab=0.0.0.0)")
    p_run.add_argument("--port", type=int, default=8765)
    p_run.add_argument("--mode", choices=list(_RUNTIME_MODES), default=None,
                       help="Force a runtime mode (local / colab). Equivalent to setting "
                            "ALS_RUNTIME_MODE; also locks the mode picker in the UI.")
    p_run.add_argument("--no-build", action="store_true",
                       help="Don't auto-build even if dist is missing")
    p_run.add_argument("--no-browser", action="store_true",
                       help="Don't auto-open a browser after starting")
    p_run.add_argument("--skip-pending", action="store_true",
                       help="Skip pending pip installs (torch reinstall, etc) and start right away")
    p_run.add_argument("--torch", metavar="TAG",
                       help="Force a torch CUDA build (cu128/cu126/cu124/cu118/cpu); "
                            "reinstalls automatically if it doesn't match. Useful on CPU rental "
                            "machines that come with GPU torch preinstalled.")
    p_run.set_defaults(func=cmd_run)

    p_dev = sub.add_parser("dev", help="Frontend + backend dev mode")
    p_dev.add_argument("--host", default=None,
                       help="Bind address (default by runtime mode: local=127.0.0.1 / colab=0.0.0.0)")
    p_dev.add_argument("--mode", choices=list(_RUNTIME_MODES), default=None,
                       help="Force a runtime mode (local / colab)")
    p_dev.add_argument("--port", type=int, default=8765,
                       help="Backend uvicorn port (default 8765)")
    p_dev.add_argument("--fe-port", type=int, default=5173,
                       help="Frontend Vite dev server port (default 5173)")
    p_dev.add_argument("--no-browser", action="store_true",
                       help="Don't auto-open a browser after starting")
    p_dev.add_argument("--skip-pending", action="store_true",
                       help="Skip pending pip installs (torch reinstall, etc) and start right away")
    p_dev.add_argument("--torch", metavar="TAG",
                       help="Force a torch CUDA build (cu128/cu126/cu124/cu118/cpu)")
    p_dev.set_defaults(func=cmd_dev)

    p_build = sub.add_parser("build", help="Build the frontend only")
    p_build.set_defaults(func=cmd_build)

    p_test = sub.add_parser("test", help="Run pytest + vitest")
    p_test.set_defaults(func=cmd_test)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    # Redirect third-party library caches into `<repo>/.cache/` (this fork's
    # convention, see infrastructure/local_cache.py).
    #
    # Must run before any pip / npm / server subprocess -- `args.func(args)`
    # below will spawn them, so the env vars need to be in place first. This
    # lives in main() rather than at import time: doing it at import time
    # would set the env vars and create .cache/ in the repo during test
    # collection too, which only benefits the non-user path of "someone
    # directly imports cmd_build". Values the user explicitly set aren't
    # overridden; `ALS_SYSTEM_CACHES=1` disables this entirely.
    from .infrastructure import local_cache

    local_cache.apply(REPO_ROOT)

    parser = build_parser()
    args_list = list(argv) if argv is not None else sys.argv[1:]
    # Default to run when no subcommand is given (e.g. studio.sh --port 6006
    # -> run --port 6006). Find the first argument not starting with '-' and
    # check whether it's a known subcommand; if not, insert 'run'.
    _subcmds = {'run', 'dev', 'build', 'test'}
    _first_pos = next((a for a in args_list if not a.startswith('-')), None)
    if _first_pos not in _subcmds:
        args_list = ['run'] + args_list
    args = parser.parse_args(args_list)
    # PR-1 C4: unified logging (ADR-0009). file=False -- the CLI process only
    # lives ~5s, so startup info doesn't go into studio.log (decided in round
    # 2 §1.3). console=True routes logger.x calls to stderr for humans; the
    # existing 48 print() call sites are left alone (folded in by the PR-3
    # _say() wrapper). Env ANIMA_LOGGING_NO_BOOTSTRAP=1 makes this a no-op
    # (test mode).
    from .infrastructure.logging import setup_logging
    setup_logging(f"cli:{args.cmd}", file=False, console=True)
    # host / browser concepts only apply to run / dev; build / test don't have these attributes.
    if args.cmd in ("run", "dev"):
        _apply_runtime_mode_defaults(args)
        _say(f"Runtime mode: {args.runtime_mode}"
             + (" (Colab / cloud notebook)" if args.runtime_mode == "colab"
                else " (local machine)"))
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
