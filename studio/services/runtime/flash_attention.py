"""Flash Attention wheel lookup and install.

Wheel naming scheme (mjun0812/flash-attention-prebuild-wheels):
  flash_attn-{fa_ver}+{cuda}{torch}-{pyver}-{pyver}-{platform}.whl
  e.g. flash_attn-2.8.3+cu130torch2.11-cp312-cp312-win_amd64.whl

Matching strategy:
- platform: must match exactly
- torch: must match exactly (2.11 != 2.10)
- CUDA: exact match > same major version (cu132 -> accepts cu130, CUDA minor versions are backward compatible)
- Python: must match exactly (a cp312 wheel can't run on cp313, different ABI)

Public API (also the entry point used by the server endpoint / CLI):
- `detect_env()` -- current Python / CUDA / PyTorch / platform
- `current_status()` -- whether flash_attn is already installed + its version
- `find_candidates(env)` -- list from GitHub releases (with score / usable / notes)
- `find_best_wheel(env)` -- the best usable wheel URL
- `install(url=None)` -- synchronous pip install; url=None -> auto-select
"""
from __future__ import annotations

import importlib.metadata
import json
import logging
import platform
import re
import subprocess
import sys
import urllib.request
from typing import Any, Optional

logger = logging.getLogger(__name__)

FA_RELEASES_URL = (
    "https://api.github.com/repos/mjun0812/flash-attention-prebuild-wheels/releases"
)


def detect_env() -> dict[str, Any]:
    """Detects the current Python / CUDA / PyTorch / platform.

    Each field is None when it can't be determined. `platform` only ever returns `linux_x86_64` / `win_amd64`;
    other platforms (macOS arm64 / linux aarch64) have no prebuilt wheels currently.
    """
    vi = sys.version_info
    python_tag = f"cp{vi.major}{vi.minor}"

    syst = platform.system().lower()
    mach = platform.machine().lower()
    if syst == "linux" and mach == "x86_64":
        plat = "linux_x86_64"
    elif syst == "windows" and mach in ("amd64", "x86_64"):
        plat = "win_amd64"
    else:
        plat = None

    # cuda_tag must match the CUDA runtime **PyTorch was built with** -- flash_attn's ABI follows
    # torch, not the driver. Prefer reading it from the `+cuXXX` suffix of `torch.__version__`.
    #
    # Historical bug (PR-7): the original implementation took cuda_tag from nvidia-smi. But the
    # "CUDA Version: X.Y" nvidia-smi prints is **the highest CUDA version the driver supports**, which is not
    # the same as the CUDA version PyTorch was locked to at build time. Example: a 5090's driver reports 13.0, but the venv
    # has torch 2.11.0+cu128 installed -> flash_attn must install the cu128 wheel; the old implementation set
    # cuda_tag to cu130 -> picked a cu130 wheel -> ABI mismatch -> flash_attn import fails.
    cuda_tag: Optional[str] = None
    cuda_ver: Optional[str] = None
    torch_tag: Optional[str] = None
    torch_ver: Optional[str] = None
    # 'cu128' / 'cu130' / 'cpu' / None (not installed) -- distinguishing a mistaken CPU install matters: flash_attn
    # is a CUDA C extension that simply can't be installed on a CPU-build torch; the UI needs to say "reinstall a CUDA build of torch first",
    # not the misleading "wheel not found".
    torch_cuda_build: Optional[str] = None
    # Historically only ImportError was caught, but on Windows a torch DLL load failure raises OSError
    # (WinError 126), and newer torch versions can also raise IndexError parsing __version__ -- no
    # exception here should ever 500 the status endpoint. Any exception falls back to None.
    try:
        import torch  # type: ignore[import-not-found]  # noqa: PLC0415
        torch_ver = torch.__version__
        parts = torch_ver.split("+")[0].split(".")
        if len(parts) >= 2:
            torch_tag = f"torch{parts[0]}.{parts[1]}"
        m = re.search(r"\+(cu\d+|cpu)", torch_ver)
        if m:
            tag = m.group(1)
            torch_cuda_build = tag
            if tag.startswith("cu"):
                cuda_tag = tag
                num = tag[2:]
                # cu128 -> 12.8, cu130 -> 13.0 (last digit is minor, the rest is major)
                if len(num) >= 2:
                    cuda_ver = f"{num[:-1]}.{num[-1]}"
        else:
            # Old builds without a +cu/+cpu suffix fall back to torch.version.cuda
            cuda_v = getattr(getattr(torch, "version", None), "cuda", None)
            if cuda_v is None:
                torch_cuda_build = "cpu"
            else:
                clean = str(cuda_v).replace(".", "")
                torch_cuda_build = f"cu{clean}"
                cuda_tag = torch_cuda_build
                cuda_ver = str(cuda_v)
    except Exception:  # noqa: BLE001
        pass

    # Still runs nvidia-smi: when torch has no +cu suffix (a CPU-only build) it's used as a fallback for cuda_tag;
    # the driver version is always stored separately as `driver_cuda_ver` for the UI to display and for troubleshooting (so the user can see
    # a case like "driver supports cu130, PyTorch is cu128" and immediately understand why the wheel should be cu128).
    driver_cuda_ver: Optional[str] = None
    # Historically only caught (subprocess.SubprocessError, OSError); on a Windows Chinese locale (cp936),
    # decoding nvidia-smi output with text=True could raise UnicodeDecodeError (not one of those subclasses),
    # causing a raw 500. Switched to errors='replace' to avoid that, and widened the except to Exception.
    try:
        r = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=10,
            errors="replace",
        )
        if r.returncode == 0:
            m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", r.stdout)
            if m:
                driver_cuda_ver = f"{m.group(1)}.{m.group(2)}"
                if cuda_tag is None:
                    cuda_tag = f"cu{m.group(1)}{m.group(2)}"
                    cuda_ver = driver_cuda_ver
    except Exception:  # noqa: BLE001
        pass

    return {
        "python_tag": python_tag,
        "cuda_tag": cuda_tag,
        "cuda_ver": cuda_ver,
        "driver_cuda_ver": driver_cuda_ver,
        "torch_tag": torch_tag,
        "torch_ver": torch_ver,
        "torch_cuda_build": torch_cuda_build,
        "platform": plat,
    }


def _parse_wheel(name: str) -> Optional[dict[str, str]]:
    """Parses version / cuda / torch / python / platform tags out of a wheel filename.

    e.g. flash_attn-2.8.3+cu130torch2.11-cp312-cp312-win_amd64.whl
    → {version: "2.8.3", cuda: "cu130", torch: "torch2.11", python: "cp312",
       platform: "win_amd64"}
    """
    m = re.search(
        r"flash_attn-([^+]+)\+(cu\d+)(torch[\d.]+)-(cp\d+)-cp\d+-([\w]+)\.whl$", name
    )
    if not m:
        return None
    return {
        "version": m.group(1),
        "cuda": m.group(2),
        "torch": m.group(3),
        "python": m.group(4),
        "platform": m.group(5),
    }


def _cuda_major(tag: str) -> int:
    """cu130 -> 13, cu124 -> 12; returns -1 on parse failure."""
    m = re.search(r"cu(\d+)", tag)
    return int(m.group(1)) // 10 if m else -1


def _version_key(version: str) -> tuple[int, ...]:
    """flash_attn version string -> a comparable integer tuple, used to break score ties in favor of newer versions.

    e.g. '2.8.3' -> (2, 8, 3); '2.7.4.post1' -> (2, 7, 4, 1); returns () if unparseable.
    """
    return tuple(int(x) for x in re.findall(r"\d+", version or ""))


def find_candidates(
    env: dict[str, Any],
) -> tuple[list[dict[str, Any]], Optional[str]]:
    """Queries GitHub Releases, returns (candidates, fetch_error).

    Each candidates entry: `{url, name, score, notes, usable, tags}`, sorted by score descending.
    `usable=True` means it can be installed directly in the current environment; False means it's ABI-incompatible (typically a Python mismatch).
    `fetch_error` non-None means the GitHub API request failed (network / rate limit / parsing);
    the UI should show "couldn't fetch the candidate list, you can paste a URL manually".
    """
    plat = env.get("platform")
    torch_tag = env.get("torch_tag")
    cuda_tag = env.get("cuda_tag")
    python_tag = env.get("python_tag")

    if not plat:
        return [], None

    try:
        req = urllib.request.Request(
            FA_RELEASES_URL + "?per_page=100",
            headers={"User-Agent": "AnimaLoraStudio"},
        )
        data = json.loads(urllib.request.urlopen(req, timeout=15).read())
    except Exception as exc:  # noqa: BLE001
        return [], str(exc)

    # GitHub returns a dict on rate limiting ({"message": "API rate limit exceeded..."}), not a list
    if not isinstance(data, list):
        msg = data.get("message", str(data)) if isinstance(data, dict) else str(data)
        return [], f"GitHub API error: {msg}"

    candidates: list[dict[str, Any]] = []
    for release in data:
        for asset in release.get("assets", []):
            tags = _parse_wheel(asset["name"])
            if not tags:
                continue
            if tags["platform"] != plat:
                continue
            if torch_tag and tags["torch"] != torch_tag:
                continue

            score = 0
            notes: list[str] = []
            usable = True

            # Python ABI: must match exactly, other versions can't be used
            if python_tag:
                if tags["python"] == python_tag:
                    score += 20
                else:
                    usable = False
                    notes.append(
                        f"Python incompatible (wheel={tags['python']}, current={python_tag})"
                    )

            # CUDA: same major version usable, but not as good as an exact match
            if cuda_tag:
                if tags["cuda"] == cuda_tag:
                    score += 20
                elif _cuda_major(tags["cuda"]) == _cuda_major(cuda_tag):
                    score += 10
                    notes.append(
                        f"CUDA minor version differs (wheel={tags['cuda']}, current={cuda_tag}, "
                        f"should still be compatible at the same major version)"
                    )
                else:
                    score -= 5
                    notes.append(
                        f"CUDA major version differs (wheel={tags['cuda']}, current={cuda_tag})"
                    )

            candidates.append({
                "url": asset["browser_download_url"],
                "name": asset["name"],
                "version": tags["version"],
                "score": score,
                "notes": notes,
                "usable": usable,
                "tags": tags,
            })

    # When scores tie (same python/cuda/torch), prefer the newer flash version. The old implementation only sorted by score, and ties
    # fell back to GitHub asset order (the oldest happened to be first) -> auto-install would pick the oldest wheel first; but on GPU
    # architectures newer than that wheel (e.g. Blackwell sm_120) the old wheel has no matching kernel, so flash falls back to SDPA entirely. Using
    # version as a secondary key (above asset order, below score) lets find_best_wheel pick the newest usable version.
    return sorted(
        candidates,
        key=lambda x: (x["score"], _version_key(x["version"])),
        reverse=True,
    ), None


def find_best_wheel(env: dict[str, Any]) -> Optional[str]:
    """Returns the best usable wheel URL; None if there are no candidates / all are unusable."""
    candidates, _ = find_candidates(env)
    for c in candidates:
        if c["usable"]:
            return c["url"]
    return None


def current_status() -> dict[str, Any]:
    """Current flash_attn install status."""
    try:
        version = importlib.metadata.version("flash_attn")
        return {"installed": True, "version": version}
    except importlib.metadata.PackageNotFoundError:
        return {"installed": False, "version": None}


def install(url: Optional[str] = None) -> dict[str, Any]:
    """Installs the flash_attn wheel; url=None auto-finds the best match from GitHub.

    Runs pip install synchronously, which can take a few minutes (the remote wheel is ~150MB). flash_attn is a C extension,
    so the process must restart after a pip reinstall to switch versions; returns `restart_required=True` so the UI can prompt.
    """
    env = detect_env()

    if url is None:
        # A CPU-build torch can't have flash_attn installed -- flash_attn is a CUDA C extension, which needs
        # a CUDA-build torch. The auto path pre-checks this and gives a clear error first; otherwise find_best_wheel would
        # misreport "wheel not found" because cuda_tag came from nvidia-smi (cu130), misleading the user into thinking it's a
        # network/repo problem. The explicit-URL path isn't blocked here, to allow forcing an install.
        if env.get("torch_cuda_build") == "cpu":
            raise RuntimeError(
                "This PyTorch is the CPU build (torch+cpu), so flash_attn cannot be installed."
                "flash_attn is a CUDA C extension, so PyTorch has to be reinstalled as a CUDA build first.\n"
                "Go to Settings → Training → PyTorch and reinstall a CUDA build (cu128 / cu130), "
                "restart the studio, then come back and install flash_attn."
            )
        if not env.get("platform"):
            raise RuntimeError("Unsupported platform (only linux_x86_64 / win_amd64)")
        if not env.get("torch_tag"):
            raise RuntimeError("PyTorch was not detected, so no wheel can be matched automatically")
        url = find_best_wheel(env)
        if not url:
            raise RuntimeError(
                f"No suitable wheel was found (Python={env.get('python_tag')}, "
                f"CUDA={env.get('cuda_tag')}, Torch={env.get('torch_tag')}).\n"
                "Pick one from the list below, or paste a URL from "
                "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases"
            )

    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", url],
        capture_output=True,
        text=True,
    )
    stdout = r.stdout + r.stderr
    tail = "\n".join(stdout.splitlines()[-40:])

    if r.returncode != 0:
        raise RuntimeError(f"pip install failed:\n{tail}")

    try:
        importlib.invalidate_caches()
        version = importlib.metadata.version("flash_attn")
    except Exception:  # noqa: BLE001
        version = None

    return {
        "installed": True,
        "version": version,
        "url": url,
        "stdout_tail": tail,
        "restart_required": True,
    }
