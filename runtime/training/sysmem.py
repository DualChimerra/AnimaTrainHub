"""System memory utilities (mainly Windows): working set trim + available memory queries.

Background (a real machine hang): safetensors uses mmap to stream-read
weight files, and pages that have been read stay resident in the process
working set -- after loading a 13GB DiT + 5GB TE, there's ~18GB of file
cache pages that are "reclaimable but won't leave on their own". On a
machine tight on physical RAM, these pages compete with other applications
and trigger a paging storm, showing up as the whole machine hanging (VRAM
stays healthy the whole time).

- ``trim_working_set``: pushes working-set pages into the standby list (the
  system can reclaim them in milliseconds when needed, with no paging I/O);
  hot pages that got pushed out are pulled back by a soft fault, at
  microsecond cost.
- ``available_ram_bytes``: reads GlobalMemoryStatusEx directly via ctypes, zero dependencies.
"""

from __future__ import annotations

import logging
import sys


logger = logging.getLogger(__name__)


def trim_working_set() -> bool:
    """Push reclaimable pages in the process working set (mmap file cache
    etc.) back to the system.

    Called after loading large weights; no-op returning False on non-Windows
    or on failure.
    """
    if sys.platform != "win32":
        return False
    try:
        import ctypes

        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ok = bool(ctypes.windll.psapi.EmptyWorkingSet(handle))
        if ok:
            logger.info("working set trimmed (mmap file cache pages returned to the system)")
        return ok
    except Exception:
        return False


def available_ram_bytes() -> int | None:
    """Current available physical RAM in bytes; returns None if the query fails."""
    if sys.platform == "win32":
        try:
            import ctypes

            class _MemStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemStatus()
            status.dwLength = ctypes.sizeof(_MemStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullAvailPhys)
            return None
        except Exception:
            return None
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        return None


#: RAM budget base: headroom for the process/torch itself (real-machine
#: measurement of process base is ~4GB + a paging safety margin)
_RAM_BASE_BYTES = 4 * 1024**3
#: VRAM budget base: CUDA context + activation headroom
_VRAM_BASE_BYTES = 3 * 1024**3


def _file_bytes(paths) -> int:
    """Total size of the weight files to be read (the budget basis); nonexistent paths are ignored."""
    import os

    total = 0
    for p in paths or ():
        try:
            path = str(p)
            if os.path.isdir(path):
                for entry in os.scandir(path):
                    if entry.is_file():
                        total += entry.stat().st_size
            elif path:
                total += os.stat(path).st_size
        except OSError:
            continue
    return total


def guard_enabled_from_env() -> bool:
    """Training-side memory watermark protection toggle (Settings -> Training -> Training parameters).

    The supervisor injects ``LORA_RAM_GUARD=0`` per the global setting when
    spawning the training / regularization AI subprocess, meaning disabled;
    default = enabled (a plain CLI run is likewise protected by default).
    """
    import os

    return str(os.environ.get("LORA_RAM_GUARD", "1")).strip().lower() not in {
        "0", "false", "off", "no",
    }


def check_load_budget(
    enabled: bool, *, weight_paths, stage: str, vram_discount_ratio: float = 0.0,
    settings_hint: str = "Settings -> VRAM strategy",
) -> None:
    """Dual-budget watermark guardrail: budgets RAM and VRAM by the actual
    size of the files about to be loaded.

    A static threshold only answers "is it fine right now"; a budget answers
    "will it still be fine after doing this":
    - RAM need ~= total file size (the instantaneous peak of an mmap file
      read, ~1:1 in real-machine measurements) + base
    - VRAM need ~= total file size (weights loaded onto the card) + base.
      The GPU free check naturally catches multi-process stacking (a second
      load fails fast if another daemon already has a model resident)
    ``vram_discount_ratio``: the **proportion** of the full model's
    parameters that belong to the part that will **not** go into VRAM.
    Block swap keeps the last N layers in RAM, and those weights never go
    onto the card -- without discounting this, a 16GB card with swap maxed
    out would be falsely rejected by this guardrail as "the full model
    doesn't fit", when it actually does. Only the VRAM side is discounted;
    the RAM side is computed in full (swapped-out layers still occupy RAM,
    and are **pinned**, which check_pinned_budget separately guards).

    Why a ratio and not a byte count: ``need`` comes from the weight file's
    **actual** size, and an fp8 checkpoint is only half the size of bf16. If
    the discount were computed from the compute dtype's byte count, the fp8
    case would over-discount (need 13GB, subtract a bf16-based estimate of
    11.3GB -> think only 1.7GB is needed, when 7.2GB is actually resident),
    making the guardrail useless. Multiplying the ratio by the actual file
    size is correct for both precisions.

    ``enabled`` comes from user config (inference side = Settings -> VRAM
    strategy; training side = Settings -> Training -> Training parameters,
    read via ``guard_enabled_from_env``); a failed query silently lets it
    through. ``settings_hint``: the "where to turn this protection off"
    pointer in the error message; the two sides have different toggle
    locations, so the caller passes in its own.
    """
    if not enabled:
        return
    need = _file_bytes(weight_paths)
    if need <= 0:
        return
    ratio = min(max(vram_discount_ratio, 0.0), 1.0)
    vram_need = int(need * (1.0 - ratio))

    avail = available_ram_bytes()
    if avail is not None and avail < need + _RAM_BASE_BYTES:
        raise RuntimeError(
            f"Not enough available system RAM ({avail / 1024**3:.1f}GB; this {stage} needs approx "
            f"{(need + _RAM_BASE_BYTES) / 1024**3:.1f}GB: weight file(s) "
            f"{need / 1024**3:.1f}GB + runtime headroom). Aborted to avoid a "
            f"whole-machine paging freeze. Please close other memory-hungry "
            f"applications and retry; to force it anyway, you can disable "
            f"the memory watermark protection at {settings_hint}."
        )

    free = gpu_free_bytes_global()
    if free is not None and free < vram_need + _VRAM_BASE_BYTES:
        raise RuntimeError(
            f"Not enough free GPU VRAM ({free / 1024**3:.1f}GB; this {stage} "
            f"needs approx {(vram_need + _VRAM_BASE_BYTES) / 1024**3:.1f}GB). "
            f"Another process may be using VRAM (another generation/training "
            f"job?) -- please free it up and retry; to force it anyway, you "
            f"can disable the memory watermark protection at {settings_hint}."
        )


#: The safety cap for pinned memory, as a fraction of **available** physical RAM.
#: Pinned memory **cannot be paged out, and trim_working_set has no effect
#: on it**; filling it up can bring down the whole machine (same root cause
#: as the mmap working-set freeze, but harder).
#: The denominator is available, not total -- what other applications are
#: using is already excluded, so this is "what fraction of the currently
#: free memory the training process is allowed to eat", leaving the rest to
#: absorb fluctuations (by convention: no other heavy memory work should run
#: at the same time as training).
_PINNED_SAFE_FRACTION = 0.8


def pinned_safe_limit(avail_bytes: int) -> int:
    """The maximum bytes allowed to be pinned, out of available memory.

    Takes the stricter of a ratio and an absolute floor. A pure ratio breaks
    down on low-memory machines: with 10GB available, 80% would allow
    pinning 8GB, leaving only 2GB for the training process's own non-pinned
    parts (Python/torch base, dataset, latents -- ``_RAM_BASE_BYTES`` is
    calibrated at ~4GB) -> straight into paging. And low-memory machines are
    exactly who block swap is meant to serve, so this can't be allowed to break there.

    Pulled out into its own function so ``check_pinned_budget`` (the reject
    path) and the block swap preflight (the recommend-blocks_to_swap path)
    use **the same watermark** -- keeping two separate copies would sooner
    or later drift into "preflight says it'll run, the guardrail rejects it on the spot".
    """
    avail = max(int(avail_bytes), 0)
    return min(
        int(avail * _PINNED_SAFE_FRACTION),
        max(avail - _RAM_BASE_BYTES, 0),
    )


def check_pinned_budget(need_bytes: int, *, blocks: int) -> None:
    """Pinned-memory budget guardrail for block swap (docs/design/block-swap.md SS3.2 (1)).

    **Cannot reuse check_load_budget**: that one budgets by "file size ~=
    mmap instantaneous peak", assuming the memory is reclaimable; pinned
    memory is **permanently locked**, ``trim_working_set`` has no effect on
    it, so the same byte count carries a different severity.

    Called only once, at training startup (B6: fail loudly on failure, no
    silent degradation; an allocation can only fail at that one moment, see
    SS8.1). A failed query silently lets it through, matching the other guardrails.
    """
    if need_bytes <= 0:
        return
    avail = available_ram_bytes()
    if avail is None:
        return
    safe = pinned_safe_limit(avail)
    if need_bytes > safe:
        raise RuntimeError(
            f"Not enough memory to swap out {blocks} layers: need to pin {need_bytes / 1024**3:.1f}GB, "
            f"currently available {avail / 1024**3:.1f}GB (safe cap {safe / 1024**3:.1f}GB). "
            f"Swapped-out layer weights are **pinned** in memory and can't be paged out; filling it up will slow down the whole machine. "
            f"Please lower blocks_to_swap, or close other memory-hungry applications and retry."
        )


def log_vram(stage: str, device=None) -> None:
    """Log one VRAM/RAM snapshot line at key points, to help judge the actual
    effect of knobs like block swap.

    Deliberately logs both torch's allocated amount and the **whole card's**
    used amount: under WDDM these can differ a lot (driver-side overhead +
    other processes), and looking only at torch's number would underestimate
    real usage. Fails silently on a query error.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return
        dev = torch.device(device) if device is not None else torch.device("cuda")
        if dev.type != "cuda":
            return
        allocated = torch.cuda.memory_allocated(dev) / 1024**3
        reserved = torch.cuda.memory_reserved(dev) / 1024**3
        free, total = torch.cuda.mem_get_info(dev)
        used = (total - free) / 1024**3
    except Exception:  # noqa: BLE001
        return
    ram = available_ram_bytes()
    ram_note = f", available RAM {ram / 1024**3:.1f}GB" if ram else ""
    logger.info(
        "[VRAM] %s: torch allocated %.2fGB / reserved %.2fGB, whole card used %.2fGB / %.1fGB%s",
        stage, allocated, reserved, used, total / 1024**3, ram_note,
    )


def gpu_free_bytes_global() -> int | None:
    """Actual free VRAM for the whole card; returns None if the query fails.

    Must go through NVML: under WDDM, ``cudaMemGetInfo`` is a **per-process
    virtualized view** and can't see other processes' usage (real-machine
    test: with another process holding 20GB, it still reports the full
    amount free) -- using it for a cross-process guardrail would be useless.
    NVML gives the whole-card view.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            return int(pynvml.nvmlDeviceGetMemoryInfo(handle).free)
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.mem_get_info()[0])
    except Exception:
        pass
    return None
