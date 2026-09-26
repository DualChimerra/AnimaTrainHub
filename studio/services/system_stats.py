"""System resource collection (CPU / RAM / GPU / VRAM).

Used by the topbar's live widget, polled every 2-3s.

Design:
    - pynvml is lazily initialized once; on failure (no NVIDIA / missing driver / library not installed) it's
      marked permanently -- all subsequent calls just return gpu=None, no retry, no log spam.
    - psutil almost never fails; still wrapped in try/except as a fallback so frontend polling never crashes on a fluke.
    - The module is stateless on export; just call collect_stats().
"""
from __future__ import annotations

import logging
import threading
from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional

import psutil

logger = logging.getLogger(__name__)

# psutil.cpu_percent(interval=None) returns 0.0 on the first call (no baseline),
# then returns the average usage since the last call. We prime it once on module import so the first
# request already gets the average from startup to that request, instead of always showing 0%.
psutil.cpu_percent(interval=None)


# -- Lazy NVML init --------------------------------------------------------
_nvml_lock = threading.Lock()
_nvml_state: dict[str, Any] = {"inited": False, "ok": False}


def _ensure_nvml() -> bool:
    with _nvml_lock:
        if _nvml_state["inited"]:
            return _nvml_state["ok"]
        _nvml_state["inited"] = True
        try:
            import pynvml  # type: ignore[import-untyped]
            pynvml.nvmlInit()
            _nvml_state["ok"] = True
        except Exception as e:
            _nvml_state["ok"] = False
            logger.info("pynvml unavailable; GPU stats disabled (%s)", e)
        return _nvml_state["ok"]


# -- Data structures ---------------------------------------------------------
@dataclass(frozen=True)
class GpuStats:
    index: int
    name: str
    util_pct: int
    vram_used_gb: float
    vram_total_gb: float
    temp_c: Optional[int] = None


@dataclass(frozen=True)
class SystemStats:
    cpu_pct: float
    ram_used_gb: float
    ram_total_gb: float
    # None = NVML unavailable; [] = NVML available but 0 GPUs (frontend hides the GPU pill in both cases)
    gpu: Optional[list[GpuStats]]


# -- Collection ---------------------------------------------------------------
def _bytes_to_gb(n: int) -> float:
    return round(n / (1024 ** 3), 2)


def _collect_gpu() -> Optional[list[GpuStats]]:
    if not _ensure_nvml():
        return None
    try:
        import pynvml  # type: ignore[import-untyped]
        count = pynvml.nvmlDeviceGetCount()
        out: list[GpuStats] = []
        for i in range(count):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode(errors="replace")
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            try:
                temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                temp = None
            out.append(GpuStats(
                index=i,
                name=name,
                util_pct=int(util.gpu),
                vram_used_gb=_bytes_to_gb(mem.used),
                vram_total_gb=_bytes_to_gb(mem.total),
                temp_c=int(temp) if temp is not None else None,
            ))
        return out
    except Exception:
        logger.exception("gpu stats collection failed")
        return None


def collect_stats() -> SystemStats:
    try:
        # interval=None: returns usage since the last call; first call returns 0.0,
        # subsequent polls get the 2-3s average, which is exactly right for live monitoring.
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        ram_used = _bytes_to_gb(mem.total - mem.available)
        ram_total = _bytes_to_gb(mem.total)
    except Exception:
        logger.exception("psutil stats collection failed")
        cpu = 0.0
        ram_used = 0.0
        ram_total = 0.0
    return SystemStats(
        cpu_pct=round(float(cpu), 1),
        ram_used_gb=ram_used,
        ram_total_gb=ram_total,
        gpu=_collect_gpu(),
    )


def stats_to_json(s: SystemStats) -> dict[str, Any]:
    return {
        "cpu_pct": s.cpu_pct,
        "ram_used_gb": s.ram_used_gb,
        "ram_total_gb": s.ram_total_gb,
        "gpu": [asdict(g) for g in s.gpu] if s.gpu is not None else None,
    }


# ── SSE sampler ──────────────────────────────────────────────────────
class SystemStatsSampler:
    """Background thread: periodically collects system resources -> callback (usually bus.publish).

    Replaces each client polling /api/system/stats independently -- avoids polluting the
    server access log, DevTools Network panel, and cross-WAN RTT overhead in cloud deployments. The frontend only
    does one cold-start GET on mount, then receives updates via SSE.
    """

    def __init__(
        self,
        on_sample: Callable[[dict[str, Any]], None],
        *,
        interval: float = 2.5,
    ) -> None:
        self._on_sample = on_sample
        self._interval = interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread:
            return
        self._thread = threading.Thread(
            target=self._run, name="system-stats-sampler", daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                payload = stats_to_json(collect_stats())
                self._on_sample(payload)
            except Exception:
                logger.exception("system stats sampler tick failed")
            self._stop.wait(self._interval)
