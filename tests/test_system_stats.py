"""services/system_stats.py -- collection + graceful NVML degradation + SSE sampler thread."""
from __future__ import annotations

import sys
import threading
import time
import types

import pytest

from studio.services import system_stats


def test_collect_stats_returns_sane_basic():
    """Real-environment collection: CPU/RAM ranges are sane, structure is complete."""
    stats = system_stats.collect_stats()
    assert 0.0 <= stats.cpu_pct <= 100.0
    assert stats.ram_used_gb >= 0.0
    assert stats.ram_total_gb > 0.0
    assert stats.ram_used_gb <= stats.ram_total_gb
    # the gpu field is usually None in CI; it's a list[GpuStats] locally when a card is present


def test_stats_to_json_no_gpu():
    s = system_stats.SystemStats(
        cpu_pct=12.5, ram_used_gb=8.0, ram_total_gb=32.0, gpu=None,
    )
    j = system_stats.stats_to_json(s)
    assert set(j.keys()) == {"cpu_pct", "ram_used_gb", "ram_total_gb", "gpu"}
    assert j["gpu"] is None


def test_stats_to_json_with_gpu():
    g = system_stats.GpuStats(
        index=0, name="Test GPU", util_pct=42,
        vram_used_gb=4.0, vram_total_gb=24.0, temp_c=55,
    )
    s = system_stats.SystemStats(
        cpu_pct=1.0, ram_used_gb=8.0, ram_total_gb=32.0, gpu=[g],
    )
    j = system_stats.stats_to_json(s)
    assert isinstance(j["gpu"], list) and len(j["gpu"]) == 1
    g0 = j["gpu"][0]
    assert g0["index"] == 0
    assert g0["name"] == "Test GPU"
    assert g0["util_pct"] == 42
    assert g0["temp_c"] == 55


def test_nvml_init_failure_returns_none(monkeypatch: pytest.MonkeyPatch):
    """Simulate nvmlInit raising: collect_gpu permanently returns None."""
    monkeypatch.setattr(
        system_stats, "_nvml_state", {"inited": False, "ok": False},
    )
    fake = types.ModuleType("pynvml")

    def boom() -> None:
        raise RuntimeError("simulated init failure")

    fake.nvmlInit = boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pynvml", fake)

    assert system_stats._collect_gpu() is None
    # the second call hits the cache and is still None; it should not raise again
    assert system_stats._collect_gpu() is None


def test_nvml_zero_devices_returns_empty_list(monkeypatch: pytest.MonkeyPatch):
    """NVML available but no card: returns [] (frontend hides the GPU pill the same as for None)."""
    monkeypatch.setattr(
        system_stats, "_nvml_state", {"inited": True, "ok": True},
    )
    fake = types.ModuleType("pynvml")
    fake.nvmlDeviceGetCount = lambda: 0  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pynvml", fake)

    assert system_stats._collect_gpu() == []


def test_nvml_one_fake_gpu(monkeypatch: pytest.MonkeyPatch):
    """A single mock card: fields map as expected."""
    monkeypatch.setattr(
        system_stats, "_nvml_state", {"inited": True, "ok": True},
    )

    class FakeMem:
        used = 4 * 1024 ** 3
        total = 24 * 1024 ** 3

    class FakeUtil:
        gpu = 67

    fake = types.ModuleType("pynvml")
    fake.NVML_TEMPERATURE_GPU = 0  # type: ignore[attr-defined]
    fake.nvmlDeviceGetCount = lambda: 1  # type: ignore[attr-defined]
    fake.nvmlDeviceGetHandleByIndex = lambda i: f"h{i}"  # type: ignore[attr-defined]
    fake.nvmlDeviceGetName = lambda h: "Mock GPU"  # type: ignore[attr-defined]
    fake.nvmlDeviceGetMemoryInfo = lambda h: FakeMem()  # type: ignore[attr-defined]
    fake.nvmlDeviceGetUtilizationRates = lambda h: FakeUtil()  # type: ignore[attr-defined]
    fake.nvmlDeviceGetTemperature = lambda h, t: 50  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pynvml", fake)

    result = system_stats._collect_gpu()
    assert result is not None and len(result) == 1
    g = result[0]
    assert g.index == 0
    assert g.name == "Mock GPU"
    assert g.util_pct == 67
    assert g.vram_used_gb == 4.0
    assert g.vram_total_gb == 24.0
    assert g.temp_c == 50


def test_sampler_emits_payloads(monkeypatch: pytest.MonkeyPatch):
    """SystemStatsSampler calls back periodically once started; stop() exits cleanly."""
    samples: list[dict] = []
    event = threading.Event()

    def on_sample(payload: dict) -> None:
        samples.append(payload)
        if len(samples) >= 2:
            event.set()

    sampler = system_stats.SystemStatsSampler(on_sample, interval=0.05)
    sampler.start()
    try:
        assert event.wait(timeout=2.0), f"only got {len(samples)} samples"
    finally:
        sampler.stop()

    assert len(samples) >= 2
    for p in samples:
        assert set(p.keys()) == {"cpu_pct", "ram_used_gb", "ram_total_gb", "gpu"}


def test_sampler_swallows_collection_errors(monkeypatch: pytest.MonkeyPatch):
    """The sampler should not crash when collection raises; it continues to the next round."""
    fail_count = [0]
    samples: list[dict] = []

    real_collect = system_stats.collect_stats

    def flaky_collect() -> system_stats.SystemStats:
        fail_count[0] += 1
        if fail_count[0] == 1:
            raise RuntimeError("simulated transient failure")
        return real_collect()

    monkeypatch.setattr(system_stats, "collect_stats", flaky_collect)

    sampler = system_stats.SystemStatsSampler(samples.append, interval=0.05)
    sampler.start()
    try:
        deadline = time.time() + 2.0
        while len(samples) < 1 and time.time() < deadline:
            time.sleep(0.05)
    finally:
        sampler.stop()

    # the first collect raises and is swallowed, the second succeeds -> samples >= 1
    assert fail_count[0] >= 2
    assert len(samples) >= 1


def test_nvml_temp_failure_keeps_other_fields(monkeypatch: pytest.MonkeyPatch):
    """A raise from a partial metric (temperature) should not drop the whole card."""
    monkeypatch.setattr(
        system_stats, "_nvml_state", {"inited": True, "ok": True},
    )

    class FakeMem:
        used = 1 * 1024 ** 3
        total = 8 * 1024 ** 3

    class FakeUtil:
        gpu = 10

    def temp_boom(*a, **k):
        raise RuntimeError("temp sensor unavailable")

    fake = types.ModuleType("pynvml")
    fake.NVML_TEMPERATURE_GPU = 0  # type: ignore[attr-defined]
    fake.nvmlDeviceGetCount = lambda: 1  # type: ignore[attr-defined]
    fake.nvmlDeviceGetHandleByIndex = lambda i: f"h{i}"  # type: ignore[attr-defined]
    fake.nvmlDeviceGetName = lambda h: "Old GPU"  # type: ignore[attr-defined]
    fake.nvmlDeviceGetMemoryInfo = lambda h: FakeMem()  # type: ignore[attr-defined]
    fake.nvmlDeviceGetUtilizationRates = lambda h: FakeUtil()  # type: ignore[attr-defined]
    fake.nvmlDeviceGetTemperature = temp_boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pynvml", fake)

    result = system_stats._collect_gpu()
    assert result is not None and len(result) == 1
    assert result[0].temp_c is None
    assert result[0].util_pct == 10
