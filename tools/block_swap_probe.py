#!/usr/bin/env python3
"""block_swap_probe -- Gate-0 feasibility probe for block swap.

Companion doc: ``docs/design/block-swap.md``. **This probe must be run before any
implementation code is written** -- if the numbers don't clear the bar, the plan is
rejected outright.

Whether block swap succeeds comes down to a single inequality (doc §2.2):

    fully maskable  <=>  T_transfer(block) < T_compute(block)
    T_transfer = bytes per block / effective PCIe bandwidth

This probe answers it in six stages, plus the hardware impact the user explicitly
cares about (doc §3):

  A link health     actual PCIe gen/width vs max, replay baseline, GPU/RAM capacity
  B bandwidth matrix pinned/pageable x H2D/D2H x several sizes; pinned alloc time
  C compute baseline real SingleStreamBlock forward/backward time at real shapes
  D masking criterion B/C ratio -> blocks_to_swap x (VRAM saved, time added) curve
  E end-to-end       wall clock diff between a real dual-stream swap loop and a fully
                      resident loop
  F stability        temperature / power / PCIe replay delta / available RAM under
                      sustained load

Usage
-----
    ./venv/Scripts/python.exe tools/block_swap_probe.py
    ./venv/Scripts/python.exe tools/block_swap_probe.py --stages ABCD  # skip the slow stages
    ./venv/Scripts/python.exe tools/block_swap_probe.py --resolution 1536 --soak-seconds 120
    ./venv/Scripts/python.exe tools/block_swap_probe.py --out probe.csv

Without CUDA, only stage A runs, then it exits. Depends on torch; pynvml is optional
(stages A/F degrade gracefully without it).

Scope: stages C/D/E/F only cover **krea2** (28 homogeneous SingleStreamBlock layers,
the plan's target family). Anima's Block forward needs a chain of prep tensors like
rope/adaln_lora, which is expensive to construct for little payoff (24GB is already
enough); this probe only measures its weight size, leaving the compute baseline for
if/when it's needed.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

# This machine's terminal is cp932; non-ASCII output crashes without reconfiguring (see windows_console_cp932_utf8)
if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (_REPO_ROOT, _REPO_ROOT / "runtime"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

MIB = 1024 ** 2
GIB = 1024 ** 3

#: every observation collected; optionally written to CSV at the end
RECORDS: list[dict] = []


def record(stage: str, metric: str, value, unit: str = "", note: str = "") -> None:
    RECORDS.append(
        {"stage": stage, "metric": metric, "value": value, "unit": unit, "note": note}
    )


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


# ------------------------------------------------------------------ NVML wrapper
class Nvml:
    """Thin pynvml wrapper: degrades entirely to None when the library / API is missing, so it never crashes the probe."""

    def __init__(self) -> None:
        self.handle = None
        self._nvml = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:  # noqa: BLE001
            self._nvml = None
            self.handle = None

    @property
    def ok(self) -> bool:
        return self.handle is not None

    def _call(self, name: str, *args):
        if not self.ok:
            return None
        fn = getattr(self._nvml, name, None)
        if fn is None:
            return None
        try:
            return fn(self.handle, *args)
        except Exception:  # noqa: BLE001
            return None

    def pcie_gen(self) -> tuple[int | None, int | None]:
        return self._call("nvmlDeviceGetCurrPcieLinkGeneration"), self._call(
            "nvmlDeviceGetMaxPcieLinkGeneration"
        )

    def pcie_width(self) -> tuple[int | None, int | None]:
        return self._call("nvmlDeviceGetCurrPcieLinkWidth"), self._call(
            "nvmlDeviceGetMaxPcieLinkWidth"
        )

    def replay_counter(self) -> int | None:
        return self._call("nvmlDeviceGetPcieReplayCounter")

    def temperature(self) -> int | None:
        if not self.ok:
            return None
        try:
            return self._nvml.nvmlDeviceGetTemperature(
                self.handle, self._nvml.NVML_TEMPERATURE_GPU
            )
        except Exception:  # noqa: BLE001
            return None

    def power_watts(self) -> float | None:
        milliwatts = self._call("nvmlDeviceGetPowerUsage")
        return None if milliwatts is None else milliwatts / 1000.0

    def name(self) -> str | None:
        raw = self._call("nvmlDeviceGetName")
        if isinstance(raw, bytes):
            return raw.decode("utf-8", "replace")
        return raw

    def shutdown(self) -> None:
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:  # noqa: BLE001
                pass


# ------------------------------------------------------------------ A link health
def stage_a(nvml: Nvml) -> dict:
    """Whether the PCIe link is degraded + capacity baseline + replay baseline (doc §3.2 (2))."""
    section("A - Link health")
    import torch

    info: dict = {}

    cuda_ok = torch.cuda.is_available()
    info["cuda"] = cuda_ok
    print(f"torch {torch.__version__}   CUDA available: {cuda_ok}")
    if cuda_ok:
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = props.name
        info["vram_total"] = props.total_memory
        print(f"GPU        : {props.name}")
        print(f"VRAM       : {props.total_memory / GIB:.1f} GB")
        record("A", "vram_total", props.total_memory / GIB, "GB", props.name)

    from training.sysmem import available_ram_bytes

    avail = available_ram_bytes()
    if avail is not None:
        info["ram_available"] = avail
        print(f"RAM avail. : {avail / GIB:.1f} GB")
        record("A", "ram_available", round(avail / GIB, 1), "GB")

    if nvml.ok:
        cur_gen, max_gen = nvml.pcie_gen()
        cur_width, max_width = nvml.pcie_width()
        info["pcie_gen"] = cur_gen
        info["pcie_width"] = cur_width
        degraded = []
        if cur_gen and max_gen and cur_gen < max_gen:
            degraded.append(f"gen {cur_gen} < max {max_gen}")
        if cur_width and max_width and cur_width < max_width:
            degraded.append(f"width x{cur_width} < max x{max_width}")
        print(f"PCIe link  : gen{cur_gen} x{cur_width} (max gen{max_gen} x{max_width})")
        record("A", "pcie_gen", cur_gen, "", f"max={max_gen}")
        record("A", "pcie_width", cur_width, "", f"max={max_width}")
        if degraded:
            # The link auto-downshifts to gen1 to save power when idle, and only ramps
            # up under load -- this is just a heads-up
            print(f"  ! Currently degraded ({'; '.join(degraded)})")
            print("    Idle power-saving downshift is normal; stage B re-measures under load; if still degraded then it's a real degradation")

        replay = nvml.replay_counter()
        info["replay_base"] = replay
        print(f"PCIe replay: {replay} (baseline, stage F looks at the delta)")
        record("A", "pcie_replay_base", replay)
    else:
        print("pynvml unavailable -- can't collect PCIe link / replay counters (stages A/F degrade)")

    return info


# ------------------------------------------------------------------ B bandwidth matrix
def _time_copy(src, dst_device: str, iters: int) -> float:
    """Returns the median time for a single copy (seconds). Uses cuda events for timing, to exclude launch jitter."""
    import torch

    events = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        src.to(dst_device, non_blocking=True)
        end.record()
        torch.cuda.synchronize()
        events.append(start.elapsed_time(end) / 1000.0)
    return statistics.median(events)


def stage_b(args) -> dict:
    """Effective bandwidth for pinned vs pageable x H2D vs D2H (doc §2.1)."""
    section("B - Transfer bandwidth matrix")
    import torch

    sizes_mib = [16, 64, 256, 1024]
    results: dict = {}

    print(f"{'size':>8} {'pinned H2D':>12} {'pageable H2D':>14} {'pinned D2H':>12}")
    print("-" * 50)
    for size_mib in sizes_mib:
        numel = size_mib * MIB // 2  # bf16

        pin_start = time.perf_counter()
        try:
            pinned = torch.empty(numel, dtype=torch.bfloat16, pin_memory=True)
        except RuntimeError as exc:
            # doc §3.2 (1): a pinned allocation failure is a hard error, the implementation must have a fallback path
            print(f"{size_mib:>6}MB  pinned allocation failed: {exc}")
            record("B", f"pin_alloc_fail_{size_mib}MB", 1, "", str(exc))
            continue
        pin_alloc = time.perf_counter() - pin_start

        pageable = torch.empty(numel, dtype=torch.bfloat16)
        on_gpu = torch.empty(numel, dtype=torch.bfloat16, device="cuda")

        # warmup
        pinned.to("cuda", non_blocking=True)
        torch.cuda.synchronize()

        t_pin_h2d = _time_copy(pinned, "cuda", args.iters)
        t_page_h2d = _time_copy(pageable, "cuda", args.iters)
        t_pin_d2h = _time_copy(on_gpu, "cpu", args.iters)

        payload = size_mib * MIB
        bw_pin = payload / t_pin_h2d / GIB
        bw_page = payload / t_page_h2d / GIB
        bw_d2h = payload / t_pin_d2h / GIB
        results[size_mib] = {
            "pinned_h2d": bw_pin,
            "pageable_h2d": bw_page,
            "pinned_d2h": bw_d2h,
            "pin_alloc_s": pin_alloc,
        }
        print(
            f"{size_mib:>6}MB {bw_pin:>10.1f}GB/s {bw_page:>12.1f}GB/s {bw_d2h:>10.1f}GB/s"
        )
        record("B", f"pinned_h2d_{size_mib}MB", round(bw_pin, 2), "GB/s")
        record("B", f"pageable_h2d_{size_mib}MB", round(bw_page, 2), "GB/s")
        record("B", f"pin_alloc_{size_mib}MB", round(pin_alloc * 1000, 1), "ms")

        del pinned, pageable, on_gpu
        torch.cuda.empty_cache()

    if results:
        big = max(results)
        peak = results[big]["pinned_h2d"]
        print(f"\nEffective bandwidth (using {big}MB pinned H2D): {peak:.1f} GB/s")
        print(
            f"pinned speedup over pageable: "
            f"{results[big]['pinned_h2d'] / results[big]['pageable_h2d']:.2f}x"
        )
        print(
            f"pinned alloc time: {results[big]['pin_alloc_s'] * 1000:.0f} ms / {big}MB"
            "  <- the implementation must pre-allocate and reuse, not alloc every step"
        )
        results["effective_bw"] = peak
    return results


# ------------------------------------------------------------------ C compute baseline
def _build_krea2_block(device, dtype):
    """Instantiates a single real SingleStreamBlock + the synthetic inputs its forward needs.

    The input construction mirrors ``SingleStreamDiT.forward`` line-for-line
    (krea2_modeling.py:444-522): combined = cat(text, image) is fed through the block,
    vec is the (B, 6F) tensor produced by tproj.
    """
    import torch
    from modeling.krea2 import KREA2_CONFIG
    from modeling.krea2.krea2_modeling import PositionalEncoding, SingleStreamBlock

    cfg = KREA2_CONFIG
    block = SingleStreamBlock(
        cfg.features, cfg.heads, cfg.multiplier, cfg.bias, cfg.kvheads
    ).to(device=device, dtype=dtype)
    return block, cfg, PositionalEncoding


def _krea2_inputs(cfg, PositionalEncoding, resolution: int, batch: int, device, dtype):
    import torch

    latent = resolution // 8              # VAE f8 (WAN21_F8C16)
    grid = latent // cfg.patch            # patch=2
    image_len = grid * grid
    text_len = 512                        # TextSpec.max_seq_len
    seq_len = text_len + image_len

    head_dim = cfg.features // cfg.heads
    axes = (
        head_dim - 12 * (head_dim // 16),
        6 * (head_dim // 16),
        6 * (head_dim // 16),
    )
    posemb = PositionalEncoding(axes, theta=cfg.theta)

    text_pos = torch.zeros(batch, text_len, 3, device=device, dtype=torch.float32)
    image_pos = torch.zeros(grid, grid, 3, device=device, dtype=torch.float32)
    image_pos[..., 1] = torch.arange(grid, device=device)[:, None]
    image_pos[..., 2] = torch.arange(grid, device=device)[None, :]
    image_pos = image_pos.reshape(1, image_len, 3).expand(batch, -1, -1)
    freqs = posemb(torch.cat((text_pos, image_pos), dim=1))

    x = torch.randn(batch, seq_len, cfg.features, device=device, dtype=dtype)
    vec = torch.randn(batch, 6 * cfg.features, device=device, dtype=dtype)
    return x, vec, freqs, seq_len


def stage_c(args) -> dict:
    """Forward / forward+backward time for a real block (T_compute)."""
    section("C - Single-block compute baseline (krea2)")
    import torch

    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    block, cfg, PositionalEncoding = _build_krea2_block(device, dtype)
    x, vec, freqs, seq_len = _krea2_inputs(
        cfg, PositionalEncoding, args.resolution, args.batch, device, dtype
    )

    param_bytes = sum(p.numel() * p.element_size() for p in block.parameters())
    param_count = sum(p.numel() for p in block.parameters())
    total_bytes = param_bytes * cfg.layers

    print(f"Resolution : {args.resolution}^2 -> seq_len {seq_len} (text 512 + image {seq_len - 512})")
    print(f"Single blk : {param_count / 1e6:.1f}M params, {param_bytes / MIB:.0f} MB @ {args.dtype}")
    print(f"{cfg.layers} layers  : {total_bytes / GIB:.2f} GB total (excludes txtfusion / embed / last)")
    record("C", "block_params", round(param_count / 1e6, 1), "M")
    record("C", "block_bytes", round(param_bytes / MIB), "MB", args.dtype)
    record("C", "dit_blocks_bytes", round(total_bytes / GIB, 2), "GB")

    # Default args bound early (same trick as krea2_modeling.py:517): deleting these
    # names at the end frees VRAM -- closure capture would otherwise leave them dangling
    def once(backward: bool, blk=block, inp=x, mod=vec, rope=freqs):
        if backward:
            out = blk(inp, mod, rope)
            out.sum().backward()
            blk.zero_grad(set_to_none=True)
        else:
            with torch.no_grad():
                blk(inp, mod, rope)

    for backward in (False, True):
        if backward:
            for p in block.parameters():
                p.requires_grad_(True)
            x.requires_grad_(True)
        for _ in range(3):  # warmup
            once(backward)
        torch.cuda.synchronize()
        samples = []
        for _ in range(args.iters):
            start = time.perf_counter()
            once(backward)
            torch.cuda.synchronize()
            samples.append(time.perf_counter() - start)
        median = statistics.median(samples)
        label = "fwd+bwd" if backward else "fwd"
        print(f"{label:>9} : {median * 1000:.2f} ms")
        record("C", "fwd_bwd_ms" if backward else "fwd_ms", round(median * 1000, 2), "ms")
        if backward:
            t_bwd = median
        else:
            t_fwd = median

    del block, x, vec, freqs
    torch.cuda.empty_cache()
    return {
        "param_bytes": param_bytes,
        "total_bytes": total_bytes,
        "layers": cfg.layers,
        "t_fwd": t_fwd,
        "t_fwd_bwd": t_bwd,
        "seq_len": seq_len,
    }


# ------------------------------------------------------------------ D masking criterion
def stage_d(bandwidth: dict, compute: dict) -> None:
    """Plugs B and C's numbers into doc §2.2's inequality, and prints the blocks_to_swap curve."""
    section("D - Masking criterion")

    bw = bandwidth.get("effective_bw")
    if not bw:
        print("No effective bandwidth from stage B, skipping")
        return

    param_bytes = compute["param_bytes"]
    t_transfer = param_bytes / (bw * GIB)
    t_fwd = compute["t_fwd"]
    t_fwd_bwd = compute["t_fwd_bwd"]

    print(f"T_transfer (1 block)  : {t_transfer * 1000:.2f} ms  "
          f"({param_bytes / MIB:.0f} MB / {bw:.1f} GB/s)")
    print(f"T_compute  forward    : {t_fwd * 1000:.2f} ms")
    print(f"T_compute  fwd+bwd    : {t_fwd_bwd * 1000:.2f} ms")
    record("D", "t_transfer_ms", round(t_transfer * 1000, 2), "ms")

    ratio_fwd = t_transfer / t_fwd
    print(f"\nTransfer/compute ratio (forward, inference basis)  : {ratio_fwd:.2f}")
    print(f"Transfer/compute ratio (fwd+bwd, training basis)    : {t_transfer / t_fwd_bwd:.2f}")
    record("D", "ratio_fwd", round(ratio_fwd, 3))
    record("D", "ratio_train", round(t_transfer / t_fwd_bwd, 3))

    if ratio_fwd < 1:
        print("-> Transfer is faster than compute: **theoretically fully maskable** (both forward and training)")
    else:
        print(f"-> Transfer is slower than compute: {(t_transfer - t_fwd) * 1000:.1f} ms exposed per block (forward basis)")

    # LoRA training: base model frozen, only H2D, 1 forward pass + 1 backward recompute (doc §2.3/2.4)
    layers = compute["layers"]
    per_block = param_bytes / GIB
    # Each block has two residency windows within a training step: once for the forward
    # pass, once for the backward pass (including checkpoint recompute). The two compute
    # times differ and must be compared against the transfer time separately (doc §2.4).
    t_bwd_only = max(t_fwd_bwd - t_fwd, 0.0)
    exposed_fwd = max(0.0, t_transfer - t_fwd)
    exposed_bwd = max(0.0, t_transfer - t_bwd_only)
    print(f"\nblocks_to_swap curve (LoRA training basis, 2 H2D passes per step)")
    print(f"  forward-pass exposure {exposed_fwd * 1000:.1f} ms/block, "
          f"backward-pass exposure {exposed_bwd * 1000:.1f} ms/block")
    print(f"{'N':>4} {'VRAM saved':>10} {'exposed/step':>12} {'vs baseline':>10}")
    print("-" * 40)
    base_step = t_fwd_bwd * layers
    for n in (0, 4, 8, 14, 20, 28):
        if n > layers:
            continue
        exposed = (exposed_fwd + exposed_bwd) * n
        saved = per_block * n
        overhead = exposed / base_step * 100 if base_step else 0
        print(f"{n:>4} {saved:>9.2f}GB {exposed * 1000:>10.1f}ms {overhead:>9.1f}%")
        record("D", f"swap_{n}_saved_gb", round(saved, 2), "GB")
        record("D", f"swap_{n}_overhead_pct", round(overhead, 1), "%")

    print("\nNote: this is a theoretical upper bound (assumes zero-overhead prefetch, perfect stream overlap). Stage E measures the real value.")


# ------------------------------------------------------------------ E end-to-end
def stage_e(args, compute: dict) -> dict:
    """Wall clock difference between a real dual-stream swap loop and a fully resident loop."""
    section("E - End-to-end swap loop vs fully resident")
    import torch
    from modeling.krea2 import KREA2_CONFIG
    from modeling.krea2.krea2_modeling import PositionalEncoding, SingleStreamBlock

    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    cfg = KREA2_CONFIG
    n = args.swap_blocks

    x0, vec, freqs, _ = _krea2_inputs(
        cfg, PositionalEncoding, args.resolution, args.batch, device, dtype
    )

    def make_block(dev):
        return SingleStreamBlock(
            cfg.features, cfg.heads, cfg.multiplier, cfg.bias, cfg.kvheads
        ).to(device=dev, dtype=dtype)

    need = compute["param_bytes"] * n / GIB
    free = torch.cuda.mem_get_info()[0] / GIB
    print(f"Control group needs {n} resident blocks ~= {need:.1f} GB, currently {free:.1f} GB free")
    if need + 3 > free:
        print("Not enough VRAM to build the fully-resident control group, reduce --swap-blocks")
        return {}

    # ---- Baseline: N blocks all resident on GPU
    resident = [make_block(device) for _ in range(n)]
    for _ in range(2):
        h = x0
        for blk in resident:
            with torch.no_grad():
                h = blk(h, vec, freqs)
    torch.cuda.synchronize()

    samples = []
    for _ in range(args.iters):
        start = time.perf_counter()
        h = x0
        for blk in resident:
            with torch.no_grad():
                h = blk(h, vec, freqs)
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - start)
    t_resident = statistics.median(samples)
    print(f"Fully resident : {t_resident * 1000:.1f} ms / {n} block")
    record("E", "resident_ms", round(t_resident * 1000, 1), "ms", f"{n} blocks")

    # ---- swap: weights pinned on CPU, 2 GPU buffers rotate + prefetch on an independent copy stream
    #
    # Measuring both transfer granularities, since the difference is this probe's most
    # important implementation guidance:
    #   per_tensor -- per-param `copy_` (naive implementation), a dozen-plus small transfers per block
    #   flat       -- weights rebound to one contiguous buffer, one big transfer per block (musubi's approach)
    cpu_weights = [
        {k: v.detach().to("cpu").pin_memory() for k, v in blk.state_dict().items()}
        for blk in resident
    ]
    del resident
    torch.cuda.empty_cache()

    def rebind_flat(block):
        """Rebind all of a block's parameters onto views of one contiguous GPU buffer."""
        entries = list(block.named_parameters())
        total = sum(p.numel() for _, p in entries)
        flat = torch.empty(total, device=device, dtype=dtype)
        offset = 0
        for _, param in entries:
            count = param.numel()
            view = flat[offset:offset + count].view(param.shape)
            view.copy_(param.detach())
            param.data = view
            offset += count
        return flat, [name for name, _ in entries]

    def build_swap(flat_mode: bool):
        buffers = [make_block(device), make_block(device)]
        copy_stream = torch.cuda.Stream()
        ready = [torch.cuda.Event() for _ in range(2)]
        done = [torch.cuda.Event() for _ in range(2)]

        if flat_mode:
            flats, order = zip(*(rebind_flat(b) for b in buffers))
            order = order[0]
            cpu_flat = []
            for weights in cpu_weights:
                parts = [weights[name].reshape(-1) for name in order]
                cpu_flat.append(torch.cat(parts).pin_memory())

            def prefetch(idx: int, slot: int) -> None:
                with torch.cuda.stream(copy_stream):
                    flats[slot].copy_(cpu_flat[idx], non_blocking=True)
                    ready[slot].record(copy_stream)
        else:
            buf_params = [dict(b.state_dict()) for b in buffers]

            def prefetch(idx: int, slot: int) -> None:
                with torch.cuda.stream(copy_stream):
                    for key, dst in buf_params[slot].items():
                        dst.copy_(cpu_weights[idx][key], non_blocking=True)
                    ready[slot].record(copy_stream)

        def swap_pass():
            h = x0
            prefetch(0, 0)
            for i in range(n):
                slot = i % 2
                if i + 1 < n:
                    # This slot's previous compute must finish first, otherwise the
                    # prefetch would overwrite weights still being read (the data race
                    # naive implementations most easily miss, and skipping it makes
                    # the timing look overly optimistic)
                    nxt = (i + 1) % 2
                    if i >= 1:
                        copy_stream.wait_event(done[nxt])
                    prefetch(i + 1, nxt)
                torch.cuda.current_stream().wait_event(ready[slot])
                with torch.no_grad():
                    h = buffers[slot](h, vec, freqs)
                done[slot].record(torch.cuda.current_stream())
            return h

        return swap_pass, buffers

    results: dict = {}
    for flat_mode, label in ((False, "per_tensor"), (True, "flat")):
        swap_pass, buffers = build_swap(flat_mode)
        for _ in range(2):
            swap_pass()
        torch.cuda.synchronize()
        samples = []
        for _ in range(args.iters):
            start = time.perf_counter()
            swap_pass()
            torch.cuda.synchronize()
            samples.append(time.perf_counter() - start)
        t_swap = statistics.median(samples)
        overhead = (t_swap - t_resident) / t_resident * 100
        print(f"swap({label:>10}) : {t_swap * 1000:.1f} ms / {n} block"
              f"   overhead {overhead:+.1f}%")
        record("E", f"swap_{label}_ms", round(t_swap * 1000, 1), "ms", f"{n} blocks")
        record("E", f"overhead_{label}_pct", round(overhead, 1), "%")
        results[label] = {"t_swap": t_swap, "overhead": overhead, "swap_pass": swap_pass}
        if flat_mode:
            best = results
        else:
            del buffers
            torch.cuda.empty_cache()

    saved = compute["param_bytes"] * n / GIB
    overhead = min(r["overhead"] for r in results.values())
    print(f"\nBest measured overhead : {overhead:+.1f}% (forward basis)")
    print(f"VRAM saved in exchange : {saved:.2f} GB ({n} blocks not resident)")
    record("E", "saved_gb", round(saved, 2), "GB")

    if overhead < 15:
        print("-> Meets doc §5's recommended threshold (<15%)")
    elif overhead > 30:
        print("-> Exceeds doc §5's rejection line (>30%)")
    else:
        print("-> Falls in the 15%~30% gray zone, needs a user call")
    print("Note: the forward basis is the **worst case** -- training's backward-pass compute time is much longer than transfer, making it easier to mask")

    winner = min(results.values(), key=lambda r: r["overhead"])
    return {
        "swap_pass": winner["swap_pass"],
        "t_swap": winner["t_swap"],
        "t_resident": t_resident,
        "overhead": overhead,
        "modes": {k: v["overhead"] for k, v in results.items()},
    }


# ------------------------------------------------------------------ G training basis
def stage_g(args, compute: dict) -> dict:
    """Training-basis end-to-end: gradient checkpointing + reverse-order backward prefetch (B10).

    Stage E only measures forward; the training basis was previously extrapolated from
    stage D's formula (doc §5.3-1). This stage measures the timing of a complete step,
    simulating **LoRA + gradient checkpointing**, the only real scenario in this repo:

      forward  base model frozen, no intermediate activations saved (checkpoint
               semantics), swapped in block by block
      backward reverse order N-1->0, each block's "recompute forward + backward" happens
               within the same residency window after being swapped in (the merge doc
               §2.4 describes), only grad_input is needed, no weight gradients

    Numerical correctness is out of scope for this stage (buffer weights get overwritten
    by rotation, so gradients are meaningless) -- only timing is measured. The LoRA
    parameters themselves stay resident on GPU and don't participate in swap; their
    compute cost is negligible relative to the base model.
    """
    section("G - Training-basis end-to-end (checkpoint + reverse-order backward prefetch)")
    import torch
    from modeling.krea2 import KREA2_CONFIG
    from modeling.krea2.krea2_modeling import PositionalEncoding, SingleStreamBlock

    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    cfg = KREA2_CONFIG
    n = args.swap_blocks

    x0, vec, freqs, _ = _krea2_inputs(
        cfg, PositionalEncoding, args.resolution, args.batch, device, dtype
    )

    def make_block(dev):
        blk = SingleStreamBlock(
            cfg.features, cfg.heads, cfg.multiplier, cfg.bias, cfg.kvheads
        ).to(device=dev, dtype=dtype)
        blk.requires_grad_(False)  # base model frozen (LoRA scenario)
        return blk

    need = compute["param_bytes"] * n / GIB
    free = torch.cuda.mem_get_info()[0] / GIB
    print(f"Control group needs {n} resident blocks ~= {need:.1f} GB, currently {free:.1f} GB free")
    if need + 4 > free:
        print("Not enough VRAM to build the fully-resident control group, reduce --swap-blocks")
        return {}

    def step(get_block, after=None):
        """Timing for one step of checkpoint training. get_block(i) returns the ready
        module for layer i; after(i) is called once that layer's compute finishes (the
        swap path uses it to mark "this slot can now be overwritten")."""
        saved = []
        h = x0
        for i in range(n):
            saved.append(h)
            with torch.no_grad():
                h = get_block(i, forward=True)(h, vec, freqs)
            if after is not None:
                after(i)
        grad = torch.ones_like(h)
        for i in reversed(range(n)):
            inp = saved[i].detach().requires_grad_(True)
            with torch.enable_grad():
                out = get_block(i, forward=False)(inp, vec, freqs)
            grad = torch.autograd.grad(out, inp, grad)[0]
            if after is not None:
                after(i)
        return grad

    # The two paths are measured **interleaved**: measuring them sequentially gets
    # contaminated by GPU clock state differences (whichever runs first is measured
    # cold, before the clock boosts) -- that's exactly how the first version's
    # sequential measurement produced a nonsensical -2.2% negative overhead
    def measure_ab(fn_a, fn_b) -> tuple[list[float], list[float]]:
        for _ in range(2):
            fn_a()
            fn_b()
        torch.cuda.synchronize()
        a_samples: list[float] = []
        b_samples: list[float] = []
        for _ in range(args.iters):
            for fn, bucket in ((fn_a, a_samples), (fn_b, b_samples)):
                start = time.perf_counter()
                fn()
                torch.cuda.synchronize()
                bucket.append(time.perf_counter() - start)
        return a_samples, b_samples

    # ---- Baseline: fully resident + the same checkpoint recompute semantics (kept
    # resident alongside swap so they can be interleaved)
    resident = [make_block(device) for _ in range(n)]

    def resident_step():
        return step(lambda i, forward: resident[i])

    # ---- swap: forward sequential prefetch + backward reverse-order prefetch
    cpu_weights = [
        {k: v.detach().to("cpu").pin_memory() for k, v in blk.state_dict().items()}
        for blk in resident
    ]

    buffers = [make_block(device), make_block(device)]
    buf_params = [dict(b.state_dict()) for b in buffers]
    copy_stream = torch.cuda.Stream()
    ready = [torch.cuda.Event() for _ in range(2)]
    done = [torch.cuda.Event() for _ in range(2)]
    fetched = [-1, -1]  # which layer each slot currently holds, to avoid re-transferring

    def prefetch(idx: int, slot: int) -> None:
        """Move layer idx's weights into slot. Waiting on a done event that was never
        recorded is a no-op, so the first round needs no special-casing."""
        if idx < 0 or idx >= n or fetched[slot] == idx:
            return
        with torch.cuda.stream(copy_stream):
            # This slot's previous compute must finish first, otherwise it would overwrite weights still being read
            copy_stream.wait_event(done[slot])
            for key, dst in buf_params[slot].items():
                dst.copy_(cpu_weights[idx][key], non_blocking=True)
            ready[slot].record(copy_stream)
        fetched[slot] = idx

    def get_swapped(i: int, forward: bool):
        slot = i % 2
        prefetch(i, slot)                       # this layer (usually already a prefetch hit from the previous round)
        prefetch(i + 1 if forward else i - 1, (i + 1) % 2 if forward else (i - 1) % 2)
        torch.cuda.current_stream().wait_event(ready[slot])
        return buffers[slot]

    def swap_step():
        # Reset every step: simulates weights not staying resident on GPU, so the
        # backward phase must re-transfer too (a real implementation could reuse the
        # two layers still in place at the end of the forward pass; this is the
        # conservative estimate)
        fetched[0] = fetched[1] = -1
        return step(get_swapped, after=lambda i: done[i % 2].record(
            torch.cuda.current_stream()
        ))

    res_samples, swap_samples = measure_ab(resident_step, swap_step)
    t_resident = statistics.median(res_samples)
    t_swap = statistics.median(swap_samples)
    spread = (
        statistics.stdev(res_samples) / t_resident * 100
        if len(res_samples) > 1 else 0.0
    )

    overhead = (t_swap - t_resident) / t_resident * 100
    saved_gb = compute["param_bytes"] * n / GIB
    print(f"resident+checkpoint : {t_resident * 1000:.1f} ms / {n} block"
          f"   (min {min(res_samples) * 1000:.1f}, jitter +-{spread:.1f}%)")
    print(f"block swap          : {t_swap * 1000:.1f} ms / {n} block"
          f"   (min {min(swap_samples) * 1000:.1f})")
    print(f"\nTraining-basis measured overhead  : {overhead:+.1f}%")
    if abs(overhead) < spread:
        print(f"  ! Overhead is smaller than the baseline's own jitter (+-{spread:.1f}%) -- "
              f"the conclusion is 'within noise', not a precise value")
    print(f"VRAM saved in exchange            : {saved_gb:.2f} GB")
    record("G", "resident_ms", round(t_resident * 1000, 1), "ms", f"{n} blocks")
    record("G", "swap_ms", round(t_swap * 1000, 1), "ms", f"{n} blocks")
    record("G", "overhead_pct", round(overhead, 1), "%")
    record("G", "baseline_spread_pct", round(spread, 1), "%")

    per_block_ms = (t_swap - t_resident) / n * 1000
    print(f"Extra per block    : {per_block_ms:+.2f} ms (forward+backward residency windows combined)")
    record("G", "per_block_extra_ms", round(per_block_ms, 2), "ms")

    # Not explicitly deleted: resident / buffers are held by the two closures above, and get GC'd together once the function returns
    return {
        "overhead": overhead,
        "t_swap": t_swap,
        "t_resident": t_resident,
        "spread": spread,
    }


# ------------------------------------------------------------------ F stability
def stage_f(args, nvml: Nvml, e_result: dict) -> None:
    """Temperature / power / replay delta / available RAM under sustained load (doc §3.2)."""
    section(f"F - Stability and hardware impact ({args.soak_seconds}s sustained swap load)")
    import torch
    from training.sysmem import available_ram_bytes

    swap_pass = e_result.get("swap_pass")
    if swap_pass is None:
        print("Stage E didn't produce a swap loop, skipping")
        return

    replay_start = nvml.replay_counter()
    ram_start = available_ram_bytes()
    temps: list[int] = []
    powers: list[float] = []
    passes = 0

    print(f"{'t':>6} {'temp':>6} {'power':>8} {'RAM avail':>10} {'replay':>8}")
    print("-" * 44)
    start = time.perf_counter()
    next_sample = 0.0
    while True:
        elapsed = time.perf_counter() - start
        if elapsed >= args.soak_seconds:
            break
        swap_pass()
        torch.cuda.synchronize()
        passes += 1
        if elapsed >= next_sample:
            temp = nvml.temperature()
            power = nvml.power_watts()
            ram = available_ram_bytes()
            replay = nvml.replay_counter()
            if temp is not None:
                temps.append(temp)
            if power is not None:
                powers.append(power)
            print(
                f"{elapsed:>5.0f}s {str(temp) + '°C':>6} "
                f"{(f'{power:.0f}W' if power else '-'):>8} "
                f"{(f'{ram / GIB:.1f}GB' if ram else '-'):>10} "
                f"{str(replay):>8}"
            )
            next_sample = elapsed + 5.0

    replay_end = nvml.replay_counter()
    ram_end = available_ram_bytes()
    print(f"\nCompleted {passes} passes, {passes * args.swap_blocks} block swap-ins")

    if replay_start is not None and replay_end is not None:
        delta = replay_end - replay_start
        print(f"PCIe replay delta : {delta}")
        record("F", "replay_delta", delta)
        if delta == 0:
            print("  -> Zero link retransmits, PCIe side is healthy (doc §3.2 (2) passes)")
        else:
            print("  ! Link retransmits occurred: check the slot / riser / whether it's on a chipset lane")
    if temps:
        print(f"Temp max/mean      : {max(temps)}C / {statistics.mean(temps):.0f}C")
        record("F", "temp_max", max(temps), "C")
    if powers:
        print(f"Power max/mean     : {max(powers):.0f}W / {statistics.mean(powers):.0f}W")
        record("F", "power_max", round(max(powers)), "W")
    if ram_start and ram_end:
        drift = (ram_start - ram_end) / MIB
        print(f"Available RAM drift : {drift:+.0f} MB (pinned memory stays resident, shouldn't keep growing)")
        record("F", "ram_drift_mb", round(drift))


# ------------------------------------------------------------------ main
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gate-0 feasibility probe for block swap (see docs/design/block-swap.md)"
    )
    parser.add_argument("--stages", default="ABCDEF", help="stages to run, e.g. ABCD")
    parser.add_argument("--resolution", type=int, default=1024, help="training/inference side length")
    parser.add_argument("--batch", type=int, default=1, help="batch size")
    parser.add_argument("--dtype", default="bf16", choices=("bf16", "fp16"))
    parser.add_argument("--iters", type=int, default=10, help="number of samples per measurement")
    parser.add_argument("--swap-blocks", type=int, default=8, help="number of blocks in stage E's control group")
    parser.add_argument("--soak-seconds", type=int, default=60, help="stage F's sustained duration")
    parser.add_argument("--out", type=Path, help="CSV path to write observations to")
    args = parser.parse_args()

    stages = args.stages.upper()
    nvml = Nvml()
    try:
        info = stage_a(nvml) if "A" in stages else {}

        import torch

        if not torch.cuda.is_available():
            print("\nNo CUDA device -- skipping everything after stage B.")
            return 0

        bandwidth = stage_b(args) if "B" in stages else {}
        compute = stage_c(args) if "C" in stages else {}
        if "D" in stages and bandwidth and compute:
            stage_d(bandwidth, compute)
        e_result = stage_e(args, compute) if "E" in stages and compute else {}
        g_result = stage_g(args, compute) if "G" in stages and compute else {}
        if "F" in stages and e_result:
            stage_f(args, nvml, e_result)

        section("Summary of conclusions")
        if e_result:
            print(f"Inference-basis (forward) measured overhead {e_result['overhead']:+.1f}%")
        if g_result:
            print(f"Training-basis (checkpoint+reverse backward) measured overhead {g_result['overhead']:+.1f}%")
        if compute:
            print(f"VRAM saved {compute['param_bytes'] * args.swap_blocks / GIB:.2f} GB "
                  f"({args.swap_blocks} blocks, {args.resolution}^2); "
                  f"all {compute['layers']} layers could save {compute['total_bytes'] / GIB:.2f} GB")
        print("See docs/design/block-swap.md §5 for thresholds and §7 for open questions")

        if args.out:
            with args.out.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(
                    fh, fieldnames=["stage", "metric", "value", "unit", "note"]
                )
                writer.writeheader()
                writer.writerows(RECORDS)
            print(f"\nObservations written to {args.out} ({len(RECORDS)} entries)")
        return 0
    finally:
        nvml.shutdown()


if __name__ == "__main__":
    sys.exit(main())
