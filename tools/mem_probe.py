#!/usr/bin/env python3
"""mem_probe -- memory/VRAM peak localization probe (Windows-first).

Purpose: localize "training sample / test generate occasionally hangs + system committed
virtual memory spikes by 20-30GB". The key discriminant is whether that chunk of memory sits
on the **GPU side** (mirrored into system commit by WDDM) or the **CPU/host side**. This probe
samples both, every --interval seconds:

  - System commit (Commit Charge): GetPerformanceInfo().CommitTotal -- the same number Task
    Manager shows under "Performance > Memory > Committed", i.e. the one the user sees spiking.
  - Target process private commit (PrivateUsage, roughly Task Manager's process "Commit size").
  - GPU VRAM: NVML (falls back to nvidia-smi if unavailable), device-level used/total, plus
    per-pid when possible.

Computes the delta each frame (difference from the previous frame). When |Δcommit| or
|Δgpu_used| exceeds --spike-gb, prints a loud warning plus a one-line **verdict**:

  - Δgpu ≈ Δcommit  -> a GPU allocation got mirrored into commit by WDDM (VRAM-side blowup,
    usually some convolution workspace / a large bucket resolution; on an 8G card this just
    falls back to chunked processing via OOM so it's fine, but a large-VRAM card instead fills
    up commit and hangs).
  - Δcommit >> Δgpu  -> a CPU/host-side allocation (pinned memory / a huge CPU tensor / a leak).

Optional --pyspy: automatically runs `py-spy dump --pid <pid>` when a spike is caught, giving a
zero-intrusion snapshot of the Python call stack executing at that moment (requires
`pip install py-spy`).

Writes a CSV the whole time (--out), for plotting/comparison afterward.

Usage
-----
1) Start training/generation first, and get that python process's PID (Task Manager / `tasklist`).
2) In another terminal:
     python tools/mem_probe.py --pid <PID> --interval 0.25 --spike-gb 4 --pyspy
   or if you don't know the PID, pick by name (selects the python process with the largest
   private commit):
     python tools/mem_probe.py --name python --interval 0.25 --spike-gb 4
3) Reproduce the hang. Watch the [SPIKE] lines in the terminal + the SUMMARY at the end, or read
   mem_probe.csv afterward.

Dependencies: psutil (strongly recommended). NVML (pynvml) is optional; falls back to
nvidia-smi if absent. System commit is read directly via ctypes, no third-party dependency
needed for that.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import os
import shutil
import subprocess
import sys
import time
from ctypes import wintypes

GB = 1024.0 ** 3

try:
    import psutil  # type: ignore
except Exception:  # noqa: BLE001
    psutil = None


# ---------------------------------------------------------------- System commit (Commit Charge)
class _PERFORMANCE_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("CommitTotal", ctypes.c_size_t),
        ("CommitLimit", ctypes.c_size_t),
        ("CommitPeak", ctypes.c_size_t),
        ("PhysicalTotal", ctypes.c_size_t),
        ("PhysicalAvailable", ctypes.c_size_t),
        ("SystemCache", ctypes.c_size_t),
        ("KernelTotal", ctypes.c_size_t),
        ("KernelPaged", ctypes.c_size_t),
        ("KernelNonpaged", ctypes.c_size_t),
        ("PageSize", ctypes.c_size_t),
        ("HandleCount", wintypes.DWORD),
        ("ProcessCount", wintypes.DWORD),
        ("ThreadCount", wintypes.DWORD),
    ]


def system_commit_bytes() -> tuple[float, float, float]:
    """Returns (CommitTotal, CommitLimit, CommitPeak) in bytes. Windows only."""
    pi = _PERFORMANCE_INFORMATION()
    pi.cb = ctypes.sizeof(pi)
    ok = ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(pi), pi.cb)
    if not ok:
        raise ctypes.WinError()
    ps = pi.PageSize
    return pi.CommitTotal * ps, pi.CommitLimit * ps, pi.CommitPeak * ps


# ---------------------------------------------------------------- Target process
def resolve_pid(pid: int | None, name: str | None) -> int:
    if pid:
        return pid
    if psutil is None:
        sys.exit("psutil is not installed and no --pid was given, cannot look up by name. Install psutil or pass --pid.")
    name_l = (name or "python").lower()
    best, best_priv = None, -1
    for p in psutil.process_iter(["name"]):
        try:
            if name_l in (p.info["name"] or "").lower():
                priv = p.memory_info().private
                if priv > best_priv:
                    best, best_priv = p.pid, priv
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if best is None:
        sys.exit(f"No process found with a name containing '{name_l}'; specify one with --pid.")
    print(f"[mem_probe] Selected PID={best} by name '{name_l}' (largest private commit)")
    return best


def proc_mem_bytes(proc) -> tuple[float, float]:
    """(private/commit, working_set) in bytes."""
    mi = proc.memory_info()
    # psutil on Windows: memory_info() has both private (PrivateUsage) and rss (WorkingSet)
    private = getattr(mi, "private", getattr(mi, "pagefile", mi.vms))
    return float(private), float(mi.rss)


# ---------------------------------------------------------------- GPU
class _GpuReader:
    def __init__(self, index: int):
        self.index = index
        self.nvml = None
        self.handle = None
        try:
            import pynvml  # type: ignore
            pynvml.nvmlInit()
            self.nvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        except Exception:  # noqa: BLE001
            self.nvml = None
        self.smi = shutil.which("nvidia-smi")

    def read(self, target_pid: int) -> tuple[float, float, float]:
        """Returns (used, total, proc_used) in bytes; proc_used is -1 when unavailable."""
        if self.nvml is not None:
            try:
                m = self.nvml.nvmlDeviceGetMemoryInfo(self.handle)
                proc_used = -1.0
                try:
                    for pr in self.nvml.nvmlDeviceGetComputeRunningProcesses(self.handle):
                        if pr.pid == target_pid and pr.usedGpuMemory not in (None, 0):
                            proc_used = float(pr.usedGpuMemory)
                except Exception:  # noqa: BLE001
                    pass  # per-pid is often unavailable under WDDM
                return float(m.used), float(m.total), proc_used
            except Exception:  # noqa: BLE001
                pass
        if self.smi:
            try:
                out = subprocess.check_output(
                    [self.smi, f"--id={self.index}",
                     "--query-gpu=memory.used,memory.total",
                     "--format=csv,noheader,nounits"],
                    text=True, timeout=5,
                ).strip().splitlines()[0]
                used_mb, total_mb = (float(x) for x in out.split(","))
                return used_mb * 1024 * 1024, total_mb * 1024 * 1024, -1.0
            except Exception:  # noqa: BLE001
                pass
        return -1.0, -1.0, -1.0


# ---------------------------------------------------------------- main loop
def main() -> None:
    # The console may be cp932/gbk or some other non-utf-8 codepage (this machine uses cp932);
    # non-ASCII print output would raise UnicodeEncodeError and crash the probe itself.
    # Force utf-8 + errors=replace uniformly.
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")  # py3.7+
        except Exception:  # noqa: BLE001
            pass

    if os.name != "nt":
        print("[mem_probe] Note: the system commit reading uses the Windows API; on non-Windows only GPU/process readings are available.")

    ap = argparse.ArgumentParser(description="Memory/VRAM peak localization probe")
    ap.add_argument("--pid", type=int, default=None, help="target process PID")
    ap.add_argument("--name", type=str, default="python", help="find by name (--pid takes priority)")
    ap.add_argument("--gpu-index", type=int, default=0)
    ap.add_argument("--interval", type=float, default=0.25, help="sampling interval in seconds")
    ap.add_argument("--spike-gb", type=float, default=4.0, help="warn when Δcommit/Δgpu exceeds this value (GB)")
    ap.add_argument("--out", type=str, default="mem_probe.csv")
    ap.add_argument("--pyspy", action="store_true", help="automatically py-spy dump the target process's stack on a spike")
    args = ap.parse_args()

    pid = resolve_pid(args.pid, args.name)
    if psutil is None:
        sys.exit("This script needs psutil to read process memory: pip install psutil")
    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        sys.exit(f"PID {pid} does not exist.")
    gpu = _GpuReader(args.gpu_index)
    pyspy = shutil.which("py-spy") if args.pyspy else None
    if args.pyspy and not pyspy:
        print("[mem_probe] py-spy not found, ignoring --pyspy (pip install py-spy)")

    fields = ["t_rel_s", "sys_commit_GB", "sys_commit_limit_GB", "sys_commit_pct",
              "proc_private_GB", "proc_wset_GB", "gpu_used_GB", "gpu_total_GB",
              "gpu_proc_GB", "d_commit_GB", "d_gpu_GB", "note"]
    f = open(args.out, "w", newline="", encoding="utf-8")
    w = csv.writer(f)
    w.writerow(fields)

    print(f"[mem_probe] PID={pid} GPU#{args.gpu_index} interval={args.interval}s "
          f"spike>{args.spike_gb}G -> {args.out}  (Ctrl+C to stop)")

    t0 = time.perf_counter()
    prev_commit = prev_gpu = None
    peak = {"commit": 0.0, "private": 0.0, "gpu": 0.0}
    spikes: list[str] = []

    try:
        while True:
            if not proc.is_running():
                print("[mem_probe] Target process has exited, stopping.")
                break
            t = time.perf_counter() - t0
            try:
                commit, climit, cpeak = system_commit_bytes()
            except Exception as e:  # noqa: BLE001
                commit = climit = cpeak = -1.0
                _ = e
            try:
                priv, wset = proc_mem_bytes(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            gused, gtotal, gproc = gpu.read(pid)

            d_commit = (commit - prev_commit) / GB if prev_commit is not None and commit >= 0 else 0.0
            d_gpu = (gused - prev_gpu) / GB if prev_gpu is not None and gused >= 0 else 0.0
            prev_commit = commit if commit >= 0 else prev_commit
            prev_gpu = gused if gused >= 0 else prev_gpu

            peak["commit"] = max(peak["commit"], commit)
            peak["private"] = max(peak["private"], priv)
            peak["gpu"] = max(peak["gpu"], gused)

            note = ""
            if abs(d_commit) >= args.spike_gb or abs(d_gpu) >= args.spike_gb:
                # verdict
                if gused >= 0 and abs(d_gpu) >= args.spike_gb and abs(d_commit - d_gpu) <= max(2.0, 0.3 * abs(d_gpu)):
                    verdict = "GPU side (WDDM-mirrored into commit)"
                elif abs(d_commit) >= args.spike_gb and (gused < 0 or abs(d_commit) - abs(d_gpu) >= args.spike_gb):
                    verdict = "CPU/host side"
                else:
                    verdict = "mixed/inconclusive"
                note = f"SPIKE {verdict} dCommit={d_commit:+.1f}G dGPU={d_gpu:+.1f}G"
                line = (f"[SPIKE] t={t:7.2f}s  {verdict}  "
                        f"Δcommit={d_commit:+.1f}G  Δgpu={d_gpu:+.1f}G  "
                        f"commit={commit/GB:.1f}G gpu_used={gused/GB if gused>=0 else -1:.1f}G "
                        f"proc_priv={priv/GB:.1f}G")
                print("\n" + "!" * 78 + "\n" + line + "\n" + "!" * 78)
                spikes.append(line)
                if pyspy:
                    try:
                        dump = subprocess.check_output(
                            [pyspy, "dump", "--pid", str(pid), "--nonblocking"],
                            text=True, timeout=15, stderr=subprocess.STDOUT)
                        with open("mem_probe_pyspy.log", "a", encoding="utf-8") as pf:
                            pf.write(f"\n===== SPIKE t={t:.2f}s {verdict} =====\n{dump}\n")
                        print("[mem_probe] py-spy stack appended to mem_probe_pyspy.log")
                    except Exception as e:  # noqa: BLE001
                        print(f"[mem_probe] py-spy dump failed: {e}")

            w.writerow([f"{t:.3f}",
                        f"{commit/GB:.3f}" if commit >= 0 else "",
                        f"{climit/GB:.3f}" if climit >= 0 else "",
                        f"{100*commit/climit:.1f}" if climit > 0 else "",
                        f"{priv/GB:.3f}", f"{wset/GB:.3f}",
                        f"{gused/GB:.3f}" if gused >= 0 else "",
                        f"{gtotal/GB:.3f}" if gtotal >= 0 else "",
                        f"{gproc/GB:.3f}" if gproc >= 0 else "",
                        f"{d_commit:+.3f}", f"{d_gpu:+.3f}", note])
            f.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[mem_probe] Stopped manually.")
    finally:
        f.close()
        print("\n" + "=" * 60 + "\nSUMMARY")
        print(f"  Peak system commit         : {peak['commit']/GB:.1f} G")
        print(f"  Peak process private commit: {peak['private']/GB:.1f} G")
        print(f"  Peak GPU used               : {peak['gpu']/GB:.1f} G" if peak['gpu'] >= 0 else "  GPU unreadable")
        print(f"  CSV                         : {os.path.abspath(args.out)}")
        if spikes:
            print(f"  Captured {len(spikes)} spike(s):")
            for s in spikes:
                print("    " + s)
        else:
            print("  No spikes captured (didn't reproduce, or --spike-gb was set too high).")
        print("=" * 60)


if __name__ == "__main__":
    main()
