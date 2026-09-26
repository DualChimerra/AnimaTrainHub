"""WD14 tagging performance diagnostics: per-stage timing + EP / preload / model / thread-count self-check.

How to run (from repo root):
    venv/bin/python tools/bench_wd14.py [<image_dir>] [--n 10] [--model <hf_id>]
    # Windows: venv\\Scripts\\python.exe tools\\bench_wd14.py ...

When no image directory is given, it scans `studio_data/projects/*/raw_*` for the most recent
batch of images and takes the first N. All logs go to both stdout and `bench_wd14.log`.

Questions it can answer:
1. Is the current onnxruntime the GPU package or the CPU package, and does the CUDA EP actually work?
2. How many torch CUDA .so files did preload manage to hit?
3. How many threads does the CPU EP actually run on?
4. Per-image breakdown: how much time does preprocess / session.run / postprocess each take?
5. How much faster is GPU than CPU in measured throughput (if both providers can create a session)?
"""
from __future__ import annotations

import argparse
import logging
import os
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

LOG_PATH = REPO_ROOT / "bench_wd14.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("bench_wd14")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def find_default_images(n: int) -> list[Path]:
    """Recursively scan studio_data/projects/ at any depth for images.

    Studio project layout:
      studio_data/projects/{id}-{slug}/download/...        -- raw images pulled from booru
      studio_data/projects/{id}-{slug}/versions/{label}/train/{folder}/...
    Either layout works; the subdirectory name is not restricted.
    """
    base = REPO_ROOT / "studio_data" / "projects"
    if not base.exists():
        return []
    candidates: list[Path] = []
    for f in base.rglob("*"):
        if f.is_file() and f.suffix.lower() in IMG_EXTS:
            candidates.append(f)
            if len(candidates) >= n:
                break
    return candidates[:n]


def report_environment() -> None:
    """Log the onnxruntime package name/version/EP/CPU/preload status."""
    log.info("=" * 60)
    log.info("ENVIRONMENT")
    log.info("=" * 60)
    log.info("python: %s", sys.version.split()[0])
    log.info("platform: %s", sys.platform)
    log.info("cpu count: %s", os.cpu_count())

    # PP9.5 -- preload already ran when this module was imported; just read the result
    from studio.services.runtime import onnxruntime as ors

    rt = ors.current_runtime()
    log.info(
        "onnxruntime: installed=%s version=%s",
        rt["installed"], rt["version"],
    )
    log.info("providers (advertised): %s", rt["providers"])
    log.info("cuda_available (advertised): %s", rt["cuda_available"])
    log.info("cuda_load_error (last): %s", rt.get("cuda_load_error"))

    pre = rt.get("preload") or {}
    log.info(
        "preload: applied=%s skip=%s candidates=%s preloaded=%d errors=%d",
        pre.get("applied"),
        pre.get("platform_skip"),
        pre.get("candidates"),
        len(pre.get("preloaded") or []),
        len(pre.get("errors") or []),
    )
    for path in pre.get("preloaded") or []:
        log.info("  preload OK: %s", os.path.basename(path))
    for path, reason in pre.get("errors") or []:
        log.info("  preload FAIL: %s -- %s", os.path.basename(path), reason)

    cuda = ors.detect_cuda()
    log.info(
        "nvidia-smi: available=%s driver=%s gpu=%s",
        cuda["available"], cuda.get("driver_version"), cuda.get("gpu_name"),
    )


def probe_provider(model_path: str, provider: str) -> tuple[bool, str, object]:
    """Actually try to create an InferenceSession; returns (ok, msg, session_or_err_str).

    onnxruntime does **not** raise when the requested EP fails to dlopen -- it silently
    falls back to the next available EP (usually CPU). So `ok` can't just check try/except;
    it also has to compare sess.get_providers()[0] against the requested provider. A mismatch
    means it was "silently downgraded", which counts as a failure here.
    """
    import onnxruntime as ort

    try:
        sess = ort.InferenceSession(model_path, providers=[provider])
    except Exception as exc:  # noqa: BLE001
        return False, str(exc), None
    actual = sess.get_providers()
    if provider not in actual:
        return False, f"silently downgraded to {actual} (requested {provider} failed)", None
    return True, f"providers={actual}", sess


def time_stage(fn, *args, **kwargs):
    t0 = time.perf_counter()
    res = fn(*args, **kwargs)
    return res, (time.perf_counter() - t0) * 1000


def bench_once(tagger, sess, paths: list[Path], label: str) -> None:
    """Run the given session over paths, timing and printing each stage."""
    import numpy as np

    log.info("-" * 60)
    log.info("RUN [%s] on %d images", label, len(paths))
    log.info("intra_op_num_threads: %s", sess.get_session_options().intra_op_num_threads or "(default)")
    log.info("inter_op_num_threads: %s", sess.get_session_options().inter_op_num_threads or "(default)")

    # Inject the session into tagger (bypassing the one prepare() would create itself)
    tagger._session = sess
    if not tagger._tags:
        # selected_tags.csv must be loaded before postprocess can run; do one prepare()
        # pass to get the tags (it creates another session, which we just discard)
        old = tagger._session
        tagger._session = None
        tagger.prepare()
        tagger._session = old
    name_in = sess.get_inputs()[0].name

    # warmup (the first run includes graph optimization / kernel compilation, must be excluded)
    from PIL import Image
    with Image.open(paths[0]) as im:
        warm = tagger._preprocess(im)
    sess.run(None, {name_in: np.stack([warm], axis=0).copy()})

    pre_ms: list[float] = []
    inf_ms: list[float] = []
    post_ms: list[float] = []
    for p in paths:
        with Image.open(p) as im:
            arr, t_pre = time_stage(tagger._preprocess, im)
        batch = np.stack([arr], axis=0).copy()
        logits, t_inf = time_stage(sess.run, None, {name_in: batch})
        _, t_post = time_stage(tagger._postprocess_one, logits[0][0])
        pre_ms.append(t_pre)
        inf_ms.append(t_inf)
        post_ms.append(t_post)

    def stats(xs: list[float]) -> str:
        return (
            f"mean={statistics.mean(xs):7.1f}ms "
            f"median={statistics.median(xs):7.1f}ms "
            f"min={min(xs):7.1f}ms max={max(xs):7.1f}ms"
        )

    log.info("preprocess  : %s", stats(pre_ms))
    log.info("session.run : %s", stats(inf_ms))
    log.info("postprocess : %s", stats(post_ms))
    total = sum(pre_ms) + sum(inf_ms) + sum(post_ms)
    log.info(
        "TOTAL %.1fs over %d images = %.2f img/s (%.1fms/img)",
        total / 1000, len(paths), len(paths) * 1000 / total, total / len(paths),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", help="image directory; auto-finds raw_* when omitted")
    parser.add_argument("--n", type=int, default=10, help="how many images to run (warmup not counted)")
    parser.add_argument(
        "--model",
        default=None,
        help="override secrets.wd14.model_id (e.g. SmilingWolf/wd-vit-tagger-v3)",
    )
    parser.add_argument(
        "--provider",
        choices=["auto", "cpu", "gpu", "both"],
        default="both",
        help="which EP to run; both = run both for comparison",
    )
    args = parser.parse_args()

    report_environment()

    if args.path:
        d = Path(args.path)
        paths = [p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS][: args.n]
    else:
        paths = find_default_images(args.n)
    if len(paths) < 2:
        log.error("Not enough images found (need >=2, found %d). Specify a path or create a project first.", len(paths))
        return 1
    log.info("images: %d, first is %s", len(paths), paths[0].name)

    from studio.services.tagging.wd14 import WD14Tagger

    overrides = {"model_id": args.model} if args.model else {}
    tagger = WD14Tagger(overrides=overrides or None)
    model_dir = tagger._resolve_model_dir()
    onnx_path = str(model_dir / "model.onnx")
    size_mb = (model_dir / "model.onnx").stat().st_size / (1024 * 1024)
    log.info("model: %s (%.1f MB)", model_dir.name, size_mb)

    log.info("=" * 60)
    log.info("PROBE PROVIDERS")
    log.info("=" * 60)
    cpu_ok, cpu_msg, cpu_sess = probe_provider(onnx_path, "CPUExecutionProvider")
    log.info("CPU EP: ok=%s %s", cpu_ok, cpu_msg if cpu_ok else cpu_msg[:200])
    gpu_ok, gpu_msg, gpu_sess = (False, "skipped (no advertised CUDA EP)", None)
    import onnxruntime as ort

    if "CUDAExecutionProvider" in ort.get_available_providers():
        gpu_ok, gpu_msg, gpu_sess = probe_provider(onnx_path, "CUDAExecutionProvider")
        log.info("GPU EP: ok=%s %s", gpu_ok, gpu_msg if gpu_ok else gpu_msg[:300])
    else:
        log.info("GPU EP: %s", gpu_msg)

    if args.provider in ("cpu", "both") and cpu_ok:
        bench_once(tagger, cpu_sess, paths, "CPU")
    if args.provider in ("gpu", "both") and gpu_ok:
        bench_once(tagger, gpu_sess, paths, "GPU")
    if args.provider == "auto":
        sess = gpu_sess if gpu_ok else cpu_sess
        bench_once(tagger, sess, paths, "GPU" if gpu_ok else "CPU")

    log.info("done. log saved to %s", LOG_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
