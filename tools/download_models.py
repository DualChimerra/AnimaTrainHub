#!/usr/bin/env python
"""Download every model + tokenizer required to train a given model family (thin CLI shell).

The actual logic lives in `studio.services.model_downloader`, shared between the CLI and the
Studio settings page UI.

Final layout on disk (by default determined by Settings' models_root):
    models/
      diffusion_models/anima-base-v1.0.safetensors
      diffusion_models/krea2-raw-bf16.safetensors
      vae/qwen_image_vae.safetensors
      text_encoders/                # Anima Qwen3 (legacy flat layout)
      text_encoders/Qwen_Qwen3-VL-4B-Instruct/
      t5_tokenizer/                 # T5 tokenizer only, no weights

Usage:
    python tools/download_models.py
    python tools/download_models.py --family krea2
    python tools/download_models.py --family krea2 --variant turbo
    python tools/download_models.py --variant preview3-base
    python tools/download_models.py --no-mirror
    python tools/download_models.py --skip-main --skip-vae
    python tools/download_models.py --output /data/anima
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Windows consoles using cp936/cp932 raise UnicodeEncodeError on non-ASCII / emoji; force UTF-8.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

# Let `python tools/download_models.py` import the studio package too
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from studio.services.models import (  # noqa: E402
    ANIMA_VARIANTS,
    KREA2_VARIANTS,
    LATEST_ANIMA,
    LATEST_KREA2,
    download_anima_main,
    download_anima_vae,
    download_krea2_main,
    download_qwen3,
    download_qwen3_vl,
    download_t5_tokenizer,
    models_root,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download the models + tokenizer required to train Anima / Krea 2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""\
Anima main model versions (--variant):
{chr(10).join(f"  {k:<14} {v}" for k, v in ANIMA_VARIANTS.items())}
  latest         (= {LATEST_ANIMA})

Krea 2 main model versions (--family krea2 --variant):
{chr(10).join(f"  {k:<14} {v['repo']}/{v['subpath']}" for k, v in KREA2_VARIANTS.items())}
  latest         (= {LATEST_KREA2})

Download source: read from secrets.huggingface.endpoint by default (empty on first install = official HF).
  - --no-mirror     force the official HuggingFace source (overrides secrets, equivalent to endpoint=https://huggingface.co)
  - --endpoint URL  custom endpoint URL (overrides secrets and --no-mirror)
  - --modelscope    download via ModelScope instead (requires pip install modelscope)

Note: as of the 0.8.2 hotfix, hf-mirror.com is temporarily unavailable (see docs/todo/hf-mirror-recheck.md).
    The Settings UI has hidden that preset, but --endpoint URL still accepts any value.
""",
    )
    parser.add_argument(
        "--family", default="anima", choices=["anima", "krea2"],
        help="model family (default: anima)",
    )
    parser.add_argument(
        "--no-mirror", action="store_true",
        help="use the official HuggingFace source (equivalent to --endpoint=https://huggingface.co)",
    )
    parser.add_argument(
        "--endpoint", default=None,
        help="custom HF endpoint URL (overrides the secrets config + --no-mirror)",
    )
    parser.add_argument(
        "--modelscope", action="store_true",
        help="download via ModelScope instead; models with no mapping fall back to HF automatically",
    )
    parser.add_argument(
        "--output", default="",
        help="target root directory (defaults to Settings' models_root)",
    )
    parser.add_argument(
        "--variant", default="latest",
        help="main model version (default: 1.0 for Anima, raw for Krea 2)",
    )
    parser.add_argument("--skip-main", action="store_true")
    parser.add_argument("--skip-vae",  action="store_true")
    parser.add_argument("--skip-qwen", action="store_true")
    parser.add_argument("--skip-t5",   action="store_true")
    args = parser.parse_args()

    import os  # noqa: PLC0415
    if args.modelscope:
        os.environ["MODELSCOPE_SOURCE"] = "modelscope"
        print("Using download source: ModelScope (models with no mapping fall back to HF automatically)")
    else:
        # Explicit CLI flags override secrets: --endpoint wins; --no-mirror sets official HF; neither given -> secrets.
        if args.endpoint:
            os.environ["HF_ENDPOINT"] = args.endpoint
        elif args.no_mirror:
            os.environ["HF_ENDPOINT"] = "https://huggingface.co"
        from studio.services.models import _resolve_endpoint  # noqa: PLC0415
        active = _resolve_endpoint() or "https://huggingface.co (HF default)"
        print(f"Using download source: HuggingFace  endpoint: {active}")

    out_root = Path(args.output) if args.output else models_root()
    print(f"📁 Target root directory: {out_root.absolute()}")

    variants = ANIMA_VARIANTS if args.family == "anima" else KREA2_VARIANTS
    latest = LATEST_ANIMA if args.family == "anima" else LATEST_KREA2
    variant = latest if args.variant == "latest" else args.variant
    if variant not in variants:
        parser.error(
            f"{args.family} does not support variant {args.variant!r}; choices: "
            f"{', '.join(variants)} / latest"
        )

    ok = True
    if not args.skip_main:
        if args.family == "anima":
            ok &= download_anima_main(out_root, variant)
        else:
            ok &= download_krea2_main(out_root, variant)
    if not args.skip_vae:
        ok &= download_anima_vae(out_root)
    if not args.skip_qwen:
        if args.family == "anima":
            ok &= download_qwen3(out_root)
        else:
            ok &= download_qwen3_vl(out_root)
    if args.family == "anima" and not args.skip_t5:
        ok &= download_t5_tokenizer(out_root)

    print()
    print("=" * 50)
    print("✅ All downloads complete!" if ok else "⚠️  Some downloads failed, see the log above for details")
    print("=" * 50)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
