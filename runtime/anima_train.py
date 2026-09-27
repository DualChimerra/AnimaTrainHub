#!/usr/bin/env python
"""Anima LoRA Trainer v2 -- main() orchestration entry point.

This module's implementation layer was split into the runtime/training/
subpackage per ADR 0003 PR-A:
  bootstrap / cli / observability / model_loading / models / text_encoding /
  state / dataset / sampling / timestep_sampling / noise / loss_weighting

The re-export block at the top keeps the anima_train.X access path unchanged
for sister scripts (anima_daemon / anima_generate / anima_reg_ai) and tests/.
New code should `from training.X import Y` directly.

LoRA / LoKr implementation: see utils.lycoris_adapter.AnimaLycorisAdapter (ADR 0001).
"""

import logging
import os
import sys
from pathlib import Path

# Small-VRAM optimization: reduces CUDA memory fragmentation, mitigating LoKr
# full-matrix OOM on 8GB cards.
# - Must be set before the torch import chain: torch reads and caches
#   PYTORCH_CUDA_ALLOC_CONF during its own import, and changing it afterward has no effect.
# - expandable_segments' CUDA backend implementation needs the
#   PYTORCH_C10_DRIVER_API_SUPPORTED macro, which PyTorch's c10/cuda/CMakeLists.txt
#   gates behind `if(NOT WIN32)`, so Windows wheels don't include that backend and
#   emit `TORCH_WARN_ONCE("expandable_segments not supported on this platform")`
#   at runtime before force-disabling it. To spare Windows users a useless warning, only set this on Linux.
# - setdefault doesn't override a value the user has already set explicitly.
if sys.platform.startswith("linux"):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# The script is launched as a bare script from runtime/ (`python runtime/anima_train.py`).
# Inject the repo root + runtime/ into sys.path so `import utils.*` / `import train_monitor` /
# `import training.*` etc. don't need to be converted to package imports.
_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (_REPO_ROOT, _REPO_ROOT / "runtime"):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

# The Windows console defaults to the ANSI codepage; logging / print writing non-ASCII text
# raises UnicodeEncodeError, and the default handler's errors='backslashreplace'
# turns it into \uXXXX escapes -- that's the source of the garbled text seen in task logs.
# Force stdout/stderr to UTF-8 + replace so non-ASCII text / emoji always print directly.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# --- Re-exports for sister script / tests (ADR 0003 PR-A) --------------------
# These names are read directly by anima_daemon / anima_generate / anima_reg_ai
# (`import anima_train as _T` then _T.X) and by tests/test_anima_train_migration.py
# etc. New code should import the submodule directly instead of relying on the anima_train top level.
from training.bootstrap import (  # noqa: E402
    apply_yaml_config,
    ensure_dependencies,
    init_progress,
    load_yaml_config,
)
from training.observability import (  # noqa: E402
    WandBMonitor,
    init_wandb_monitor,
    render_curve_panel,
    render_loss_curve,
)
from training.model_loading import (  # noqa: E402
    _load_safetensors_state_dict,
    _load_weights_best_effort,
    _pick_best_prefix_remap,
    _strip_prefixes,
    enable_xformers,
    find_diffusion_pipe_root,
    forward_with_optional_checkpoint,
    resolve_path_best_effort,
)
from training.families.anima.text_encoding import (  # noqa: E402
    _build_qwen_text_from_prompt,
    _parse_weighted_tag,
    encode_qwen,
    tokenize_t5_weighted,
)
from training.state import load_training_state, save_training_state  # noqa: E402
from training.model_loading import ensure_models_namespace  # noqa: E402
from training.vae import load_vae  # noqa: E402
from training.families import get_family, resolve_family  # noqa: E402  # dispatch choke point (D8')
from training.families.anima.loader import (  # noqa: E402
    load_anima_model,
    load_text_encoders,
)
from training.families.anima.sampling import sample_image  # noqa: E402
from training.dataset import (  # noqa: E402
    BucketBatchSampler,
    BucketManager,
    CachedLatentDataset,
    ImageDataset,
    MergedDataset,
    RepeatDataset,
    collate_fn,
    collate_fn_cached,
)
from training.cli import (  # noqa: E402
    parse_args,
    prompt_for_args,
)
from training.timestep_sampling import sample_t  # noqa: E402
from training.noise import make_noise  # noqa: E402
from training.loss_weighting import compute_loss_weight  # noqa: E402


# ============================================================================
# Main function
# ============================================================================

def main():
    """ADR 0003 PR-B: main() now only orchestrates phases.

    Each phase is a `run(ctx)` function that mutates TrainingContext in place, in order.
    The actual implementation lives in runtime/training/phases/.
    """
    from training import phases
    from training.context import TrainingContext
    from training import loop

    args = parse_args()
    ctx = TrainingContext(args=args)
    phases.bootstrap.run(ctx)
    phases.models.run(ctx)
    phases.dataset.run(ctx)
    phases.text_cache.run(ctx)
    phases.models.finish(ctx)
    phases.optimizer.run(ctx)
    phases.resume.run(ctx)
    loop.run(ctx)
    phases.finalize.run(ctx)


if __name__ == "__main__":
    main()
