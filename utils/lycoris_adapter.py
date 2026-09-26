"""Anima-friendly wrapper around LycorisNetwork.

Replaces LoRAInjector / LoRALayer / LoKrLayer / LoRALinear from anima_train.py.

The API is equivalent to (drop-in for) the original LoRAInjector, and keeps the
optimization that excludes w1 from weight_decay.

The T-LoRA timestep rank mask schedule (the ``(1-t)^alpha`` formula and batch-mean
aggregation in _install_tlora_masks / _set_tlora_mask) is taken from
sorryhyun/anima_lora (MIT, Copyright (c) 2026 Seunghyun Ji, see
THIRD_PARTY_NOTICES.md); the mask injection mechanism (patching lycoris'
make_weight) is this repo's own implementation. The algorithmic idea comes from
the T-LoRA paper (ControlGenAI/T-LoRA).
"""
from __future__ import annotations

import json
import logging
import math
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from safetensors.torch import save_file
from safetensors import safe_open

from utils.lycoris_patch import apply_lokr_device_patch

logger = logging.getLogger(__name__)

_FP8_DTYPES = frozenset({torch.float8_e4m3fn, torch.float8_e5m2})
_ADAPTER_DTYPES = frozenset({torch.float16, torch.bfloat16, torch.float32, torch.float64})

# One-time fix for the lycoris-lora 3.4.0 LokrModule.get_weight rank_dropout device bug.
# Called at module level: any path that reaches lycoris_adapter (CLI training / Studio
# worker / tests) triggers the patch once. The return value is available for test
# assertions; on a normal import path the result just goes to the logger.
_LOKR_PATCH_STATUS = apply_lokr_device_patch()

# When dropout>0, lycoris' LokrModule/LohaModule print one line per module instance:
#   "[WARN]LoHa/LoKr haven't implemented normal dropout yet."
# With 280 layers that's 280 lines. Behaviorally this is equivalent to silently ignoring
# normal dropout (rank/module dropout are unaffected). During injection we temporarily
# filter stdout line-by-line, swallow and count these lines, then roll them up into a
# single logger record at the end.
_LOKR_DROPOUT_MARKER = "haven't implemented normal dropout yet"


class _LineFilteredStdout:
    """Wraps stdout line-by-line, dropping and counting any line containing the marker; everything else passes through unchanged."""

    def __init__(self, wrapped: Any, marker: str) -> None:
        self._wrapped = wrapped
        self._marker = marker
        self._buf = ""
        self.dropped = 0

    def write(self, s: str) -> int:
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if self._marker in line:
                self.dropped += 1
            else:
                self._wrapped.write(line + "\n")
        return len(s)

    def flush(self) -> None:
        if self._buf:
            if self._marker in self._buf:
                self.dropped += 1
            else:
                self._wrapped.write(self._buf)
            self._buf = ""
        self._wrapped.flush()

    def __getattr__(self, name: str) -> Any:  # pass through encoding/fileno/isatty/etc.
        return getattr(self._wrapped, name)


@contextmanager
def _suppress_lokr_dropout_spam():
    """Temporarily collapse lycoris' normal-dropout print spam during injection."""
    original = sys.stdout
    flt = _LineFilteredStdout(original, _LOKR_DROPOUT_MARKER)
    sys.stdout = flt
    try:
        yield flt
    finally:
        flt.flush()
        sys.stdout = original


class LycorisAdapter:
    """Equivalent wrapper around LycorisNetwork; the external interface matches the original LoRAInjector.

    Compared to the original LoRAInjector:
    - inject()/get_params()/get_param_groups()/state_dict()/save()/load() are equivalent
    - additionally supports algo: lora/lokr/loha + native LyCORIS parameters like DoRA/dropout/rs_lora
    - keeps the optimization that excludes w1 from weight_decay
    - the saved key prefix is lora_unet_*, fully compatible with existing ComfyUI workflows
    """

    def __init__(
        self,
        *,
        preset: dict | None = None,
        algo: str = "lokr",
        rank: int = 32,
        alpha: float = 16.0,
        factor: int = 8,
        dropout: float = 0.0,
        rank_dropout: float = 0.0,
        module_dropout: float = 0.0,
        weight_decompose: bool = False,
        rs_lora: bool = False,
        lora_reg_dims: Optional[dict[str, int]] = None,
        tlora_min_rank: int = 8,
        tlora_alpha_rank_scale: float = 1.0,
    ):
        # Family knowledge (target/exclude/prefix) is injected by the caller; see families/<fam>/preset.py
        self._preset = preset
        self.algo = algo
        self.rank = rank
        self.alpha = alpha
        self.factor = factor
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        self.weight_decompose = weight_decompose
        self.rs_lora = rs_lora
        # Layered rank: a dict of regex -> rank, matched against lora_name with re.fullmatch
        self.lora_reg_dims: Optional[dict[str, int]] = lora_reg_dims or None
        self.use_timestep_mask = (algo == "tlora")
        self.tlora_min_rank = max(1, min(int(tlora_min_rank), int(rank)))
        self.tlora_alpha_rank_scale = max(0.0, float(tlora_alpha_rank_scale))
        self._tlora_modules: list[nn.Module] = []
        self._tlora_mask: Optional[torch.Tensor] = None
        self._tlora_arange: Optional[torch.Tensor] = None

        # use_lokr is a field from the original LoRAInjector; anima_train.py branches on it
        # in several places. Keeping this field avoids touching too many call sites.
        self.use_lokr = (algo == "lokr")
        self.network = None  # lazy init in inject()
        # commit 20: used to undo the hook in detach() -- records the model reference and
        # the original model.train function at inject time, restored on detach.
        self._injected_model: Optional[nn.Module] = None
        self._orig_train: Optional[Any] = None

    # ------------------------------------------------------------- disabled
    @contextmanager
    def disabled(self):
        """Temporarily zero out the whole network's scale -- equivalent to "no LoRA on this model".

        What DOP's reference forward pass needs is exactly what the base model would output
        on its own (see training/dop.py). Uses multiplier rather than detach(): detach()
        unhooks / mutates the module tree, and attaching/detaching every step is both slow
        and prone to mismatches during checkpoint recompute; multiplier is just a scalar.

        The finally block restores unconditionally -- training must keep using the real
        scale, and exception paths are no exception.
        """
        if self.network is None:
            yield
            return
        previous = getattr(self.network, "multiplier", 1.0)
        try:
            self.network.set_multiplier(0.0)
            yield
        finally:
            self.network.set_multiplier(previous)

    # --------------------------------------------------------------- inject
    def inject(self, model: nn.Module) -> dict[str, nn.Module]:
        """Inject the lycoris adapter into the model."""
        from lycoris import LycorisNetwork

        if self._preset is None:
            raise ValueError(
                "LycorisAdapter requires an explicit preset (family knowledge is no longer "
                "built in; see runtime/training/families/<fam>/preset.py)"
            )
        LycorisNetwork.apply_preset(self._preset)

        # algo name mapping: anima_train uses 'lora'/'tlora', lycoris uses 'locon' (with conv disabled it's equivalent to lora)
        net_module = self.algo
        if net_module in {"lora", "tlora"}:
            net_module = "locon"

        # extra kwargs: only pass the corresponding field when the algorithm supports it
        extra: dict[str, Any] = {}
        if self.algo == "lokr":
            extra["factor"] = self.factor
        if self.weight_decompose:
            extra["weight_decompose"] = True
        if self.rs_lora:
            extra["rs_lora"] = True

        # algo='lora' (LoCon) defaults to bypass_mode: lycoris' LoConModule default forward
        # rebuilds ΔW=up@down (out,in) and then runs F.linear again, i.e. roughly 2x FLOPs
        # per layer. bypass_mode=True goes through
        # bypass_forward_diff = org_forward(x) + lora_up(lora_down(x)), which is the standard
        # forward pass from the LoRA paper + sd-scripts + PEFT; externally equivalent
        # behavior but ~2x faster.
        # The DoRA (weight_decompose) path mathematically requires a rebuild -- lycoris'
        # bypass forward doesn't go through the wd branch, which would silently disable
        # DoRA; guarded here. See lycoris docs/Network-Args.md "Bypass Mode".
        if self.algo in {"lora", "tlora"} and not self.weight_decompose:
            extra["bypass_mode"] = True

        # Krea2 FP8 checkpoints keep frozen Linear weights in float8 and
        # monkeypatch Linear.forward to dequantize them to the input dtype.
        # LyCORIS only recognizes dedicated quantized Linear subclasses, so a
        # monkeypatched nn.Linear is otherwise mistaken for a regular layer and
        # LoKr takes the rebuild path. That path casts the dense LoKr delta to
        # the raw base-weight dtype; torch has no float8 mul/add kernels for it.
        # Bypass preserves the patched base forward and computes LoKr separately.
        has_fp8_base = any(
            isinstance(module, nn.Linear) and module.weight.dtype in _FP8_DTYPES
            for module in model.modules()
        )
        if self.algo == "lokr" and has_fp8_base:
            extra["bypass_mode"] = True
            logger.info("FP8 base detected: forcing LoKr bypass forward")

        with _suppress_lokr_dropout_spam() as _dropout_filter:
            self.network = LycorisNetwork(
                model,
                multiplier=1.0,
                lora_dim=self.rank,
                alpha=self.alpha,
                dropout=self.dropout,
                rank_dropout=self.rank_dropout,
                module_dropout=self.module_dropout,
                network_module=net_module,
                **extra,
            )
            self.network.apply_to()

        if _dropout_filter.dropped:
            logger.info(
                "LoKr/LoHa doesn't support normal dropout (lora_dropout=%s); silently ignored; "
                "%d upstream per-layer warnings collapsed. rank_dropout/module_dropout are unaffected.",
                self.dropout,
                _dropout_filter.dropped,
            )

        if self.lora_reg_dims:
            _apply_reg_dims_(self.network, self.lora_reg_dims)
        if self.use_timestep_mask:
            self._install_tlora_masks()

        # lycoris creates modules on CPU by default; the model is likely already on CUDA --
        # device/dtype must be synced explicitly, otherwise the first forward pass raises
        # "tensors on cuda:0 and cpu". Inferred from the model's first parameter. The
        # inference-parity path converts the network to fp32 again before load_state_dict,
        # to match ComfyUI's LoRA patch intermediate precision.
        try:
            ref = next(model.parameters())
        except StopIteration:
            ref = None
        if ref is not None:
            # The first parameter may itself be FP8. Adapter parameters must
            # stay in a trainable compute dtype; prefer the model's first
            # non-FP8 floating dtype and fall back to fp32 for an all-FP8 model.
            adapter_dtype = next(
                (p.dtype for p in model.parameters() if p.dtype in _ADAPTER_DTYPES),
                torch.float32,
            )
            self.network.to(device=ref.device, dtype=adapter_dtype)

        # LycorisNetwork is a standalone nn.Module, not part of the model's submodule tree;
        # model.eval()/.train() doesn't cascade to the lycoris module (self.training stays
        # True forever). This means sampling still goes through the rank_dropout branch,
        # triggering an upstream lycoris bug: torch.rand(...) doesn't pass device, so the
        # CPU mask multiplied with a CUDA weight raises a device mismatch (lokr.py:380).
        # Fix: hijack model.train() so the network follows it, and immediately sync the
        # current mode.
        # commit 20: saves _orig_train + _injected_model so detach() can restore them.
        _orig_train = model.train
        _network = self.network

        def _train_with_lycoris(mode: bool = True):
            _network.train(mode)
            return _orig_train(mode)

        model.train = _train_with_lycoris  # type: ignore[method-assign]
        self.network.train(model.training)
        self._orig_train = _orig_train
        self._injected_model = model

        n = len(self.network.loras)
        forward_path = "bypass (low-rank)" if extra.get("bypass_mode") else "rebuild (ΔW)"
        logger.info(f"Injected {self.algo.upper()} into {n} layers (lycoris-lora, forward={forward_path})")
        if self.use_lokr:
            full_matrix = [lora for lora in self.network.loras if getattr(lora, "use_w2", False)]
            if full_matrix:
                logger.info(
                    "LoKr dim/rank=%s triggered LyCORIS full dimension: the second block of "
                    "%s/%s layers is no longer decomposed; alpha will be ignored.",
                    self.rank,
                    len(full_matrix),
                    n,
                )
        return {lora.lora_name: lora for lora in self.network.loras}

    def _install_tlora_masks(self) -> None:
        if self.network is None:
            return
        self._tlora_modules = [
            lora for lora in self.network.loras
            if hasattr(lora, "lora_down") and hasattr(lora, "lora_up") and callable(getattr(lora, "make_weight", None))
        ]
        for lora in self._tlora_modules:
            if getattr(lora, "_anima_tlora_patched", False):
                continue
            original_make_weight = lora.make_weight

            def _make_weight_with_tlora(
                device=None,
                *,
                _lora=lora,
                _original_make_weight=original_make_weight,
            ):
                wa = _lora.lora_up.weight.to(device)
                wb = _lora.lora_down.weight.to(device)
                mask = getattr(_lora, "_anima_tlora_mask", None)
                if mask is not None:
                    mask = mask.to(device=device, dtype=wa.dtype)
                    view_up = (1, -1) + (1,) * max(0, wa.dim() - 2)
                    view_down = (-1, 1) + (1,) * max(0, wb.dim() - 2)
                    wa = wa * mask.view(*view_up)
                    wb = wb * mask.view(*view_down)
                # LyCORIS 4 functional API owns eager/compile/Triton/TileLang
                # selection.  Keeping the timestep mask on the factors and
                # delegating the rebuild preserves T-LoRA semantics while
                # allowing the new fused merge forward/backward kernels.
                from lycoris.functional.locon import diff_weight

                t = (
                    _lora.lora_mid.weight.to(device)
                    if getattr(_lora, "tucker", False)
                    else None
                )
                weight = diff_weight(wb, wa, t, gamma=1.0).view(_lora.shape)
                if _lora.training and _lora.rank_dropout:
                    drop = (
                        torch.rand(weight.size(0), device=device) > _lora.rank_dropout
                    ).to(weight.dtype)
                    drop = drop.view(-1, *[1] * len(weight.shape[1:]))
                    if _lora.rank_dropout_scale:
                        drop /= drop.mean()
                    weight *= drop
                return weight * _lora.scalar.to(device)

            lora._anima_original_make_weight = original_make_weight
            lora.make_weight = _make_weight_with_tlora

            if getattr(lora, "bypass_mode", False):
                original_bypass_forward_diff = lora.bypass_forward_diff

                def _bypass_forward_diff_with_tlora(
                    x,
                    scale=1,
                    *,
                    _lora=lora,
                ):
                    wa = _lora.lora_up.weight.to(x)
                    wb = _lora.lora_down.weight.to(x)
                    mask = getattr(_lora, "_anima_tlora_mask", None)
                    if mask is not None:
                        mask = mask.to(device=x.device, dtype=wa.dtype)
                        view_up = (1, -1) + (1,) * max(0, wa.dim() - 2)
                        view_down = (-1, 1) + (1,) * max(0, wb.dim() - 2)
                        wa = wa * mask.view(*view_up)
                        wb = wb * mask.view(*view_down)

                    if _lora.training and _lora.rank_dropout:
                        drop = (
                            torch.rand(_lora.lora_dim, device=x.device)
                            > _lora.rank_dropout
                        ).to(wb.dtype)
                        if _lora.rank_dropout_scale:
                            drop /= drop.mean()
                        wb = wb * drop.view(
                            -1,
                            *[1] * (wb.dim() - 1),
                        )

                    t = (
                        _lora.lora_mid.weight.to(x)
                        if getattr(_lora, "tucker", False)
                        else None
                    )
                    op = (
                        _lora.lora_mid
                        if getattr(_lora, "tucker", False)
                        else _lora.lora_down
                    )
                    extra_args = (
                        {
                            "stride": op.stride,
                            "padding": op.padding,
                            "dilation": op.dilation,
                            "groups": op.groups,
                        }
                        if _lora.isconv
                        else {}
                    )
                    from lycoris.functional.locon import bypass_forward_diff

                    diff = bypass_forward_diff(
                        x,
                        None,
                        wb,
                        wa,
                        t,
                        gamma=_lora.scale * scale,
                        extra_args=extra_args,
                    )
                    scalar = _lora.scalar.to(device=diff.device, dtype=diff.dtype)
                    return _lora.dropout(diff * scalar)

                lora._anima_original_bypass_forward_diff = original_bypass_forward_diff
                lora.bypass_forward_diff = _bypass_forward_diff_with_tlora
            lora._anima_tlora_patched = True
        logger.info(
            "T-LoRA timestep rank mask enabled on %s/%s LoRA layers (min_rank=%s, alpha_rank_scale=%s)",
            len(self._tlora_modules),
            len(self.network.loras),
            self.tlora_min_rank,
            self.tlora_alpha_rank_scale,
        )

    def _set_tlora_mask(self, sigma_t: torch.Tensor) -> None:
        if not self._tlora_modules:
            return
        device = sigma_t.device
        rank = self.rank
        if self._tlora_mask is None or self._tlora_mask.device != device:
            self._tlora_mask = torch.ones(rank, device=device)
            self._tlora_arange = torch.arange(rank, device=device)
            for lora in self._tlora_modules:
                lora._anima_tlora_mask = self._tlora_mask
        # Matches ControlGenAI/T-LoRA's official (arxiv 2507.05964) SDXL get_mask_by_timestep:
        # r = ((max_t - t)/max_t)^alpha * (rank - min_rank) + min_rank
        # In this PR sigma_t is in [0,1] with t=1=noisy (locked in training_loop.py:120),
        # so (max_t - t)/max_t == (1 - t). alpha=1.0 degenerates to the FLUX path's linear schedule.
        # Paper motivation: "higher diffusion timesteps are more prone to overfitting" --
        # high-noise timesteps get a restricted rank, low-noise timesteps get the full rank.
        t = sigma_t.float().mean().clamp(min=0.0, max=1.0)
        frac = (1.0 - t).pow(self.tlora_alpha_rank_scale)
        active_rank = frac * (rank - self.tlora_min_rank) + self.tlora_min_rank
        active_rank = active_rank.clamp(min=float(self.tlora_min_rank), max=float(rank))
        self._tlora_mask.copy_((self._tlora_arange < active_rank).to(self._tlora_mask.dtype))

    def clear_timestep_mask(self) -> None:
        if self._tlora_mask is not None:
            self._tlora_mask.fill_(1)

    # --------------------------------------------------------------- detach
    def detach(self) -> bool:
        """Undo inject: restore the model.train hook + call LycorisNetwork.restore (if available).

        Lets the daemon switch LoRA without reloading the entire transformer. Return value:
          - True: success; the old hook has been undone, safe to inject a new LoRA
          - False: the current lycoris version doesn't expose a restore interface, the hook
                  remains; the caller should fall back to a full model reload (crude but safe)

        Idempotent across multiple calls (once self.network is None it's a straight no-op).
        """
        if self.network is None:
            return True

        # First try lycoris' built-in restore; the interface name differs across versions, try each
        ok = True
        for restore_attr in ("restore", "restore_apply", "remove_apply"):
            fn = getattr(self.network, restore_attr, None)
            if callable(fn):
                try:
                    fn()
                    break
                except Exception as e:
                    logger.warning(f"LycorisNetwork.{restore_attr}() failed: {e}")
                    ok = False
                    break
        else:
            # None of the three interfaces exist -> the current lycoris version doesn't support hot-unload
            logger.warning("LycorisNetwork has no restore/restore_apply/remove_apply interface; hook remains")
            ok = False

        # Restore the model.train hijack (should be restored regardless of whether restore() succeeded)
        if self._injected_model is not None and self._orig_train is not None:
            try:
                self._injected_model.train = self._orig_train  # type: ignore[method-assign]
            except Exception as e:
                logger.warning(f"Failed to restore model.train: {e}")
                ok = False

        # Drop the reference so GC can clean up LycorisNetwork (including the closure's _network reference)
        self.network = None
        self._injected_model = None
        self._orig_train = None
        return ok

    # --------------------------------------------------------------- params
    def get_params(self) -> list[nn.Parameter]:
        """All trainable parameters (equivalent to the original LoRAInjector.get_params)"""
        if self.network is None:
            return []
        return [p for p in self.network.parameters() if p.requires_grad]

    def get_param_groups(self, weight_decay: float) -> list[dict]:
        """In LoKr mode, excludes w1 from weight_decay (equivalent to the original LoRAInjector).

        For other algorithms there's no grouping; all parameters share weight_decay.
        """
        if self.network is None:
            return [{"params": [], "weight_decay": weight_decay}]

        if not self.use_lokr or weight_decay == 0:
            return [{"params": self.get_params(), "weight_decay": weight_decay}]

        no_decay = []  # lokr_w1 (full-matrix branch) / lokr_w1_a/b (if decompose_both is enabled)
        decay = []
        for lora in self.network.loras:
            for n, p in lora.named_parameters():
                if not p.requires_grad:
                    continue
                # 'lokr_w1' / 'lokr_w1_a' / 'lokr_w1_b' are all treated as part of the w1 family
                if "lokr_w1" in n:
                    no_decay.append(p)
                else:
                    decay.append(p)
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    # --------------------------------------------------------------- state I/O
    def state_dict(self) -> dict[str, torch.Tensor]:
        """LoRA weight state_dict (with the lora_unet_* prefix, ComfyUI-compatible).

        lycoris already outputs the correct prefix via LORA_PREFIX ('lora_prefix=lora_unet' in the preset).
        """
        if self.network is None:
            return {}
        return self.network.state_dict()

    def load_state_dict(self, sd: dict[str, torch.Tensor], strict: bool = True) -> Any:
        if self.network is None:
            raise RuntimeError("AnimaLycorisAdapter.inject() must be called first")
        return self.network.load_state_dict(sd, strict=strict)

    # --------------------------------------------------------------- safetensors
    def save(self, path: str | Path) -> None:
        """Save as safetensors (with ss_* metadata, ComfyUI/sd-scripts compatible)"""
        sd = self.state_dict()
        # Recompute each layer's .alpha from the current (scale, lora_dim) so that the
        # downstream standard formula alpha/rank recovers the training scale. The .alpha
        # written by lycoris init is already scale*lora_dim; but _apply_reg_dims_ changes
        # lora_dim without touching the self.scale / self.alpha buffer, so a layered-rank
        # layer's .alpha no longer matches its per-layer rank -> the scale ComfyUI computes
        # via alpha/rank ends up tens of times off from the actual training value, producing
        # noisy output images.
        # Rewritten here from each lora's current real (scale, lora_dim): a no-op for
        # non-layered layers, a correction for layered ones.
        _rewrite_per_layer_alpha_(self.network, sd)
        # lora_reg_dims must be written: training changed some layers' rank by pattern, and
        # rebuilding the network for inference needs the same dict so lokr_w2_a/b shapes
        # line up with the checkpoint.
        ss_args: dict[str, Any] = {
            "algo": self.algo,
            "factor": self.factor,
            "dropout": self.dropout,
            "rank_dropout": self.rank_dropout,
            "module_dropout": self.module_dropout,
            "weight_decompose": self.weight_decompose,
            "rs_lora": self.rs_lora,
        }
        if self.lora_reg_dims:
            ss_args["lora_reg_dims"] = self.lora_reg_dims
        # Multi-model D13: family marker (injected by phases/models via
        # family.lora_metadata()); existing artifacts without a marker are grandfathered
        # in as anima on the reading side.
        ss_args.update(getattr(self, "metadata_extra", None) or {})
        meta = {
            "ss_network_dim": str(self.rank),
            "ss_network_alpha": str(self.alpha),
            "ss_network_module": "lycoris.kohya",
            "ss_network_args": json.dumps(ss_args),
        }
        save_file(sd, str(path), metadata=meta)
        logger.info(f"LoRA saved to: {path}")

    def load(self, path: str | Path) -> None:
        """Load existing LoRA weights from safetensors (used to resume training)"""
        logger.info(f"Loading existing LoRA weights: {path}")
        sd: dict[str, torch.Tensor] = {}
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for k in f.keys():
                sd[k] = f.get_tensor(k)

        # Fallback for the old self-implemented format (lora_unet_*.lokr_w2_a/b low-rank vs.
        # lokr_w2 full-matrix): lycoris expects the format it writes itself (same
        # lora_unet_* prefix, but internally it may produce lokr_w2 instead of
        # lokr_w2_a/b if dim is too large and it takes the full_matrix path).
        # Uses strict=False directly to let lycoris tolerate missing keys, and prints the
        # missing count.
        result = self.load_state_dict(sd, strict=False)
        missing = len(getattr(result, "missing_keys", [])) if hasattr(result, "missing_keys") else 0
        unexpected = len(getattr(result, "unexpected_keys", [])) if hasattr(result, "unexpected_keys") else 0
        logger.info(
            f"Loaded {len(sd)} weight tensors, "
            f"missing={missing}, unexpected={unexpected}"
        )

    # --- ADR 0003 PR-C: no-op implementations of AdapterProtocol's optional hooks ---
    # Lets the LyCORIS adapter satisfy the runtime_checkable Protocol; paper-level variants
    # (T-LoRA / OFT / Ortho-Hydra) can override these in their own build wrapper.

    def on_step_begin(self, ctx) -> None:
        """Called before each micro-batch forward pass; T-LoRA updates the rank mask from sigma_t."""
        if self.use_timestep_mask:
            self._set_tlora_mask(ctx.sigma_t)
        return None

    def regularization_loss(self, ctx):
        """LoKr/LoRA/LoHa have no extra regularization term."""
        return None

    def excludes_weight_decay(self, param_name: str) -> bool:
        """w1-family parameters are excluded from weight_decay; nothing else is.

        Note: get_param_groups already internally groups by the "lokr_w1" substring and
        sets weight_decay to 0 for those params; this method just exposes the Protocol
        interface for external callers (e.g. phase log decisions) to query.
        """
        return self.use_lokr and "lokr_w1" in param_name


def _rewrite_per_layer_alpha_(network: Optional[nn.Module], sd: dict[str, torch.Tensor]) -> None:
    """Recompute each layer's .alpha tensor in sd as lora.scale x lora.lora_dim.

    Preserves the downstream standard LoRA formula alpha/rank recovering the training
    scale. A no-op for layers that didn't trigger lora_reg_dims (lycoris init already
    writes .alpha = scale x lora_dim); a correction for layered-rank layers.
    """
    if network is None:
        return
    loras = getattr(network, "loras", None) or []
    for lora in loras:
        name = getattr(lora, "lora_name", None)
        if not name:
            continue
        alpha_key = f"{name}.alpha"
        if alpha_key not in sd:
            continue
        try:
            scale = float(getattr(lora, "scale"))
            dim = int(getattr(lora, "lora_dim"))
        except (AttributeError, TypeError, ValueError):
            continue
        old = sd[alpha_key]
        sd[alpha_key] = torch.tensor(scale * dim, dtype=old.dtype, device=old.device)


def _apply_reg_dims_(network: nn.Module, lora_reg_dims: dict[str, int]) -> None:
    """Layered rank: reallocate rank for modules whose lora_name fully matches the regex.

    Supports LoRA/LoCoN (either the lora_A / lora_B parameter style, or the lora_up /
    lora_down submodule style -- lycoris names things differently across code paths) and
    LoKr's non-full-matrix branch (lokr_w2_a / lokr_w2_b). LoKr's full-matrix branch
    (lokr_w2, use_w2=True) doesn't support layered rank overrides; skipped with a warning.

    The re-init strategy matches lycoris' original initialization:
      - A/w2_a: kaiming_uniform_(a=sqrt(5)) (same as nn.Linear weight init)
      - B/w2_b: zeros (ensures the initial ΔW=0)
    """
    patterns = list(lora_reg_dims.items())
    changed = 0
    skipped = 0

    loras = getattr(network, "loras", None) or []
    for lora_mod in loras:
        name: str = getattr(lora_mod, "lora_name", "") or ""
        new_dim: Optional[int] = None
        for pat, dim in patterns:
            if re.fullmatch(pat, name):
                new_dim = int(dim)
                break
        if new_dim is None:
            continue

        old_dim = getattr(lora_mod, "lora_dim", None)
        if old_dim == new_dim:
            continue

        # -- LoRA / LoCoN --------------------------------------------------------
        if hasattr(lora_mod, "lora_A") and hasattr(lora_mod, "lora_B"):
            in_f = lora_mod.lora_A.shape[1]
            out_f = lora_mod.lora_B.shape[0]
            dev, dt = lora_mod.lora_A.device, lora_mod.lora_A.dtype
            new_A = torch.empty(new_dim, in_f, device=dev, dtype=dt)
            nn.init.kaiming_uniform_(new_A, a=math.sqrt(5))
            lora_mod.lora_A = nn.Parameter(new_A)
            lora_mod.lora_B = nn.Parameter(torch.zeros(out_f, new_dim, device=dev, dtype=dt))
            lora_mod.lora_dim = new_dim
            changed += 1

        # -- LoRA/LoCoN submodule style (this lycoris version's LoConModule: lora_up/lora_down
        #    are nn.Linear). Only Linear is handled; conv (lora_mid/Conv2d) doesn't support
        #    layered overrides --
        elif (
            isinstance(getattr(lora_mod, "lora_down", None), nn.Linear)
            and isinstance(getattr(lora_mod, "lora_up", None), nn.Linear)
        ):
            in_f = lora_mod.lora_down.in_features
            out_f = lora_mod.lora_up.out_features
            p = lora_mod.lora_down.weight
            dev, dt = p.device, p.dtype
            new_down = nn.Linear(in_f, new_dim, bias=False, device=dev, dtype=dt)
            nn.init.kaiming_uniform_(new_down.weight, a=math.sqrt(5))
            new_up = nn.Linear(new_dim, out_f, bias=False, device=dev, dtype=dt)
            nn.init.zeros_(new_up.weight)
            lora_mod.lora_down = new_down
            lora_mod.lora_up = new_up
            lora_mod.lora_dim = new_dim
            changed += 1

        # -- LoKr low-rank branch (use_w2=False): w2_a [d_out//f, dim], w2_b [dim, d_in//f] --
        elif hasattr(lora_mod, "lokr_w2_a") and hasattr(lora_mod, "lokr_w2_b"):
            d0 = lora_mod.lokr_w2_a.shape[0]   # d_out // factor
            d1 = lora_mod.lokr_w2_b.shape[1]   # d_in  // factor
            dev, dt = lora_mod.lokr_w2_a.device, lora_mod.lokr_w2_a.dtype
            new_w2a = torch.empty(d0, new_dim, device=dev, dtype=dt)
            nn.init.kaiming_uniform_(new_w2a, a=math.sqrt(5))
            lora_mod.lokr_w2_a = nn.Parameter(new_w2a)
            lora_mod.lokr_w2_b = nn.Parameter(torch.zeros(new_dim, d1, device=dev, dtype=dt))
            lora_mod.lora_dim = new_dim
            changed += 1

        # -- LoKr full-matrix branch (use_w2=True): the rank concept doesn't apply, skip ---
        elif hasattr(lora_mod, "lokr_w2"):
            logger.warning(
                "[lora_reg_dims] %s: full-matrix LoKr branch (use_w2=True) — "
                "rank override not applicable, skipped",
                name,
            )
            skipped += 1

        else:
            skipped += 1

    if changed:
        logger.info("[lora_reg_dims] applied rank overrides to %d modules", changed)
    if skipped:
        logger.info("[lora_reg_dims] skipped %d modules (full-matrix or unsupported)", skipped)


# Compatibility alias (renamed in the multi-model PR-2b; remove once call sites have migrated)
AnimaLycorisAdapter = LycorisAdapter
