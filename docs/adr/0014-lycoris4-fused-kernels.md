# 0014 — LyCORIS 4 fused kernels and Windows Triton

**Status**: Accepted

**Date**: 2026-09-20
**Decision makers**: project maintainers

## Background

The training stack integrates LyCORIS 3.4 via `utils/lycoris_adapter.py`. Plain LoRA already uses bypass forward, but the rebuild paths for LoHa / LoKr / DoRA and plain T-LoRA can still materialize the full `ΔW`. LyCORIS 4.0 adds Triton / TileLang fused kernels, selecting and falling back per call in the order Triton → TileLang → `torch.compile` → eager.

PyTorch on Windows doesn't ship an importable Triton, and its absence also causes xformers to print a non-fatal traceback at startup. This project's target local environment is PyTorch 2.11 + CUDA 12.8, which corresponds to Triton 3.6 per the triton-windows compatibility table.

Separately, LyCORIS 4.0 removes the internal `make_kron` / `rebuild_tucker` imports that the old patch relied on, but 4.0.0's LoKr `rank_dropout` still creates its mask on CPU.

## Candidate approaches

1. **Stay on LyCORIS 3.4**: lowest risk, but no access to fused kernels, and the startup warning remains.
2. **Upgrade directly and drop the local adapter/patch**: least code, but loses the family preset, checkpoint metadata, training hooks, and T-LoRA timestep mask, and re-exposes the LoKr device bug.
3. **Upgrade to 4.0, keep a thin integration adapter, and delegate the math to the official functional API**: requires small compatibility changes, but keeps the project's own contracts and gains official kernel dispatch.

## Decision

- Pin `lycoris-lora==4.0.0` to avoid silent drift from future major API changes.
- The Windows dependency declaration allows Triton 3.6–3.8; the launcher automatically narrows/fixes the version based on the installed PyTorch minor version before xformers/flash-attn are imported (2.10/2.11→3.6, 2.12/2.13→3.7, 2.14→3.8).
- Keep `LycorisAdapter`: it's responsible only for the AdapterProtocol, model-family target presets, saved metadata, sample/eval state, and the T-LoRA step hook; the LoRA math itself is delegated to LyCORIS.
- Plain T-LoRA applies the timestep mask on the two low-rank factors. Forward calls the official `functional.locon.bypass_forward_diff` and does not materialize `ΔW`; the merge/save path calls `functional.locon.diff_weight`. Both paths go through LyCORIS 4's fused/compile/eager dispatch.
- The LoKr device patch now wraps the upstream `get_weight`: it temporarily disables only the broken dropout branch, calls upstream (preserving fused dispatch), then replays the dropout on `weight.device`.
- OrthoLoRA remains a project-owned implementation. LyCORIS 4 has no official adapter equivalent to its Cayley/SVD parameterization, so the algorithm and checkpoint semantics can't be changed just to drop the wrapper.

## Rationale

Approach 3 leaves as much of the external algorithm implementation to upstream as possible, while preserving the boundaries needed for Anima/Krea2 integration. Explicitly pinning the version makes the applicable version range for the device patch auditable. When Triton is unavailable or a given shape is unsupported, LyCORIS falls back on its own, so a failed acceleration attempt never changes training correctness.

## Consequences

- First install on Windows downloads roughly 50 MB more, and pays a JIT/tuning cost the first time it encounters a new shape.
- Fused-kernel microbenchmark gains don't translate proportionally to whole-step training time; the DiT backbone still dominates.
- Calls outside fused scope — `rank_dropout`, convolutions, overly large ranks, etc. — fall back automatically.
- Triton has paired constraints with PyTorch minor versions; a new PyTorch release outside the support table falls back conservatively and requires updating `_WINDOWS_TRITON_FOR_TORCH` rather than guessing ABI compatibility.

## References

- [LyCORIS 4.0 fused kernels](https://github.com/KohakuBlueleaf/LyCORIS/blob/v4.0.0/docs/kernels/README.md)
- [LyCORIS backend selection](https://github.com/KohakuBlueleaf/LyCORIS/blob/v4.0.0/docs/kernels/backends.md)
- [triton-windows PyTorch compatibility](https://github.com/triton-lang/triton-windows#3-pytorch)
