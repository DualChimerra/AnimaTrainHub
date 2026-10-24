# 0016 — Simplify MOSS text configuration

**Status**: Accepted
**Date**: 2026-10-24
**Decision makers**: user request

## Decision

Remove `moss_max_token_length` and `moss_tokenizer_path` from TrainingConfig.
Use the official 4096-token limit and automatic tokenizer discovery beside the
text encoder, in a nested or sibling tokenizer directory, then the official
download. The generic Anima T5 tokenizer field stays available internally and is hidden in the form.

Old configs still load: TrainingConfig ignores the removed keys. Internal text
stack arguments remain available for callers that construct the stack directly.
Cache identity includes the resolved tokenizer and encoding limit, so changed
encoding settings invalidate old caption features.

## Related controls

Move text caching to System, group quantization separately in Model, expose
Caption Dropout, keep InfoNoise unavailable for MOSS, and allow DoRA on unquantized compatible adapters. Muon exposes Use Riemannian instead of separate adapter/optimizer choices. Low VRAM enables the memory profile described in the MOSS guide.
qfloat8 is opt-in scaled FP8 weight storage, implemented with existing PyTorch;
it does not add a dependency or change the default `none` storage setting.
