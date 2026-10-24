---
date: 2026-10-24
tag: notice
title: MOSS training controls and upstream inference updates
---

- Muon now offers a Use Riemannian switch. Old Riemann/Riemannion configs migrate automatically.
- MOSS supports qfloat8 weight storage, Caption Dropout and Low VRAM. DoRA remains available on unquantized compatible adapters.
- Quantization has its own Model group; text caching and Low VRAM are in System and output. The trigger-word section appears only with DOP enabled.
- Settings selects one model per category using the global root, with complete shard sets shown as a single model.
- MOSS generation adopts upstream INT8 and LoRA merge behavior closer to ComfyUI. Training supports mixed-size packs, packed multi-image forwards and updated noise defaults for new configs.
- Fixed ConvRot cache reuse after inference and validation of incompatible latent caches.

Frontend build and Python syntax checks passed. Tests and full-size GPU training
were not run for this update. See the MOSS guide and the upstream activity review
for implementation details and compatibility limits.
