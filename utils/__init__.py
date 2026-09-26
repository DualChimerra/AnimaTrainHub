"""Legacy ``utils`` directory.

Don't add new code here -- training-related code goes in ``runtime/training/``.
Where algorithm implementations (lycoris / future T-LoRA etc.) should ultimately live is
pending the full [[utils-full-refactor-plan-postponed]] refactor decision (see memory).

This module is **deliberately left empty**: early versions eagerly re-exported five
submodules here (dataset / model_utils / checkpoint / comfyui_loader / optimizer_utils),
which triggered a chained torchvision import and was a long-standing pain point for the
test infrastructure. The dead code has been removed; the remaining live submodules are
still reachable via ``from utils.X import ...`` submodule paths.
"""
