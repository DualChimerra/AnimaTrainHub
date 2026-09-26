"""runtime/training subpackage: modular split of the anima_train training code (ADR 0003).

PR-A: split the original runtime/anima_train.py's 53 defs/classes into this subpackage by responsibility.
PR-B: introduce TrainingContext + split main() into phases.
PR-C: introduce 4 plugin subpackages (adapters / optimizers / schedulers / inference_samplers).

See docs/adr/0003-anima-train-refactor.md for the detailed design.
"""
