"""Phase split of training main() (ADR 0003 PR-B).

Each phase is a `run(ctx: TrainingContext) -> None` function that mutates ctx
in place. main() orchestrates: bootstrap -> models.run -> dataset -> text_cache
-> models.finish -> optimizer -> resume -> train_loop -> finalize.
``models.finish`` only fills in the model stack for families that lazy-load
the DiT after caching; for the rest it's a no-op.
"""

from training.phases import bootstrap, dataset, finalize, models, optimizer, resume, text_cache

__all__ = [
    "bootstrap", "models", "dataset", "text_cache", "optimizer", "resume", "finalize",
]
