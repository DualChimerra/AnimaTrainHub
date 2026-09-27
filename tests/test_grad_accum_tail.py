"""Regression test for grad_accum tail-group handling (runtime/training/loop.py::_accumulation_step).

Fix: the last batch of an epoch now steps even if it doesn't fill a full
grad_accum group (matching kohya-ss / HF Trainer) -- otherwise the gradients
of the trailing `len % ga` batches were dropped (single epoch) or leaked into
the first step of the next epoch (multi-epoch); an incomplete tail group is
normalized by its actual micro-batch count.

Extracts the function via AST and runs it standalone (_accumulation_step is a
pure function with no dependencies), avoiding importing the whole training
stack (torch + model). Same lightweight approach as test_find_diffusion_pipe_root.
"""
from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "runtime" / "training" / "loop.py"


def _load_fn():
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_accumulation_step":
            mod = ast.Module(body=[node], type_ignores=[])
            ns: dict = {}
            exec(compile(mod, "<test>", "exec"), ns)
            return ns["_accumulation_step"]
    raise RuntimeError("_accumulation_step not found in loop.py")


_accumulation_step = _load_fn()


def _plan(dl_len, ga):
    return [_accumulation_step(i, dl_len, ga) for i in range(dl_len)]


def _step_idxs(plan):
    return [i for i, (_, end) in enumerate(plan) if end]


def test_divisible_unchanged():
    # 8 batches / ga4: two full groups, step at idx3/idx7, group_size always 4 (behavior unchanged)
    plan = _plan(8, 4)
    assert _step_idxs(plan) == [3, 7]
    assert all(gs == 4 for gs, _ in plan)


def test_tail_steps_and_normalizes_by_actual_size():
    # 7 batches / ga4: idx3 steps (full group of 4) + idx6 steps (tail group of 3) -- tail batches are no longer dropped
    plan = _plan(7, 4)
    assert _step_idxs(plan) == [3, 6]
    assert [gs for gs, _ in plan] == [4, 4, 4, 4, 3, 3, 3]


def test_fewer_batches_than_ga_still_trains():
    # 3 batches / ga4: old floor=0 steps (no training at all!); now 1 step, group_size=3
    plan = _plan(3, 4)
    assert _step_idxs(plan) == [2]
    assert all(gs == 3 for gs, _ in plan)


def test_ga_one_steps_every_batch():
    plan = _plan(5, 1)
    assert all(end for _, end in plan)
    assert all(gs == 1 for gs, _ in plan)


def test_no_len_falls_back_to_old_behavior():
    # dl_len=None: always grad_accum, steps only on exact multiples (last batch is unknown, tail handling not possible)
    assert _accumulation_step(3, None, 4) == (4, True)
    assert _accumulation_step(4, None, 4) == (4, False)
    assert _accumulation_step(6, None, 4) == (4, False)  # tail batch but no len -> old behavior, no step
