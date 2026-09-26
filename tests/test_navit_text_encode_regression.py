"""NaViT text-encoding regression (0.20.0 t5_attn NameError).

After PR #407 pushed text encoding down into ``AnimaFamily.encode_text_for_batch`` (returning only
an opaque cross), loop.py's NaViT branch still referenced the old local variable ``t5_attn`` -> every
user training with ``navit_packing=true`` was guaranteed to crash with a NameError on the first step. CI has no GPU and
never reaches that branch, hence two lines of defense:

1. A family-private contract unit test for ``encode_text_for_batch(return_t5_attn=True)``;
2. A ``symtable`` static scan of every function scope under runtime/training/ -- a name that is referenced but
   unbound anywhere in the scope chain (i.e. a pyflakes-F821-style error) fails the test outright. A
   branch-gated code path can be caught without ever being executed.
"""
from __future__ import annotations

import builtins
import symtable
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from training.families.anima.family import AnimaFamily


# -- 1. return_t5_attn contract -------------------------------------------


def _stub_text_encoding(monkeypatch, attn: "torch.Tensor") -> None:
    import training.families.anima.text_encoding as te

    B, L = attn.shape
    monkeypatch.setattr(
        te, "encode_qwen", lambda model, tok, texts, device: (torch.zeros(B, 8, 4), None)
    )
    monkeypatch.setattr(
        te,
        "tokenize_t5_comfy_literal",
        lambda tok, captions, max_length: (
            torch.zeros(B, L, dtype=torch.long),
            attn,
            torch.ones(B, L),
        ),
    )


def _stub_dit(max_len: int):
    return SimpleNamespace(
        preprocess_text_embeds=lambda qwen_emb, t5_ids, t5xxl_weights: torch.zeros(
            t5_ids.shape[0], max_len, 4
        )
    )


def test_encode_text_for_batch_returns_t5_attn_for_navit(monkeypatch) -> None:
    family = AnimaFamily()
    max_len = family.spec.text.max_seq_len
    attn = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.long)
    _stub_text_encoding(monkeypatch, attn)

    result = family.encode_text_for_batch(
        (None, None, None), _stub_dit(max_len), ["a", "b"], "cpu", torch.float32,
        return_t5_attn=True,
    )
    assert isinstance(result, tuple) and len(result) == 2
    cross, t5_attn = result
    assert cross.shape[1] == max_len
    torch.testing.assert_close(t5_attn, attn)


def test_encode_text_for_batch_default_stays_bare_cross(monkeypatch) -> None:
    family = AnimaFamily()
    max_len = family.spec.text.max_seq_len
    _stub_text_encoding(monkeypatch, torch.ones(2, 4, dtype=torch.long))

    cross = family.encode_text_for_batch(
        (None, None, None), _stub_dit(max_len), ["a", "b"], "cpu", torch.float32,
    )
    assert isinstance(cross, torch.Tensor)


# -- 2. symtable static scan for undefined names ---------------------------

_TRAINING_ROOT = Path(__file__).resolve().parents[1] / "runtime" / "training"

_MODULE_DUNDERS = {
    "__name__", "__file__", "__doc__", "__package__", "__spec__",
    "__loader__", "__builtins__", "__debug__", "__annotations__",
    "__dict__", "__class__", "__module__", "__qualname__",
}


def _module_bindings(table: symtable.SymbolTable) -> set[str]:
    names = {
        sym.get_name()
        for sym in table.get_symbols()
        if sym.is_assigned() or sym.is_imported()
    }
    names |= {child.get_name() for child in table.get_children()}
    return names


def _walk_undefined(table: symtable.SymbolTable, module_names: set[str]) -> list[str]:
    bad = []
    if table.get_type() != "module":
        for sym in table.get_symbols():
            if not (sym.is_referenced() and sym.is_global()):
                continue
            name = sym.get_name()
            if name in module_names or name in _MODULE_DUNDERS:
                continue
            if hasattr(builtins, name):
                continue
            bad.append(f"{table.get_name()}:{table.get_lineno()} references undefined name {name!r}")
    for child in table.get_children():
        bad.extend(_walk_undefined(child, module_names))
    return bad


@pytest.mark.parametrize(
    "py_file",
    sorted(_TRAINING_ROOT.rglob("*.py")),
    ids=lambda p: str(p.relative_to(_TRAINING_ROOT)),
)
def test_training_package_has_no_undefined_names(py_file: Path) -> None:
    """A name loaded via LOAD_GLOBAL in a function scope must resolve to a module-level binding or a builtin.

    This is exactly the kind of "leftover reference after a refactor that a branch-gated test never
    reaches" NameError that 0.20.0's ``t5_attn`` was. Flow-insensitive (like pyflakes): late module-level bindings are not false-flagged.
    """
    source = py_file.read_text(encoding="utf-8")
    table = symtable.symtable(source, str(py_file), "exec")
    bad = _walk_undefined(table, _module_bindings(table))
    assert not bad, "\n".join(f"{py_file.name}: {b}" for b in bad)
