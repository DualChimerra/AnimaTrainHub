from __future__ import annotations

import ast
import os
from pathlib import Path


SRC_REL = Path("runtime") / "training" / "model_loading.py"
REPO_ROOT = Path(__file__).resolve().parent.parent


class _RecordingLogger:
    def __init__(self):
        self.warnings: list[str] = []

    def warning(self, msg, *args):
        self.warnings.append(msg % args if args else str(msg))


def _make_fn(logger):
    src_path = REPO_ROOT / SRC_REL
    tree = ast.parse(src_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "find_diffusion_pipe_root":
            mod = ast.Module(body=[node], type_ignores=[])
            ns: dict = {"Path": Path, "os": os, "logger": logger, "__file__": str(src_path)}
            exec(compile(mod, str(src_path), "exec"), ns)
            return ns["find_diffusion_pipe_root"]
    raise RuntimeError(f"find_diffusion_pipe_root not found in {SRC_REL}")


def test_returns_modeling_anima_dir(monkeypatch):
    monkeypatch.delenv("DIFFUSION_PIPE_ROOT", raising=False)
    logger = _RecordingLogger()
    fn = _make_fn(logger)
    root = fn()
    assert root == REPO_ROOT / "modeling" / "anima"
    assert (root / "anima_modeling.py").exists()
    assert (root / "cosmos_predict2_modeling.py").exists()
    assert logger.warnings == []


def test_diffusion_pipe_root_env_is_ignored_with_warning(tmp_path, monkeypatch):
    monkeypatch.setenv("DIFFUSION_PIPE_ROOT", str(tmp_path))
    logger = _RecordingLogger()
    fn = _make_fn(logger)
    root = fn()
    assert root == REPO_ROOT / "modeling" / "anima"
    assert len(logger.warnings) == 1
    assert "DIFFUSION_PIPE_ROOT" in logger.warnings[0]
