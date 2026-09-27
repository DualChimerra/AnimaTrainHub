from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import launcher  # noqa: E402


def make_repo(root: Path) -> Path:
    (root / "studio").mkdir(parents=True, exist_ok=True)
    (root / "studio" / "__init__.py").touch()
    (root / "requirements.txt").write_text("packaging\n", encoding="utf-8")
    return root


@pytest.fixture
def isolate_lookup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    empty = tmp_path / "elsewhere"
    empty.mkdir()

    def _set(where: Path) -> None:
        monkeypatch.setattr(launcher, "launcher_dir", lambda: where)
        monkeypatch.chdir(where)

    _set(empty)
    return _set


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_looks_like_repo_needs_both_markers(tmp_path: Path) -> None:
    (tmp_path / "requirements.txt").touch()
    assert launcher.looks_like_repo(tmp_path) is False
    (tmp_path / "studio").mkdir()
    (tmp_path / "studio" / "__init__.py").touch()
    assert launcher.looks_like_repo(tmp_path) is True


def test_find_repo_next_to_launcher(tmp_path: Path, isolate_lookup) -> None:
    repo = make_repo(tmp_path / "AnimaLoraStudio")
    isolate_lookup(repo)
    assert launcher.find_repo(None) == repo


def test_find_repo_walks_up_from_subdirectory(tmp_path: Path, isolate_lookup) -> None:
    repo = make_repo(tmp_path / "AnimaLoraStudio")
    nested = repo / "tools" / "bin"
    nested.mkdir(parents=True)
    isolate_lookup(nested)
    assert launcher.find_repo(None) == repo


def test_find_repo_explicit_path(tmp_path: Path, isolate_lookup) -> None:  # noqa: ARG001
    repo = make_repo(tmp_path / "somewhere")
    assert launcher.find_repo(str(repo)) == repo


def test_find_repo_explicit_path_rejects_wrong_folder(
    tmp_path: Path, isolate_lookup, capsys: pytest.CaptureFixture[str]  # noqa: ARG001
) -> None:
    with pytest.raises(SystemExit) as exc:
        launcher.find_repo(str(tmp_path))
    assert exc.value.code == 1
    assert "is not an AnimaLoraStudio checkout" in capsys.readouterr().err


def test_find_repo_reports_actionable_error_when_missing(
    isolate_lookup, capsys: pytest.CaptureFixture[str]  # noqa: ARG001
) -> None:
    with pytest.raises(SystemExit):
        launcher.find_repo(None)
    err = capsys.readouterr().err
    assert "could not find the AnimaLoraStudio files" in err
    assert "--repo" in err


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_venv_python_path_is_platform_correct(tmp_path: Path) -> None:
    py = launcher.venv_python(tmp_path / "venv")
    if launcher.os.name == "nt":
        assert py == tmp_path / "venv" / "Scripts" / "python.exe"
    else:
        assert py == tmp_path / "venv" / "bin" / "python"


def test_find_existing_venv_prefers_venv_over_dotvenv(tmp_path: Path) -> None:
    for name in (".venv", "venv"):
        py = launcher.venv_python(tmp_path / name)
        py.parent.mkdir(parents=True)
        py.touch()
    assert launcher.find_existing_venv(tmp_path) == tmp_path / "venv"


def test_find_existing_venv_none_when_absent(tmp_path: Path) -> None:
    assert launcher.find_existing_venv(tmp_path) is None


def test_find_existing_venv_ignores_empty_directory(tmp_path: Path) -> None:
    (tmp_path / "venv").mkdir()
    assert launcher.find_existing_venv(tmp_path) is None


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_unknown_args_are_passed_through() -> None:
    args, passthrough = launcher.build_parser().parse_known_args(
        ["--mode", "colab", "--port", "8800", "--no-browser"]
    )
    assert args.mode == "colab"
    assert passthrough == ["--port", "8800", "--no-browser"]


def test_mode_defaults_to_none() -> None:
    args, _ = launcher.build_parser().parse_known_args([])
    assert args.mode is None


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_printed_strings_are_pure_ascii() -> None:
    import ast

    source = (Path(__file__).resolve().parent.parent / "tools" / "launcher.py")
    tree = ast.parse(source.read_text(encoding="utf-8"))

    printed: list[tuple[int, str]] = []

    def collect(node: ast.AST) -> None:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                printed.append((sub.lineno, sub.value))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = getattr(func, "id", None) or getattr(func, "attr", None)
        if name in {"say", "warn", "die", "print", "input"}:
            for arg in node.args:
                collect(arg)
        for kw in node.keywords:
            if kw.arg in {"help", "description"}:
                collect(kw.value)

    offenders = [
        (line, text) for line, text in printed if not text.isascii()
    ]
    assert not offenders, (
        "non-ASCII in launcher output (crashes frozen exe on non-UTF-8 consoles):\n"
        + "\n".join(f"  line {line}: {text!r}" for line, text in offenders)
    )


# ---------------------------------------------------------------------------
# --check
# ---------------------------------------------------------------------------


def test_check_reports_missing_venv(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo = make_repo(tmp_path / "repo")
    assert launcher.run_check(repo) == 0
    out = capsys.readouterr().out
    assert str(repo) in out
    assert "not created yet" in out
    assert "gpu" in out
