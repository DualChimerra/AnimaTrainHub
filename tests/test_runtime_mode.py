from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from studio import secrets, server
from studio.cli import build_parser, _apply_runtime_mode_defaults
from studio.infrastructure import runtime_mode


@pytest.fixture
def secrets_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    sf = tmp_path / "secrets.json"
    monkeypatch.setattr(secrets, "SECRETS_FILE", sf)
    return sf


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("ALS_RUNTIME_MODE", "COLAB_RELEASE_TAG", "COLAB_GPU",
                "COLAB_JUPYTER_IP", "COLAB_BACKEND_VERSION",
                "KAGGLE_KERNEL_RUN_TYPE", "KAGGLE_URL_BASE",
                "KAGGLE_DATA_PROXY_TOKEN"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def client(secrets_file: Path, clean_env: None) -> TestClient:  # noqa: ARG001
    return TestClient(server.app)


# ---------------------------------------------------------------------------
# normalize / detect
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("local", "local"),
        ("colab", "colab"),
        ("  COLAB  ", "colab"),
        ("kaggle", "colab"),
        ("cloud", "colab"),
        ("notebook", "colab"),
        ("pc", "local"),
        ("desktop", "local"),
        ("bogus", ""),
        ("", ""),
        (None, ""),
    ],
)
def test_normalize(raw: object, expected: str) -> None:
    assert runtime_mode.normalize(raw) == expected


def test_detect_local_by_default(clean_env: None) -> None:  # noqa: ARG001
    assert runtime_mode.detect() == "local"


@pytest.mark.parametrize(
    "env_key",
    ["COLAB_RELEASE_TAG", "COLAB_GPU", "COLAB_JUPYTER_IP", "KAGGLE_KERNEL_RUN_TYPE"],
)
def test_detect_colab_from_env(
    clean_env: None, monkeypatch: pytest.MonkeyPatch, env_key: str  # noqa: ARG001
) -> None:
    monkeypatch.setenv(env_key, "1")
    assert runtime_mode.detect() == "colab"


def test_content_dir_alone_does_not_flip_to_colab(
    clean_env: None, monkeypatch: pytest.MonkeyPatch  # noqa: ARG001
) -> None:
    monkeypatch.setattr(
        runtime_mode.Path, "is_dir", lambda self: str(self) == "/content"
    )
    assert runtime_mode.detect_signals()["content_dir"] is True
    assert runtime_mode.detect() == "local"


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_resolve_unset_when_never_chosen(secrets_file: Path, clean_env: None) -> None:  # noqa: ARG001
    assert runtime_mode.resolve() == ""
    assert runtime_mode.effective() == "local"


def test_stored_choice_wins_over_detection(
    secrets_file: Path, clean_env: None, monkeypatch: pytest.MonkeyPatch  # noqa: ARG001
) -> None:
    monkeypatch.setenv("COLAB_RELEASE_TAG", "release-colab-2026")
    secrets.update({"runtime": {"mode": "local"}})
    assert runtime_mode.detect() == "colab"
    assert runtime_mode.resolve() == "local"
    assert runtime_mode.effective() == "local"


def test_env_override_wins_over_stored(
    secrets_file: Path, clean_env: None, monkeypatch: pytest.MonkeyPatch  # noqa: ARG001
) -> None:
    secrets.update({"runtime": {"mode": "local"}})
    monkeypatch.setenv("ALS_RUNTIME_MODE", "colab")
    assert runtime_mode.resolve() == "colab"
    assert runtime_mode.describe()["locked"] is True


def test_invalid_env_override_is_ignored(
    secrets_file: Path, clean_env: None, monkeypatch: pytest.MonkeyPatch  # noqa: ARG001
) -> None:
    secrets.update({"runtime": {"mode": "local"}})
    monkeypatch.setenv("ALS_RUNTIME_MODE", "banana")
    assert runtime_mode.env_override() == ""
    assert runtime_mode.resolve() == "local"


def test_invalid_stored_mode_falls_back_to_unset(secrets_file: Path, clean_env: None) -> None:  # noqa: ARG001
    secrets_file.write_text('{"runtime": {"mode": "banana", "asked": true}}', encoding="utf-8")
    assert secrets.load().runtime.mode == ""
    assert runtime_mode.resolve() == ""


def test_choosing_a_mode_sets_asked(secrets_file: Path, clean_env: None) -> None:  # noqa: ARG001
    s = secrets.update({"runtime": {"mode": "colab"}})
    assert s.runtime.mode == "colab"
    assert s.runtime.asked is True


# ---------------------------------------------------------------------------
# /api/runtime
# ---------------------------------------------------------------------------


def test_get_runtime_reports_unset(client: TestClient) -> None:
    body = client.get("/api/runtime").json()
    assert body["mode"] == ""
    assert body["stored"] == ""
    assert body["detected"] == "local"
    assert body["effective"] == "local"
    assert body["locked"] is False
    assert body["modes"] == ["local", "colab"]
    assert "studio_data" in body["environment"]


def test_put_runtime_persists(client: TestClient) -> None:
    body = client.put("/api/runtime", json={"mode": "colab"}).json()
    assert body["mode"] == "colab"
    assert body["stored"] == "colab"
    assert secrets.load().runtime.mode == "colab"
    assert client.get("/api/runtime").json()["mode"] == "colab"


def test_put_runtime_rejects_unknown_mode(client: TestClient) -> None:
    r = client.put("/api/runtime", json={"mode": "banana"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "runtime.invalid_mode"
    assert secrets.load().runtime.mode == ""


def test_put_runtime_conflicts_with_env_pin(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ALS_RUNTIME_MODE", "colab")
    r = client.put("/api/runtime", json={"mode": "local"})
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "runtime.mode_locked"
    assert secrets.load().runtime.mode == ""
    assert client.put("/api/runtime", json={"mode": "colab"}).status_code == 200


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def _parse(argv: list[str]):
    args = build_parser().parse_args(argv)
    _apply_runtime_mode_defaults(args)
    return args


def test_cli_local_binds_loopback_and_opens_browser(
    secrets_file: Path, clean_env: None  # noqa: ARG001
) -> None:
    args = _parse(["run", "--mode", "local"])
    assert args.host == "127.0.0.1"
    assert args.no_browser is False


def test_cli_colab_binds_all_and_skips_browser(
    secrets_file: Path, clean_env: None  # noqa: ARG001
) -> None:
    args = _parse(["run", "--mode", "colab"])
    assert args.host == "0.0.0.0"
    assert args.no_browser is True


def test_cli_explicit_host_beats_mode_default(
    secrets_file: Path, clean_env: None  # noqa: ARG001
) -> None:
    args = _parse(["run", "--mode", "colab", "--host", "127.0.0.1"])
    assert args.host == "127.0.0.1"


def test_cli_mode_flag_propagates_via_env(
    secrets_file: Path, clean_env: None, monkeypatch: pytest.MonkeyPatch  # noqa: ARG001
) -> None:
    _parse(["run", "--mode", "colab"])
    assert runtime_mode.env_override() == "colab"


def test_cli_follows_stored_choice_without_flag(
    secrets_file: Path, clean_env: None  # noqa: ARG001
) -> None:
    secrets.update({"runtime": {"mode": "colab"}})
    args = _parse(["run"])
    assert args.host == "0.0.0.0"  # noqa: S104
    assert args.no_browser is True
