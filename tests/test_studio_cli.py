from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import pytest

from studio import cli


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_calls(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake(cmd, **kwargs: Any) -> int:
        calls.append(list(cmd))
        return 0

    monkeypatch.setattr(cli.subprocess, "call", fake)
    # Tests using this fixture exercise command dispatch only.  Avoid importing
    # the real CUDA/Triton stack in those unit tests.
    monkeypatch.setattr(cli, "_check_torch_cuda", lambda: None)
    monkeypatch.setattr(cli, "_ensure_windows_triton", lambda: None)
    monkeypatch.setattr(cli, "_try_enable_flash_attn", lambda: None)
    monkeypatch.setattr(cli, "_report_lycoris_kernels", lambda: None)
    monkeypatch.setattr(cli, "_check_onnxruntime", lambda: None)
    return calls


@pytest.fixture
def fake_npm(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setattr(cli, "find_npm", lambda: "fake-npm")
    return "fake-npm"


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------


def test_parser_has_all_subcommands() -> None:
    p = cli.build_parser()
    args = p.parse_args(["run"])
    assert args.cmd == "run"
    args = p.parse_args(["dev"])
    assert args.cmd == "dev"
    args = p.parse_args(["build"])
    assert args.cmd == "build"
    args = p.parse_args(["test"])
    assert args.cmd == "test"


def test_run_args_default_host_port() -> None:
    p = cli.build_parser()
    args = p.parse_args(["run"])
    assert args.host is None
    assert args.port == 8765

    resolved = p.parse_args(["run", "--mode", "local"])
    cli._apply_runtime_mode_defaults(resolved)
    assert resolved.host == "127.0.0.1"


def test_run_custom_host_port() -> None:
    p = cli.build_parser()
    args = p.parse_args(["run", "--host", "0.0.0.0", "--port", "9000"])
    assert args.host == "0.0.0.0"
    assert args.port == 9000


def test_default_command_is_run() -> None:
    p = cli.build_parser()
    args = p.parse_args([])
    assert getattr(args, "cmd", None) is None
    args2 = p.parse_args(["run"])
    assert args2.cmd == "run"


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def test_build_runs_npm_run_build(fake_calls, fake_npm, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "NODE_MODULES", Path("/fake/exists"))
    monkeypatch.setattr(cli.Path, "exists", lambda self: True)
    rc = cli.main(["build"])
    assert rc == 0
    assert any(c[:3] == ["fake-npm", "run", "build"] for c in fake_calls)


def test_build_installs_when_node_modules_missing(
    fake_calls, fake_npm, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "NODE_MODULES", tmp_path / "absent")

    class FakePopen:
        def __init__(self, cmd, **kwargs):
            fake_calls.append(list(cmd))
            self.args = cmd
            self.returncode = 0
            self.stdin = None
            self.stdout = None
            self.stderr = None

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

        def poll(self):
            return 0

        def communicate(self, input=None, timeout=None):
            return ("", "")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(cli.subprocess, "Popen", FakePopen)

    rc = cli.main(["build"])
    assert rc == 0
    assert any(c[:2] == ["fake-npm", "install"] for c in fake_calls)
    assert any(c[:3] == ["fake-npm", "run", "build"] for c in fake_calls)


def test_build_installs_when_package_file_newer_than_node_modules(
    fake_calls, fake_npm, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    web_dir = tmp_path / "web"
    node_modules = web_dir / "node_modules"
    bin_dir = node_modules / ".bin"
    bin_dir.mkdir(parents=True)
    marker = node_modules / ".package-lock.json"
    marker.write_text("{}", encoding="utf-8")
    (bin_dir / ("eslint.cmd" if cli.os.name == "nt" else "eslint")).write_text("", encoding="utf-8")
    package_json = web_dir / "package.json"
    package_json.write_text('{"dependencies":{"i18next":"latest"}}', encoding="utf-8")
    package_lock = web_dir / "package-lock.json"
    package_lock.write_text("{}", encoding="utf-8")
    os.utime(marker, (100, 100))
    os.utime(package_json, (200, 200))
    os.utime(package_lock, (100, 100))

    monkeypatch.setattr(cli, "WEB_DIR", web_dir)
    monkeypatch.setattr(cli, "NODE_MODULES", node_modules)

    class FakePopen:
        def __init__(self, cmd, **kwargs):
            fake_calls.append(list(cmd))
            self.args = cmd
            self.returncode = 0
            self.stdin = None
            self.stdout = None
            self.stderr = None

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

        def poll(self):
            return 0

        def communicate(self, input=None, timeout=None):
            return ("", "")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(cli.subprocess, "Popen", FakePopen)

    rc = cli.main(["build"])
    assert rc == 0
    assert any(c[:2] == ["fake-npm", "install"] for c in fake_calls)
    assert any(c[:3] == ["fake-npm", "run", "build"] for c in fake_calls)


def test_build_no_npm_returns_2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "find_npm", lambda: None)
    assert cli.main(["build"]) == 2


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def test_run_starts_backend_and_skips_build_when_dist_exists(
    fake_calls, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_dist = tmp_path / "dist"
    fake_dist.mkdir()
    (fake_dist / "index.html").write_text("<html/>")
    monkeypatch.setattr(cli, "WEB_DIST", fake_dist)
    monkeypatch.setattr(cli, "_web_dist_is_stale", lambda: False)
    rc = cli.main(["run"])
    assert rc == 0
    assert not any("run" in c and "build" in c for c in fake_calls)
    assert any(
        "studio.server" in " ".join(c) and "--port" in c for c in fake_calls
    )


def test_run_no_build_skips_when_dist_missing(
    fake_calls, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(cli, "WEB_DIST", tmp_path / "absent")
    monkeypatch.setattr(cli, "find_npm", lambda: "fake-npm")
    rc = cli.main(["run", "--no-build"])
    assert rc == 0
    assert not any(c[:3] == ["fake-npm", "run", "build"] for c in fake_calls)


def test_run_passes_host_port(
    fake_calls, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_dist = tmp_path / "dist"
    fake_dist.mkdir()
    monkeypatch.setattr(cli, "WEB_DIST", fake_dist)
    monkeypatch.setattr(cli, "_web_dist_is_stale", lambda: False)
    cli.main(["run", "--host", "0.0.0.0", "--port", "9999"])
    server_call = next(
        c for c in fake_calls if "studio.server" in " ".join(c)
    )
    assert "0.0.0.0" in server_call
    assert "9999" in server_call


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_test_subcommand_runs_pytest(
    fake_calls, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "find_npm", lambda: None)
    rc = cli.main(["test"])
    assert rc == 0
    assert any("pytest" in " ".join(c) for c in fake_calls)


def test_test_runs_vitest_when_npm_available(
    fake_calls, fake_npm, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_node_modules = tmp_path / "nm"
    fake_node_modules.mkdir()
    monkeypatch.setattr(cli, "NODE_MODULES", fake_node_modules)
    rc = cli.main(["test"])
    assert rc == 0
    assert any(c[:3] == ["fake-npm", "run", "test"] for c in fake_calls)


def test_test_pytest_failure_short_circuits(
    fake_npm, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []
    def fake(cmd, **_: Any) -> int:
        calls.append(list(cmd))
        return 7 if "pytest" in " ".join(cmd) else 0
    monkeypatch.setattr(cli.subprocess, "call", fake)
    rc = cli.main(["test"])
    assert rc == 7
    assert all("vitest" not in " ".join(c) for c in calls)
    assert all(c[:3] != ["fake-npm", "run", "test"] for c in calls)


# ---------------------------------------------------------------------------
# PR-4 — _check_torch_cuda
# ---------------------------------------------------------------------------


class _FakeTorchVersion:
    def __init__(self, cuda):
        self.cuda = cuda


class _FakeTorch:
    def __init__(self, *, available: bool, cuda_build, version: str = "2.5.0", device_name: str = "RTX 5090"):
        self._available = available
        self._device_name = device_name
        self.__version__ = version
        self.version = _FakeTorchVersion(cuda_build)
        outer = self
        class _Cuda:
            @staticmethod
            def is_available():
                return outer._available
            @staticmethod
            def get_device_name(_idx):
                return outer._device_name
        self.cuda = _Cuda()


def _install_fake_torch(monkeypatch: pytest.MonkeyPatch, torch_module) -> None:
    import sys as _sys
    monkeypatch.setitem(_sys.modules, "torch", torch_module)


def test_check_torch_cuda_silent_when_torch_missing(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    import sys as _sys
    monkeypatch.delitem(_sys.modules, "torch", raising=False)

    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def _fake_import(name, *a, **k):
        if name == "torch":
            raise ImportError("simulated")
        return real_import(name, *a, **k)

    monkeypatch.setattr("builtins.__import__", _fake_import)
    cli._check_torch_cuda()
    out = capsys.readouterr()
    assert out.out == ""
    assert out.err == ""


def test_check_torch_cuda_prints_ok_when_available(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    _install_fake_torch(
        monkeypatch,
        _FakeTorch(available=True, cuda_build="12.8", version="2.5.0", device_name="RTX 5090"),
    )
    cli._check_torch_cuda()
    out = capsys.readouterr()
    assert "RTX 5090" in out.out
    assert "2.5.0" in out.out
    assert out.err == ""


def test_check_torch_cuda_warns_on_cpu_only_with_gpu(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    _install_fake_torch(
        monkeypatch,
        _FakeTorch(available=False, cuda_build=None, version="2.5.0+cpu"),
    )
    from studio.services.runtime import onnxruntime as onnxruntime_setup
    monkeypatch.setattr(
        onnxruntime_setup,
        "detect_cuda",
        lambda: {"available": True, "driver_version": "551.86", "gpu_name": "RTX 5090"},
    )
    cli._check_torch_cuda()
    out = capsys.readouterr()
    assert out.out == ""
    assert "CPU-only" in out.err
    assert "pip install torch" in out.err
    assert "--index-url" in out.err
    assert "✓" not in out.err
    assert "⚠" not in out.err


def test_check_torch_cuda_info_on_cpu_only_without_gpu(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    _install_fake_torch(
        monkeypatch,
        _FakeTorch(available=False, cuda_build=None, version="2.5.0+cpu"),
    )
    from studio.services.runtime import onnxruntime as onnxruntime_setup
    monkeypatch.setattr(
        onnxruntime_setup,
        "detect_cuda",
        lambda: {"available": False, "driver_version": None, "gpu_name": None},
    )
    cli._check_torch_cuda()
    out = capsys.readouterr()
    assert "CPU-only build" in out.out
    assert "no NVIDIA GPU detected" in out.out
    assert out.err == ""


def test_check_torch_cuda_warns_on_cuda_build_but_unavailable(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    _install_fake_torch(
        monkeypatch,
        _FakeTorch(available=False, cuda_build="12.8", version="2.5.0+cu128"),
    )
    cli._check_torch_cuda()
    out = capsys.readouterr()
    assert "CUDA 12.8 build" in out.err
    assert "is_available()=False" in out.err
    assert "pip install torch" not in out.err


def test_report_lycoris_kernels_prints_preferred_backend(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    import importlib.metadata
    import sys as _sys
    import types

    versions = {
        "lycoris-lora": "4.0.0",
        "triton-windows": "3.6.0.post26",
    }

    def _version(name: str) -> str:
        if name in versions:
            return versions[name]
        raise importlib.metadata.PackageNotFoundError(name)

    kernels = types.ModuleType("lycoris.kernels")
    kernels.available_backends = lambda: ("triton", "compile", "torch")
    kernels.resolve_backend = lambda: "triton"
    lycoris = types.ModuleType("lycoris")
    lycoris.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setattr(importlib.metadata, "version", _version)
    monkeypatch.setitem(_sys.modules, "lycoris", lycoris)
    monkeypatch.setitem(_sys.modules, "lycoris.kernels", kernels)

    cli._report_lycoris_kernels()
    out = capsys.readouterr()
    assert "LyCORIS 4.0.0" in out.out
    assert "preferred=triton" in out.out
    assert "fused=triton" in out.out
    assert "Triton 3.6.0.post26" in out.out
    assert out.err == ""


def test_report_lycoris_kernels_silent_when_not_installed(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    import importlib.metadata

    def _missing(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", _missing)
    cli._report_lycoris_kernels()
    out = capsys.readouterr()
    assert out.out == ""
    assert out.err == ""


def test_ensure_windows_triton_installs_torch_211_pair(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    import importlib.metadata

    _install_fake_torch(
        monkeypatch,
        _FakeTorch(available=True, cuda_build="12.8", version="2.11.0+cu128"),
    )
    monkeypatch.setattr(cli.sys, "platform", "win32")

    def _missing(_name: str) -> str:
        raise importlib.metadata.PackageNotFoundError

    installs: list[list[str]] = []
    monkeypatch.setattr(importlib.metadata, "version", _missing)
    monkeypatch.setattr(
        cli,
        "_pip_install",
        lambda args: installs.append(args) or 0,
    )

    cli._ensure_windows_triton()

    assert installs == [["triton-windows>=3.6,<3.7"]]
    assert "torch 2.11 requires Triton 3.6" in capsys.readouterr().out


def test_ensure_windows_triton_keeps_matching_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib.metadata

    _install_fake_torch(
        monkeypatch,
        _FakeTorch(available=True, cuda_build="12.8", version="2.11.0+cu128"),
    )
    monkeypatch.setattr(cli.sys, "platform", "win32")
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "3.6.0.post26")
    monkeypatch.setattr(
        cli,
        "_pip_install",
        lambda _args: pytest.fail("matching Triton must not be reinstalled"),
    )

    cli._ensure_windows_triton()


# ---------------------------------------------------------------------------
# PR-5 — _print_npm_install_hint
# ---------------------------------------------------------------------------


def test_npm_hint_windows(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(cli.os, "name", "nt")
    cli._print_npm_install_hint()
    err = capsys.readouterr().err
    assert "Node.js 18+" in err
    assert "winget install" in err
    assert "nodejs.org" in err
    assert "nodesource" not in err
    assert "nvm" not in err


def test_npm_hint_linux_non_root(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(cli.os, "name", "posix")
    monkeypatch.setattr(cli.os, "getuid", lambda: 1000, raising=False)
    cli._print_npm_install_hint()
    err = capsys.readouterr().err
    assert "nodesource.com" in err
    assert "sudo bash" in err
    assert "nvm" in err
    assert "winget" not in err


def test_npm_hint_linux_root_drops_sudo(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setattr(cli.os, "name", "posix")
    monkeypatch.setattr(cli.os, "getuid", lambda: 0, raising=False)
    cli._print_npm_install_hint()
    err = capsys.readouterr().err
    assert " sudo " not in err
    assert "sudo bash" not in err
    assert "nodesource.com" in err


def test_cmd_build_no_npm_prints_install_hint(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setattr(cli, "find_npm", lambda: None)
    monkeypatch.setattr(cli.os, "name", "nt")
    rc = cli.main(["build"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "winget" in err
    assert "Node.js 18+" in err


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_installer_hashes_sha256_per_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli_py = tmp_path / "cli.py"
    sh = tmp_path / "studio.sh"
    bat = tmp_path / "studio.bat"
    cli_py.write_bytes(b"print('hello')\n")
    sh.write_bytes(b"#!/bin/sh\n")
    monkeypatch.setattr(cli, "_INSTALLER_FILES", (cli_py, sh, bat))
    h = cli._installer_hashes()
    assert h == {
        "cli.py": hashlib.sha256(b"print('hello')\n").hexdigest(),
        "studio.sh": hashlib.sha256(b"#!/bin/sh\n").hexdigest(),
        "studio.bat": None,
    }


def test_installer_hashes_changes_when_content_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = tmp_path / "cli.py"
    p.write_bytes(b"v1")
    monkeypatch.setattr(cli, "_INSTALLER_FILES", (p,))
    before = cli._installer_hashes()
    p.write_bytes(b"v2")
    assert cli._installer_hashes() != before


def _stub_run_bootstrap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Path:
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html/>")
    monkeypatch.setattr(cli, "WEB_DIST", dist)
    monkeypatch.setattr(cli, "_web_dist_is_stale", lambda: False)
    monkeypatch.setattr(cli, "_ensure_python_deps", lambda: 0)
    monkeypatch.setattr(cli, "_apply_update_pending", lambda: None, raising=False)
    monkeypatch.setattr(cli, "_apply_pending_install", lambda: None)
    monkeypatch.setattr(cli, "_check_torch_cuda", lambda: None)
    monkeypatch.setattr(cli, "_ensure_windows_triton", lambda: None)
    monkeypatch.setattr(cli, "_try_enable_flash_attn", lambda: None)
    monkeypatch.setattr(cli, "_report_lycoris_kernels", lambda: None)
    monkeypatch.setattr(cli, "_check_onnxruntime", lambda: None)
    monkeypatch.setattr(cli, "_spawn_browser_opener", lambda *a, **k: None)
    return dist


def test_cmd_run_returns_42_when_installer_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    _stub_run_bootstrap(monkeypatch, tmp_path)

    fake_flag = tmp_path / "restart"
    fake_flag.touch()
    monkeypatch.setattr(cli, "_RESTART_FLAG", fake_flag)

    call_count = [0]
    def fake_hashes() -> dict[str, str]:
        call_count[0] += 1
        return {"cli.py": "B"} if call_count[0] > 1 else {"cli.py": "A"}
    monkeypatch.setattr(cli, "_installer_hashes", fake_hashes)

    monkeypatch.setattr(cli.subprocess, "call", lambda *a, **k: 0)

    rc = cli.main(["run", "--no-browser"])
    assert rc == cli._INSTALLER_RELOAD_EXIT_CODE == 42
    assert fake_flag.exists(), "the flag must be kept for the wrapper to take the exec-self path"
    assert "update to a launcher file" in capsys.readouterr().out


def test_cmd_run_normal_restart_when_installer_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_run_bootstrap(monkeypatch, tmp_path)

    fake_flag = tmp_path / "restart"
    monkeypatch.setattr(cli, "_RESTART_FLAG", fake_flag)
    monkeypatch.setattr(cli, "_installer_hashes", lambda: {"cli.py": "stable"})

    server_calls = [0]
    def fake(cmd, **_: Any) -> int:
        if "studio.server" in " ".join(cmd):
            if server_calls[0] == 0:
                fake_flag.touch()
            server_calls[0] += 1
        return 0
    monkeypatch.setattr(cli.subprocess, "call", fake)

    rc = cli.main(["run", "--no-browser"])
    assert rc == 0
    assert not fake_flag.exists(), "the flag should be removed after a normal restart completes"
    assert server_calls[0] == 2, "a normal restart should loop twice"


def test_cmd_run_ctrl_c_returns_130_without_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    _stub_run_bootstrap(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "_RESTART_FLAG", tmp_path / "restart")

    def fake(cmd, **_: Any) -> int:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli.subprocess, "call", fake)

    rc = cli.main(["run", "--no-browser"])
    assert rc == 130
    assert "Ctrl+C" in capsys.readouterr().out


def test_cmd_run_no_flag_no_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_run_bootstrap(monkeypatch, tmp_path)

    fake_flag = tmp_path / "restart"
    monkeypatch.setattr(cli, "_RESTART_FLAG", fake_flag)

    hash_calls = [0]
    def fake_hashes() -> dict[str, str]:
        hash_calls[0] += 1
        return {"cli.py": "v"} if hash_calls[0] == 1 else {"cli.py": "v-changed"}
    monkeypatch.setattr(cli, "_installer_hashes", fake_hashes)

    monkeypatch.setattr(cli.subprocess, "call", lambda *a, **k: 7)

    rc = cli.main(["run", "--no-browser"])
    assert rc == 7
    assert hash_calls[0] == 1
