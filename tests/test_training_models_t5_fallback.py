from __future__ import annotations

import logging
import sys
import types

import pytest


class _SelfReturningModel:

    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self

    def requires_grad_(self, *args, **kwargs):
        return self


def _make_fake_transformers(t5_from_pretrained):
    mod = types.ModuleType("transformers")
    t5_calls: list[str] = []

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(path, **kwargs):
            return object()

    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(path, **kwargs):
            return _SelfReturningModel()

    class T5Tokenizer:
        @staticmethod
        def from_pretrained(path, **kwargs):
            t5_calls.append(str(path))
            return t5_from_pretrained(str(path))

    class T5TokenizerFast(T5Tokenizer):
        pass

    mod.AutoTokenizer = AutoTokenizer
    mod.AutoModelForCausalLM = AutoModelForCausalLM
    mod.T5Tokenizer = T5Tokenizer
    mod.T5TokenizerFast = T5TokenizerFast
    return mod, t5_calls


@pytest.fixture()
def tm():
    from training.families.anima import loader as models

    return models


def test_missing_t5_dir_download_error_is_wrapped(tm, monkeypatch, tmp_path, caplog):

    def _boom(path):
        raise ConnectionError("[WinError 10060] the connected party did not properly respond after a period of time")

    fake, t5_calls = _make_fake_transformers(_boom)
    monkeypatch.setitem(sys.modules, "transformers", fake)

    missing = tmp_path / "no_such_dir"
    with caplog.at_level(logging.WARNING):
        with pytest.raises(RuntimeError) as excinfo:
            tm.load_text_encoders(str(tmp_path), str(missing), "cpu", None)

    msg = str(excinfo.value)
    assert "Failed to download the T5 tokenizer" in msg
    assert "google/t5-v1_1-xxl" in msg
    assert "10060" in msg
    assert "t5_tokenizer_path" in msg
    assert isinstance(excinfo.value.__cause__, ConnectionError)
    assert t5_calls == ["google/t5-v1_1-xxl"]
    assert any("local directory missing" in r.getMessage() for r in caplog.records)


def test_missing_t5_dir_fallback_success_logs_warning(tm, monkeypatch, tmp_path, caplog):
    fake, t5_calls = _make_fake_transformers(lambda path: "tok")
    monkeypatch.setitem(sys.modules, "transformers", fake)

    with caplog.at_level(logging.WARNING):
        _, _, t5_tokenizer = tm.load_text_encoders(
            str(tmp_path), str(tmp_path / "no_such_dir"), "cpu", None
        )

    assert t5_tokenizer == "tok"
    assert t5_calls == ["google/t5-v1_1-xxl"]
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("local directory missing" in m and "google/t5-v1_1-xxl" in m for m in warnings)


def test_local_t5_dir_never_falls_back(tm, monkeypatch, tmp_path, caplog):
    fake, t5_calls = _make_fake_transformers(lambda path: "tok")
    monkeypatch.setitem(sys.modules, "transformers", fake)

    local = tmp_path / "t5_tokenizer"
    local.mkdir()
    with caplog.at_level(logging.WARNING):
        _, _, t5_tokenizer = tm.load_text_encoders(str(tmp_path), str(local), "cpu", None)

    assert t5_tokenizer == "tok"
    assert t5_calls == [str(local)]
    assert not any("local directory missing" in r.getMessage() for r in caplog.records)
