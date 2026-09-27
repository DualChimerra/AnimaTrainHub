from __future__ import annotations

from unittest.mock import MagicMock

from utils.lycoris_adapter import AnimaLycorisAdapter
from training.families.anima.preset import ANIMA_PRESET


def test_detach_noop_when_not_injected() -> None:
    a = AnimaLycorisAdapter(preset=ANIMA_PRESET, )
    assert a.detach() is True
    assert a.network is None


def test_detach_calls_restore_and_clears_state() -> None:
    a = AnimaLycorisAdapter(preset=ANIMA_PRESET, )
    fake_network = MagicMock()
    fake_network.restore = MagicMock()
    a.network = fake_network

    fake_orig_train = MagicMock()
    fake_model = MagicMock()
    a._orig_train = fake_orig_train
    a._injected_model = fake_model

    assert a.detach() is True
    fake_network.restore.assert_called_once()
    assert fake_model.train is fake_orig_train
    assert a.network is None
    assert a._injected_model is None
    assert a._orig_train is None


def test_detach_falls_back_to_restore_apply() -> None:
    a = AnimaLycorisAdapter(preset=ANIMA_PRESET, )
    fake_network = MagicMock(spec=["restore_apply", "remove_apply"])
    fake_network.restore_apply = MagicMock()
    a.network = fake_network
    a._orig_train = MagicMock()
    a._injected_model = MagicMock()

    assert a.detach() is True
    fake_network.restore_apply.assert_called_once()


def test_detach_returns_false_when_no_restore_interface() -> None:
    a = AnimaLycorisAdapter(preset=ANIMA_PRESET, )
    fake_network = MagicMock(spec=[])
    a.network = fake_network
    a._orig_train = MagicMock()
    a._injected_model = MagicMock()

    assert a.detach() is False
    assert a.network is None


def test_detach_idempotent() -> None:
    a = AnimaLycorisAdapter(preset=ANIMA_PRESET, )
    fake_network = MagicMock()
    fake_network.restore = MagicMock()
    a.network = fake_network
    a._orig_train = MagicMock()
    a._injected_model = MagicMock()

    a.detach()
    assert a.detach() is True
