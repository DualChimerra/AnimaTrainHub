"""Suppresses LoKr/LoHa normal-dropout warning spam (utils.lycoris_adapter).

lycoris's LokrModule/LohaModule, when `dropout>0`, has **every module instance**
print a line: "[WARN]LoHa/LoKr haven't implemented normal dropout yet." -- with
280 layers that's 280 lines. During injection, stdout is filtered line by line:
any full line containing the marker is swallowed and counted, everything else
passes through unchanged, and on exit the count is rolled up into a single
logger record.

Covers: whole-line dropping + counting, non-marker lines passed through
unchanged, a half-line split across writes, flush of a trailing line with no
newline, sys.stdout restoration on context exit (including on exception), and
attribute passthrough.
"""
from __future__ import annotations

import io
import sys

import pytest

from utils.lycoris_adapter import (
    _LOKR_DROPOUT_MARKER,
    _LineFilteredStdout,
    _suppress_lokr_dropout_spam,
)

SPAM = f"[WARN]LoHa/LoKr {_LOKR_DROPOUT_MARKER}"


def test_drops_marker_lines_and_counts_them() -> None:
    sink = io.StringIO()
    flt = _LineFilteredStdout(sink, _LOKR_DROPOUT_MARKER)

    for _ in range(280):
        flt.write(SPAM + "\n")

    assert flt.dropped == 280
    assert sink.getvalue() == ""


def test_passes_other_lines_through_unchanged() -> None:
    sink = io.StringIO()
    flt = _LineFilteredStdout(sink, _LOKR_DROPOUT_MARKER)

    flt.write("training step 1\n")
    flt.write(SPAM + "\n")
    flt.write("training step 2\n")

    assert flt.dropped == 1
    assert sink.getvalue() == "training step 1\ntraining step 2\n"


def test_filters_across_write_boundaries() -> None:
    """print doesn't guarantee one write per line -- a half-line must still be judged by line, so the marker can't slip through."""
    sink = io.StringIO()
    flt = _LineFilteredStdout(sink, _LOKR_DROPOUT_MARKER)

    head, tail = SPAM[:10], SPAM[10:]
    flt.write(head)
    flt.write(tail + "\nkeep me\n")

    assert flt.dropped == 1
    assert sink.getvalue() == "keep me\n"


def test_write_returns_input_length() -> None:
    """IO protocol: write returns the number of characters written, which callers (print) rely on."""
    flt = _LineFilteredStdout(io.StringIO(), _LOKR_DROPOUT_MARKER)
    assert flt.write(SPAM + "\n") == len(SPAM) + 1
    assert flt.write("plain\n") == 6


def test_flush_handles_trailing_partial_line() -> None:
    """When the last line has no trailing newline, flush must still check it against the marker instead of letting it through as plain output."""
    sink = io.StringIO()
    flt = _LineFilteredStdout(sink, _LOKR_DROPOUT_MARKER)
    flt.write(SPAM)  # no newline
    flt.flush()

    assert flt.dropped == 1
    assert sink.getvalue() == ""

    sink2 = io.StringIO()
    flt2 = _LineFilteredStdout(sink2, _LOKR_DROPOUT_MARKER)
    flt2.write("half line")  # no newline, not the marker
    flt2.flush()

    assert flt2.dropped == 0
    assert sink2.getvalue() == "half line"


def test_forwards_unknown_attributes_to_wrapped() -> None:
    """encoding / isatty etc. are provided by the wrapped object -- in the training process stdout is a real TextIO."""
    flt = _LineFilteredStdout(sys.__stdout__, _LOKR_DROPOUT_MARKER)
    assert flt.encoding == sys.__stdout__.encoding
    assert flt.isatty() == sys.__stdout__.isatty()


def test_context_restores_stdout_and_reports_count(capsys) -> None:
    original = sys.stdout
    with _suppress_lokr_dropout_spam() as flt:
        assert sys.stdout is not original
        print(SPAM)
        print(SPAM)
        print("real output")
    assert sys.stdout is original
    assert flt.dropped == 2
    assert capsys.readouterr().out == "real output\n"


def test_context_restores_stdout_on_exception() -> None:
    """stdout must be swapped back even when the injected code raises, or all later output would keep going through the filter."""
    original = sys.stdout
    with pytest.raises(RuntimeError):
        with _suppress_lokr_dropout_spam():
            print(SPAM)
            raise RuntimeError("injection blew up")
    assert sys.stdout is original
