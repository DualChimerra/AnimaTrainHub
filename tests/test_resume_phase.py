from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest


class _FakeModel:
    def train(self):
        return None


def _ctx(tmp_path: Path, *, sample_every: int, sample_steps: int = 0):
    from training.context import TrainingContext

    args = Namespace(
        no_progress=True,
        loss_curve_steps=100,
        no_live_curve=True,
        resume_state="",
        sample_prompts=["p1", "p2"],
        sample_prompt="fallback",
        sample_steps=sample_steps,
        sample_every=sample_every,
        optimizer_type="adamw",
    )
    ctx = TrainingContext(args=args)
    ctx.total_steps = 10
    ctx.sample_dir = tmp_path
    ctx.optimizer = None
    ctx.model = _FakeModel()
    ctx.monitor_server = False
    ctx.wandb_monitor = Namespace(log_samples=False)
    return ctx


def test_resume_phase_runs_startup_baseline_at_step0(tmp_path, monkeypatch):
    """When periodic sampling is enabled, a fresh run (global_step == 0) runs the step 0 baseline sample."""
    pytest.importorskip("torch")
    from training.phases import resume

    calls = []
    monkeypatch.setattr(resume, "run_sample", lambda *args, **kwargs: calls.append(kwargs))

    ctx = _ctx(tmp_path, sample_every=2)
    resume.run(ctx)

    assert [call["prompt"] for call in calls] == ["p1", "p2"]
    assert [call["sample_path"].name for call in calls] == [
        "step_0_baseline_0.png",
        "step_0_baseline_1.png",
    ]


def test_resume_phase_skips_baseline_when_sampling_disabled(tmp_path, monkeypatch):
    """No baseline sample when sample_every / sample_steps are both 0 (periodic sampling disabled)."""
    pytest.importorskip("torch")
    from training.phases import resume

    calls = []
    monkeypatch.setattr(resume, "run_sample", lambda *args, **kwargs: calls.append(kwargs))

    ctx = _ctx(tmp_path, sample_every=0, sample_steps=0)
    resume.run(ctx)

    assert calls == []


def _resumed_ctx(tmp_path: Path, monkeypatch, *, with_config: bool):
    from training.phases import resume

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    state = state_dir / "auto_epoch_state.pt"
    state.write_bytes(b"x")
    if with_config:
        (state_dir / "auto_epoch_state.config.json").write_text("{}")

    events = []
    monkeypatch.setattr(resume, "emit_event", lambda name, payload: events.append((name, payload)))
    monkeypatch.setattr(resume, "load_training_state", lambda *a, **k: (2, 202, [], None))

    ctx = _ctx(tmp_path, sample_every=0)
    ctx.args.resume_state = str(state)
    ctx.task_archive_state_dir = state_dir
    ctx.family = Namespace(spec=Namespace(family_id="test"))
    resume.run(ctx)
    return ctx, events


def test_resume_from_auto_epoch_state_is_pausable_immediately(tmp_path, monkeypatch):
    """Resuming from auto_epoch_state.pt reuses it as the pause point instead of locking pause until the next epoch."""
    pytest.importorskip("torch")

    ctx, events = _resumed_ctx(tmp_path, monkeypatch, with_config=True)

    assert ctx.last_auto_epoch_state_path == tmp_path / "state" / "auto_epoch_state.pt"
    assert ctx.last_auto_epoch_config_path == tmp_path / "state" / "auto_epoch_state.config.json"
    names = [name for name, _ in events]
    assert names.index("auto_epoch_backup_written") < names.index("train_loop_started")
    payload = dict(events)["auto_epoch_backup_written"]
    assert (payload["epoch"], payload["step"]) == (2, 202)


def test_resume_without_config_snapshot_keeps_pause_locked(tmp_path, monkeypatch):
    """No config snapshot next to the state -> not a complete pause point, wait for the next epoch backup."""
    pytest.importorskip("torch")

    ctx, events = _resumed_ctx(tmp_path, monkeypatch, with_config=False)

    assert ctx.last_auto_epoch_state_path is None
    assert "auto_epoch_backup_written" not in [name for name, _ in events]
