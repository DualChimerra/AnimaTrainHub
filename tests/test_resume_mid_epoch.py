"""Mid-epoch step recovery point resume, step-by-step parity (runtime/training/loop.py + state.py).

Background: `save_state_every_steps` stores the current epoch index, and the old
resume ran `for epoch in range(start_epoch, ...)` from batch 0 of that epoch --
batches trained before the recovery point were trained again, step N+1 after the
resume got the data of step 1, and per-step loss drifted from an uninterrupted
run. Also each DataLoader iter() drew a worker seed from the global torch CPU RNG,
so rebuilding the iterator on resume shifted CPU-side random streams (InfoNoise
sampling etc.) by one.

Fix: the recovery point stores `epoch_batch_offset`, the sampler skips trained
batches at the index level, and the DataLoader gets its own generator. This file
runs an uninterrupted and a resumed trajectory with the real DataLoader /
BucketBatchSampler / save+load_training_state and asserts the per-step loss after
resume and the final weights match bit for bit.
"""
from __future__ import annotations

import random
import types

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402

from runtime.training import loop as loop_mod  # noqa: E402
from runtime.training.dataset import BucketBatchSampler, NavitPackBatchSampler  # noqa: E402
from runtime.training.phases import resume as resume_mod  # noqa: E402
from runtime.training.phases.dataset import build_dataloader  # noqa: E402
from tests.test_loop_bucket_switch_cache import _make_ctx  # noqa: E402

N_ITEMS = 6


class _IdxDataset(Dataset):
    """Each latent is set by its index; __getitem__ rolls Python random once to mimic a flip
    augmentation, so whether skipped batches go through __getitem__ shows up in the random stream."""

    def __len__(self):
        return N_ITEMS

    def __getitem__(self, idx):
        sign = -1.0 if random.random() < 0.5 else 1.0
        g = torch.Generator().manual_seed(1000 + idx)
        return {"latent": sign * torch.randn(4, 1, 8, 8, generator=g), "caption": str(idx)}


def _collate(items):
    return {
        "latents": torch.stack([it["latent"] for it in items]),
        "captions": [it["caption"] for it in items],
    }


class _CpuRandTimestepSampler:
    """Samples from the global CPU RNG (like InfoNoise): a DataLoader drawing from it changes t."""

    def sample(self, bs, device):
        return torch.rand(bs).clamp(0.05, 0.95)

    def record(self, t, mse):
        pass

    def maybe_refresh(self, step):
        pass

    def status(self):
        return {"kind": "fake"}


class _StateInjector:
    """Injector stand-in with state_dict / load_state_dict for the real save/load_training_state."""

    def __init__(self, model):
        self.model = model

    def state_dict(self):
        return {k: v.detach().clone() for k, v in self.model.state_dict().items()}

    def load_state_dict(self, sd, strict=False):
        return self.model.load_state_dict(sd, strict=strict)

    def on_step_begin(self, step_ctx):
        pass

    def regularization_loss(self, step_ctx):
        return None

    def save(self, path):
        pass


def _ctx(tmp_path, monkeypatch, *, grad_accum, save_every, resume_state=""):
    ctx = _make_ctx(
        tmp_path, None, monkeypatch,
        epochs=2, grad_accum=grad_accum, save_state_every_steps=save_every,
        resume_state=resume_state, seed=7, batch_size=1,
        no_progress=True, no_live_curve=True,
        sample_prompts=[], sample_prompt="",
    )
    # _make_ctx stubs out save_training_state; this test saves and loads for real
    monkeypatch.undo()
    monkeypatch.setattr(loop_mod, "write_config_snapshot", lambda *a, **k: None)
    monkeypatch.setattr(loop_mod, "emit_event", lambda *a, **k: None)
    monkeypatch.setattr(resume_mod, "emit_event", lambda *a, **k: None)
    monkeypatch.setattr(resume_mod.signal, "signal", lambda *a, **k: None)

    ds = _IdxDataset()
    sampler = BucketBatchSampler(ds, batch_size=1, drop_last=False, shuffle=True, seed=7)
    ctx.dataloader = build_dataloader(ds, sampler, _collate, num_workers=0, seed=7)
    ctx.timestep_sampler = _CpuRandTimestepSampler()
    ctx.injector = _StateInjector(ctx.model)
    ctx.optimizer = torch.optim.AdamW(ctx.model.parameters(), lr=0.05)
    ctx.monitor_server = None
    ctx.family.spec = types.SimpleNamespace(family_id="anima")
    return ctx


def _run(ctx, seed=7):
    torch.manual_seed(seed)
    random.seed(seed)
    resume_mod.run(ctx)
    loop_mod.run(ctx)
    return list(ctx.loss_history), float(ctx.model.w.detach())


@pytest.mark.parametrize(
    "grad_accum,save_every,resume_step",
    [
        (1, 2, 2),   # middle of epoch 0
        (1, 2, 8),   # middle of epoch 1
        (1, 3, 6),   # exactly the last batch of epoch 0 (offset == dl_len)
        (2, 1, 2),   # gradient accumulation: offset = 4 micro-batches
    ],
)
def test_mid_epoch_resume_matches_uninterrupted(tmp_path, monkeypatch, grad_accum, save_every, resume_step):
    ref_ctx = _ctx(tmp_path / "ref", monkeypatch, grad_accum=grad_accum, save_every=save_every)
    ref_losses, ref_w = _run(ref_ctx)
    state = ref_ctx.state_dir() / f"training_state_step{resume_step}.pt"
    assert state.exists()

    res_ctx = _ctx(
        tmp_path / "res", monkeypatch,
        grad_accum=grad_accum, save_every=0, resume_state=str(state),
    )
    # scramble the global RNG first: after resume the random stream may only come from the recovery point
    res_losses, res_w = _run(res_ctx, seed=12345)

    assert res_losses == ref_losses
    assert res_w == ref_w
    assert res_ctx.global_step == ref_ctx.global_step


def test_bucket_sampler_skip_batches_is_one_shot():
    ds = _IdxDataset()
    sampler = BucketBatchSampler(ds, batch_size=1, drop_last=False, shuffle=True, seed=3)
    sampler.set_epoch(1)
    full = list(sampler)
    sampler.skip_batches(4)
    assert list(sampler) == full[4:]
    assert len(sampler) == len(full)  # length is still the full epoch (batch_idx starts at the offset)
    assert list(sampler) == full      # one-shot


def test_navit_sampler_skip_batches_is_one_shot():
    ds = _IdxDataset()
    ds.token_count_for_index = [16 + 8 * i for i in range(N_ITEMS)]
    sampler = NavitPackBatchSampler(ds, token_budget=48, shuffle=True, seed=3)
    sampler.set_epoch(2)
    full = list(sampler)
    assert len(full) > 2
    sampler.set_epoch(2)
    sampler.skip_batches(2)
    assert list(sampler) == full[2:]
    assert list(sampler) == full


def test_dataloader_iter_does_not_consume_global_rng():
    ds = _IdxDataset()
    sampler = BucketBatchSampler(ds, batch_size=1, drop_last=False, shuffle=False)
    dl = build_dataloader(ds, sampler, _collate, num_workers=0, seed=0)
    torch.manual_seed(0)
    expected = torch.rand(1)
    torch.manual_seed(0)
    iter(dl)
    assert torch.equal(torch.rand(1), expected)
