from __future__ import annotations

import types

import pytest
import torch
from torch import nn

from utils.soap_optimizer import SOAP, SOAPScheduleFree
from utils.optimizer_utils import create_optimizer, optimizer_eval_mode


def _toy_problem(seed: int = 0):
    torch.manual_seed(seed)
    model = nn.Linear(8, 4, bias=False)
    target = nn.Linear(8, 4, bias=False)
    for p in target.parameters():
        p.requires_grad_(False)
    x = torch.randn(32, 8)
    y = target(x).detach()
    return model, x, y


def _train_n_steps(model, x, y, optimizer, n: int) -> list[float]:
    losses = []
    for _ in range(n):
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


def test_soap_reduces_loss() -> None:
    model, x, y = _toy_problem()
    opt = SOAP(model.parameters(), lr=1e-2, precondition_frequency=2, max_precond_dim=64)
    losses = _train_n_steps(model, x, y, opt, 40)
    assert losses[-1] < losses[0] * 0.5, f"SOAP failed to converge: {losses[0]} -> {losses[-1]}"


def test_soap_sf_reduces_loss() -> None:
    model, x, y = _toy_problem()
    opt = SOAPScheduleFree(model.parameters(), lr=2e-2, precondition_frequency=2, max_precond_dim=64)
    losses = _train_n_steps(model, x, y, opt, 60)
    assert losses[-1] < losses[0] * 0.5, f"SOAP-SF failed to converge: {losses[0]} -> {losses[-1]}"


def test_soap_sf_eval_swaps_to_average_and_train_swaps_back() -> None:
    model, x, y = _toy_problem()
    opt = SOAPScheduleFree(model.parameters(), lr=2e-2, precondition_frequency=2, max_precond_dim=64)
    _train_n_steps(model, x, y, opt, 20)

    w_train = next(model.parameters()).detach().clone()  # y
    opt.eval()
    w_eval = next(model.parameters()).detach().clone()   # x (averaged)
    assert not torch.allclose(w_train, w_eval), "eval() did not switch to averaged weights"
    opt.eval()
    assert torch.allclose(w_eval, next(model.parameters()).detach()), "eval() is not idempotent"
    opt.train()
    w_back = next(model.parameters()).detach().clone()   # y again
    assert torch.allclose(w_train, w_back, atol=1e-5), "train() did not switch back to the gradient point y"


def test_soap_sf_step_in_eval_mode_raises() -> None:
    model, x, y = _toy_problem()
    opt = SOAPScheduleFree(model.parameters(), lr=2e-2, max_precond_dim=64)
    _train_n_steps(model, x, y, opt, 3)
    opt.eval()
    with pytest.raises(RuntimeError, match="eval mode"):
        opt.step()


def test_optimizer_eval_mode_wraps_soap_sf() -> None:
    model, x, y = _toy_problem()
    opt = SOAPScheduleFree(model.parameters(), lr=2e-2, max_precond_dim=64)
    _train_n_steps(model, x, y, opt, 10)
    with optimizer_eval_mode(opt):
        assert opt.param_groups[0]["train_mode"] is False
    assert opt.param_groups[0]["train_mode"] is True


def test_soap_precond_in_state_false_strips_and_resumes() -> None:
    model, x, y = _toy_problem()
    opt = SOAP(model.parameters(), lr=1e-2, precondition_frequency=2,
               max_precond_dim=64, precond_in_state=False)
    initial_losses = _train_n_steps(model, x, y, opt, 10)
    sd = opt.state_dict()
    for pstate in sd["state"].values():
        assert "GG" not in pstate and "Q" not in pstate and "has_preconditioner" not in pstate

    opt2 = SOAP(model.parameters(), lr=1e-2, precondition_frequency=2,
                max_precond_dim=64, precond_in_state=False)
    opt2.load_state_dict(sd)
    losses = _train_n_steps(model, x, y, opt2, 40)
    assert all(torch.isfinite(torch.tensor(losses)))
    assert losses[-1] < initial_losses[0] * 0.8, (
        f"did not keep decreasing after resume: start={initial_losses[0]} -> end={losses[-1]}"
    )


@pytest.mark.parametrize("opt_type", ["soap", "soap_sf"])
def test_create_optimizer_dispatch(opt_type: str) -> None:
    model, _, _ = _toy_problem()
    opt = create_optimizer(opt_type, model.parameters(), learning_rate=1e-2, weight_decay=0.0)
    expected = SOAP if opt_type == "soap" else SOAPScheduleFree
    assert isinstance(opt, expected)


@pytest.mark.parametrize("opt_type", ["soap", "soap_sf"])
def test_build_optimizer_registry(opt_type: str) -> None:
    from training.optimizers import build_optimizer
    model, _, _ = _toy_problem()
    args = types.SimpleNamespace(optimizer_type=opt_type)
    opt = build_optimizer(args, list(model.parameters()), lr=1e-2, weight_decay=0.0)
    expected = SOAP if opt_type == "soap" else SOAPScheduleFree
    assert isinstance(opt, expected)


def test_soap_sf_validate_rejects_scheduler() -> None:
    from training.optimizers import soap_sf
    soap_sf.validate(types.SimpleNamespace(lr_scheduler="none"))  # ok
    with pytest.raises(SystemExit, match="lr_scheduler=none"):
        soap_sf.validate(types.SimpleNamespace(lr_scheduler="cosine"))


def test_optimizer_schema_registry_in_sync() -> None:
    from training.optimizers import validate_schema_consistency
    validate_schema_consistency()  # raises if Literal != BUILDERS
