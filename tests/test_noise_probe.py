"""noise_probe math + noise_gap_probe glue, on CPU with analytic flow models."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dual_peak_sampling import DualPeakConfig
from noise_probe import (
    bin_edges,
    compare_images,
    config_bin_masses,
    denoise_mse,
    euler_denoise,
    fit_dual_peak,
    gap_profile,
    seeded_noise,
    shifted_sigmas,
    trait_target,
    truncate_schedule,
)

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "noise_gap_probe.py"


def _tool():
    spec = importlib.util.spec_from_file_location("noise_gap_probe", _TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gaussian_velocity(scale: float):
    """Optimal rectified-flow velocity for data ~ N(0, scale^2 I)."""
    def velocity(x, t, cond):
        tt = t.view(-1, *([1] * (x.dim() - 1))).to(x.dtype)
        s2 = scale ** 2
        x0_hat = (1 - tt) * s2 / ((1 - tt) ** 2 * s2 + tt ** 2) * x
        return (x - x0_hat) / tt
    return velocity


def test_schedule_shape_and_truncation():
    sigmas = shifted_sigmas(30, 3.0)
    assert float(sigmas[0]) == 1.0 and float(sigmas[-1]) == 0.0
    assert bool((sigmas[1:] < sigmas[:-1]).all())
    rung = truncate_schedule(sigmas, 0.6)
    assert float(rung[0]) == 0.6 and float(rung[-1]) == 0.0
    assert bool((rung[1:] < rung[:-1]).all())
    assert torch.equal(truncate_schedule(sigmas, 1.0), sigmas)
    with pytest.raises(ValueError):
        truncate_schedule(sigmas, 0.0)


def test_euler_recovers_point_mass_from_any_t0():
    x0 = torch.randn(1, 4, 1, 8, 8)
    oracle = lambda x, t, cond: (x - x0) / t.view(-1, 1, 1, 1, 1)  # noqa: E731
    noise = seeded_noise(x0, 3)
    for t0 in (0.3, 0.8, 1.0):
        x_t = (1 - t0) * x0 + t0 * noise
        out = euler_denoise(oracle, x_t, truncate_schedule(shifted_sigmas(10, 3.0), t0), None)
        torch.testing.assert_close(out, x0, atol=1e-4, rtol=0)


def test_cfg_combines_branches():
    x = torch.ones(1, 1, 1, 2, 2)
    v = lambda x, t, cond: torch.full_like(x, float(cond))  # noqa: E731
    out = euler_denoise(v, x, torch.tensor([1.0, 0.0], dtype=torch.float64), 2.0, uncond=1.0, cfg_scale=3.0)
    torch.testing.assert_close(out, x - (1.0 + 3.0 * (2.0 - 1.0)))


def test_seeded_noise_is_device_and_dtype_independent():
    like = torch.zeros(1, 2, 1, 3, 3)
    torch.testing.assert_close(seeded_noise(like, 5), seeded_noise(like.double(), 5).float())
    assert not torch.equal(seeded_noise(like, 5), seeded_noise(like, 6))


def test_gap_is_zero_for_identical_sets_and_positive_for_off_prior_data():
    velocity = gaussian_velocity(1.0)
    gen = torch.Generator().manual_seed(0)
    own = [(torch.randn(1, 4, 1, 16, 16, generator=gen), None) for _ in range(6)]
    same = gap_profile(velocity, own, own, edges=bin_edges(10), seeds=(0,))
    torch.testing.assert_close(same.relative_gap, torch.zeros(10, dtype=torch.float64))
    torch.testing.assert_close(same.target_masses(), torch.full((10,), 0.1, dtype=torch.float64))

    # The "artist" has 3x the contrast the model expects: harder everywhere
    # except near pure noise, where the target is mostly the noise itself.
    artist = [(3 * torch.randn(1, 4, 1, 16, 16, generator=gen), None) for _ in range(6)]
    prof = gap_profile(velocity, artist, own, edges=bin_edges(10), seeds=(0, 1))
    assert float(prof.relative_gap[:8].min()) > 0
    masses = prof.target_masses()
    assert abs(float(masses.sum()) - 1) < 1e-9 and float(masses.min()) > 0


def test_denoise_mse_is_zero_for_oracle():
    x0 = torch.randn(1, 2, 1, 4, 4)
    oracle = lambda x, t, cond: (x - x0) / t.view(-1, 1, 1, 1, 1)  # noqa: E731
    mse = denoise_mse(oracle, x0, None, [0.2, 0.5, 0.9], seeds=(0, 1))
    assert float(mse.max()) < 1e-10


def test_fit_recovers_shipped_default_shape():
    edges = bin_edges(20)
    target = config_bin_masses(DualPeakConfig(), edges)
    assert abs(float(target.sum()) - 1) < 1e-9
    fitted, err = fit_dual_peak(target, edges)
    assert err < 0.05
    assert fitted.peak1_position < fitted.peak2_position
    assert abs(sum(fitted.weights) - 1) < 0.01


def test_fit_moves_toward_a_low_noise_target():
    edges = bin_edges(20)
    target = torch.zeros(20, dtype=torch.float64)
    target[6:10] = 1.0  # all mass on t 0.3-0.5
    fitted, _ = fit_dual_peak(target, edges)
    masses = config_bin_masses(fitted, edges)
    assert float(masses[6:10].sum()) > 0.5


def test_metrics_identity_and_directions():
    gen = torch.Generator().manual_seed(1)
    img = torch.rand(3, 128, 96, generator=gen)
    same = compare_images(img, img)
    assert same["palette"] < 1e-6 and same["shapes"] > 0.999 and same["layout"] > 0.999
    assert abs(same["detail"] - 1) < 1e-6
    blurred = torch.nn.functional.avg_pool2d(img[None], 7, stride=1, padding=3, count_include_pad=False)[0]
    assert compare_images(img, blurred)["detail"] < 0.5
    tinted = (img * torch.tensor([1.0, 0.7, 0.5]).view(3, 1, 1)).clamp(0, 1)
    tint = compare_images(img, tinted)
    assert tint["palette"] > 5 and tint["shapes"] > 0.9
    with pytest.raises(ValueError):
        compare_images(img, img[:, :64])


def test_strip_trigger_and_evenly():
    tool = _tool()
    assert tool.strip_trigger("@artist, 1girl, solo", "@artist") == "1girl, solo"
    assert tool.strip_trigger("1girl, @artist, solo", "@artist") == "1girl, solo"
    assert tool.strip_trigger("1girl", "") == "1girl"
    assert tool.evenly(list(range(10)), 4) == [0, 2, 5, 7]
    assert tool.evenly([1, 2], 5) == [1, 2]


class _FakeModel:
    """Duck-types the tool's Model with a Gaussian-prior velocity and a sigmoid 'VAE'."""

    def __init__(self):
        self.ctx = SimpleNamespace(device="cpu", dtype=torch.float32)
        self._v = gaussian_velocity(1.0)

    def cond(self, caption):
        return caption

    def velocity(self, x, t, cond):
        return self._v(x, t, cond)

    def decode(self, latent):
        return torch.sigmoid(latent[0, :3, 0].float())

    def roundtrip(self, latent):
        return latent


def test_tool_pipeline_end_to_end_on_cpu(tmp_path):
    tool = _tool()
    gen = torch.Generator().manual_seed(2)
    images = [{"image": f"img{k}.png", "caption": f"cap {k}",
               "latent": 2.5 * torch.randn(1, 4, 1, 24, 16, generator=gen)} for k in range(3)]
    opts = tool.parse_args(["--config", "x.yaml", "--bins", "10", "--seeds", "1",
                            "--ladder-images", "2", "--ladder-t", "0.5,0.9", "--anchor", "0.3"])
    sampling = {"steps": 8, "cfg": 1.0, "shift": 3.0, "negative": ""}
    model = _FakeModel()

    profile = tool.run_profile(model, images, opts, tmp_path, sampling)
    assert (tmp_path / "reference_latents.pt").exists()
    assert len(list((tmp_path / "reference").glob("ref_*.png"))) == 3
    tool.write_profile(profile, tmp_path)
    assert len((tmp_path / "profile.csv").read_text(encoding="utf-8").splitlines()) == 11
    restored = tool.read_profile(tmp_path)
    torch.testing.assert_close(restored.relative_gap, profile.relative_gap, atol=1e-3, rtol=1e-3)

    opts.reuse_reference = True
    again = tool.run_profile(model, images, opts, tmp_path, sampling)
    torch.testing.assert_close(again.mse_reference, profile.mse_reference)

    rows = tool.run_ladder(model, images, opts, tmp_path, sampling)
    assert len(rows) == 2 * 3  # 2 images x (2 rungs + full generation)
    assert len(list((tmp_path / "ladder").glob("ladder_*.png"))) == 2
    summary = tool.ladder_summary(rows)
    assert [s["t0"] for s in summary] == [0.5, 0.9, 1.0]
    # Lower t0 keeps more of the original's layout.
    assert summary[0]["layout"] > summary[-1]["layout"]

    restored_ladder = tool.read_ladder(tmp_path)
    assert [s["t0"] for s in restored_ladder] == [s["t0"] for s in summary]
    assert restored_ladder[0]["layout"] == pytest.approx(summary[0]["layout"], abs=1e-6)

    fit = tool.write_fit(tmp_path, opts, profile, summary)
    assert isinstance(fit["fitted"], DualPeakConfig) and fit["source"].startswith("лестница")
    yaml_text = (tmp_path / "suggested_dual_peak.yaml").read_text(encoding="utf-8")
    assert "timestep_sampling: dual_peak" in yaml_text and "dual_peak_peak1_position" in yaml_text
    assert tool.write_fit(tmp_path, opts, profile, None)["source"] == "профиль разрыва"

    meta = {"config": "x.yaml", "family": "fake", "images": 3, "steps": 8, "cfg": 1.0,
            "shift": 3.0, "minutes": 0.1}
    (tmp_path / "meta.json").write_text(__import__("json").dumps(meta), encoding="utf-8")
    tool.write_report(tmp_path, meta, fit, profile, summary)
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Профиль разрыва" in report and "Лестница зашумления" in report and "dual_peak" in report

    # GPU-free refit from the CSVs of the run above.
    tool.main(["--config", "x.yaml", "--out", str(tmp_path), "--refit-only", "--bins", "10",
               "--trait-weights", "palette=1,shapes=1,layout=0"])
    assert "layout=0" in (tmp_path / "report.md").read_text(encoding="utf-8")


def test_trait_target_places_mass_where_each_trait_changes():
    edges = bin_edges(20)
    t0s = [0.3, 0.6, 0.8, 0.9, 1.0]
    # Palette only changes above 0.8; layout only between 0.3 and 0.6.
    values = {"palette": [0, 0, 0, 5, 10], "layout": [1, 0.0, 0.0, 0.0, 0.0]}
    palette = trait_target(t0s, values, edges, {"palette": 1})
    assert float(palette[16:].sum()) == pytest.approx(1.0)
    layout = trait_target(t0s, values, edges, {"layout": 1})
    assert float(layout[6:12].sum()) == pytest.approx(1.0)
    mixed = trait_target(t0s, values, edges, {"palette": 1, "layout": 1})
    torch.testing.assert_close(mixed, (palette + layout) / 2)
    # Non-monotone noise in a metric never produces negative mass.
    noisy = trait_target(t0s, {"shapes": [0.9, 0.95, 0.6, 0.5, 0.4]}, edges, {"shapes": 1})
    assert float(noisy.min()) >= 0 and float(noisy.sum()) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        trait_target(t0s, {"palette": [0, 0, 0, 0, 0]}, edges, {"palette": 1})


def test_scheduled_velocity_and_effect_share():
    from noise_probe import effect_share, scheduled_velocity

    seen = []
    state = {"scale": None}
    velocity = scheduled_velocity(lambda x, t, c: x * state["scale"],
                                  lambda s: (state.__setitem__("scale", s), seen.append(s)),
                                  lambda t: 1.0 if t >= 0.5 else 0.0)
    x = torch.ones(1, 1, 1, 2, 2)
    assert float(velocity(x, torch.tensor([0.7]), None).sum()) == 4.0
    assert float(velocity(x, torch.tensor([0.2]), None).sum()) == 0.0
    assert seen == [1.0, 0.0]

    none = {"palette": 8.0, "shapes": 0.4, "layout": 0.2}
    assert effect_share({"palette": 0.0, "shapes": 1.0, "layout": 1.0}, none) == {
        "palette": 1.0, "shapes": 1.0, "layout": 1.0}
    half = effect_share({"palette": 4.0, "shapes": 0.7, "layout": 0.6}, none)
    assert half == pytest.approx({"palette": 0.5, "shapes": 0.5, "layout": 0.5})


def test_lora_mode_end_to_end_on_cpu(tmp_path, monkeypatch):
    """A fake 'LoRA' that shifts the model's prior; windows and profile run through."""
    tool = _tool()
    import studio.services.inference.core as inference_core

    state = {"scale": 0.0}

    class _Adapter:
        def set_multiplier(self, s):
            state["scale"] = s

    monkeypatch.setattr(inference_core, "apply_loras", lambda *a, **k: [_Adapter()])

    class _LoraModel(_FakeModel):
        def __init__(self):
            super().__init__()
            self.ctx = SimpleNamespace(device="cpu", dtype=torch.float32, model=None, text_stack=None,
                                       family=SimpleNamespace(spec=SimpleNamespace(family_id="anima")))

        def velocity(self, x, t, cond):
            # The "LoRA" teaches a higher-contrast prior (closer to the 2.5x data below).
            return gaussian_velocity(1.0 + 1.5 * state["scale"])(x, t, cond)

    gen = torch.Generator().manual_seed(4)
    images = [{"image": f"img{k}.png", "caption": f"cap {k}",
               "latent": 2.5 * torch.randn(1, 4, 1, 24, 16, generator=gen)} for k in range(2)]
    opts = tool.parse_args(["--config", "x.yaml", "--lora", "fake.safetensors", "--bins", "10", "--seeds", "1",
                            "--ladder-images", "2", "--windows", "0.8-1,!0.8-1"])
    sampling = {"steps": 8, "cfg": 1.0, "shift": 3.0, "negative": ""}
    result = tool.run_lora_probe(_LoraModel(), images, opts, tmp_path, sampling)

    assert float(result["learned"][:8].mean()) > 0  # the LoRA's prior fits its data better
    variants = [s["variant"] for s in result["summary"]]
    assert variants == ["none", "full", "0.8-1", "!0.8-1"]
    assert len(list((tmp_path / "lora").glob("windows_*.png"))) == 2
    report = (tmp_path / "lora" / "report.md").read_text(encoding="utf-8")
    assert "Где LoRA научилась" in report and "0.8-1" in report
    assert tool.parse_windows("0.85-1, !0-0.3") == [(0.85, 1.0, False), (0.0, 0.3, True)]
    inside = tool._window_scale(0.3, 0.6, False, 1.0)
    outside = tool._window_scale(0.3, 0.6, True, 1.0)
    assert [inside(t) for t in (0.2, 0.3, 0.59, 0.6)] == [0.0, 1.0, 1.0, 0.0]
    assert [outside(t) for t in (0.2, 0.3, 0.59, 0.6)] == [1.0, 0.0, 0.0, 1.0]
    assert tool._window_scale(0.85, 1.0, False, 1.0)(1.0) == 1.0
    opts.skip_profile = True
    assert tool.run_lora_probe(_LoraModel(), images, opts, tmp_path, sampling)["learned"] is None
    with pytest.raises(ValueError):
        tool.parse_windows("0.9-0.5")
