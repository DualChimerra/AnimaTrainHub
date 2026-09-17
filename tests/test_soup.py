"""Checkpoint soup: the merge must be arithmetically exact and must refuse
anything that would produce a file that only *looks* valid.

The dangerous failure mode here is a silent one — averaging two adapters with
different ranks or different target modules writes a file that loads and then
generates noise. So most of these tests are about the refusals.
"""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
from safetensors.torch import load_file, save_file  # noqa: E402

from studio.services import soup  # noqa: E402


@pytest.fixture(autouse=True)
def _tmp_soup_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(soup, "SOUP_DIR", tmp_path / "soup")
    monkeypatch.setattr(soup, "UPLOAD_DIR", tmp_path / "soup" / "uploads")
    monkeypatch.setattr(soup, "OUTPUT_DIR", tmp_path / "soup" / "output")
    soup.ensure_dirs()
    return tmp_path


def _write(path, tensors, meta=None):
    save_file(tensors, str(path), metadata=meta or {})
    return path


def _lora(tmp_path, name, *, value=1.0, rank=4, alpha=4.0, algo="lokr", dtype=torch.bfloat16):
    tensors = {
        "lora_unet_blocks_0.lokr_w1": torch.full((rank, rank), value, dtype=dtype),
        "lora_unet_blocks_0.lokr_w2": torch.full((rank, 8), value, dtype=dtype),
        "lora_unet_blocks_0.alpha": torch.tensor(alpha, dtype=torch.float32),
    }
    meta = {
        "ss_network_dim": str(rank),
        "ss_network_alpha": str(alpha),
        "ss_network_module": "lycoris.kohya",
        "ss_network_args": json.dumps({"algo": algo, "factor": 8}),
    }
    return _write(tmp_path / name, tensors, meta)


# ── naming / paths ───────────────────────────────────────────────────────────

def test_filename_is_forced_to_safetensors():
    assert soup.safe_filename("my soup") == "my soup.safetensors"
    assert soup.safe_filename("my soup.safetensors") == "my soup.safetensors"


@pytest.mark.parametrize("bad", ["", "   ", "../escape", "a/b", "nul\0l", "x" * 200])
def test_bad_names_are_rejected(bad):
    with pytest.raises(soup.SoupError):
        soup.safe_filename(bad)


def test_traversal_cannot_escape_the_output_dir():
    """A crafted name must never resolve outside the soup folder."""
    with pytest.raises(soup.SoupError):
        soup.output_path("../../etc/passwd")


def test_unique_name_does_not_overwrite(tmp_path):
    first = soup.output_path("soup")
    first.write_bytes(b"x")
    assert soup.unique_output_path("soup").name == "soup (2).safetensors"


# ── inspection ───────────────────────────────────────────────────────────────

def test_inspect_reads_the_fingerprint_without_loading(tmp_path):
    p = _lora(tmp_path, "a.safetensors", rank=8, alpha=4.0, algo="lokr")
    info = soup.inspect(p)
    assert info["algo"] == "lokr"
    assert info["rank"] == 8.0
    assert info["alpha"] == 4.0
    assert info["tensor_count"] == 3
    assert info["dtypes"] == ["BF16", "F32"] or "BF16" in info["dtypes"]


def test_inspect_rejects_non_safetensors(tmp_path):
    p = tmp_path / "weights.ckpt"
    p.write_bytes(b"x")
    with pytest.raises(soup.SoupError):
        soup.inspect(p)


def test_inspect_rejects_a_truncated_file(tmp_path):
    p = tmp_path / "broken.safetensors"
    p.write_bytes(b"not really a safetensors header")
    with pytest.raises(soup.SoupError):
        soup.inspect(p)


# ── compatibility ────────────────────────────────────────────────────────────

def test_identical_adapters_are_compatible(tmp_path):
    a = _lora(tmp_path, "a.safetensors")
    b = _lora(tmp_path, "b.safetensors", value=2.0)
    v = soup.compatibility([a, b])
    assert v["ok"] and not v["errors"] and not v["warnings"]


def test_different_rank_is_refused(tmp_path):
    a = _lora(tmp_path, "a.safetensors", rank=4)
    b = _lora(tmp_path, "b.safetensors", rank=8)
    v = soup.compatibility([a, b])
    assert not v["ok"]
    assert "rank" in v["errors"][0].lower()


def test_different_target_modules_are_refused(tmp_path):
    a = _lora(tmp_path, "a.safetensors")
    b = _lora(tmp_path, "b.safetensors")
    extra = load_file(str(b))
    extra["lora_unet_blocks_9.lokr_w1"] = torch.zeros(4, 4)
    save_file(extra, str(b))
    v = soup.compatibility([a, b])
    assert not v["ok"]
    assert "different modules" in v["errors"][0]


def test_different_algo_is_refused(tmp_path):
    a = _lora(tmp_path, "a.safetensors", algo="lokr")
    b = _lora(tmp_path, "b.safetensors", algo="lora")
    v = soup.compatibility([a, b])
    assert not v["ok"]


def test_different_alpha_warns_but_allows(tmp_path):
    """Alpha only rescales; the user can compensate with LoRA strength, so this
    is information, not a refusal."""
    a = _lora(tmp_path, "a.safetensors", alpha=4.0)
    b = _lora(tmp_path, "b.safetensors", alpha=8.0)
    v = soup.compatibility([a, b])
    assert v["ok"]
    assert any("alpha" in w for w in v["warnings"])


def test_one_file_is_not_a_soup(tmp_path):
    a = _lora(tmp_path, "a.safetensors")
    with pytest.raises(soup.SoupError):
        soup.compatibility([a])


# ── weights ──────────────────────────────────────────────────────────────────

def test_average_normalizes_to_one():
    assert soup.normalize_weights([1.0, 1.0], "average") == [0.5, 0.5]
    assert soup.normalize_weights([3.0, 1.0], "average") == [0.75, 0.25]


def test_sum_leaves_weights_alone():
    assert soup.normalize_weights([0.7, 0.4], "sum") == [0.7, 0.4]


def test_weights_summing_to_zero_are_refused():
    with pytest.raises(soup.SoupError):
        soup.normalize_weights([1.0, -1.0], "average")


# ── the merge ────────────────────────────────────────────────────────────────

def test_equal_average_is_the_midpoint(tmp_path):
    a = _lora(tmp_path, "a.safetensors", value=1.0)
    b = _lora(tmp_path, "b.safetensors", value=3.0)
    out = soup.merge([{"path": str(a), "weight": 1}, {"path": str(b), "weight": 1}], "mix")
    merged = load_file(out["path"])
    assert torch.allclose(
        merged["lora_unet_blocks_0.lokr_w1"].float(),
        torch.full((4, 4), 2.0),
    )


def test_weights_are_respected(tmp_path):
    a = _lora(tmp_path, "a.safetensors", value=0.0)
    b = _lora(tmp_path, "b.safetensors", value=4.0)
    out = soup.merge([{"path": str(a), "weight": 3}, {"path": str(b), "weight": 1}], "mix")
    merged = load_file(out["path"])
    assert torch.allclose(merged["lora_unet_blocks_0.lokr_w1"].float(), torch.full((4, 4), 1.0))


def test_sum_mode_does_not_normalize(tmp_path):
    a = _lora(tmp_path, "a.safetensors", value=1.0)
    b = _lora(tmp_path, "b.safetensors", value=1.0)
    out = soup.merge(
        [{"path": str(a), "weight": 1}, {"path": str(b), "weight": 1}],
        "mix", method="sum",
    )
    merged = load_file(out["path"])
    assert torch.allclose(merged["lora_unet_blocks_0.lokr_w1"].float(), torch.full((4, 4), 2.0))


def test_alpha_is_carried_over_not_averaged(tmp_path):
    """Averaging alpha would silently rescale every layer of the result."""
    a = _lora(tmp_path, "a.safetensors", alpha=4.0)
    b = _lora(tmp_path, "b.safetensors", alpha=64.0)
    out = soup.merge([{"path": str(a), "weight": 1}, {"path": str(b), "weight": 1}], "mix")
    merged = load_file(out["path"])
    assert float(merged["lora_unet_blocks_0.alpha"]) == 4.0


def test_dtype_of_the_first_file_is_preserved(tmp_path):
    a = _lora(tmp_path, "a.safetensors", dtype=torch.bfloat16)
    b = _lora(tmp_path, "b.safetensors", dtype=torch.bfloat16)
    out = soup.merge([{"path": str(a), "weight": 1}, {"path": str(b), "weight": 1}], "mix")
    merged = load_file(out["path"])
    assert merged["lora_unet_blocks_0.lokr_w1"].dtype == torch.bfloat16


def test_merge_records_its_recipe(tmp_path):
    """A merged file is anonymous three weeks later without this."""
    from safetensors import safe_open
    a = _lora(tmp_path, "a.safetensors")
    b = _lora(tmp_path, "b.safetensors")
    out = soup.merge([{"path": str(a), "weight": 3}, {"path": str(b), "weight": 1}], "mix")
    with safe_open(out["path"], framework="pt") as f:
        meta = f.metadata()
    assert meta["ss_soup_method"] == "average"
    sources = json.loads(meta["ss_soup_sources"])
    assert [s["name"] for s in sources] == ["a.safetensors", "b.safetensors"]
    assert sources[0]["effective_weight"] == 0.75
    # The network fingerprint must survive, or inference cannot rebuild it.
    assert meta["ss_network_module"] == "lycoris.kohya"
    assert meta["ss_network_dim"] == "4"


def test_merge_refuses_incompatible_inputs(tmp_path):
    a = _lora(tmp_path, "a.safetensors", rank=4)
    b = _lora(tmp_path, "b.safetensors", rank=8)
    with pytest.raises(soup.SoupError):
        soup.merge([{"path": str(a), "weight": 1}, {"path": str(b), "weight": 1}], "mix")


def test_merge_does_not_clobber_by_default(tmp_path):
    a = _lora(tmp_path, "a.safetensors")
    b = _lora(tmp_path, "b.safetensors")
    first = soup.merge([{"path": str(a), "weight": 1}, {"path": str(b), "weight": 1}], "mix")
    second = soup.merge([{"path": str(a), "weight": 1}, {"path": str(b), "weight": 1}], "mix")
    assert first["name"] != second["name"]


def test_three_way_soup(tmp_path):
    files = [_lora(tmp_path, f"{n}.safetensors", value=v) for n, v in
             (("a", 1.0), ("b", 2.0), ("c", 6.0))]
    out = soup.merge([{"path": str(p), "weight": 1} for p in files], "mix")
    merged = load_file(out["path"])
    assert torch.allclose(merged["lora_unet_blocks_0.lokr_w1"].float(), torch.full((4, 4), 3.0))


# ── uploads ──────────────────────────────────────────────────────────────────

def test_upload_round_trip(tmp_path):
    src = _lora(tmp_path, "mine.safetensors")
    saved = soup.save_upload("mine", src.read_bytes())
    assert saved["name"] == "mine.safetensors"
    assert [i["name"] for i in soup.list_uploads()] == ["mine.safetensors"]
    soup.delete_upload("mine")
    assert soup.list_uploads() == []


def test_upload_rejects_a_file_that_is_not_an_adapter(tmp_path):
    """Catch it at the door, not two screens later inside the merge."""
    with pytest.raises(soup.SoupError):
        soup.save_upload("junk", b"this is not a safetensors file")
    assert soup.list_uploads() == []


def test_upload_rejects_an_empty_file():
    with pytest.raises(soup.SoupError):
        soup.save_upload("empty", b"")


def test_deleting_a_missing_file_is_an_error():
    with pytest.raises(soup.SoupError):
        soup.delete_output("never-existed")
