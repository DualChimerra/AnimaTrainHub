from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from studio import secrets
from studio.services.tagging import wd14 as wd14_tagger


@pytest.fixture
def isolated_secrets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    sf = tmp_path / "secrets.json"
    monkeypatch.setattr(secrets, "SECRETS_FILE", sf)
    return tmp_path


def _make_local_model(model_dir: Path, tags: list[tuple[str, int]]) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.onnx").write_bytes(b"fake-onnx")
    with open(model_dir / "selected_tags.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag_id", "name", "category"])
        for i, (n, c) in enumerate(tags):
            w.writerow([i, n, c])


def test_resolve_default_models_root(
    isolated_secrets: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = isolated_secrets / "models"
    monkeypatch.setattr(wd14_tagger.model_downloader, "models_root", lambda: root)
    model_id = secrets.load().wd14.model_id
    target = wd14_tagger.model_downloader.wd14_target_dir(root, model_id)
    _make_local_model(target, [("a", 0)])
    t = wd14_tagger.WD14Tagger()
    assert t._resolve_model_dir() == target


def test_postprocess_filters_by_threshold(isolated_secrets: Path) -> None:
    secrets.update({
        "wd14": {
            "threshold_general": 0.5,
            "threshold_character": 0.85,
            "blacklist_tags": ["banned"],
        }
    })
    t = wd14_tagger.WD14Tagger()
    t._tags = ["1girl", "solo", "banned", "char_a", "rating"]
    t._tag_categories = [0, 0, 0, 4, 9]  # 9 = rating, 4 = character
    scores = np.array([0.9, 0.4, 0.99, 0.7, 0.95])
    tags, raw = t._postprocess_one(scores)
    # 1girl (0.9 > 0.5 ✓), solo (0.4 < 0.5 ✗), banned (blacklist),
    # char_a (0.7 < 0.85 ✗), rating (cat=9 → drop)
    assert tags == ["1girl"]
    assert raw == {"1girl": pytest.approx(0.9)}


def test_postprocess_blacklist_underscore_and_case_insensitive(isolated_secrets: Path) -> None:
    secrets.update({
        "wd14": {
            "threshold_general": 0.1, "threshold_character": 0.1,
            "blacklist_tags": ["cat_girl", "Blue Eyes"],
        }
    })
    t = wd14_tagger.WD14Tagger()
    t._tags = ["cat girl", "blue eyes", "1girl"]
    t._tag_categories = [0, 0, 0]
    scores = np.array([0.9, 0.9, 0.9])
    tags, _ = t._postprocess_one(scores)
    assert tags == ["1girl"]


def test_postprocess_sorts_by_score_desc(isolated_secrets: Path) -> None:
    secrets.update({"wd14": {"threshold_general": 0.1, "threshold_character": 0.1}})
    t = wd14_tagger.WD14Tagger()
    t._tags = ["a", "b", "c"]
    t._tag_categories = [0, 0, 0]
    scores = np.array([0.3, 0.9, 0.5])
    tags, _ = t._postprocess_one(scores)
    assert tags == ["b", "c", "a"]


def test_preprocess_pads_to_square(isolated_secrets: Path) -> None:
    t = wd14_tagger.WD14Tagger()
    t._input_size = 16
    img = Image.new("RGB", (10, 4), (255, 0, 0))
    arr = t._preprocess(img)
    assert arr.shape == (16, 16, 3)
    assert arr[8, 8, 2] == pytest.approx(255.0, abs=1.0)


def test_overrides_replace_thresholds(isolated_secrets: Path) -> None:
    secrets.update({
        "wd14": {
            "threshold_general": 0.5,
            "threshold_character": 0.85,
            "blacklist_tags": [],
        }
    })
    t = wd14_tagger.WD14Tagger(
        overrides={"threshold_general": 0.2, "blacklist_tags": ["solo"]}
    )
    t._tags = ["1girl", "solo", "rare"]
    t._tag_categories = [0, 0, 0]
    scores = np.array([0.3, 0.9, 0.25])
    tags, _ = t._postprocess_one(scores)
    assert sorted(tags) == ["1girl", "rare"]
    assert secrets.load().wd14.threshold_general == 0.5
    assert secrets.load().wd14.blacklist_tags == []


def test_overrides_none_falls_back_to_global(isolated_secrets: Path) -> None:
    secrets.update({"wd14": {"threshold_general": 0.7}})
    t = wd14_tagger.WD14Tagger(overrides={"threshold_general": None})
    cfg = t._cfg()
    assert cfg.threshold_general == 0.7


def test_tag_iterator_handles_io_error(
    isolated_secrets: Path, tmp_path: Path
) -> None:
    t = wd14_tagger.WD14Tagger()
    t._session = MagicMock()
    t._session.get_providers.return_value = ["CPUExecutionProvider"]
    t._session.run.return_value = (np.array([[0.9]]),)
    t._tags = ["x"]
    t._tag_categories = [0]
    t._input_name = "input"
    t._input_size = 4

    secrets.update({"wd14": {"threshold_general": 0.1, "threshold_character": 0.1}})

    good = tmp_path / "good.png"
    Image.new("RGB", (8, 8)).save(good)
    bad = tmp_path / "ghost.png"

    results = list(t.tag([good, bad]))
    assert len(results) == 2
    assert results[0]["tags"] == ["x"]
    assert "error" in results[1]


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_tag_batch_processes_chunks(
    isolated_secrets: Path, tmp_path: Path
) -> None:
    secrets.update({
        "wd14": {
            "threshold_general": 0.1,
            "threshold_character": 0.1,
            "batch_size": 4,
        }
    })
    t = wd14_tagger.WD14Tagger()
    t._session = MagicMock()
    t._session.get_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def _fake_run(_outputs, feeds):
        n = feeds["input"].shape[0]
        return (np.array([[0.5 + i * 0.1] for i in range(n)]),)

    t._session.run.side_effect = _fake_run
    t._tags = ["x"]
    t._tag_categories = [0]
    t._input_name = "input"
    t._input_size = 4

    paths = []
    for i in range(5):
        p = tmp_path / f"img{i}.png"
        Image.new("RGB", (8, 8), (i * 30, 0, 0)).save(p)
        paths.append(p)

    results = list(t.tag(paths))
    assert len(results) == 5
    assert all(r["tags"] == ["x"] for r in results)
    assert t._session.run.call_count == 2
    first_call_batch = t._session.run.call_args_list[0][0][1]["input"]
    assert first_call_batch.shape == (4, 4, 4, 3)
    second_call_batch = t._session.run.call_args_list[1][0][1]["input"]
    assert second_call_batch.shape == (1, 4, 4, 3)


def test_tag_batch_falls_back_to_one_on_cpu(
    isolated_secrets: Path, tmp_path: Path
) -> None:
    secrets.update({"wd14": {"threshold_general": 0.1, "batch_size": 8}})
    t = wd14_tagger.WD14Tagger()
    t._session = MagicMock()
    t._session.get_providers.return_value = ["CPUExecutionProvider"]
    t._session.run.return_value = (np.array([[0.9]]),)
    t._tags = ["x"]
    t._tag_categories = [0]
    t._input_name = "input"
    t._input_size = 4

    paths = []
    for i in range(3):
        p = tmp_path / f"img{i}.png"
        Image.new("RGB", (8, 8)).save(p)
        paths.append(p)

    list(t.tag(paths))
    assert t._session.run.call_count == 3
    for call in t._session.run.call_args_list:
        assert call[0][1]["input"].shape == (1, 4, 4, 3)


def test_tag_batch_keeps_decode_errors(
    isolated_secrets: Path, tmp_path: Path
) -> None:
    secrets.update({"wd14": {"threshold_general": 0.1, "batch_size": 4}})
    t = wd14_tagger.WD14Tagger()
    t._session = MagicMock()
    t._session.get_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    t._session.run.return_value = (np.array([[0.9], [0.95]]),)
    t._tags = ["x"]
    t._tag_categories = [0]
    t._input_name = "input"
    t._input_size = 4

    good1 = tmp_path / "g1.png"
    good2 = tmp_path / "g2.png"
    bad = tmp_path / "missing.png"
    Image.new("RGB", (8, 8)).save(good1)
    Image.new("RGB", (8, 8)).save(good2)

    results = list(t.tag([good1, bad, good2]))
    assert len(results) == 3
    assert results[0]["tags"] == ["x"]
    assert "error" in results[1]
    assert results[2]["tags"] == ["x"]
    assert t._session.run.call_count == 1
    fed = t._session.run.call_args[0][1]["input"]
    assert fed.shape == (2, 4, 4, 3)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_preprocess_runs_concurrently_on_gpu_ep(
    isolated_secrets: Path, tmp_path: Path
) -> None:
    import threading

    secrets.update({"wd14": {"threshold_general": 0.1, "batch_size": 4}})
    t = wd14_tagger.WD14Tagger()
    t._session = MagicMock()
    t._session.get_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    t._session.run.return_value = (np.array([[0.9], [0.9], [0.9], [0.9]]),)
    t._tags = ["x"]
    t._tag_categories = [0]
    t._input_name = "input"
    t._input_size = 4

    barrier = threading.Barrier(parties=4, timeout=2.0)
    seen_threads: set[str] = set()
    seen_threads_lock = threading.Lock()

    real_preprocess = t._preprocess

    def gated_preprocess(img):
        with seen_threads_lock:
            seen_threads.add(threading.current_thread().name)
        barrier.wait()
        return real_preprocess(img)

    t._preprocess = gated_preprocess  # type: ignore[method-assign]

    paths = []
    for i in range(4):
        p = tmp_path / f"img{i}.png"
        Image.new("RGB", (8, 8)).save(p)
        paths.append(p)

    results = list(t.tag(paths))
    assert len(results) == 4
    assert all(r["tags"] == ["x"] for r in results)
    prep_threads = {n for n in seen_threads if n.startswith("wd14-prep")}
    assert len(prep_threads) >= 2, f"expected concurrent preprocess threads, saw {seen_threads}"


def test_cpu_ep_path_does_not_spawn_pool(
    isolated_secrets: Path, tmp_path: Path
) -> None:
    import threading

    secrets.update({"wd14": {"threshold_general": 0.1, "batch_size": 8}})
    t = wd14_tagger.WD14Tagger()
    t._session = MagicMock()
    t._session.get_providers.return_value = ["CPUExecutionProvider"]
    t._session.run.return_value = (np.array([[0.9]]),)
    t._tags = ["x"]
    t._tag_categories = [0]
    t._input_name = "input"
    t._input_size = 4

    seen_threads: set[str] = set()
    real_preprocess = t._preprocess

    def tracking_preprocess(img):
        seen_threads.add(threading.current_thread().name)
        return real_preprocess(img)

    t._preprocess = tracking_preprocess  # type: ignore[method-assign]

    paths = []
    for i in range(3):
        p = tmp_path / f"img{i}.png"
        Image.new("RGB", (8, 8)).save(p)
        paths.append(p)

    list(t.tag(paths))
    prep_threads = {n for n in seen_threads if n.startswith("wd14-prep")}
    assert prep_threads == set(), f"CPU 路径不应开 pool，saw {seen_threads}"


def test_preprocess_concurrent_preserves_chunk_order(
    isolated_secrets: Path, tmp_path: Path
) -> None:
    secrets.update({"wd14": {"threshold_general": 0.1, "batch_size": 4}})
    t = wd14_tagger.WD14Tagger()
    t._session = MagicMock()
    t._session.get_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    t._session.run.return_value = (
        np.array([[0.5], [0.6], [0.7], [0.8]]),
    )
    t._tags = ["x"]
    t._tag_categories = [0]
    t._input_name = "input"
    t._input_size = 4

    paths = []
    for i in range(4):
        p = tmp_path / f"img{i}.png"
        Image.new("RGB", (8, 8), (i * 60, 0, 0)).save(p)
        paths.append(p)

    results = list(t.tag(paths))
    for k, r in enumerate(results):
        assert r["image"] == paths[k]
