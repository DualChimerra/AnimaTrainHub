from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from studio import db
from studio.domain.errors import InvalidPathError, NotFoundError
from studio.services.preprocess import duplicates as duplicate_finder
from studio.services.preprocess import manifest as preprocess_manifest
from studio.services.projects import projects, versions


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    monkeypatch.setattr(projects, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    with db.connection_for(dbfile) as conn:
        p = projects.create_project(conn, title="P")
        v = versions.create_version(conn, project_id=p["id"], label="v1")
    pdir = projects.project_dir(p["id"], p["slug"])
    sub = pdir / "versions" / "v1" / "train" / "1_data"
    sub.mkdir(parents=True, exist_ok=True)
    return {"db": dbfile, "p": p, "v": v, "pdir": pdir, "sub": sub}


def _png(path: Path, color: tuple[int, int, int] = (255, 0, 0)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color).save(path, "PNG")


# ---------------------------------------------------------------------------
# _resolve_train_sources
# ---------------------------------------------------------------------------


def test_resolve_train_sources_lists_subfolder_images(env) -> None:
    _png(env["sub"] / "X.png")
    _png(env["sub"] / "Y.png")
    with db.connection_for(env["db"]) as conn:
        out = duplicate_finder._resolve_train_sources(
            conn, env["p"]["id"], env["v"]["id"], env["pdir"],
        )
    names = [n for n, _ in out]
    assert names == ["1_data/X.png", "1_data/Y.png"]


def test_resolve_train_sources_skips_duplicate_removed(env) -> None:
    _png(env["sub"] / "X.png")
    _png(env["sub"] / "Y.png")
    preprocess_manifest.train_mark_duplicate_removed(
        env["pdir"], env["v"]["label"], ["1_data/Y.png"],
    )
    with db.connection_for(env["db"]) as conn:
        out = duplicate_finder._resolve_train_sources(
            conn, env["p"]["id"], env["v"]["id"], env["pdir"],
        )
    names = [n for n, _ in out]
    assert names == ["1_data/X.png"]


def test_resolve_train_sources_skips_non_image(env) -> None:
    _png(env["sub"] / "X.png")
    (env["sub"] / "X.txt").write_text("caption")
    with db.connection_for(env["db"]) as conn:
        out = duplicate_finder._resolve_train_sources(
            conn, env["p"]["id"], env["v"]["id"], env["pdir"],
        )
    names = [n for n, _ in out]
    assert names == ["1_data/X.png"]


def test_resolve_train_sources_empty_when_no_train(env) -> None:
    import shutil
    shutil.rmtree(env["pdir"] / "versions")
    with db.connection_for(env["db"]) as conn:
        out = duplicate_finder._resolve_train_sources(
            conn, env["p"]["id"], env["v"]["id"], env["pdir"],
        )
    assert out == []


def test_resolve_train_sources_mismatched_version_raises(env) -> None:
    with db.connection_for(env["db"]) as conn:
        other_p = projects.create_project(conn, title="Other")
        other_v = versions.create_version(
            conn, project_id=other_p["id"], label="v1"
        )
    with db.connection_for(env["db"]) as conn:
        with pytest.raises(NotFoundError) as exc:
            duplicate_finder._resolve_train_sources(
                conn, env["p"]["id"], other_v["id"], env["pdir"],
            )
        assert exc.value.code == "version.not_found"


# ---------------------------------------------------------------------------
# apply_train_duplicate_removals
# ---------------------------------------------------------------------------


def test_apply_train_duplicate_marks_manifest(env) -> None:
    _png(env["sub"] / "A.png")
    _png(env["sub"] / "B.png")
    with db.connection_for(env["db"]) as conn:
        result = duplicate_finder.apply_train_duplicate_removals(
            conn, env["p"]["id"], env["v"]["id"],
            names=["1_data/A.png", "1_data/B.png"],
        )
    assert sorted(result["removed"]) == ["1_data/A.png", "1_data/B.png"]
    assert not (env["sub"] / "A.png").exists()
    assert not (env["sub"] / "B.png").exists()
    entry = preprocess_manifest.train_get_entry(
        env["pdir"], env["v"]["label"], "1_data/A.png"
    )
    assert entry["kind"] == preprocess_manifest.DUPLICATE_REMOVED_KIND


def test_apply_train_duplicate_rejects_invalid_rel_name(env) -> None:
    with db.connection_for(env["db"]) as conn:
        with pytest.raises(InvalidPathError) as exc:
            duplicate_finder.apply_train_duplicate_removals(
                conn, env["p"]["id"], env["v"]["id"],
                names=["../etc/passwd"],
            )
        assert exc.value.code == "path.invalid"


def test_apply_train_duplicate_rejects_flat_name(env) -> None:
    with db.connection_for(env["db"]) as conn:
        with pytest.raises(InvalidPathError) as exc:
            duplicate_finder.apply_train_duplicate_removals(
                conn, env["p"]["id"], env["v"]["id"],
                names=["X.png"],
            )
        assert exc.value.code == "path.invalid"


def test_apply_train_duplicate_mismatched_version_raises(env) -> None:
    with db.connection_for(env["db"]) as conn:
        other_p = projects.create_project(conn, title="Other")
        other_v = versions.create_version(
            conn, project_id=other_p["id"], label="v1"
        )
    with db.connection_for(env["db"]) as conn:
        with pytest.raises(NotFoundError) as exc:
            duplicate_finder.apply_train_duplicate_removals(
                conn, env["p"]["id"], other_v["id"],
                names=["1_data/A.png"],
            )
        assert exc.value.code == "version.not_found"


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def _info(name: str, w: int, h: int, size: int) -> duplicate_finder.ImageInfo:
    return duplicate_finder.ImageInfo(
        name=name, path=Path(name), width=w, height=h, size=size,
        phash=None, soft_phash=None, dhash=None, ahash=None, colorhash=None,
        grayprint=None,  # type: ignore[arg-type]
    )


def _crop_rel(
    source: duplicate_finder.ImageInfo,
    crop: duplicate_finder.ImageInfo,
    score: float = 0.85,
) -> duplicate_finder.CropRelation:
    return duplicate_finder.CropRelation(
        source=source, crop=crop, score=score,
        source_window=(0, 0, source.width, source.height),
        window_ratio=0.5, segment_matches=3, segment_coverage=0.4, note="",
    )


def test_merge_crop_relations_folds_candidate_into_existing_group() -> None:
    a = _info("1_data/a.png", 1000, 1000, 100_000)
    b = _info("1_data/b.png", 1000, 1000, 90_000)
    c = _info("1_data/c.png", 500, 500, 30_000)
    merged, metrics = duplicate_finder.merge_crop_relations_into_groups(
        [a, b, c], [[a, b]], {}, [_crop_rel(a, c)],
    )
    assert len(merged) == 1
    assert sorted(img.name for img in merged[0]) == [
        "1_data/a.png", "1_data/b.png", "1_data/c.png",
    ]
    assert merged[0][0].name in ("1_data/a.png", "1_data/b.png")
    key = ("1_data/a.png", "1_data/c.png")
    assert key in metrics
    assert metrics[key].match_type == "crop-variant"


def test_merge_crop_relations_creates_new_group() -> None:
    a = _info("1_data/a.png", 1000, 1000, 100_000)
    c = _info("1_data/c.png", 400, 400, 20_000)
    d = _info("1_data/d.png", 800, 800, 50_000)
    merged, _ = duplicate_finder.merge_crop_relations_into_groups(
        [a, c, d], [], {}, [_crop_rel(a, c)],
    )
    assert len(merged) == 1
    assert sorted(img.name for img in merged[0]) == ["1_data/a.png", "1_data/c.png"]


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_options_from_payload_sensitivity_maps_scores() -> None:
    loose = duplicate_finder.options_from_payload({"match_scope": "both", "sensitivity": "loose"})
    std = duplicate_finder.options_from_payload({"match_scope": "both", "sensitivity": "standard"})
    strict = duplicate_finder.options_from_payload({"match_scope": "both", "sensitivity": "strict"})
    assert loose.variant_score < std.variant_score < strict.variant_score
    assert loose.crop_score < std.crop_score < strict.crop_score
    assert std.variant_score == duplicate_finder.DEFAULT_VARIANT_SCORE
    assert std.crop_score == duplicate_finder.DEFAULT_CROP_SCORE


def test_options_from_payload_detect_crops_follows_scope() -> None:
    assert duplicate_finder.options_from_payload({"match_scope": "both"}).detect_crops is True
    assert duplicate_finder.options_from_payload({"match_scope": "strict"}).detect_crops is False


def test_options_from_payload_ignores_stray_tuning_fields() -> None:
    opts = duplicate_finder.options_from_payload({
        "match_scope": "both", "sensitivity": "standard",
        "variant_score": 10, "crop_score": 0.1, "hash_size": 4, "structure_threshold": 99,
    })
    assert opts.variant_score == duplicate_finder.DEFAULT_VARIANT_SCORE
    assert opts.crop_score == duplicate_finder.DEFAULT_CROP_SCORE
    assert opts.hash_size == duplicate_finder.DEFAULT_HASH_SIZE
    assert opts.structure_threshold == duplicate_finder.DEFAULT_STRUCTURE_THRESHOLD


def test_options_from_payload_rejects_bad_sensitivity() -> None:
    with pytest.raises(duplicate_finder.DuplicateFinderError) as exc:
        duplicate_finder.options_from_payload({"match_scope": "both", "sensitivity": "wat"})
    assert exc.value.code == "duplicate.sensitivity_invalid"


