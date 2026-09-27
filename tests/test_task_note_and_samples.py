"""Queue task notes (v20 tasks.note) + per-task sample image listing endpoint.

Both endpoints serve the same UI change: each row in the queue list can flip through its sample images directly (opening a lightbox),
and right-click lets you attach a note to the task, which is also visible / editable on the task detail page.

Reuses the isolation approach from tests/test_studio_queue_endpoints.py (tmp db + stub supervisor,
no lifespan run).
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from studio import db, server


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from studio.api.routers.queue import lifecycle as _queue_lifecycle

    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    presets = tmp_path / "presets"
    presets.mkdir()
    (presets / "good.yaml").write_text("epochs: 1\n", encoding="utf-8")

    monkeypatch.setattr(server, "STUDIO_DB", dbfile)
    monkeypatch.setattr(server.db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(_queue_lifecycle, "USER_PRESETS_DIR", presets)
    return tmp_path


class _StubSupervisor:
    def is_task_pausable(self, task_id: int) -> bool:
        return False


@pytest.fixture
def client(isolated: Path) -> TestClient:
    server.app.state.supervisor = _StubSupervisor()
    return TestClient(server.app)


def _enqueue(client: TestClient, name: str = "t") -> int:
    return client.post(
        "/api/queue", json={"config_name": "good", "name": name}
    ).json()["id"]


# ── note ────────────────────────────────────────────────────────────────────


def test_note_defaults_to_null(client: TestClient) -> None:
    tid = _enqueue(client)
    assert client.get(f"/api/queue/{tid}").json()["note"] is None


def test_set_and_read_note(client: TestClient) -> None:
    tid = _enqueue(client)
    resp = client.put(f"/api/queue/{tid}/note", json={"note": "alpha=16, dataset v3"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["note"] == "alpha=16, dataset v3"
    # the list row also carries it (readable even without refreshing the queue page after writing via right-click)
    items = client.get("/api/queue").json()["items"]
    assert [i["note"] for i in items] == ["alpha=16, dataset v3"]
    # same for the detail row
    assert client.get(f"/api/queue/{tid}").json()["note"] == "alpha=16, dataset v3"


def test_blank_note_clears(client: TestClient) -> None:
    tid = _enqueue(client)
    client.put(f"/api/queue/{tid}/note", json={"note": "temp"})
    resp = client.put(f"/api/queue/{tid}/note", json={"note": "   "})
    assert resp.status_code == 200
    assert resp.json()["note"] is None


def test_note_truncated_not_rejected(client: TestClient) -> None:
    tid = _enqueue(client)
    resp = client.put(f"/api/queue/{tid}/note", json={"note": "x" * 900})
    assert resp.status_code == 200
    assert len(resp.json()["note"]) == 500


def test_note_missing_task_404(client: TestClient) -> None:
    assert client.put("/api/queue/9999/note", json={"note": "hi"}).status_code == 404


def test_note_survives_status_change(client: TestClient) -> None:
    """A note is UI metadata unrelated to task state: a note can still be added to a finished historical task."""
    tid = _enqueue(client)
    with db.connection_for() as conn:
        db.update_task(conn, tid, status="done", finished_at=time.time())
    resp = client.put(f"/api/queue/{tid}/note", json={"note": "best one so far"})
    assert resp.status_code == 200
    assert resp.json()["note"] == "best one so far"
    assert resp.json()["status"] == "done"


# -- samples listing --------------------------------------------------------


def _make_task_with_samples(
    tmp_path: Path, filenames: list[str], sub: str = "samples"
) -> int:
    """Create a task with a monitor_state_path, and lay out a few sample images alongside the state file."""
    monitor_dir = tmp_path / "monitor"
    (monitor_dir / sub).mkdir(parents=True, exist_ok=True)
    state = monitor_dir / "monitor_state.json"
    state.write_text("{}", encoding="utf-8")
    with db.connection_for() as conn:
        tid = db.create_task(conn, name="train", config_name="good")
        db.update_task(conn, tid, monitor_state_path=str(state))
    for i, fn in enumerate(filenames):
        f = monitor_dir / sub / fn
        f.write_bytes(b"png")
        # mtime increases -> listing order is predictable (matches the training timeline)
        ts = 1_700_000_000 + i
        import os
        os.utime(f, (ts, ts))
    return tid


def test_samples_empty_without_monitor_path(client: TestClient) -> None:
    """Non-training task / not yet started -> an empty listing instead of 404 (the queue page requests per-row and shouldn't flood the console with red errors)."""
    tid = _enqueue(client)
    resp = client.get(f"/api/queue/{tid}/samples")
    assert resp.status_code == 200
    assert resp.json() == {"items": [], "total": 0}


def test_samples_missing_task_is_empty(client: TestClient) -> None:
    resp = client.get("/api/queue/9999/samples")
    assert resp.status_code == 200
    assert resp.json()["items"] == []


def test_samples_listed_in_time_order_with_marks(
    client: TestClient, isolated: Path
) -> None:
    tid = _make_task_with_samples(
        isolated, ["epoch_1_a.png", "epoch_2_a.png", "step_1200_b.png"]
    )
    body = client.get(f"/api/queue/{tid}/samples").json()
    assert [i["filename"] for i in body["items"]] == [
        "epoch_1_a.png", "epoch_2_a.png", "step_1200_b.png",
    ]
    assert [i["epoch"] for i in body["items"]] == [1, 2, None]
    assert [i["step"] for i in body["items"]] == [None, None, 1200]
    assert body["total"] == 3


def test_samples_skip_non_images(client: TestClient, isolated: Path) -> None:
    tid = _make_task_with_samples(isolated, ["epoch_1.png", "prompt.txt"])
    items = client.get(f"/api/queue/{tid}/samples").json()["items"]
    assert [i["filename"] for i in items] == ["epoch_1.png"]


def test_samples_find_legacy_output_layout(client: TestClient, isolated: Path) -> None:
    """Pre-PP6.1 layout: images live in output/samples/ alongside the monitor."""
    tid = _make_task_with_samples(isolated, ["epoch_1.png"], sub="output/samples")
    items = client.get(f"/api/queue/{tid}/samples").json()["items"]
    assert [i["filename"] for i in items] == ["epoch_1.png"]


def test_samples_served_by_image_route(client: TestClient, isolated: Path) -> None:
    """A filename from the listing can be fed straight into /samples/{filename}?task_id= to fetch the image."""
    tid = _make_task_with_samples(isolated, ["epoch_1.png"])
    fn = client.get(f"/api/queue/{tid}/samples").json()["items"][0]["filename"]
    assert client.get(f"/samples/{fn}?task_id={tid}").status_code == 200
