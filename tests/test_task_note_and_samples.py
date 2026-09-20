"""队列任务备注（v20 tasks.note）+ 每任务采样图清单端点。

两个端点服务同一个 UI 改动：队列列表里每行能直接翻采样图（点开灯箱），
右键能给任务贴一句备注，备注在任务详情页也看得见 / 改得了。

复用 tests/test_studio_queue_endpoints.py 的隔离手法（tmp db + stub supervisor，
不跑 lifespan）。
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
    # 列表行也带上（右键写完队列页不刷新也能读到）
    items = client.get("/api/queue").json()["items"]
    assert [i["note"] for i in items] == ["alpha=16, dataset v3"]
    # 详情行同理
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
    """备注是 UI 元数据，跟 task 状态无关：跑完的历史任务也能补备注。"""
    tid = _enqueue(client)
    with db.connection_for() as conn:
        db.update_task(conn, tid, status="done", finished_at=time.time())
    resp = client.put(f"/api/queue/{tid}/note", json={"note": "best one so far"})
    assert resp.status_code == 200
    assert resp.json()["note"] == "best one so far"
    assert resp.json()["status"] == "done"


# ── samples 清单 ────────────────────────────────────────────────────────────


def _make_task_with_samples(
    tmp_path: Path, filenames: list[str], sub: str = "samples"
) -> int:
    """建一个带 monitor_state_path 的 task，并在 state 文件同级铺几张采样图。"""
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
        # mtime 递增 → 清单顺序可预测（与训练时间轴一致）
        ts = 1_700_000_000 + i
        import os
        os.utime(f, (ts, ts))
    return tid


def test_samples_empty_without_monitor_path(client: TestClient) -> None:
    """非训练任务 / 还没开跑 → 空清单而不是 404（队列页逐行请求，不刷红控制台）。"""
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
    """pre-PP6.1 布局：图在 monitor 同级 output/samples/。"""
    tid = _make_task_with_samples(isolated, ["epoch_1.png"], sub="output/samples")
    items = client.get(f"/api/queue/{tid}/samples").json()["items"]
    assert [i["filename"] for i in items] == ["epoch_1.png"]


def test_samples_served_by_image_route(client: TestClient, isolated: Path) -> None:
    """清单里的 filename 直接喂 /samples/{filename}?task_id= 能取到图。"""
    tid = _make_task_with_samples(isolated, ["epoch_1.png"])
    fn = client.get(f"/api/queue/{tid}/samples").json()["items"][0]["filename"]
    assert client.get(f"/samples/{fn}?task_id={tid}").status_code == 200
