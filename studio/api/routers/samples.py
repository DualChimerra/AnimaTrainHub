"""采样图代理（PR-6 commit 1 从 server.py 抽出）。

2 routes：
    GET /samples/{filename}        带 task_id 时按 monitor_state_path 多候选目录解析；
                                   不给走全局 OUTPUT_DIR/samples/ 兜底；可选 ?w=N 缩略图
    GET /api/queue/{task_id}/samples  某 task 的采样图清单（队列页内联采样条用）
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter
from fastapi.responses import FileResponse

from .. import errors as _errors
from ..responses import _thumb_response
from ... import db
from ...domain.errors import NotFoundError
from ...paths import OUTPUT_DIR, task_samples_dir
from ...services.dataset.scan import IMAGE_EXTS

router = APIRouter()
logger = logging.getLogger(__name__)

# 队列页每行内联的采样条上限。monitor state 自己 cap 50（train_monitor.py），
# 这里扫盘能看到全部历史图，给同一个数量级的上限防超长训练一次吐几百条。
_MAX_LIST = 200

_EPOCH_RE = re.compile(r"^epoch_(\d+)", re.IGNORECASE)
_STEP_RE = re.compile(r"^step_(\d+)", re.IGNORECASE)


def _sample_dirs(monitor_state_path: str, task_id: int) -> list[Path]:
    """某 task 的采样图目录候选，按新→旧布局排序。

    - **新（task-scoped）** `studio_data/tasks/<task_id>/samples/`
    - `monitor_state.json` 同级 `samples/`（PP6.1 v0.5.0+ 老 task 兼容；
      state file 在 versions/<v>/monitor/task_<id>/ 时，samples 也在那）
    - `monitor_state.json` 同级 `output/samples/`（pre-PP6.1 老 task；
      sample_dir = output_dir/samples，output_dir 通常是 versions/{label}/output）
    - 同级 `output/<任意子目录>/samples/`（兜底防 anima_train 用别的 output 名）
    """
    monitor_dir = Path(monitor_state_path).parent
    dirs = [
        task_samples_dir(task_id),
        monitor_dir / "samples",
        monitor_dir / "output" / "samples",
    ]
    output_root = monitor_dir / "output"
    if output_root.is_dir():
        for sub in sorted(output_root.iterdir()):
            if sub.is_dir():
                dirs.append(sub / "samples")
    return dirs


def _monitor_state_path(task_id: int) -> Optional[str]:
    with db.connection_for() as conn:
        row = conn.execute(
            "SELECT monitor_state_path FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
    if not row or not row["monitor_state_path"]:
        return None
    return str(row["monitor_state_path"])


def _marks(filename: str) -> tuple[Optional[int], Optional[int]]:
    """从文件名解析 (epoch, step)。`epoch_3_xxx.png` / `step_1200_xxx.png`。"""
    ep = _EPOCH_RE.match(filename)
    st = _STEP_RE.match(filename)
    return (
        int(ep.group(1)) if ep else None,
        int(st.group(1)) if st else None,
    )


@router.get("/samples/{filename}")
def get_sample(
    filename: str,
    task_id: Optional[int] = None,
    w: Optional[int] = None,
) -> FileResponse:
    """采样图代理。

    `?task_id=N` 给了 → 按 `_sample_dirs()` 的候选目录逐个查找。
    没给 task_id → 兜底全局 OUTPUT_DIR/samples/（旧训练直接命令行的兼容）。

    `?w=N` 给了 → 走 thumb_cache 生成 N px 缩略图（用于监控页缩略图条）；
    不给 → 返回原图。两种都走 _thumb_response 的弱 etag + no-cache，浏览器
    304 命中即可，避免「重启窗口期失败响应被永久缓存」问题。
    """
    _errors._validate_component_or_400(filename)

    resolved: Optional[Path] = None
    if task_id is not None:
        state_path = _monitor_state_path(task_id)
        if not state_path:
            raise NotFoundError("Sample image not found", code="sample.not_found")
        candidates = [d / filename for d in _sample_dirs(state_path, task_id)]
        for p in candidates:
            if p.exists():
                resolved = p
                break
        if resolved is None:
            logger.info(
                "sample 404: task_id=%s file=%s tried=%s",
                task_id, filename, [str(p) for p in candidates],
            )
            raise NotFoundError("Sample image not found", code="sample.not_found")
    else:
        path = OUTPUT_DIR / "samples" / filename
        if not path.exists():
            raise NotFoundError("Sample image not found", code="sample.not_found")
        resolved = path

    # w 给了走缩略图；w<=0 或没给 → 原图。复用 thumb_cache，盘上落 .jpg。
    # task-scoped 采样图内容不可变（文件名带 epoch/step，重训得新 task_id），
    # URL (`/samples/{file}?task_id=N&w=W`) 是稳定唯一 key → immutable 长缓存，
    # 浏览器重开不再回源（云端隧道场景每张图省一次 304 RTT）。无 task_id 的
    # 兜底（CLI OUTPUT_DIR/samples，同名可能被覆盖）保持 no-cache 重验。
    immutable = task_id is not None
    size = w if (w is not None and w > 0) else 0
    return _thumb_response(resolved, size, immutable=immutable)


@router.get("/api/queue/{task_id}/samples")
def list_task_samples(task_id: int) -> dict[str, Any]:
    """某 task 的采样图清单，按 mtime 升序（训练时间轴）。

    队列页每行的内联采样条 + 灯箱用。刻意**扫盘**而不是读 monitor_state.json：
    - 已结束的 task 也能看（state 里的 samples 数组还在，但读整个 state 文件
      为了拿 50 条路径太重——10k 步训练的 losses 数组几 MB）；
    - 扫盘能看到全部历史图，不受 monitor 那边 cap 50 的限制。

    task 不存在 / 没 monitor_state_path / 目录还没建 → `{"items": []}`，不报错
    （队列里一堆非训练任务，前端逐行请求，404 只会刷红控制台）。
    """
    state_path = _monitor_state_path(task_id)
    if not state_path:
        return {"items": [], "total": 0}

    seen: set[str] = set()
    found: list[tuple[float, str, int]] = []  # (mtime, filename, size)
    for d in _sample_dirs(state_path, task_id):
        if not d.is_dir():
            continue
        for f in d.iterdir():
            if not f.is_file() or f.suffix.lower() not in IMAGE_EXTS:
                continue
            if f.name in seen:
                continue  # 老/新布局同名文件：先命中的目录优先（同 get_sample 顺序）
            seen.add(f.name)
            try:
                stat = f.stat()
            except OSError:
                continue
            found.append((stat.st_mtime, f.name, stat.st_size))

    found.sort(key=lambda r: (r[0], r[1]))
    total = len(found)
    # 超上限时保留**最新**的一批（用户要看的是训练最近长什么样）。
    if total > _MAX_LIST:
        found = found[-_MAX_LIST:]

    items = []
    for mtime, name, size in found:
        epoch, step = _marks(name)
        items.append({
            "filename": name,
            "mtime": mtime,
            "size": size,
            "epoch": epoch,
            "step": step,
        })
    return {"items": items, "total": total}
