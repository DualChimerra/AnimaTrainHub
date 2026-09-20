"""v19 → v20: tasks.note —— 队列任务备注。

加列 `tasks.note TEXT`（NULL / 空串 = 没备注）。用户在队列页右键任一任务写
一句话（"这次试了 alpha=16"、"数据集换了 v3"），备注跟着 task 走：队列列表
行上显示徽标，任务详情页 overview 顶部完整显示 + 可编辑。

纯 UI 元数据，不参与调度，也不进 config snapshot。
"""
from __future__ import annotations

import sqlite3

from ._v2_projects import _add_column_if_missing


def migrate(conn: sqlite3.Connection) -> None:
    _add_column_if_missing(conn, "tasks", "note", "note TEXT")
