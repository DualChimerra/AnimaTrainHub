"""v13 -> v14: adds generate_params / generate_cover to tasks (0.17 P-I forward-write).

Paves the way for a future "pure-DB generation timeline": from now on every generate writes
a params snapshot + cover image location to the DB, though the frontend **doesn't read it
yet** (the right panel still uses the live queue union cache/disk scanning). Once it switches
to DB-driven, the data will already be there, no data migration needed.

- generate_params: the params_snapshot written at enqueue time (JSON with mode/prompt/lora/...),
  used to backfill + display the future timeline.
- generate_cover: the **cover image location** written when generation finishes -- a disk write
  uses the persistent PNG url, a temp result uses a cache reference (session-scoped). The future
  timeline uses this to locate the image and check whether it still exists (present = show,
  gone = already released).

All NULLABLE, no backfill needed; old generate tasks (and every non-generate task) stay NULL.
Old generate images have no task_id so they can't be backfilled -- this is inherently forward-
only (see the 0.17 discussion).
"""
from __future__ import annotations

import sqlite3

from ._v2_projects import _add_column_if_missing


def migrate(conn: sqlite3.Connection) -> None:
    _add_column_if_missing(conn, "tasks", "generate_params", "generate_params TEXT")
    _add_column_if_missing(conn, "tasks", "generate_cover", "generate_cover TEXT")
