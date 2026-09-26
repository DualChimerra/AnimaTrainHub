"""Queue endpoints (extracted from server.py in PR-6 commit 6; 20 routes split into 3 files internally).

Split by responsibility:
    lifecycle.py  task state machine, 12 routes: list / enqueue / hold / release / reorder /
                  get / cancel / pause / resume / retry / delete
    io.py          data import/export, 3 routes: export / import / snapshot/config
    outputs.py     training outputs, 5 routes: outputs / outputs.zip / export-outputs /
                   output/{filename} / open-folder

The 3 sub-routers are independent; api/app.py calls include 3 times.
"""
