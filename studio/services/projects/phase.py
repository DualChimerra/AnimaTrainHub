"""Phase advancement / completion validation -- ADR-0007 sections 11.5-A / 11.5-B.

Phase enum, see ``versions.VersionPhase``:
``curating -> tagging -> editing -> regularizing -> ready``.

Completion criteria (section 11.5-B):
- ``curating``: at least 1 image in ``train/``
- ``tagging``: 100% caption coverage (every train image has a same-named .txt)
- ``editing``: same as tagging (fallback, in case the user deleted a caption)
- ``regularizing``: no reg_build job pending/running (skippable, section 11.5-A SKIPPABLE)
- ``ready``: training config file exists + schema validation passes

Cursor advancement rules (section 11.5-A):
- one-directional (forward only)
- the header "Next" button is always clickable; validation failures show the user a hint
- a mandatory phase failing validation -> does not advance
- a skippable phase's validation = no concurrent job (regularizing has no confirm dialog)
- the cursor never moves backward on its own (section 11.5-C; only prompted on the next `next` check after the user deletes data)
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any, Optional

from . import projects as _projects
from . import versions as _versions
from .. import version_config as _version_config


@dataclass(frozen=True)
class CheckResult:
    """Phase validation result. When ok=True, reason may be empty; when ok=False, reason is a user-readable message."""
    ok: bool
    reason: str = ""


# ---------------------------------------------------------------------------
# Per-phase validation functions (take the minimal args needed, for easy unit-test isolation)
# ---------------------------------------------------------------------------


def check_curating(stats: dict[str, Any]) -> CheckResult:
    """At least 1 image in train/ (section 11.5-B)."""
    if stats.get("train_image_count", 0) < 1:
        return CheckResult(False, "Training set is empty, please select training images first")
    return CheckResult(True)


def check_tagging(stats: dict[str, Any]) -> CheckResult:
    """100% caption coverage (section 11.5-B)."""
    total = int(stats.get("train_image_count", 0))
    tagged = int(stats.get("tagged_image_count", 0))
    if total < 1:
        return CheckResult(False, "Training set is empty, please select training images first")
    if tagged < total:
        missing = total - tagged
        return CheckResult(False, f"{missing} image(s) still have no caption generated; please rerun or delete them")
    return CheckResult(True)


def check_editing(stats: dict[str, Any]) -> CheckResult:
    """editing is the same as tagging (fallback, section 11.5-B; passes automatically in most cases)."""
    return check_tagging(stats)


def check_regularizing(
    conn: sqlite3.Connection, version_id: int
) -> CheckResult:
    """No reg_build job pending/running (section 11.5-B; skippable = doesn't require a non-empty reg set)."""
    from . import jobs as project_jobs
    if project_jobs.count_active(conn, version_id=version_id, kind="reg_build") > 0:
        return CheckResult(False, "A regularization task is in progress, please wait for it to finish")
    return CheckResult(True)




def check_preprocessing(
    conn: sqlite3.Connection, version_id: int
) -> CheckResult:
    """No preprocess job pending/running (ADR 0010; skippable = doesn't require everything to be processed).

    Same pattern as `check_regularizing` -- preprocessing is an optional phase (upscale / crop /
    dedup etc. can all be skipped, falling back to the default upscale algorithm at training time); this check only guards against a concurrent job collision.
    """
    from . import jobs as project_jobs
    if project_jobs.count_active(conn, version_id=version_id, kind="preprocess") > 0:
        return CheckResult(False, "A preprocessing task is in progress, please wait for it to finish")
    return CheckResult(True)


def check_ready(
    project: dict[str, Any], version: dict[str, Any]
) -> CheckResult:
    """training config exists + schema validation passes (section 11.5-B)."""
    try:
        _version_config.read_version_config(project, version)
    except _version_config.VersionConfigError as exc:
        return CheckResult(False, f"Please finish the training config first: {exc}")
    return CheckResult(True)


# ---------------------------------------------------------------------------
# Dispatcher + advancement
# ---------------------------------------------------------------------------


def check_phase(
    conn: sqlite3.Connection, version_id: int, phase: str
) -> CheckResult:
    """Selects the matching check function for the phase; returns a CheckResult even if the version / project doesn't exist."""
    v = _versions.get_version(conn, version_id)
    if not v:
        return CheckResult(False, "Version does not exist")
    p = _projects.get_project(conn, int(v["project_id"]))
    if not p:
        return CheckResult(False, "Project does not exist")

    P = _versions.VersionPhase
    if phase == P.CURATING:
        return check_curating(_versions.stats_for_version(p, v))
    if phase == P.PREPROCESSING:
        return check_preprocessing(conn, version_id)
    if phase == P.EDITING:
        return check_editing(_versions.stats_for_version(p, v))
    if phase == P.REGULARIZING:
        return check_regularizing(conn, version_id)
    if phase == P.READY:
        return check_ready(p, v)
    return CheckResult(False, f"Unknown phase: {phase}")


def advance_phase(
    conn: sqlite3.Connection, version_id: int
) -> tuple[bool, CheckResult, Optional[str]]:
    """Attempts to advance the phase cursor to the next phase (used for both mandatory and skippable phases).

    Returns ``(advanced, result, new_phase)``:
    - ``advanced=True``: the cursor advanced; ``new_phase`` is the new phase name
    - ``advanced=False``: ``result.reason`` holds the failure reason; ``new_phase=None``

    ``ready`` is the last phase -- after the caller receives ``(False, ok-but-end, None)``,
    it should transition status: ``preparing -> training`` + submit a task (i.e. enqueue training).
    """
    v = _versions.get_version(conn, version_id)
    if not v:
        return False, CheckResult(False, "Version does not exist"), None

    current_phase = _versions.get_phase(v)
    order = _versions.VersionPhase.ORDER

    # phase not in the known set -> reject via fallback
    if current_phase not in order:
        return False, CheckResult(False, f"Unknown phase: {current_phase}"), None

    # already at the last phase (ready) -> phase no longer advances; the caller triggers the status transition
    idx = order.index(current_phase)
    if idx >= len(order) - 1:
        result = check_phase(conn, version_id, current_phase)
        # even at ready, run a validation pass so the caller can decide whether to move into training
        return False, result, None

    # run validation
    result = check_phase(conn, version_id, current_phase)
    if not result.ok:
        return False, result, None

    next_phase = order[idx + 1]
    _versions.update_version(conn, version_id, phase=next_phase)
    return True, CheckResult(True), next_phase


def skip_phase(
    conn: sqlite3.Connection, version_id: int
) -> tuple[bool, CheckResult, Optional[str]]:
    """Skips the current phase (only allowed for the ``SKIPPABLE`` set; currently = preprocessing /
    tagging / regularizing).

    Difference from ``advance_phase``: doesn't require the "completion condition" to be met (e.g. doesn't require
    generating a reg set / doesn't require every image to be preprocessed / doesn't require every image to have a caption); it only checks "no concurrent
    job" to prevent state corruption.
    """
    v = _versions.get_version(conn, version_id)
    if not v:
        return False, CheckResult(False, "Version does not exist"), None

    current_phase = _versions.get_phase(v)
    if current_phase not in _versions.VersionPhase.SKIPPABLE:
        return False, CheckResult(False, f"phase {current_phase} cannot be skipped"), None

    # even when skipping, still check "no concurrent job" to prevent state corruption
    if current_phase == _versions.VersionPhase.REGULARIZING:
        result = check_regularizing(conn, version_id)
        if not result.ok:
            return False, result, None
    elif current_phase == _versions.VersionPhase.PREPROCESSING:
        result = check_preprocessing(conn, version_id)
        if not result.ok:
            return False, result, None

    order = _versions.VersionPhase.ORDER
    idx = order.index(current_phase)
    if idx >= len(order) - 1:
        return False, CheckResult(False, "Already at the last phase"), None
    next_phase = order[idx + 1]
    _versions.update_version(conn, version_id, phase=next_phase)
    return True, CheckResult(True), next_phase
