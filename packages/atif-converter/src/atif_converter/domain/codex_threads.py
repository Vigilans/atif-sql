# SPDX-License-Identifier: Apache-2.0

"""One Codex session from its rollouts: a thread's continuation segments.

A Codex thread is written as one or more rollouts. Its first rollout is
``rollout-<ts>-<thread>.jsonl``; each later segment is
``rollout-<ts>-<thread>_<segment>.jsonl``, written on the day it started, and
its ``session_meta`` names the rollout it continues in ``history_base``
(``thread_id`` is that rollout's id, ``end_ordinal_exclusive`` the first
ordinal it does NOT continue from).

:func:`assemble_codex_session` turns the rollouts of one session into the
single record stream the converter reads, with what each record is:

* the rollouts come oldest first;
* the newest rollout of a thread is its live history. Walking its
  ``history_base`` chain cuts each ancestor at ``end_ordinal_exclusive``: the
  ancestor's records from that ordinal on are a branch the thread left, and
  are REVERTED. A rollout of the thread that the chain never reaches is
  reverted in full.

:func:`mark_codex_session` then writes ``extra.reverted`` onto each step of
the converted trajectory whose records are all reverted.

A session of one rollout passes through both unchanged.

Pure module: no filesystem access, no env, no logging.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

#: ``rollout-<ts>-<thread>[_<segment>].jsonl[.zst]``; each id is five
#: hyphen-separated groups and the timestamp takes everything before them.
_ROLLOUT_NAME = re.compile(
    r"^rollout-.+-(?P<thread>[^-_/]+(?:-[^-_/]+){4})"
    r"(?:_(?P<segment>[^-_/]+(?:-[^-_/]+){4}))?\.jsonl(?:\.zst)?$"
)


def rollout_ids(name: str) -> tuple[str, str] | None:
    """``(thread id, rollout id)`` a rollout filename carries, or ``None`` for any other name.

    The rollout id is what ``history_base.thread_id`` refers to: the segment
    id of a continuation, the thread id of a thread's first rollout.
    """
    match = _ROLLOUT_NAME.match(PurePosixPath(name).name)
    if match is None:
        return None
    thread_id = str(match.group("thread"))
    segment = match.group("segment")
    return thread_id, str(segment) if segment else thread_id


@dataclass(frozen=True, slots=True)
class RolloutFile:
    """One rollout of the session: its name relative to the session base and its records."""

    source_file: str
    #: The file's object records, in file order.
    records: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class CodexSessionStream:
    """The session's records in conversion order, with what each one is."""

    root_thread_id: str
    #: ``(record, source_file)`` pairs in the order the converter reads them.
    records: list[tuple[dict[str, Any], str]]
    #: Per entry of :attr:`records`: whether the record is on a branch its
    #: thread left.
    reverted: list[bool]


def _session_meta(records: list[dict[str, Any]]) -> dict[str, Any]:
    first = records[0] if records else None
    if first is None or first.get("type") != "session_meta":
        return {}
    payload = first.get("payload")
    return payload if isinstance(payload, dict) else {}


def _str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


@dataclass(slots=True)
class _Rollout:
    file: RolloutFile
    thread_id: str
    rollout_id: str
    meta: dict[str, Any]


def _cutoffs(rollouts: list[_Rollout]) -> dict[str, int]:
    """Rollout id -> first reverted ordinal, for one thread's rollouts (oldest first)."""
    cutoffs: dict[str, int] = {}
    if not rollouts:
        return cutoffs
    by_id = {r.rollout_id: r for r in rollouts}
    head = rollouts[-1]
    live = {head.rollout_id}
    current = head
    while True:
        base = current.meta.get("history_base")
        base_id = _str(base.get("thread_id")) if isinstance(base, dict) else None
        end = _int(base.get("end_ordinal_exclusive")) if isinstance(base, dict) else None
        if base_id is None or end is None or base_id in live or base_id not in by_id:
            break
        live.add(base_id)
        cutoffs[base_id] = end
        current = by_id[base_id]
    for rollout in rollouts:
        if rollout.rollout_id not in live:
            cutoffs[rollout.rollout_id] = 0
    return cutoffs


def assemble_codex_session(files: Sequence[RolloutFile]) -> CodexSessionStream:
    """The session's record stream (module docstring) from its rollout files.

    The first file is the thread's first rollout; the rest come in any order.
    A file's thread is its ``session_meta`` id, else its filename's thread id,
    else the first file's thread.
    """
    threads: dict[str, list[_Rollout]] = {}
    root_thread_id = ""
    for index, file in enumerate(files):
        meta = _session_meta(file.records)
        ids = rollout_ids(file.source_file)
        thread_id = _str(meta.get("id")) or (ids[0] if ids else root_thread_id)
        if index == 0:
            root_thread_id = thread_id
        rollout_id = ids[1] if ids else thread_id
        threads.setdefault(thread_id, []).append(_Rollout(file, thread_id, rollout_id, meta))
    for rollouts in threads.values():
        rollouts.sort(key=lambda r: PurePosixPath(r.file.source_file).name)

    others = sorted(
        (thread_id for thread_id in threads if thread_id != root_thread_id),
        key=lambda thread_id: PurePosixPath(threads[thread_id][0].file.source_file).name,
    )
    records: list[tuple[dict[str, Any], str]] = []
    reverted: list[bool] = []
    for thread_id in [root_thread_id, *others]:
        rollouts = threads.get(thread_id, [])
        cutoffs = _cutoffs(rollouts)
        for rollout in rollouts:
            cutoff = cutoffs.get(rollout.rollout_id)
            for record in rollout.file.records:
                ordinal = _int(record.get("ordinal"))
                records.append((record, rollout.file.source_file))
                reverted.append(
                    cutoff == 0
                    or (cutoff is not None and ordinal is not None and ordinal >= cutoff)
                )
    return CodexSessionStream(root_thread_id=root_thread_id, records=records, reverted=reverted)


def _step_extra(step: dict[str, Any]) -> dict[str, Any]:
    extra = step.get("extra")
    if not isinstance(extra, dict):
        extra = {}
        step["extra"] = extra
    return extra


def mark_codex_session(
    trajectory: dict[str, Any], stream: CodexSessionStream, record_ids: Sequence[str]
) -> dict[str, Any]:
    """Write ``extra.reverted`` onto ``trajectory``'s steps in place.

    ``record_ids`` are the edge ids of :attr:`CodexSessionStream.records`, in
    the same order: the ids the enrichment pass wrote into each step's
    ``extra.source_uuids``. A step is reverted when every record behind it is.
    """
    record_reverted = dict(zip(record_ids, stream.reverted, strict=True))
    steps = trajectory.get("steps")
    for step in steps if isinstance(steps, list) else []:
        if not isinstance(step, dict):
            continue
        extra = step.get("extra")
        uuids = extra.get("source_uuids") if isinstance(extra, dict) else None
        facts = [
            record_reverted[uuid]
            for uuid in (uuids if isinstance(uuids, list) else [])
            if isinstance(uuid, str) and uuid in record_reverted
        ]
        if facts and all(facts):
            _step_extra(step)["reverted"] = True
    return trajectory


__all__ = [
    "CodexSessionStream",
    "RolloutFile",
    "assemble_codex_session",
    "mark_codex_session",
    "rollout_ids",
]
