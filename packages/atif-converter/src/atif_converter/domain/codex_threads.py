# SPDX-License-Identifier: Apache-2.0

"""One Codex session from its rollouts: continuation segments and spawned threads.

A Codex thread is written as one or more rollouts. Its first rollout is
``rollout-<ts>-<thread>.jsonl``; each later segment is
``rollout-<ts>-<thread>_<segment>.jsonl``, written on the day it started, and
its ``session_meta`` names the rollout it continues in ``history_base``
(``thread_id`` is that rollout's id, ``end_ordinal_exclusive`` the first
ordinal it does NOT continue from). A thread spawned as a sub-agent writes its
own rollouts, whose ``session_meta`` names the root thread in ``session_id``
and the spawning thread in ``parent_thread_id``, and whose records before
``subagent_history_start_ordinal`` are a copy of the parent's history.

:func:`assemble_codex_session` turns the rollouts of one session into the
single record stream the converter reads, with what each record is:

* the root thread's rollouts come first, oldest first, then each spawned
  thread's, in the order the threads began;
* the newest rollout of a thread is its live history. Walking its
  ``history_base`` chain cuts each ancestor at ``end_ordinal_exclusive``: the
  ancestor's records from that ordinal on are a branch the thread left, and
  are REVERTED. A rollout of the thread that the chain never reaches is
  reverted in full;
* a spawned thread's copied parent history is dropped, since the parent's own
  records already carry it.

:func:`mark_codex_session` then writes the result onto the converted
trajectory: ``extra.reverted`` on a step whose records are all reverted,
``extra.is_sidechain`` and ``extra.agent_id`` on a spawned thread's steps (the
shape Claude Code's inlined subagent steps have, which atif-duck's subagent
views read), and one ``trajectory.extra.subagents`` entry per spawned thread.

A session of one rollout passes through both unchanged.

Pure module: no filesystem access, no env, no logging.
"""

from __future__ import annotations

import json
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
class SpawnedThread:
    """A sub-agent thread of the session, as its ``session_meta`` describes it."""

    thread_id: str
    #: The thread that spawned it, when its rollout says.
    parent_thread_id: str | None
    agent_role: str | None
    spawn_depth: int | None


@dataclass(frozen=True, slots=True)
class CodexSessionStream:
    """The session's records in conversion order, with what each one is."""

    root_thread_id: str
    #: ``(record, source_file)`` pairs in the order the converter reads them.
    records: list[tuple[dict[str, Any], str]]
    #: Per entry of :attr:`records`: whether the record is on a branch its
    #: thread left.
    reverted: list[bool]
    #: The thread each rollout file belongs to, keyed by ``source_file``.
    thread_of_file: dict[str, str]
    #: The spawned threads, in the order they began.
    spawned: tuple[SpawnedThread, ...] = ()
    #: The rollout files of the spawned threads.
    sidechain_files: frozenset[str] = frozenset()
    #: Copied parent-history records dropped from spawned threads.
    copied_records_dropped: int = 0


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

    The first file is the root thread's first rollout; the rest come in any
    order. A file's thread is its ``session_meta`` id, else its filename's
    thread id, else the root thread.
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

    spawned_ids = sorted(
        (thread_id for thread_id in threads if thread_id != root_thread_id),
        key=lambda thread_id: PurePosixPath(threads[thread_id][0].file.source_file).name,
    )
    records: list[tuple[dict[str, Any], str]] = []
    reverted: list[bool] = []
    dropped = 0
    for thread_id in [root_thread_id, *spawned_ids]:
        rollouts = threads.get(thread_id, [])
        cutoffs = _cutoffs(rollouts)
        for rollout in rollouts:
            cutoff = cutoffs.get(rollout.rollout_id)
            copied_before = (
                _int(rollout.meta.get("subagent_history_start_ordinal"))
                if thread_id != root_thread_id
                else None
            )
            for record in rollout.file.records:
                ordinal = _int(record.get("ordinal"))
                if (
                    copied_before is not None
                    and ordinal is not None
                    and ordinal < copied_before
                    and record.get("type") != "session_meta"
                ):
                    dropped += 1
                    continue
                records.append((record, rollout.file.source_file))
                reverted.append(
                    cutoff == 0
                    or (cutoff is not None and ordinal is not None and ordinal >= cutoff)
                )

    spawned: list[SpawnedThread] = []
    for thread_id in spawned_ids:
        meta = threads[thread_id][0].meta
        source = meta.get("source")
        subagent = source.get("subagent") if isinstance(source, dict) else None
        spawn = subagent.get("thread_spawn") if isinstance(subagent, dict) else None
        spawn = spawn if isinstance(spawn, dict) else {}
        spawned.append(
            SpawnedThread(
                thread_id=thread_id,
                parent_thread_id=_str(meta.get("parent_thread_id"))
                or _str(spawn.get("parent_thread_id")),
                agent_role=_str(meta.get("agent_role")) or _str(spawn.get("agent_role")),
                spawn_depth=_int(spawn.get("depth")),
            )
        )
    return CodexSessionStream(
        root_thread_id=root_thread_id,
        records=records,
        reverted=reverted,
        thread_of_file={
            r.file.source_file: thread_id for thread_id, rs in threads.items() for r in rs
        },
        spawned=tuple(spawned),
        copied_records_dropped=dropped,
        sidechain_files=frozenset(
            r.file.source_file for thread_id in spawned_ids for r in threads[thread_id]
        ),
    )


def _output_text(output: object) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        parts: list[str] = [
            text
            for block in output
            if isinstance(block, dict) and isinstance(text := block.get("text"), str)
        ]
        return "\n".join(parts)
    return json.dumps(output)


def _spawn_links(stream: CodexSessionStream) -> dict[str, tuple[str, str]]:
    """Spawned thread id -> ``(spawning tool call id, link source)``.

    ``spawn_event``: the spawning thread's ``collab_agent_spawn_end`` event
    names the new thread and its call. ``tool_output``: the first output in
    the spawning thread that reports the new thread as an ``agent_id`` (the
    ``spawn_agent`` result, also when a code-mode script made the call).
    """
    parents = {
        thread.thread_id: thread.parent_thread_id or stream.root_thread_id
        for thread in stream.spawned
    }
    patterns = {
        thread_id: re.compile(r'"agent_id"\s*:\s*"' + re.escape(thread_id) + '"')
        for thread_id in parents
    }
    links: dict[str, tuple[str, str]] = {}
    for record, source_file in stream.records:
        payload = record.get("payload")
        if not isinstance(payload, dict):
            continue
        thread_id = stream.thread_of_file.get(source_file)
        item_type = payload.get("type")
        call_id = _str(payload.get("call_id"))
        if call_id is None:
            continue
        if record.get("type") == "event_msg" and item_type == "collab_agent_spawn_end":
            child = _str(payload.get("new_thread_id"))
            if child in parents and parents[child] == thread_id:
                links.setdefault(child, (call_id, "spawn_event"))
        elif record.get("type") == "response_item" and item_type in {
            "function_call_output",
            "custom_tool_call_output",
        }:
            text = _output_text(payload.get("output"))
            for child, pattern in patterns.items():
                if child not in links and parents[child] == thread_id and pattern.search(text):
                    links[child] = (call_id, "tool_output")
    return links


def _step_extra(step: dict[str, Any]) -> dict[str, Any]:
    extra = step.get("extra")
    if not isinstance(extra, dict):
        extra = {}
        step["extra"] = extra
    return extra


def mark_codex_session(
    trajectory: dict[str, Any], stream: CodexSessionStream, record_ids: Sequence[str]
) -> dict[str, Any]:
    """Write reverted, sidechain and subagent attribution onto ``trajectory`` in place.

    ``record_ids`` are the edge ids of :attr:`CodexSessionStream.records`, in
    the same order: the ids the enrichment pass wrote into each step's
    ``extra.source_uuids``. A step with none takes the thread of the step
    before it.
    """
    record_facts = {
        record_id: (stream.thread_of_file.get(source_file, stream.root_thread_id), is_reverted)
        for record_id, (_record, source_file), is_reverted in zip(
            record_ids, stream.records, stream.reverted, strict=True
        )
    }
    steps = trajectory.get("steps")
    thread_id = stream.root_thread_id
    for step in steps if isinstance(steps, list) else []:
        if not isinstance(step, dict):
            continue
        extra = step.get("extra")
        uuids = extra.get("source_uuids") if isinstance(extra, dict) else None
        facts = [
            record_facts[uuid]
            for uuid in (uuids if isinstance(uuids, list) else [])
            if isinstance(uuid, str) and uuid in record_facts
        ]
        if facts:
            thread_id = facts[0][0]
            if all(is_reverted for _thread, is_reverted in facts):
                _step_extra(step)["reverted"] = True
        if thread_id != stream.root_thread_id:
            step_extra = _step_extra(step)
            step_extra["is_sidechain"] = True
            step_extra["agent_id"] = thread_id

    if stream.spawned:
        links = _spawn_links(stream)
        entries: list[dict[str, Any]] = []
        for thread in stream.spawned:
            entry: dict[str, Any] = {"agent_id": thread.thread_id}
            if thread.agent_role is not None:
                entry["agent_type"] = thread.agent_role
            link = links.get(thread.thread_id)
            if link is not None:
                entry["parent_tool_call_id"], entry["link_source"] = link
            if thread.spawn_depth is not None:
                entry["spawn_depth"] = thread.spawn_depth
            if thread.parent_thread_id not in {None, stream.root_thread_id}:
                entry["parent_agent_id"] = thread.parent_thread_id
            entries.append(entry)
        extra = trajectory.get("extra")
        if not isinstance(extra, dict):
            extra = {}
            trajectory["extra"] = extra
        extra["subagents"] = entries
    return trajectory


__all__ = [
    "CodexSessionStream",
    "RolloutFile",
    "SpawnedThread",
    "assemble_codex_session",
    "mark_codex_session",
    "rollout_ids",
]
