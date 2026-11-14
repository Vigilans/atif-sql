# SPDX-License-Identifier: Apache-2.0
# Ported from harbor 0.26.0, src/harbor/agents/installed/codex.py (Apache-2.0,
# Copyright the Harbor authors; harbor #3435), onto the ATIF models vendored in
# atif_converter.domain.atif, so atif-converter doesn't depend on harbor.

"""A Codex session's threads as one trajectory, each spawned thread nested in its spawner.

harbor 0.26.0's ``Codex._convert_events_to_trajectory`` converts every thread
of a session on its own and embeds each spawned thread's trajectory in the
``subagent_trajectories`` of the thread that spawned it, recursively:

* a spawned trajectory takes the root thread's id as its ``session_id``, its
  own thread id as its ``trajectory_id``, and its ``thread_spawn`` identity
  (``agent_path``, ``agent_nickname``) in ``agent.extra``;
* the spawning call's result names it in ``subagent_trajectory_ref``;
* a trajectory with children takes its own thread id as its
  ``trajectory_id`` and adds each child's totals to its ``final_metrics``.

The threads come from :func:`~atif_converter.domain.codex_threads.assemble_codex_session`,
so a thread's continuation segments are joined and a spawned thread's copied
parent history is already dropped before it converts. Two things go past
harbor, each where harbor's session has no answer:

* harbor links a child only from a ``spawn_agent`` call whose JSON result's
  ``task_name`` is the child's ``agent_path``. A child that rule does not link
  is linked from the call :func:`~atif_converter.domain.codex_threads.spawn_links`
  names (the spawn event's call, or the first output reporting the child's
  ``agent_id``, which is how a code-mode ``exec`` script's spawn shows).
* harbor converts the one root thread and drops a thread whose spawner is not
  in the session. Such a thread nests under the root here.

Pure module: no filesystem access, no env.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from atif_converter.domain.atif import FinalMetrics, SubagentTrajectoryRef, Trajectory
from atif_converter.domain.codex_conversion import convert_codex_records
from atif_converter.domain.codex_threads import spawn_links, thread_records

if TYPE_CHECKING:
    from atif_converter.domain.codex_threads import CodexSessionStream, SpawnedThread


def _attach_subagent_ref(
    parent: Trajectory, child: Trajectory, agent_path: str | None, link_call_id: str | None
) -> None:
    """Name ``child`` in the result of the call that spawned it (harbor's rule, then the link)."""
    ref = [SubagentTrajectoryRef(trajectory_id=child.trajectory_id, session_id=child.session_id)]
    if agent_path is not None:
        for step in parent.steps:
            if not step.tool_calls or step.observation is None:
                continue
            spawn_calls = {
                call.tool_call_id for call in step.tool_calls if call.function_name == "spawn_agent"
            }
            for result in step.observation.results:
                if result.source_call_id not in spawn_calls or not isinstance(result.content, str):
                    continue
                try:
                    output = json.loads(result.content)
                except json.JSONDecodeError:
                    continue
                if isinstance(output, dict) and output.get("task_name") == agent_path:
                    result.subagent_trajectory_ref = ref
                    return
    if link_call_id is None:
        return
    for step in parent.steps:
        for result in step.observation.results if step.observation else ():
            if result.source_call_id == link_call_id:
                result.subagent_trajectory_ref = ref
                return


def _include_subagent_metrics(parent: Trajectory, children: list[Trajectory]) -> None:
    """Add each child's totals to ``parent.final_metrics`` (harbor's, verbatim)."""
    if parent.final_metrics is None:
        # The children can report usage, but without the parent's own
        # totals their sum is not a total for the whole trajectory.
        parent.final_metrics = FinalMetrics(total_steps=len(parent.steps))
    # Each child total already includes its descendants.
    metrics = [parent.final_metrics, *(child.final_metrics for child in children)]

    def total(field: str) -> int | float | None:
        values: list[int | float | None] = [
            getattr(metric, field) if metric else None for metric in metrics
        ]
        present: list[int | float] = [value for value in values if value is not None]
        return sum(present) if len(present) == len(values) else None

    for field in (
        "total_prompt_tokens",
        "total_completion_tokens",
        "total_cached_tokens",
        "total_cost_usd",
    ):
        setattr(parent.final_metrics, field, total(field))

    extra = parent.final_metrics.extra or {}
    for field in (
        "reasoning_output_tokens",
        "total_cache_write_input_tokens",
        "total_tokens",
    ):
        values = [(metric.extra or {}).get(field) if metric else None for metric in metrics]
        present = [value for value in values if value is not None]
        if len(present) == len(values):
            extra[field] = sum(present)
        else:
            extra.pop(field, None)
    parent.final_metrics.extra = extra or None


def _thread_session_id(records: list[dict[str, Any]], fallback: str) -> str:
    """The ``session_id`` :func:`convert_codex_records` gives a thread's trajectory."""
    meta = next((record for record in records if record.get("type") == "session_meta"), None)
    payload = meta.get("payload", {}) if meta else {}
    meta_id = payload.get("id") if isinstance(payload, dict) else None
    return meta_id if isinstance(meta_id, str) and meta_id else fallback


def convert_codex_threads(
    stream: CodexSessionStream,
    *,
    fallback_session_id: str,
    model_name: str | None = None,
) -> Trajectory | None:
    """Convert every thread of ``stream`` and nest each spawned one in its spawner.

    ``fallback_session_id`` is the root thread's ``session_id`` when its
    ``session_meta`` names none (the first rollout's stem). ``None`` exactly
    when the root thread converts to nothing; a spawned thread that converts
    to nothing is left out, as harbor leaves it out.
    """
    positions = thread_records(stream)
    records = {
        thread_id: [stream.records[position][0] for position in thread_positions]
        for thread_id, thread_positions in positions.items()
    }
    spawned: dict[str, SpawnedThread] = {thread.thread_id: thread for thread in stream.spawned}
    children_by_parent: dict[str, list[str]] = {}
    for thread in stream.spawned:
        parent = thread.parent_thread_id
        if parent is None or parent == thread.thread_id or parent not in records:
            parent = stream.root_thread_id
        children_by_parent.setdefault(parent, []).append(thread.thread_id)
    links = spawn_links(stream)
    root_records = records[stream.root_thread_id]
    root_session_id = _thread_session_id(root_records, fallback_session_id)

    def build(thread_id: str, seen: frozenset[str]) -> Trajectory | None:
        trajectory = convert_codex_records(
            records[thread_id],
            fallback_session_id=fallback_session_id
            if thread_id == stream.root_thread_id
            else thread_id,
            model_name=model_name,
        )
        if trajectory is None:
            return None

        children: list[Trajectory] = []
        for child_id in children_by_parent.get(thread_id, []):
            if child_id in seen:
                continue
            child = build(child_id, seen | {child_id})
            if child is None:
                continue
            thread = spawned[child_id]
            child.session_id = root_session_id
            child.trajectory_id = child_id
            identity: dict[str, str] = {}
            if thread.agent_path is not None:
                identity["agent_path"] = thread.agent_path
            if thread.agent_nickname is not None:
                identity["agent_nickname"] = thread.agent_nickname
            if identity:
                child.agent.extra = {**(child.agent.extra or {}), **identity}
            link = links.get(child_id)
            _attach_subagent_ref(trajectory, child, thread.agent_path, link[0] if link else None)
            children.append(child)

        if children:
            trajectory.trajectory_id = thread_id
            trajectory.subagent_trajectories = children
            _include_subagent_metrics(trajectory, children)
        return trajectory

    return build(stream.root_thread_id, frozenset({stream.root_thread_id}))


__all__ = ["convert_codex_threads"]
