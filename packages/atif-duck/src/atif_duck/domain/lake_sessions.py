# SPDX-License-Identifier: Apache-2.0

"""The statements that read one corpus's sessions back out of the lake, a batch at a time.

The LLM pipelines (atif-analytics) walk a corpus session by session: bounds
for every session, then the steps of the ones a gate or the renderer needs.
:class:`atif_duck.infrastructure.lake_sessions.LakeSessionReader` runs these
statements for them (atif-cli adapts its rows to atif-analytics' session
port, since the two packages may not import each other).

Every statement is a module constant over the lake tables of
:mod:`atif_duck.domain.lake`, so the AST audit in
``tests/test_sql_text_boundaries.py`` reads them like any other constant. The
corpus and the session ids are bound parameters; a batch of ids is one
``VARCHAR[]`` parameter.

Row order. A step's tool calls and results have no ordinal column; their
order is the order the writer inserted them in, which the lake keeps as
``rowid`` (DuckLake numbers a data file's rows from its ``row_id_start``, and
merges keep the numbers). The writer inserts the step-level tables
(:data:`atif_duck.domain.lake.SOURCE_ORDERED_TABLES`) on one thread so that order is the source
order: the root's steps, then each embedded trajectory's. The statements below
sort a session's trajectories by their first row, then by ``step_id`` and
``rowid``, and return each row's ``trajectory_id``, since a step id names one
step only within its trajectory.

Whether the lake's rows for a session are its current ones is
:func:`is_current`: the session's ``meta.json`` against what
:data:`SESSION_FRESHNESS_SQL` reads back.

Pure: no duckdb import, no filesystem access.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from atif_duck.domain.lake import LAKE_ALIAS, SESSION_KEY
from atif_duck.domain.raw_readers import CORPUS_COLUMN
from atif_duck.domain.sql_literal import SqlFragment

if TYPE_CHECKING:
    from collections.abc import Iterable

_SCOPE: str = f"{CORPUS_COLUMN} = ? AND {SESSION_KEY} IN (SELECT unnest(?::VARCHAR[]))"
_ORDER: str = (
    f"ORDER BY {SESSION_KEY}, min(rowid) OVER (PARTITION BY {SESSION_KEY}, trajectory_id), "
    "step_id, rowid"
)

#: ``(session_id, materialized_at, source_mtime_ns, source_present,
#: source_removed_at)`` for every session of a corpus (param: corpus). What a
#: session's ``meta.json`` said when the lake loaded it, so a reader can tell a
#: current session from one the lake holds an older write of.
SESSION_FRESHNESS_SQL: SqlFragment = SqlFragment(
    f"SELECT {SESSION_KEY}, materialized_at, source_mtime_ns, source_present, source_removed_at "  # noqa: S608  # nosec B608 - constants only; corpus is bound
    f"FROM {LAKE_ALIAS}.session_meta WHERE {CORPUS_COLUMN} = ?"
)

#: ``(session_id, newest record ts)`` for every session of a corpus (param:
#: corpus): the checkpoint's first bound, from ``edges`` like the file path.
LAST_EDGE_TS_SQL: SqlFragment = SqlFragment(
    f"SELECT {SESSION_KEY}, max(ts) FROM {LAKE_ALIAS}.edges "  # noqa: S608  # nosec B608 - constants only; corpus is bound
    f"WHERE {CORPUS_COLUMN} = ? GROUP BY 1"
)


@dataclass(frozen=True, slots=True)
class SessionFreshness:
    """What a session's ``meta.json`` said when the lake loaded it.

    A conversion writes a new ``materialized_at``. Marking a session whose
    transcript vanished rewrites only ``source_present`` and
    ``source_removed_at``, so those are compared too.
    """

    materialized_at: str | None
    source_mtime_ns: int | None
    source_present: bool | None
    source_removed_at: str | None


def session_freshness(rows: Iterable[tuple[Any, ...]]) -> dict[str, SessionFreshness]:
    """:data:`SESSION_FRESHNESS_SQL`'s rows as ``{session_id: SessionFreshness}``."""
    return {
        str(sid): SessionFreshness(
            materialized_at=None if materialized is None else str(materialized),
            source_mtime_ns=None if mtime is None else int(mtime),
            source_present=None if present is None else bool(present),
            source_removed_at=None if removed is None else str(removed),
        )
        for sid, materialized, mtime, present, removed in rows
    }


def is_current(meta: object, recorded: SessionFreshness | None) -> bool:
    """True when the lake's row for a session is its current ``meta.json``'s (``meta``, parsed)."""
    if recorded is None or not isinstance(meta, dict):
        return False
    return bool(
        meta.get("materialized_at") == recorded.materialized_at
        and meta.get("source_mtime_ns") == recorded.source_mtime_ns
        and meta.get("source_present") == recorded.source_present
        and meta.get("source_removed_at") == recorded.source_removed_at
    )


#: Every step of a batch of sessions without its tool payloads (params:
#: corpus, ids): ``(session_id, trajectory_id, step_id, ts, source, message,
#: is_sidechain, is_compact_summary, source_uuids)``.
TURNS_SQL: SqlFragment = SqlFragment(
    "SELECT "  # noqa: S608  # nosec B608 - constants only; corpus and ids are bound
    f"{SESSION_KEY}, trajectory_id, step_id, ts, source, message, is_sidechain, "
    "is_compact_summary, "
    f"CAST(source_uuids AS VARCHAR) FROM {LAKE_ALIAS}.steps WHERE {_SCOPE} {_ORDER}"
)

#: ``(session_id, trajectory_id, step_id)`` of every step with an error result
#: in a batch (params: corpus, ids). The gates need the flag, not the result
#: bodies.
ERROR_STEPS_SQL: SqlFragment = SqlFragment(
    f"SELECT DISTINCT {SESSION_KEY}, trajectory_id, step_id FROM {LAKE_ALIAS}.tool_results "  # noqa: S608  # nosec B608 - constants only; corpus and ids are bound
    f"WHERE {_SCOPE} AND is_error"
)

#: Every tool call of a batch, in source order (params: corpus, ids):
#: ``(session_id, trajectory_id, step_id, tool_name, tool_input)`` with the
#: input as JSON text.
TOOL_CALLS_SQL: SqlFragment = SqlFragment(
    f"SELECT {SESSION_KEY}, trajectory_id, step_id, tool_name, CAST(tool_input AS VARCHAR) "  # noqa: S608  # nosec B608 - constants only; corpus and ids are bound
    f"FROM {LAKE_ALIAS}.tool_calls WHERE {_SCOPE} {_ORDER}"
)

#: Every tool result of a batch, in source order (params: corpus, ids):
#: ``(session_id, trajectory_id, step_id, tool_use_id, content, is_error)``
#: with the content as JSON text.
TOOL_RESULTS_SQL: SqlFragment = SqlFragment(
    f"SELECT {SESSION_KEY}, trajectory_id, step_id, tool_use_id, CAST(content AS VARCHAR), "  # noqa: S608  # nosec B608 - constants only; corpus and ids are bound
    "is_error "
    f"FROM {LAKE_ALIAS}.tool_results WHERE {_SCOPE} {_ORDER}"
)

#: The non-empty raw-record uuids of one session (params: corpus, session id).
EDGE_UUIDS_SQL: SqlFragment = SqlFragment(
    f"SELECT DISTINCT uuid FROM {LAKE_ALIAS}.edges "  # noqa: S608  # nosec B608 - constants only; corpus and the id are bound
    f"WHERE {CORPUS_COLUMN} = ? AND {SESSION_KEY} = ? AND uuid IS NOT NULL AND uuid <> ''"
)

__all__ = [
    "EDGE_UUIDS_SQL",
    "ERROR_STEPS_SQL",
    "LAST_EDGE_TS_SQL",
    "SESSION_FRESHNESS_SQL",
    "TOOL_CALLS_SQL",
    "TOOL_RESULTS_SQL",
    "TURNS_SQL",
    "SessionFreshness",
    "is_current",
    "session_freshness",
]
