# SPDX-License-Identifier: Apache-2.0

"""The step-level column expressions, written once and used from two places.

The ``steps`` / ``tool_calls`` / ``tool_results`` views turn ATIF step JSON
into typed columns. The same columns are written to the columnar artifacts at
materialize time. Both readers must agree byte for byte, so the expressions
live here and each caller supplies only how it reaches a step's members:

* The registry's JSON path has one ``step`` JSON column holding the whole
  step, so a member is ``json_extract(step, '$.<member>')``.
* The columnar producer receives one JSON column per top-level step member
  (``step_id``, ``timestamp``, ``message``, ...), so the same member is
  ``json_extract(<member>, '$')``.

Both readers also supply, beside each step, the two columns of the
trajectory that holds it (:data:`TRAJECTORY_KEY`, :data:`EMBEDDED`): a step
of an embedded ``subagent_trajectories`` entry is read like a step of the
root, keyed by its own trajectory.

Everything below the member boundary is identical text: the message
flattener, the coalesced booleans, the token casts, and the per-call and
per-result projections (a ``call`` or ``res`` column always holds one whole
tool call or observation result as JSON on both paths).

Pure string building. No duckdb import; the callers own the connection.
Every expression comes back typed :data:`~atif_duck.domain.sql_literal.SqlFragment`:
this module is one of the three producers of SQL text (with the catalog and
``sql_literal``), and nothing here reads a value from outside the process.
"""

from __future__ import annotations

from dataclasses import dataclass

from atif_duck.domain.sql_literal import SqlFragment


@dataclass(frozen=True, slots=True)
class StepMemberAccess:
    """How to spell "the JSON value of top-level step member ``name``" in SQL.

    ``column`` names the SQL column that holds the JSON; ``path_prefix`` is
    the JSONPath that reaches the member inside it (``'$.'`` when the column
    holds a whole step, ``'$'`` when the column already is the member, in
    which case ``name`` is the column itself).
    """

    #: Column holding the whole step, or ``None`` when each member is its own column.
    whole_step_column: str | None

    def json(self, member: str, path: str = "") -> SqlFragment:
        """SQL for ``json_extract`` of ``member`` (optionally a sub-path below it)."""
        column, root = self._locate(member)
        return SqlFragment(f"json_extract({column}, '{root}{path}')")

    def string(self, member: str, path: str = "") -> SqlFragment:
        """SQL for ``json_extract_string`` of ``member`` (optionally a sub-path)."""
        column, root = self._locate(member)
        return SqlFragment(f"json_extract_string({column}, '{root}{path}')")

    def _locate(self, member: str) -> tuple[str, str]:
        if self.whole_step_column is not None:
            return self.whole_step_column, f"$.{member}"
        return member, "$"


#: The registry's JSON path: ``UNNEST(t.steps) AS s(step)`` yields ``step``.
WHOLE_STEP: StepMemberAccess = StepMemberAccess(whole_step_column="step")

#: The producer's path: one JSON column per member, named after the member.
MEMBER_COLUMNS: StepMemberAccess = StepMemberAccess(whole_step_column=None)

#: The source column naming the trajectory that holds a step: the embedded
#: trajectory's ``trajectory_id``, and for the root its ``trajectory_id`` when
#: it has one, else the session id. The views return it as ``trajectory_id``.
TRAJECTORY_KEY: str = "trajectory_key"

#: The source column saying whether that trajectory is embedded in another
#: (true) or is the root (false). Both readers may hand it in as text.
EMBEDDED: str = "embedded"

#: Top-level step members the ``steps`` view reads. ``tool_calls`` and
#: ``observation`` are deliberately absent: they feed the other two views.
STEP_MEMBERS: tuple[str, ...] = (
    "step_id",
    "timestamp",
    "source",
    "model_name",
    "message",
    "metrics",
    "llm_call_count",
    "extra",
    "reasoning_content",
)


#: The prefix that makes a blob path relative to a session directory.
_SESSION_BLOB_PREFIX = "../../blobs/"


def content_text_sql(content: str) -> SqlFragment:
    """SQL rendering ATIF content (``str | ContentPart[]``) as one text body.

    ``content`` is SQL for the JSON value (``Step.message``,
    ``ObservationResult.content``). A JSON string is returned unquoted. A
    ``ContentPart`` array joins its parts with blank lines: a ``text`` part is
    its text, and an ``image`` or ``audio`` part is ``[<type> <path>
    <media_type>]``, its ``source.path`` read from the corpus root
    (``../../blobs/...`` is ``blobs/...``). A part of any other shape renders
    nothing.
    """
    path = "json_extract_string(part, '$.source.path')"
    # ``substr(..., 7)`` drops the leading ``../../``.
    shown = (
        f"CASE WHEN starts_with({path}, '{_SESSION_BLOB_PREFIX}') "
        f"THEN substr({path}, 7) ELSE {path} END"
    )
    part = (
        "CASE WHEN json_extract_string(part, '$.type') = 'text' "
        "AND json_type(part, '$.text') = 'VARCHAR' "
        "THEN json_extract_string(part, '$.text') "
        "WHEN json_extract_string(part, '$.type') IN ('image', 'audio') "
        "AND json_type(part, '$.source.path') = 'VARCHAR' "
        "AND json_type(part, '$.source.media_type') = 'VARCHAR' "
        "THEN '[' || json_extract_string(part, '$.type') || ' ' || "
        f"{shown} || ' ' || json_extract_string(part, '$.source.media_type') || ']' END"
    )
    value = f"CAST({content} AS JSON)"
    return SqlFragment(
        f"CASE WHEN json_type({value}) = 'ARRAY' "
        f"THEN array_to_string(list_transform(CAST(json_extract({value}, '$[*]') AS JSON[]), "
        f"lambda part: {part}), '\n\n') "
        f"ELSE json_extract_string({value}, '$') END"
    )


#: The body of the ``content_text(content)`` macro.
CONTENT_TEXT_SQL: SqlFragment = content_text_sql("content")


def step_columns(access: StepMemberAccess) -> tuple[tuple[str, str], ...]:
    """``(expression, alias)`` pairs for every ``steps`` column after ``session_id``.

    Column semantics a transcript-shaped reader would misread:

    * ``message`` is flattened text: ATIF ``Step.message`` is
      ``str | ContentPart[]``, rendered by :func:`content_text_sql`, so a
      pasted image or audio clip reads as its blob path.
    * ``prompt_tokens`` is the ATIF TOTAL (input + cache_read +
      cache_creation); ``cached_tokens`` is the cache-read subset;
      ``cache_creation`` is dug out of ``metrics.extra`` because harbor only
      preserves the creation split there (fidelity gap 6). The two adapters
      spell that key differently (``cache_creation_input_tokens`` for Claude
      Code, ``cache_write_input_tokens`` for Codex), so the column coalesces
      them: one name reaches SQL, and a cost query does not branch on which
      agent wrote the session.
    * ``is_compact_summary`` / ``source_uuids`` come from ``step.extra`` per
      the CONTRACT enrichment pass. ``is_sidechain`` is ``step.extra``'s, and
      true on every step of an embedded trajectory.
    * ``agent_id`` names the subagent a sidechain step belongs to (NULL on the
      main chain): ``step.extra``'s, else the embedded trajectory's id;
      ``images`` lists the attachments pasted into a user message, each
      already moved to the corpus blob store.
    * ``trajectory_id`` is the :data:`TRAJECTORY_KEY` of the trajectory
      holding the step, so ``(session_id, trajectory_id, step_id)`` names one
      step.
    * ``reasoning_content`` is ATIF ``Step.reasoning_content`` as written,
      NULL when the step carries none.
    """
    return (
        (f"{access.json('step_id')}::BIGINT", "step_id"),
        (f"{access.string('timestamp')}::TIMESTAMP", "ts"),
        (access.string("source"), "source"),
        (access.string("model_name"), "model_name"),
        (content_text_sql(access.json("message")), "message"),
        (
            (
                f"(coalesce({access.json('extra', '.is_sidechain')}::BOOLEAN, false) "
                f"OR {EMBEDDED}::BOOLEAN)"
            ),
            "is_sidechain",
        ),
        (
            f"coalesce({access.json('extra', '.is_compact_summary')}::BOOLEAN, false)",
            "is_compact_summary",
        ),
        (f"{access.json('metrics', '.prompt_tokens')}::BIGINT", "prompt_tokens"),
        (f"{access.json('metrics', '.completion_tokens')}::BIGINT", "completion_tokens"),
        (f"{access.json('metrics', '.cached_tokens')}::BIGINT", "cached_tokens"),
        (
            (
                f"coalesce({access.json('metrics', '.extra.cache_creation_input_tokens')}, "
                f"{access.json('metrics', '.extra.cache_write_input_tokens')})::BIGINT"
            ),
            "cache_creation",
        ),
        (f"{access.json('llm_call_count')}::BIGINT", "llm_call_count"),
        (access.json("extra", ".source_uuids"), "source_uuids"),
        (
            (
                f"coalesce({access.string('extra', '.agent_id')}, "
                f"CASE WHEN {EMBEDDED}::BOOLEAN THEN {TRAJECTORY_KEY} END)"
            ),
            "agent_id",
        ),
        (access.json("extra", ".images"), "images"),
        (TRAJECTORY_KEY, "trajectory_id"),
        (access.string("reasoning_content"), "reasoning_content"),
    )


def step_key_columns(access: StepMemberAccess) -> tuple[tuple[str, str], ...]:
    """The ``(step_id, ts, trajectory_id)`` every tool call and result row carries."""
    return (
        (f"{access.json('step_id')}::BIGINT", "step_id"),
        (f"{access.string('timestamp')}::TIMESTAMP", "ts"),
        (TRAJECTORY_KEY, "trajectory_id"),
    )


#: Per-call columns over a ``call`` JSON column holding one ATIF ``ToolCall``.
#: Renamed off the ATIF field names: function_name -> tool_name,
#: tool_call_id -> tool_use_id, arguments -> tool_input. ``tool_input`` is an
#: already parsed JSON value, NOT a JSON string, so a query must not
#: json_extract it twice.
CALL_COLUMNS: tuple[tuple[str, str], ...] = (
    ("json_extract_string(call, '$.function_name')", "tool_name"),
    ("json_extract_string(call, '$.tool_call_id')", "tool_use_id"),
    ("json_extract(call, '$.arguments')", "tool_input"),
)

#: Per-result columns over a ``res`` JSON column holding one ATIF
#: ``ObservationResult``. ``source_call_id`` is ATIF's join key back to the
#: tool_calls array, surfaced under the SAME name ``tool_use_id`` that
#: ``tool_calls`` exposes, so the two views join with ``USING (session_id,
#: trajectory_id, tool_use_id)``.
#: The four typed columns after ``content`` come from ``extra``, where the
#: converter's result-signals pass writes them: NULL means the transcript did
#: not say, never "false" or "0" (see ``atif_converter.domain.result_signals``).
RESULT_COLUMNS: tuple[tuple[str, str], ...] = (
    ("json_extract_string(res, '$.source_call_id')", "tool_use_id"),
    ("json_extract(res, '$.content')", "content"),
    ("json_extract(res, '$.extra.is_error')::BOOLEAN", "is_error"),
    ("json_extract(res, '$.extra.exit_code')::BIGINT", "exit_code"),
    ("json_extract(res, '$.extra.interrupted')::BOOLEAN", "interrupted"),
    ("json_extract(res, '$.extra.images')", "images"),
)


#: ``session_events`` columns after ``session_id``, over an ``ev`` JSON column
#: holding one whole ``session_events.jsonl`` line. The registry's JSON path
#: reads the lines with ``read_json_objects`` and the producer hands each line
#: in as text, so both spell every column through these same expressions.
#: ``payload`` is the bounded record body (a parsed JSON value); ``ts`` is NULL
#: for the record types that carry no timestamp (``cost-state``, ``mode``).
EVENT_COLUMNS: tuple[tuple[str, str], ...] = (
    ("json_extract(ev, '$.seq')::BIGINT", "seq"),
    ("json_extract_string(ev, '$.ts')::TIMESTAMP", "ts"),
    ("json_extract_string(ev, '$.event_type')", "event_type"),
    ("json_extract_string(ev, '$.subtype')", "subtype"),
    ("json_extract_string(ev, '$.uuid')", "uuid"),
    ("json_extract_string(ev, '$.parent_uuid')", "parent_uuid"),
    ("json_extract_string(ev, '$.tool_use_id')", "tool_use_id"),
    ("coalesce(json_extract(ev, '$.is_sidechain')::BOOLEAN, false)", "is_sidechain"),
    ("json_extract_string(ev, '$.source_file')", "source_file"),
    ("json_extract(ev, '$.payload')", "payload"),
    ("json_extract(ev, '$.payload_bytes')::BIGINT", "payload_bytes"),
    ("coalesce(json_extract(ev, '$.payload_truncated')::BOOLEAN, false)", "payload_truncated"),
)


def render(columns: tuple[tuple[str, str], ...]) -> SqlFragment:
    """Join ``(expression, alias)`` pairs into a SELECT list body."""
    return SqlFragment(",\n    ".join(f"{expression} AS {alias}" for expression, alias in columns))


__all__ = [
    "CALL_COLUMNS",
    "CONTENT_TEXT_SQL",
    "EMBEDDED",
    "EVENT_COLUMNS",
    "MEMBER_COLUMNS",
    "RESULT_COLUMNS",
    "STEP_MEMBERS",
    "TRAJECTORY_KEY",
    "WHOLE_STEP",
    "StepMemberAccess",
    "content_text_sql",
    "render",
    "step_columns",
    "step_key_columns",
]
