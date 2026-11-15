# SPDX-License-Identifier: Apache-2.0

"""Derived, test-executed example queries for the atif-duck SQL surface.

Operator's principle: DON'T hardcode example strings — they rot the moment a
macro signature or view changes. Every example here is DERIVED from the
static catalogs (:data:`~atif_duck.domain.catalog.VIEW_NAMES`,
:data:`~atif_duck.domain.catalog.ANALYTICS_VIEW_NAMES`,
:data:`~atif_duck.domain.catalog.MACRO_SIGNATURES`,
:data:`~atif_duck.domain.catalog.ANALYTICS_MACRO_SIGNATURES`) plus one small
exemplar table mapping parameter NAMES to literal SQL argument text
(:data:`ARG_EXEMPLARS`). The catalogs are already drift-pinned against the
real DDL by the tests in ``tests/test_duck_views.py`` /
``tests/test_analytics_views.py``, so the examples inherit that freshness.

THE TEST IS THE CONTRACT: ``tests/test_examples.py`` EXECUTES every example
:func:`build_examples` emits against the fixture corpus (analytics parquets
registered, tiny Lance store attached). An example that stops running stops
shipping — the CLI surface (``atif-sql examples``) can honestly claim
"test-executed against this version".

The only hand-maintained prose is
:data:`~atif_duck.domain.catalog.DESCRIPTIONS` in ``catalog.py``; a coverage
drift test forces one description per catalog object.

:data:`FILTER_EXAMPLES` are the exception to derivation: ``materialize
--filter`` scripts, which are policies over the public views rather than
objects of the catalog. Each leaves the ``directives`` relation the filter
stage reads, and each is executed twice over: by ``tests/test_examples.py``
against the fixture corpus, and by atif-cli's tests through ``materialize
--filter``.

Pure domain module: no duckdb import, no I/O — safe on the CLI's lean
import path (``schema`` / ``examples`` / ``--help``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast, get_args

from atif_duck.domain.catalog import (
    ANALYTICS_MACRO_SIGNATURES,
    ANALYTICS_VIEW_NAMES,
    ANALYTICS_VIEW_SCHEMA,
    DESCRIPTIONS,
    MACRO_SIGNATURES,
    TABLE_MACRO_NAMES,
    VIEW_NAMES,
    VIEW_SCHEMA,
)

#: What an example needs registered before it runs: the base corpus views
#: (``core``), the analytics parquet views (``analytics``), or the
#: Lance-backed embeddings surface (``vss``).
Requires = Literal["core", "analytics", "vss"]

#: Structural kind, derived from which catalog the object lives in and
#: whether its DDL is ``AS TABLE`` (see
#: :data:`~atif_duck.domain.catalog.TABLE_MACRO_NAMES`); ``filter`` for a
#: ``materialize --filter`` script.
Category = Literal["view", "table-macro", "scalar-macro", "filter"]

REQUIRES_VALUES: tuple[str, ...] = cast("tuple[str, ...]", get_args(Requires))
CATEGORY_VALUES: tuple[str, ...] = cast("tuple[str, ...]", get_args(Category))

#: Exemplar SQL argument text per macro parameter NAME. The single seam
#: between the derived examples and real macro signatures: a new parameter
#: name lands here (one line) or ``build_examples`` fails loud — and the
#: executing test in ``tests/test_examples.py`` fails with it.
#:
#: * ``sid`` — a real session id, picked from the corpus itself so the
#:   example runs verbatim on ANY materialized corpus.
#: * ``query_vec`` — a stored embedding L2-NORMALIZED into a unit-norm probe,
#:   matching what ``embed_query`` returns, so the example runs on any corpus
#:   without fabricating a dim-wide literal. The normalization is not
#:   cosmetic: stored vectors are un-normalized, and a RAW stored vector is
#:   its own nearest neighbor under every metric, so an un-normalized probe
#:   would make this example score sim == 1.0 no matter which metric the macro
#:   ranks by — it could not tell a correct macro from a broken one. For TEXT
#:   queries the flow is two-step: embed the query text first (``atif-sql
#:   search`` wraps both steps); see
#:   :data:`~atif_duck.domain.catalog.DESCRIPTIONS`.
ARG_EXEMPLARS: dict[str, str] = {
    "sid": "(SELECT session_id FROM sessions LIMIT 1)",
    "since_days": "30",
    "last_n_days": "30",
    "k": "10",
    "n": "10",
    "label_name": "'correction'",
    "signal_name": "'correction'",
    "src": "'user'",
    "msg": "'Stop hook feedback: the claim has no evidence'",
    "interval_text": "'7 days'",
    "query_vec": (
        "(SELECT list_transform("
        "me.embedding, lambda v: (v / sqrt(list_sum("
        "list_transform(me.embedding, lambda x: x::DOUBLE * x))))::FLOAT"
        ") FROM message_embeddings me LIMIT 1)"
    ),
}

#: Catalog objects DELIBERATELY without an example, mapped to the reason.
#: The coverage drift test treats these as documented exclusions; everything
#: else in the catalogs MUST yield an example. Currently empty — every view
#: and macro is exampled — but the seam stays so a future exclusion is a
#: reviewed one-liner instead of a silent coverage hole.
EXCLUSIONS: dict[str, str] = {}


@dataclass(frozen=True, slots=True)
class ExampleQuery:
    """One runnable example query, derived from the catalog.

    Attributes
    ----------
    name
        The catalog object (view or macro) the example demonstrates.
    sql
        A complete, runnable statement for ``atif-sql query``.
    description
        Agent-facing prose from :data:`~atif_duck.domain.catalog.DESCRIPTIONS`.
    requires
        ``core`` | ``analytics`` | ``vss`` — what must be registered/populated.
    category
        ``view`` | ``table-macro`` | ``scalar-macro`` | ``filter``.
    """

    name: str
    sql: str
    description: str
    requires: Requires
    category: Category


#: Catalog objects that need the embeddings store (``atif-sql embed``).
_VSS_OBJECTS: frozenset[str] = frozenset({"message_embeddings", "semantic_search"})


def object_requires(name: str) -> Requires:
    """What must be registered/populated before catalog object ``name`` binds.

    ``analytics`` for the parquet-gated analytics views and macros, ``vss``
    for the embeddings surface, ``core`` for everything the corpus alone
    registers (the authorship views included). ``atif-sql schema`` prints
    this beside every view and macro, and each derived example carries it.
    """
    if name in ANALYTICS_VIEW_SCHEMA or name in ANALYTICS_MACRO_SIGNATURES:
        return "analytics"
    return "vss" if name in _VSS_OBJECTS else "core"


def _view_sql(name: str) -> str:
    """Canonical top-N SELECT for one view.

    When the static :data:`~atif_duck.domain.catalog.VIEW_SCHEMA` (or
    :data:`~atif_duck.domain.catalog.ANALYTICS_VIEW_SCHEMA`) declares a
    timestamp column, the example orders by the FIRST one descending
    (newest-first is the agent's default question); otherwise a plain LIMIT.
    """
    schema = VIEW_SCHEMA.get(name) or ANALYTICS_VIEW_SCHEMA.get(name, ())
    ts_col = next((col for col, typ in schema if typ.startswith("TIMESTAMP")), None)
    if ts_col is not None:
        # The view name and its timestamp column both come from the static catalog.
        return f"SELECT * FROM {name} ORDER BY {ts_col} DESC LIMIT 10"  # noqa: S608  # nosec B608 - view name and column are VIEW_SCHEMA constants
    return f"SELECT * FROM {name} LIMIT 10"  # noqa: S608  # nosec B608 - view name is a VIEW_SCHEMA constant


def _macro_sql(name: str, params: tuple[str, ...]) -> str:
    """Runnable invocation for one macro, args filled from the exemplars."""
    try:
        args = ", ".join(ARG_EXEMPLARS[param] for param in params)
    except KeyError as exc:
        msg = (
            f"macro {name!r} has parameter {exc.args[0]!r} with no entry in "
            "ARG_EXEMPLARS — add an exemplar literal so build_examples() can "
            "derive a runnable example"
        )
        raise KeyError(msg) from exc
    if name in TABLE_MACRO_NAMES:
        # The macro name comes from MACRO_SIGNATURES and args from ARG_EXEMPLARS.
        return f"SELECT * FROM {name}({args}) LIMIT 10"  # noqa: S608  # nosec B608 - macro name from MACRO_SIGNATURES, args from ARG_EXEMPLARS
    return f"SELECT {name}({args}) AS {name}"


def _description(name: str) -> str:
    """DESCRIPTIONS lookup that fails loud on a missing catalog object."""
    try:
        return DESCRIPTIONS[name]
    except KeyError as exc:
        msg = (
            f"catalog object {name!r} has no entry in "
            "atif_duck.domain.catalog.DESCRIPTIONS — add one (the drift test "
            "in tests/test_examples.py pins this coverage)"
        )
        raise KeyError(msg) from exc


def build_examples() -> tuple[ExampleQuery, ...]:
    """Derive the full example inventory from the catalogs.

    Order: core views, VSS view, analytics views, then core macros, VSS
    macro, analytics macros — stable, so the CLI listing and the JSON array
    are deterministic across runs.
    """
    examples: list[ExampleQuery] = []

    for view in VIEW_NAMES:
        if view in EXCLUSIONS:
            continue
        examples.append(
            ExampleQuery(
                name=view,
                sql=_view_sql(view),
                description=_description(view),
                requires=object_requires(view),
                category="view",
            )
        )

    for view in ANALYTICS_VIEW_NAMES:
        if view in EXCLUSIONS:
            continue
        examples.append(
            ExampleQuery(
                name=view,
                sql=_view_sql(view),
                description=_description(view),
                requires=object_requires(view),
                category="view",
            )
        )

    for macro, params in MACRO_SIGNATURES.items():
        if macro in EXCLUSIONS:
            continue
        examples.append(
            ExampleQuery(
                name=macro,
                sql=_macro_sql(macro, params),
                description=_description(macro),
                requires=object_requires(macro),
                category="table-macro" if macro in TABLE_MACRO_NAMES else "scalar-macro",
            )
        )

    for macro, params in ANALYTICS_MACRO_SIGNATURES.items():
        if macro in EXCLUSIONS:
            continue
        examples.append(
            ExampleQuery(
                name=macro,
                sql=_macro_sql(macro, params),
                description=_description(macro),
                requires=object_requires(macro),
                category="table-macro" if macro in TABLE_MACRO_NAMES else "scalar-macro",
            )
        )

    return tuple(examples)


#: ``materialize --filter`` scripts for the common policies, each leaving the
#: ``directives`` relation. Run over one converted session, so a keep list or
#: a threshold is the only thing a caller adapts.
FILTER_EXAMPLES: tuple[ExampleQuery, ...] = (
    ExampleQuery(
        name="prune_uncited_tool_results",
        sql="""\
CREATE TEMP VIEW directives AS
WITH keep(trajectory_id, tool_use_id) AS (
    SELECT trajectory_id, tool_use_id FROM tool_results LIMIT 1
)
SELECT 'prune_tool_result' AS action, r.trajectory_id, r.step_id, r.tool_use_id
FROM tool_results r
WHERE NOT EXISTS (
    SELECT 1 FROM keep k
    WHERE k.trajectory_id = r.trajectory_id AND k.tool_use_id = r.tool_use_id
);""",
        description=(
            "Prune every tool result except the (trajectory_id, tool_use_id) pairs in keep "
            "(replace the subquery with VALUES ... to pin cited calls)."
        ),
        requires="core",
        category="filter",
    ),
    ExampleQuery(
        name="prune_large_tool_results",
        sql="""\
CREATE TEMP VIEW directives AS
SELECT 'prune_tool_result' AS action, trajectory_id, step_id, tool_use_id
FROM tool_results
WHERE strlen(content::VARCHAR) > 16384;""",
        description="Prune every tool result whose content is over 16 KiB of JSON text.",
        requires="core",
        category="filter",
    ),
    ExampleQuery(
        name="redact_credentials",
        sql="""\
CREATE TEMP VIEW secrets AS
SELECT DISTINCT unnest(regexp_extract_all(text, '\\b(sk-[A-Za-z0-9_-]{16,})', 1)) AS secret
FROM (
    SELECT message AS text FROM steps
    UNION ALL SELECT tool_input::VARCHAR FROM tool_calls
    UNION ALL SELECT content::VARCHAR FROM tool_results
    UNION ALL SELECT payload::VARCHAR FROM session_events
);
CREATE TEMP VIEW directives AS
SELECT DISTINCT 'replace' AS action, variant AS find, '[REDACTED]' AS replacement
FROM secrets,
     LATERAL (VALUES
         (secret),
         (url_encode(secret)),
         (to_base64(encode(secret))),
         (substr(to_json(secret)::VARCHAR, 2, length(to_json(secret)::VARCHAR) - 2))
     ) AS v(variant);""",
        description=(
            "Find provider-style keys with an RE2 capture group over messages, tool "
            "inputs, tool results and event payloads, and replace each with "
            "[REDACTED] as written, URL-encoded (url_encode), base64-encoded "
            "(to_base64) and JSON-escaped (to_json)."
        ),
        requires="core",
        category="filter",
    ),
    ExampleQuery(
        name="replace_home_paths",
        sql="""\
CREATE TEMP VIEW directives AS
SELECT DISTINCT 'replace' AS action, home AS find, '~' AS replacement
FROM (SELECT regexp_extract(cwd, '^(/home/[^/]+|/Users/[^/]+)', 1) AS home FROM sessions)
WHERE home <> '';""",
        description=(
            "Replace the home directory the session's cwd starts with (/home/<user> or "
            "/Users/<user>) with ~ everywhere in the session."
        ),
        requires="core",
        category="filter",
    ),
    ExampleQuery(
        name="drop_reasoning_content",
        sql="""\
CREATE TEMP VIEW directives AS
SELECT 'drop_reasoning' AS action, trajectory_id, step_id
FROM steps
WHERE reasoning_content IS NOT NULL;""",
        description="Clear reasoning_content on every step that has any.",
        requires="core",
        category="filter",
    ),
    ExampleQuery(
        name="drop_images_and_blobs",
        sql="""\
CREATE TEMP VIEW directives AS
SELECT DISTINCT 'drop_blob' AS action, blob_sha256
FROM (
    SELECT sha256 AS blob_sha256 FROM images
    UNION ALL
    SELECT unnest(regexp_extract_all(text, 'sha256:([0-9a-f]{64})', 1))
    FROM (SELECT message AS text FROM steps UNION ALL SELECT content::VARCHAR FROM tool_results)
);""",
        description=(
            "Drop every attachment: each image the images view lists and each blob a "
            "sha256 placeholder names in a message or a tool result, so none is published."
        ),
        requires="core",
        category="filter",
    ),
)


__all__ = [
    "ARG_EXEMPLARS",
    "CATEGORY_VALUES",
    "EXCLUSIONS",
    "FILTER_EXAMPLES",
    "REQUIRES_VALUES",
    "Category",
    "ExampleQuery",
    "Requires",
    "build_examples",
    "object_requires",
]
