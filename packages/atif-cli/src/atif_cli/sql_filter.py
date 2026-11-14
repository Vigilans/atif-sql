# SPDX-License-Identifier: Apache-2.0

"""``materialize --filter``: caller SQL scripts decide what of each session publishes.

:class:`SqlArtifactFilter` is atif-corpus's
:class:`~atif_corpus.domain.ports.ArtifactFilter`. For each converted session
it stages the session's artifacts in a private directory under the system
temp dir (mode 0700, outside the corpus): the five parquet files
atif-duck's :class:`~atif_duck.infrastructure.columnar.ColumnarArtifactProducer`
types from the trajectory, plus the session events, loss report and a minimal
``meta.json``. It stages no edges, since a filtered session publishes none, so
``messages`` is empty on both passes, as it is once the session publishes.
Each script then runs on a connection of its own over that one
session, registered like ``query``'s per-session path and locked by the same
sandbox (:func:`atif_cli.app.sandbox_corpus_connection`): the public views and
macros are all a script sees, and the statement kinds ``query`` refuses are
refused here too. A script leaves a relation named ``directives``
(:mod:`atif_cli.filter_directives`), read before its connection closes, so
every script sees the same unmodified session and a helper relation one
script creates never meets the next. A script stops publication by calling
DuckDB's ``error()``.

The merged directives apply to a copy of the session. The views are then
bound over that filtered copy and every script runs again: if the directives
of that second pass change any byte of what would publish, the filter has not
converged, and the session fails rather than publishing a result the same
scripts would not leave as it is.

Picklable: an instance holds only the scripts' text, so a materialize pool
worker receives its own copy, and ``duckdb`` and atif-duck load on first use.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from atif_cli.filter_directives import (
    Directives,
    FilterError,
    Targets,
    apply_directives,
    merge_directives,
    parse_directives,
    serialized,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from atif_corpus.domain.ports import ConversionOutput

#: The relation every filter script leaves behind.
DIRECTIVES_RELATION = "directives"


def _stage(corpus_root: Path, session_id: str, output: ConversionOutput) -> None:
    """Write ``output`` as the one session of a corpus atif-duck reads from parquet."""
    from atif_duck.domain.artifacts import LOSS_REPORT_JSON, META_JSON, SESSION_EVENTS_JSONL
    from atif_duck.infrastructure.columnar import ColumnarArtifactProducer

    session_dir = corpus_root / "sessions" / session_id
    session_dir.mkdir(parents=True)
    (session_dir / SESSION_EVENTS_JSONL).write_text(
        "".join(f"{line}\n" for line in output.events_lines), encoding="utf-8"
    )
    (session_dir / LOSS_REPORT_JSON).write_text(
        json.dumps(output.loss_report_dict), encoding="utf-8"
    )
    extras = ColumnarArtifactProducer(durable=False).produce(
        session_dir, session_id=session_id, trajectory=output.trajectory_dict
    )
    (session_dir / META_JSON).write_text(
        json.dumps({"session_id": session_id, **extras}), encoding="utf-8"
    )


class SqlArtifactFilter:
    """Filter each converted session through ``scripts``, in command-line order."""

    def __init__(self, scripts: Sequence[str]) -> None:
        #: Each script's text; at least one.
        self.scripts = tuple(scripts)

    def apply(self, output: ConversionOutput, *, session_id: str) -> ConversionOutput:
        """Run the scripts, apply their directives, and check a second pass changes nothing."""
        targets = Targets.of(output, session_id)
        directives = self._directives(output, session_id)
        targets.check(directives)
        filtered = apply_directives(output, directives, session_id=session_id)
        again = self._directives(filtered, session_id)
        targets.check(again)
        if serialized(apply_directives(filtered, again, session_id=session_id)) != serialized(
            filtered
        ):
            msg = (
                "the filter did not converge: running the scripts over the filtered "
                "session changed it again"
            )
            raise FilterError(msg)
        return filtered

    def _directives(self, output: ConversionOutput, session_id: str) -> Directives:
        """Every script's directives over ``output``, merged."""
        import duckdb

        from atif_cli.app import refused_statement_kinds, sandbox_corpus_connection

        parsed: list[list[tuple[str, tuple[object, ...]]]] = []
        with tempfile.TemporaryDirectory(prefix="atif-sql-filter-") as scratch:
            corpus_root = Path(scratch) / "corpus"
            spill_dir = Path(scratch) / "spill"
            spill_dir.mkdir(mode=0o700)
            _stage(corpus_root, session_id, output)
            for number, script in enumerate(self.scripts, start=1):
                con = sandbox_corpus_connection(corpus_root, spill_dir)
                try:
                    refused = refused_statement_kinds(con, script)
                    if refused:
                        msg = (
                            f"filter script {number}: the query sandbox does not run "
                            f"{', '.join(refused)} statements"
                        )
                        raise FilterError(msg)
                    con.execute(script)
                    try:
                        cursor = con.execute(f"SELECT * FROM {DIRECTIVES_RELATION}")  # noqa: S608  # nosec B608 - module constant
                    except duckdb.CatalogException as exc:
                        msg = f"filter script {number} left no relation named {DIRECTIVES_RELATION}"
                        raise FilterError(msg) from exc
                    columns = [column[0] for column in cursor.description or ()]
                    parsed.append(parse_directives(columns, cursor.fetchall(), script=number))
                finally:
                    con.close()
        return merge_directives(parsed)


__all__ = ["DIRECTIVES_RELATION", "SqlArtifactFilter"]
