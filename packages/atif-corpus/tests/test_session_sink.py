# SPDX-License-Identifier: Apache-2.0

"""The ``SessionSink`` port: when materialize calls it, and what a failure leaves.

atif-corpus knows nothing about what the sink keeps (atif-cli plugs in
atif-duck's DuckLake writer, which compares the corpus with the lake), so
these tests use a recording fake and pin the contract any sink gets: one call
per pass, with the corpus root and the agent, after every swap and
source-removed mark and in this process; and a sink failure never fails the
pass, which reports it with the sessions it published.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from corpus_fixtures import NOW_NS, SESSION_A, SESSION_B, STALE_NS, write_session

from atif_corpus.application.materialize import MaterializationReport, materialize
from atif_corpus.domain.layout import CorpusLayout
from atif_corpus.infrastructure.fake_converter import FakeConverter

SESSION_C = "33333333-3333-3333-3333-333333333333"
SESSION_D = "44444444-4444-4444-4444-444444444444"
SESSIONS = (SESSION_A, SESSION_B, SESSION_C, SESSION_D)

VERSIONS: dict[str, Any] = {
    "materialized_at": "2026-01-02T00:00:00+00:00",
    "harbor_version": "0.23.0",
    "converter_version": "0.1.0",
}


class RecordingSink:
    """Records the published ``meta.json`` of every session at each call; may raise."""

    def __init__(self, *, error: type[BaseException] | None = None) -> None:
        self.calls: list[tuple[Path, str, dict[str, dict[str, Any]]]] = []
        self.error = error

    def sync_corpus(self, *, corpus_root: Path, agent: str) -> int:
        sessions = CorpusLayout(corpus_root=corpus_root).sessions_dir
        metas = {
            entry.name: json.loads((entry / "meta.json").read_text())
            for entry in sorted(sessions.iterdir())
            if (entry / "meta.json").is_file()
        }
        self.calls.append((corpus_root, agent, metas))
        if self.error is not None:
            msg = "scripted sink failure"
            raise self.error(msg)
        return len(metas)


def run(
    source_root: Path,
    corpus_root: Path,
    sink: RecordingSink | None,
    *,
    converter: FakeConverter | None = None,
    workers: int = 1,
) -> MaterializationReport:
    return materialize(
        source_root=source_root,
        corpus_root=corpus_root,
        converter=converter or FakeConverter(),
        now_ns=NOW_NS,
        session_sink=sink,
        workers=workers,
        **VERSIONS,
    )


def _write_all(source_root: Path) -> None:
    for session_id in SESSIONS:
        write_session(source_root, session_id, mtime_ns=STALE_NS)


class TestWhenTheSinkIsCalled:
    @pytest.mark.parametrize("workers", [1, 3])
    def test_once_after_every_session_is_published(
        self, source_root: Path, corpus_root: Path, workers: int
    ) -> None:
        _write_all(source_root)
        sink = RecordingSink()
        report = run(source_root, corpus_root, sink, workers=workers)
        assert [(root, agent, tuple(metas)) for root, agent, metas in sink.calls] == [
            (corpus_root, "claude-code", SESSIONS)
        ]
        assert report.sink_synced_count == len(SESSIONS)
        assert report.sink_pending_session_ids == ()
        assert report.sink_error is None

    def test_an_up_to_date_pass_still_calls_it(self, source_root: Path, corpus_root: Path) -> None:
        """The sink may be behind the corpus without any source moving."""
        _write_all(source_root)
        run(source_root, corpus_root, None)
        sink = RecordingSink()
        report = run(source_root, corpus_root, sink)
        assert report.materialized_count == 0
        assert len(sink.calls) == 1

    def test_a_source_removed_session_is_marked_before_the_call(
        self, source_root: Path, corpus_root: Path
    ) -> None:
        _write_all(source_root)
        run(source_root, corpus_root, None)
        (source_root / "-proj-a" / f"{SESSION_B}.jsonl").unlink()
        sink = RecordingSink()
        report = run(source_root, corpus_root, sink)
        assert report.removed_session_ids == (SESSION_B,)
        ((_, _, metas),) = sink.calls
        assert metas[SESSION_B]["source_present"] is False


class TestAFailingSink:
    def test_the_failure_is_reported_with_the_sessions_published(
        self, source_root: Path, corpus_root: Path
    ) -> None:
        _write_all(source_root)
        converter = FakeConverter(
            fail_sessions=frozenset({SESSION_B}), empty_sessions=frozenset({SESSION_C})
        )
        report = run(
            source_root, corpus_root, RecordingSink(error=RuntimeError), converter=converter
        )
        assert report.sink_pending_session_ids == (SESSION_A, SESSION_D)
        assert report.sink_synced_count == 0
        assert report.sink_error is not None
        assert "scripted sink failure" in report.sink_error
        # The pass itself went on: artifacts published, the failure is the converter's.
        assert [f.session_id for f in report.failures] == [SESSION_B]
        assert report.materialized_count == 2

    def test_the_next_pass_calls_it_again_with_no_source_change(
        self, source_root: Path, corpus_root: Path
    ) -> None:
        _write_all(source_root)
        run(source_root, corpus_root, RecordingSink(error=RuntimeError))
        retry = RecordingSink()
        report = run(source_root, corpus_root, retry)
        assert report.materialized_count == 0
        assert report.sink_synced_count == len(SESSIONS)
        assert report.sink_pending_session_ids == ()

    def test_an_interrupt_is_not_swallowed(self, source_root: Path, corpus_root: Path) -> None:
        _write_all(source_root)
        with pytest.raises(KeyboardInterrupt):
            run(source_root, corpus_root, RecordingSink(error=KeyboardInterrupt))
