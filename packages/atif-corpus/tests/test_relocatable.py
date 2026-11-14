# SPDX-License-Identifier: Apache-2.0

"""A relocatable pass: a corpus several machines can share.

Each session a relocatable pass converts records the machine that converted
it as ``meta.host``, beside the same artifacts an ordinary pass writes. A
pass marks, retains, re-converts from an archive and guards only its own
machine's sessions; an ordinary pass keeps the host-less ones.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from corpus_fixtures import NOW_NS, SESSION_A, SESSION_B, STALE_NS, write_session

from atif_corpus.application.materialize import (
    MaterializationReport,
    SuspiciousEmptyScanError,
    materialize,
    preview_pass,
)
from atif_corpus.domain.layout import CorpusLayout
from atif_corpus.infrastructure.fake_converter import FakeConverter


def run(
    source_root: Path,
    corpus_root: Path,
    *,
    relocatable: bool = True,
    converter_version: str = "0.1.0",
    materialized_at: str = "2026-01-02T00:00:00+00:00",
) -> MaterializationReport:
    return materialize(
        source_root=source_root,
        corpus_root=corpus_root,
        converter=FakeConverter(),
        materialized_at=materialized_at,
        harbor_version="0.22.0",
        converter_version=converter_version,
        now_ns=NOW_NS,
        relocatable=relocatable,
    )


def meta_of(corpus_root: Path, session_id: str) -> dict[str, Any]:
    meta = json.loads(CorpusLayout(corpus_root=corpus_root).meta_path(session_id).read_text())
    assert isinstance(meta, dict)
    return meta


def delete_session_source(source_root: Path, session_id: str) -> None:
    project = source_root / "-proj-a"
    (project / f"{session_id}.jsonl").unlink()
    shutil.rmtree(project / session_id, ignore_errors=True)


class TestHost:
    def test_a_relocatable_pass_records_the_machine_hostname(
        self, source_root: Path, corpus_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "atif_corpus.application.materialize.machine_hostname", lambda: "ws.example.com"
        )
        write_session(source_root, SESSION_A, mtime_ns=STALE_NS, with_side_files=True)
        run(source_root, corpus_root)
        meta = meta_of(corpus_root, SESSION_A)
        assert meta["host"] == "ws.example.com"
        # The raw source archive is written as in an ordinary pass.
        assert CorpusLayout(corpus_root=corpus_root).source_archive_dir(SESSION_A).is_dir()
        assert "source_archive" in meta

    def test_an_ordinary_pass_records_no_host(self, source_root: Path, corpus_root: Path) -> None:
        write_session(source_root, SESSION_A, mtime_ns=STALE_NS)
        run(source_root, corpus_root, relocatable=False)
        assert "host" not in meta_of(corpus_root, SESSION_A)


SESSION_C = "33333333-3333-3333-3333-333333333333"
SESSION_D = "44444444-4444-4444-4444-444444444444"


def run_on(
    host: str,
    monkeypatch: pytest.MonkeyPatch,
    source_root: Path,
    corpus_root: Path,
    *,
    converter_version: str = "0.1.0",
) -> MaterializationReport:
    """A relocatable pass on machine ``host``."""
    monkeypatch.setattr("atif_corpus.application.materialize.machine_hostname", lambda: host)
    return run(source_root, corpus_root, converter_version=converter_version)


def preview_on(
    host: str, monkeypatch: pytest.MonkeyPatch, source_root: Path, corpus_root: Path
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """``(retained, from archive)`` a relocatable pass on ``host`` would report at 0.2.0."""
    monkeypatch.setattr("atif_corpus.application.materialize.machine_hostname", lambda: host)
    preview = preview_pass(
        source_root=source_root,
        corpus_root=corpus_root,
        converter_version="0.2.0",
        now_ns=NOW_NS,
        relocatable=True,
    )
    return preview.retained_session_ids, preview.archive_session_ids


class TestEachMachineMarksItsOwnSessions:
    def test_a_pass_marks_only_its_machines_removed_sessions(
        self, tmp_path: Path, corpus_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source_a, source_b = tmp_path / "a", tmp_path / "b"
        write_session(source_a, SESSION_A, mtime_ns=STALE_NS)
        write_session(source_a, SESSION_C, mtime_ns=STALE_NS)
        write_session(source_b, SESSION_B, mtime_ns=STALE_NS)
        run_on("host-a", monkeypatch, source_a, corpus_root)
        run_on("host-b", monkeypatch, source_b, corpus_root)

        # host-b's session is absent from host-a's scan, which proves nothing.
        report = run_on("host-a", monkeypatch, source_a, corpus_root)
        assert report.removed_session_ids == ()
        assert meta_of(corpus_root, SESSION_B)["source_present"] is True

        delete_session_source(source_a, SESSION_A)
        report = run_on("host-a", monkeypatch, source_a, corpus_root)
        assert report.removed_session_ids == (SESSION_A,)
        assert report.retained_count == 1
        assert meta_of(corpus_root, SESSION_A)["source_present"] is False
        assert meta_of(corpus_root, SESSION_B)["source_present"] is True

    def test_ordinary_and_relocatable_passes_keep_to_their_own_sessions(
        self, tmp_path: Path, corpus_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source_a, source_b = tmp_path / "a", tmp_path / "b"
        write_session(source_a, SESSION_A, mtime_ns=STALE_NS)
        write_session(source_b, SESSION_B, mtime_ns=STALE_NS)
        run(source_a, corpus_root, relocatable=False)
        run_on("host-b", monkeypatch, source_b, corpus_root)

        assert run(source_a, corpus_root, relocatable=False).removed_session_ids == ()
        assert meta_of(corpus_root, SESSION_B)["source_present"] is True

        delete_session_source(source_b, SESSION_B)
        write_session(source_b, SESSION_C, mtime_ns=STALE_NS)
        assert run_on("host-b", monkeypatch, source_b, corpus_root).removed_session_ids == (
            SESSION_B,
        )
        assert meta_of(corpus_root, SESSION_A)["source_present"] is True

    def test_the_preview_retains_only_its_machines_sessions(
        self, tmp_path: Path, corpus_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source_a, source_b = tmp_path / "a", tmp_path / "b"
        write_session(source_a, SESSION_A, mtime_ns=STALE_NS)
        write_session(source_b, SESSION_B, mtime_ns=STALE_NS)
        run_on("host-a", monkeypatch, source_a, corpus_root)
        run_on("host-b", monkeypatch, source_b, corpus_root)
        delete_session_source(source_a, SESSION_A)
        delete_session_source(source_b, SESSION_B)
        write_session(source_a, SESSION_C, mtime_ns=STALE_NS)
        write_session(source_b, SESSION_D, mtime_ns=STALE_NS)
        run_on("host-a", monkeypatch, source_a, corpus_root)
        run_on("host-b", monkeypatch, source_b, corpus_root)

        monkeypatch.setattr(
            "atif_corpus.application.materialize.machine_hostname", lambda: "host-a"
        )
        preview = preview_pass(
            source_root=source_a,
            corpus_root=corpus_root,
            converter_version="0.1.0",
            now_ns=NOW_NS,
            relocatable=True,
        )
        assert preview.retained_session_ids == (SESSION_A,)


class TestEachMachineReconvertsItsOwnArchives:
    def test_this_machines_expired_session_is_reconverted_from_its_archive(
        self, tmp_path: Path, corpus_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source_a = tmp_path / "a"
        write_session(source_a, SESSION_A, mtime_ns=STALE_NS)
        write_session(source_a, SESSION_C, mtime_ns=STALE_NS)
        run_on("host-a", monkeypatch, source_a, corpus_root)
        delete_session_source(source_a, SESSION_A)
        assert preview_on("host-a", monkeypatch, source_a, corpus_root) == (
            (SESSION_A,),
            (SESSION_A,),
        )

        report = run_on("host-a", monkeypatch, source_a, corpus_root, converter_version="0.2.0")

        assert report.archive_session_ids == (SESSION_A,)
        meta = meta_of(corpus_root, SESSION_A)
        assert (meta["converter_version"], meta["host"]) == ("0.2.0", "host-a")
        assert meta["source_present"] is False
        assert "source_archive" in meta
        assert CorpusLayout(corpus_root=corpus_root).source_archive_dir(SESSION_A).is_dir()

    def test_another_machines_expired_session_is_left_to_that_machine(
        self, tmp_path: Path, corpus_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source_a, source_b = tmp_path / "a", tmp_path / "b"
        write_session(source_a, SESSION_A, mtime_ns=STALE_NS)
        write_session(source_b, SESSION_B, mtime_ns=STALE_NS)
        write_session(source_b, SESSION_D, mtime_ns=STALE_NS)
        run_on("host-a", monkeypatch, source_a, corpus_root)
        run_on("host-b", monkeypatch, source_b, corpus_root)
        delete_session_source(source_b, SESSION_B)
        run_on("host-b", monkeypatch, source_b, corpus_root)
        assert preview_on("host-a", monkeypatch, source_a, corpus_root) == ((), ())

        report = run_on("host-a", monkeypatch, source_a, corpus_root, converter_version="0.2.0")

        assert report.archive_session_ids == ()
        assert meta_of(corpus_root, SESSION_B)["converter_version"] == "0.1.0"
        report = run_on("host-b", monkeypatch, source_b, corpus_root, converter_version="0.2.0")
        assert report.archive_session_ids == (SESSION_B,)
        assert meta_of(corpus_root, SESSION_B)["converter_version"] == "0.2.0"

    def test_an_archived_ordinary_session_is_left_to_ordinary_passes(
        self, source_root: Path, corpus_root: Path
    ) -> None:
        write_session(source_root, SESSION_A, mtime_ns=STALE_NS)
        write_session(source_root, SESSION_B, mtime_ns=STALE_NS)
        run(source_root, corpus_root, relocatable=False)
        delete_session_source(source_root, SESSION_A)
        run(source_root, corpus_root, relocatable=False)
        preview = preview_pass(
            source_root=source_root,
            corpus_root=corpus_root,
            converter_version="0.2.0",
            now_ns=NOW_NS,
            relocatable=True,
        )
        assert preview.retained_session_ids == ()
        assert preview.archive_session_ids == ()

        report = run(source_root, corpus_root, converter_version="0.2.0")

        assert report.archive_session_ids == ()
        assert report.retained_count == 0
        # The host-less session keeps the generation it was converted under;
        # the live one is re-converted under this host.
        assert meta_of(corpus_root, SESSION_A)["converter_version"] == "0.1.0"
        assert meta_of(corpus_root, SESSION_B)["converter_version"] == "0.2.0"
        report = run(source_root, corpus_root, relocatable=False, converter_version="0.2.0")
        assert report.archive_session_ids == (SESSION_A,)


class TestTheEmptyScanGuard:
    def test_another_machines_sessions_do_not_trip_it(
        self, tmp_path: Path, corpus_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source_a, source_b = tmp_path / "a", tmp_path / "b"
        source_a.mkdir()
        write_session(source_b, SESSION_B, mtime_ns=STALE_NS)
        run_on("host-b", monkeypatch, source_b, corpus_root)
        assert run_on("host-a", monkeypatch, source_a, corpus_root).removed_session_ids == ()

    def test_its_own_machines_sessions_trip_it(
        self, tmp_path: Path, corpus_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source_a = tmp_path / "a"
        write_session(source_a, SESSION_A, mtime_ns=STALE_NS)
        run_on("host-a", monkeypatch, source_a, corpus_root)
        delete_session_source(source_a, SESSION_A)
        with pytest.raises(SuspiciousEmptyScanError, match="refusing to mark them"):
            run_on("host-a", monkeypatch, source_a, corpus_root)
