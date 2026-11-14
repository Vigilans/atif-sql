# SPDX-License-Identifier: Apache-2.0

"""The artifact filter port: what a filtered pass publishes, and what an unfiltered one refuses.

A filter sees each converted session before its blobs are stored or anything
is staged, and only its output publishes. A filtered pass writes no raw
source archive or edges and re-converts nothing from an archive, and every
session it writes says ``filtered: true``. An unfiltered pass records a
failure for each filtered session it plans instead of overwriting it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, override

import pytest
from corpus_fixtures import (
    NOW_NS,
    SESSION_A,
    SESSION_B,
    STALE_NS,
    read_artifact_text,
    write_session,
)

from atif_corpus.application.materialize import (
    MaterializationReport,
    Provenance,
    materialize,
    read_watermark,
)
from atif_corpus.domain.layout import CorpusLayout
from atif_corpus.domain.ports import BlobOutput, ConversionOutput
from atif_corpus.infrastructure.fake_converter import FakeConverter

KEPT = b"\x89PNG kept"
DROPPED = b"\x89PNG dropped"


def _blob(data: bytes) -> BlobOutput:
    return BlobOutput(sha256=hashlib.sha256(data).hexdigest(), extension="png", data=data)


class BlobConverter(FakeConverter):
    """The fake converter, plus two attachments per session."""

    @override
    def convert(
        self,
        session_jsonl: Path,
        *,
        source_files: Sequence[Path] = (),
        archive_dir: Path | None = None,
    ) -> ConversionOutput:
        base = super().convert(session_jsonl, source_files=source_files, archive_dir=archive_dir)
        return dataclasses.replace(base, blobs=(_blob(KEPT), _blob(DROPPED)))


class TagFilter:
    """Tags the trajectory, drops the ``DROPPED`` blob, and stamps its pid. Picklable."""

    def apply(self, output: ConversionOutput, *, session_id: str) -> ConversionOutput:
        trajectory = {
            **output.trajectory_dict,
            "filtered_by": session_id,
            "filter_pid": os.getpid(),
        }
        return dataclasses.replace(
            output,
            trajectory_dict=trajectory,
            blobs=tuple(blob for blob in output.blobs if blob.data != DROPPED),
        )


class FailingFilter:
    """Rejects every session it sees."""

    def apply(self, output: ConversionOutput, *, session_id: str) -> ConversionOutput:
        msg = f"policy violation in {session_id}"
        raise ValueError(msg)


def run(
    source_root: Path,
    corpus_root: Path,
    *,
    artifact_filter: Any = None,
    converter: FakeConverter | None = None,
    force: bool = False,
    relocatable: bool = False,
    converter_version: str = "0.1.0",
    workers: int = 1,
    now_ns: int = NOW_NS,
) -> MaterializationReport:
    return materialize(
        source_root=source_root,
        corpus_root=corpus_root,
        converter=converter or BlobConverter(),
        provenance=Provenance(
            materialized_at="2026-01-02T00:00:00+00:00",
            harbor_version="0.22.0",
            converter_version=converter_version,
        ),
        now_ns=now_ns,
        force=force,
        relocatable=relocatable,
        artifact_filter=artifact_filter,
        workers=workers,
    )


def meta_of(corpus_root: Path, session_id: str) -> dict[str, Any]:
    meta = json.loads(CorpusLayout(corpus_root=corpus_root).meta_path(session_id).read_text())
    assert isinstance(meta, dict)
    return meta


def trajectory_of(corpus_root: Path, session_id: str) -> dict[str, Any]:
    layout = CorpusLayout(corpus_root=corpus_root)
    document = json.loads(read_artifact_text(layout.trajectory_path(session_id)))
    assert isinstance(document, dict)
    return document


def stored_blobs(corpus_root: Path) -> set[str]:
    blobs_dir = CorpusLayout(corpus_root=corpus_root).blobs_dir
    return {path.stem for path in blobs_dir.rglob("*") if path.is_file()}


class TestFilteredPass:
    def test_only_the_filter_output_publishes(self, source_root: Path, corpus_root: Path) -> None:
        write_session(source_root, SESSION_A, mtime_ns=STALE_NS, with_side_files=True)
        report = run(source_root, corpus_root, artifact_filter=TagFilter())
        assert report.materialized_count == 1
        assert trajectory_of(corpus_root, SESSION_A)["filtered_by"] == SESSION_A
        # The blob the filter left out never reached the shared store.
        assert stored_blobs(corpus_root) == {_blob(KEPT).sha256}
        meta = meta_of(corpus_root, SESSION_A)
        assert meta["filtered"] is True

    @pytest.mark.parametrize("relocatable", [False, True])
    def test_a_filtered_pass_writes_no_source_archive_or_edges(
        self, source_root: Path, corpus_root: Path, relocatable: bool
    ) -> None:
        write_session(source_root, SESSION_A, mtime_ns=STALE_NS, with_side_files=True)
        run(source_root, corpus_root, artifact_filter=TagFilter(), relocatable=relocatable)
        layout = CorpusLayout(corpus_root=corpus_root)
        assert not layout.source_archive_dir(SESSION_A).exists()
        assert "source_archive" not in meta_of(corpus_root, SESSION_A)
        assert not layout.edges_path(SESSION_A).exists()
        assert sorted(path.name for path in layout.session_dir(SESSION_A).iterdir()) == [
            "loss_report.json",
            "meta.json",
            "session_events.jsonl.zst",
            "trajectory.json.zst",
        ]

    def test_refiltering_an_unfiltered_session_drops_its_archive_and_edges(
        self, source_root: Path, corpus_root: Path
    ) -> None:
        write_session(source_root, SESSION_A, mtime_ns=STALE_NS, with_side_files=True)
        run(source_root, corpus_root)
        layout = CorpusLayout(corpus_root=corpus_root)
        assert layout.source_archive_dir(SESSION_A).is_dir()
        assert layout.edges_path(SESSION_A).is_file()
        assert "filtered" not in meta_of(corpus_root, SESSION_A)

        run(source_root, corpus_root, artifact_filter=TagFilter(), force=True)
        assert not layout.source_archive_dir(SESSION_A).exists()
        assert not layout.edges_path(SESSION_A).exists()
        meta = meta_of(corpus_root, SESSION_A)
        assert "source_archive" not in meta
        assert meta["filtered"] is True

    def test_a_filter_failure_publishes_nothing(self, source_root: Path, corpus_root: Path) -> None:
        write_session(source_root, SESSION_A, mtime_ns=STALE_NS)
        report = run(source_root, corpus_root, artifact_filter=FailingFilter())
        assert [failure.session_id for failure in report.failures] == [SESSION_A]
        assert report.failures[0].error == f"ValueError: policy violation in {SESSION_A}"
        layout = CorpusLayout(corpus_root=corpus_root)
        assert not layout.session_dir(SESSION_A).exists()
        assert stored_blobs(corpus_root) == set()
        assert read_watermark(layout.watermark_path) == {}

    def test_a_filtered_pass_reconverts_nothing_from_an_archive(
        self, source_root: Path, corpus_root: Path
    ) -> None:
        write_session(source_root, SESSION_A, mtime_ns=STALE_NS)
        write_session(source_root, SESSION_B, mtime_ns=STALE_NS)
        run(source_root, corpus_root)
        (source_root / "-proj-a" / f"{SESSION_A}.jsonl").unlink()
        # A converter version change makes the retained session stale; an
        # unfiltered pass would re-convert it from its archive.
        report = run(
            source_root, corpus_root, artifact_filter=TagFilter(), converter_version="0.2.0"
        )
        assert report.archive_session_ids == ()
        assert report.retained_count == 1
        meta = meta_of(corpus_root, SESSION_A)
        assert meta["converter_version"] == "0.1.0"
        assert "filtered" not in meta

    def test_the_pool_applies_the_filter_in_every_worker(
        self, source_root: Path, corpus_root: Path
    ) -> None:
        for session_id in (SESSION_A, SESSION_B):
            write_session(source_root, session_id, mtime_ns=STALE_NS)
        report = run(
            source_root,
            corpus_root,
            artifact_filter=TagFilter(),
            converter=BlobConverter(delay_seconds=0.3),
            workers=2,
        )
        assert report.workers == 2
        assert report.materialized_count == 2
        for session_id in (SESSION_A, SESSION_B):
            trajectory = trajectory_of(corpus_root, session_id)
            assert trajectory["filtered_by"] == session_id
            assert trajectory["filter_pid"] != os.getpid()
            assert meta_of(corpus_root, session_id)["filtered"] is True


class TestUnfilteredPassOverAFilteredSession:
    def _filtered_corpus(self, source_root: Path, corpus_root: Path) -> bytes:
        write_session(source_root, SESSION_A, mtime_ns=STALE_NS)
        write_session(source_root, SESSION_B, mtime_ns=STALE_NS)
        run(source_root, corpus_root, artifact_filter=TagFilter())
        layout = CorpusLayout(corpus_root=corpus_root)
        return layout.meta_path(SESSION_A).read_bytes()

    def _assert_untouched(self, corpus_root: Path, meta_bytes: bytes) -> None:
        layout = CorpusLayout(corpus_root=corpus_root)
        assert layout.meta_path(SESSION_A).read_bytes() == meta_bytes
        assert trajectory_of(corpus_root, SESSION_A)["filtered_by"] == SESSION_A
        assert not layout.source_archive_dir(SESSION_A).exists()

    def test_force_fails_the_filtered_session_and_leaves_it(
        self, source_root: Path, corpus_root: Path
    ) -> None:
        meta_bytes = self._filtered_corpus(source_root, corpus_root)
        report = run(source_root, corpus_root, force=True)
        assert [failure.session_id for failure in report.failures] == [SESSION_A, SESSION_B]
        assert all(
            failure.error.startswith("FilteredSessionError: ") for failure in report.failures
        )
        self._assert_untouched(corpus_root, meta_bytes)

    def test_a_grown_transcript_fails_and_stays_stale(
        self, source_root: Path, corpus_root: Path
    ) -> None:
        meta_bytes = self._filtered_corpus(source_root, corpus_root)
        layout = CorpusLayout(corpus_root=corpus_root)
        recorded = read_watermark(layout.watermark_path)
        later = STALE_NS + 1_000_000_000
        write_session(source_root, SESSION_A, mtime_ns=later)
        report = run(source_root, corpus_root)
        assert [failure.session_id for failure in report.failures] == [SESSION_A]
        assert report.up_to_date_count == 1
        self._assert_untouched(corpus_root, meta_bytes)
        # The failure leaves the watermark where it was, so the session is
        # still stale for the next filtered pass.
        assert read_watermark(layout.watermark_path) == recorded
        report = run(source_root, corpus_root, artifact_filter=TagFilter())
        assert report.materialized_count == 1
        assert meta_of(corpus_root, SESSION_A)["source_mtime_ns"] == later

    def test_a_generation_change_fails_every_filtered_session(
        self, source_root: Path, corpus_root: Path
    ) -> None:
        meta_bytes = self._filtered_corpus(source_root, corpus_root)
        write_session(source_root, "33333333-3333-3333-3333-333333333333", mtime_ns=STALE_NS)
        report = run(source_root, corpus_root, converter_version="0.2.0")
        assert sorted(failure.session_id for failure in report.failures) == [SESSION_A, SESSION_B]
        # A session no filtered pass wrote still converts.
        assert report.materialized_count == 1
        self._assert_untouched(corpus_root, meta_bytes)

    def test_an_untouched_filtered_session_is_up_to_date(
        self, source_root: Path, corpus_root: Path
    ) -> None:
        meta_bytes = self._filtered_corpus(source_root, corpus_root)
        report = run(source_root, corpus_root)
        assert report.failures == ()
        assert report.up_to_date_count == 2
        self._assert_untouched(corpus_root, meta_bytes)

    def test_a_marked_filtered_session_keeps_its_marker(
        self, source_root: Path, corpus_root: Path
    ) -> None:
        self._filtered_corpus(source_root, corpus_root)
        (source_root / "-proj-a" / f"{SESSION_A}.jsonl").unlink()
        report = run(source_root, corpus_root)
        assert report.removed_session_ids == (SESSION_A,)
        meta = meta_of(corpus_root, SESSION_A)
        assert meta["source_present"] is False
        assert meta["filtered"] is True
