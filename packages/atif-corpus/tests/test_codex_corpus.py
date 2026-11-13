# SPDX-License-Identifier: Apache-2.0

"""Codex discovery and materialization: the depth-3 layout end to end.

The Claude Code equivalents live in ``test_infrastructure.py`` and
``test_materialize.py``; this module covers what a three-level source root
changes, and pins the behaviors that would silently destroy a corpus if the
generalized scanner lost them: a directory that cannot be LISTED at ANY level
must not make its sessions look deleted.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any

import pytest
from corpus_fixtures import LIVE_NS, NOW_NS, STALE_NS, write_session

from atif_corpus.application.materialize import (
    MaterializationReport,
    SuspiciousEmptyScanError,
    materialize,
    preview_pass,
)
from atif_corpus.domain.agents import AgentSource
from atif_corpus.domain.layout import CorpusLayout
from atif_corpus.domain.source_layout import CLAUDE_CODE_LAYOUT, CODEX_LAYOUT
from atif_corpus.infrastructure.fake_converter import FakeConverter
from atif_corpus.infrastructure.scanner import scan_source_root, scan_sources

SESSION_A = "01a09182-2858-7f42-936b-7f027b341fdf"
SESSION_B = "01a08db0-09a9-7b02-b690-62b5323644ff"


def write_rollout(
    sessions_root: Path,
    session_id: str,
    *,
    mtime_ns: int,
    day: tuple[str, str, str] = ("2026", "09", "11"),
) -> Path:
    """Create one rollout in Codex's ``<YYYY>/<MM>/<DD>/`` layout with a fixed mtime."""
    day_dir = sessions_root.joinpath(*day)
    day_dir.mkdir(parents=True, exist_ok=True)
    rollout = day_dir / f"rollout-{day[0]}-{day[1]}-{day[2]}T00-00-00-{session_id}.jsonl"
    rollout.write_text(json.dumps({"type": "session_meta", "payload": {"id": session_id}}) + "\n")
    os.utime(rollout, ns=(mtime_ns, mtime_ns))
    return rollout


@pytest.fixture
def codex_source_root(tmp_path: Path) -> Path:
    root = tmp_path / "sessions"
    root.mkdir()
    return root


class TestCodexDiscovery:
    def test_finds_rollouts_three_levels_down(self, codex_source_root: Path) -> None:
        write_rollout(codex_source_root, SESSION_A, mtime_ns=STALE_NS)
        write_rollout(codex_source_root, SESSION_B, mtime_ns=STALE_NS, day=("2026", "09", "10"))
        sessions = scan_source_root(codex_source_root, CODEX_LAYOUT)
        assert {session.session_id for session in sessions} == {SESSION_A, SESSION_B}

    def test_session_id_is_the_filename_uuid_not_the_stem(self, codex_source_root: Path) -> None:
        write_rollout(codex_source_root, SESSION_A, mtime_ns=STALE_NS)
        session = scan_source_root(codex_source_root, CODEX_LAYOUT)[0]
        assert session.session_id == SESSION_A
        assert Path(session.session_jsonl).stem != SESSION_A

    def test_a_rollout_carries_exactly_one_source_file(self, codex_source_root: Path) -> None:
        """No side-file tree exists for a rollout, so the watermark has one entry."""
        rollout = write_rollout(codex_source_root, SESSION_A, mtime_ns=STALE_NS)
        session = scan_source_root(codex_source_root, CODEX_LAYOUT)[0]
        assert session.source_files == (str(rollout),)

    def test_a_sibling_directory_named_after_a_rollout_is_not_walked(
        self, codex_source_root: Path
    ) -> None:
        """``has_side_files=False`` means a lookalike dir cannot inflate the watermark."""
        rollout = write_rollout(codex_source_root, SESSION_A, mtime_ns=STALE_NS)
        decoy = rollout.parent / rollout.stem
        decoy.mkdir()
        (decoy / "agent-x.jsonl").write_text("{}\n")
        session = scan_source_root(codex_source_root, CODEX_LAYOUT)[0]
        assert session.source_files == (str(rollout),)

    def test_a_non_rollout_jsonl_is_skipped(self, codex_source_root: Path) -> None:
        """Codex keeps other JSONL beside its rollouts; only rollouts are sessions."""
        write_rollout(codex_source_root, SESSION_A, mtime_ns=STALE_NS)
        (codex_source_root / "2026" / "09" / "11" / "history.jsonl").write_text("{}\n")
        sessions = scan_source_root(codex_source_root, CODEX_LAYOUT)
        assert [session.session_id for session in sessions] == [SESSION_A]

    def test_a_missing_root_yields_an_empty_scan(self, tmp_path: Path) -> None:
        assert scan_source_root(tmp_path / "nope", CODEX_LAYOUT) == ()

    def test_scan_is_sorted_by_session_id(self, codex_source_root: Path) -> None:
        write_rollout(codex_source_root, SESSION_B, mtime_ns=STALE_NS)
        write_rollout(codex_source_root, SESSION_A, mtime_ns=STALE_NS)
        sessions = scan_source_root(codex_source_root, CODEX_LAYOUT)
        assert [session.session_id for session in sessions] == sorted(
            [session.session_id for session in sessions]
        )


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
class TestUnlistableIntermediateDirectories:
    def test_an_unlistable_day_dir_is_reported_not_treated_as_empty(
        self, codex_source_root: Path
    ) -> None:
        """Three levels means three chances to mistake 'cannot open' for 'empty'."""
        write_rollout(codex_source_root, SESSION_A, mtime_ns=STALE_NS)
        day_dir = codex_source_root / "2026" / "09" / "11"
        day_dir.chmod(0o000)
        try:
            scan = scan_sources(codex_source_root, CODEX_LAYOUT)
        finally:
            day_dir.chmod(stat.S_IRWXU)
        assert scan.sessions == ()
        assert str(day_dir) in scan.unlistable_dirs

    def test_an_unlistable_month_dir_is_reported(self, codex_source_root: Path) -> None:
        write_rollout(codex_source_root, SESSION_A, mtime_ns=STALE_NS)
        month_dir = codex_source_root / "2026" / "09"
        month_dir.chmod(0o000)
        try:
            scan = scan_sources(codex_source_root, CODEX_LAYOUT)
        finally:
            month_dir.chmod(stat.S_IRWXU)
        assert scan.sessions == ()
        assert str(month_dir) in scan.unlistable_dirs

    def test_an_unlistable_day_dir_does_not_ghost_its_sessions(
        self, codex_source_root: Path, tmp_path: Path
    ) -> None:
        """The data-loss case: absence behind a closed door is not deletion."""
        corpus_root = tmp_path / "corpus"
        write_rollout(codex_source_root, SESSION_A, mtime_ns=STALE_NS)
        first = _materialize_codex(codex_source_root, corpus_root)
        assert first.materialized_count == 1
        session_dir = CorpusLayout(corpus_root=corpus_root).session_dir(SESSION_A)
        assert session_dir.is_dir()

        day_dir = codex_source_root / "2026" / "09" / "11"
        day_dir.chmod(0o000)
        try:
            second = _materialize_codex(codex_source_root, corpus_root)
        finally:
            day_dir.chmod(stat.S_IRWXU)
        assert second.removed_session_ids == ()
        assert session_dir.is_dir(), "artifacts were removed over a permission error"
        assert SESSION_A in second.unreadable_session_ids


def _materialize_codex(source_root: Path, corpus_root: Path, **kwargs: object):
    """One Codex materialization pass with the fake converter and a pinned clock."""
    return materialize(
        source_root=source_root,
        corpus_root=corpus_root,
        converter=FakeConverter(),
        materialized_at="2026-09-11T00:00:00Z",
        harbor_version="0.22.0",
        converter_version="0.1.0",
        now_ns=NOW_NS,
        source_layout=CODEX_LAYOUT,
        **kwargs,  # type: ignore[arg-type]
    )


class TestCodexMaterialization:
    def test_materializes_a_rollout_with_the_full_contract_layout(
        self, codex_source_root: Path, tmp_path: Path
    ) -> None:
        corpus_root = tmp_path / "corpus"
        write_rollout(codex_source_root, SESSION_A, mtime_ns=STALE_NS)
        report = _materialize_codex(codex_source_root, corpus_root)
        assert report.materialized_count == 1
        layout = CorpusLayout(corpus_root=corpus_root)
        for path in (
            layout.trajectory_path(SESSION_A),
            layout.loss_report_path(SESSION_A),
            layout.edges_path(SESSION_A),
            layout.meta_path(SESSION_A),
        ):
            assert path.is_file(), f"missing artifact {path.name}"

    def test_meta_json_records_the_agent(self, codex_source_root: Path, tmp_path: Path) -> None:
        """Provenance: a corpus copied out of place still says what wrote it."""
        corpus_root = tmp_path / "corpus"
        write_rollout(codex_source_root, SESSION_A, mtime_ns=STALE_NS)
        _materialize_codex(codex_source_root, corpus_root)
        meta = json.loads(CorpusLayout(corpus_root=corpus_root).meta_path(SESSION_A).read_text())
        assert meta["agent"] == AgentSource.CODEX.value
        assert meta["session_id"] == SESSION_A

    def test_a_live_rollout_is_skipped_then_materialized_once_quiescent(
        self, codex_source_root: Path, tmp_path: Path
    ) -> None:
        corpus_root = tmp_path / "corpus"
        rollout = write_rollout(codex_source_root, SESSION_A, mtime_ns=LIVE_NS)
        first = _materialize_codex(codex_source_root, corpus_root)
        assert first.materialized_count == 0
        assert first.skipped_live_count == 1
        os.utime(rollout, ns=(STALE_NS, STALE_NS))
        second = _materialize_codex(codex_source_root, corpus_root)
        assert second.materialized_count == 1

    def test_a_second_pass_is_a_noop(self, codex_source_root: Path, tmp_path: Path) -> None:
        corpus_root = tmp_path / "corpus"
        write_rollout(codex_source_root, SESSION_A, mtime_ns=STALE_NS)
        _materialize_codex(codex_source_root, corpus_root)
        second = _materialize_codex(codex_source_root, corpus_root)
        assert second.materialized_count == 0
        assert second.up_to_date_count == 1

    def test_a_deleted_rollout_keeps_its_corpus_dir_marked(
        self, codex_source_root: Path, tmp_path: Path
    ) -> None:
        corpus_root = tmp_path / "corpus"
        rollout = write_rollout(codex_source_root, SESSION_A, mtime_ns=STALE_NS)
        write_rollout(codex_source_root, SESSION_B, mtime_ns=STALE_NS)
        _materialize_codex(codex_source_root, corpus_root)
        rollout.unlink()
        report = _materialize_codex(codex_source_root, corpus_root)
        assert report.removed_session_ids == (SESSION_A,)
        layout = CorpusLayout(corpus_root=corpus_root)
        assert layout.trajectory_path(SESSION_A).is_file()
        meta = json.loads(layout.meta_path(SESSION_A).read_text(encoding="utf-8"))
        assert meta["source_present"] is False

    def test_a_touched_rollout_rematerializes(
        self, codex_source_root: Path, tmp_path: Path
    ) -> None:
        corpus_root = tmp_path / "corpus"
        rollout = write_rollout(codex_source_root, SESSION_A, mtime_ns=STALE_NS)
        _materialize_codex(codex_source_root, corpus_root)
        newer = STALE_NS + 1_000_000_000
        os.utime(rollout, ns=(newer, newer))
        report = _materialize_codex(codex_source_root, corpus_root)
        assert report.materialized_count == 1


#: A session of each agent that keeps its source, so a pass whose other
#: session vanished still scans something.
KEPT_CLAUDE = "33333333-3333-3333-3333-333333333333"
KEPT_CODEX = "01a0a000-0000-7000-8000-000000000003"


def _materialize_claude(
    source_root: Path, corpus_root: Path, converter: FakeConverter | None = None
) -> MaterializationReport:
    """One Claude Code materialization pass with the fake converter and a pinned clock."""
    return materialize(
        source_root=source_root,
        corpus_root=corpus_root,
        converter=converter or FakeConverter(),
        materialized_at="2026-09-11T00:00:00Z",
        harbor_version="0.22.0",
        converter_version="0.1.0",
        now_ns=NOW_NS,
        source_layout=CLAUDE_CODE_LAYOUT,
    )


def _meta(corpus_root: Path, session_id: str) -> Any:
    return json.loads(CorpusLayout(corpus_root).meta_path(session_id).read_text())


class TestSharedCorpus:
    """Two agents materialize into one corpus root, each owning only its own sessions.

    Every session of the other agent is absent from this pass's scan by
    construction, so a pass that judged them would mark all of them
    source-removed; each session's ``meta.agent`` says whose it is.
    """

    @pytest.fixture
    def claude_source(self, tmp_path: Path) -> Path:
        root = tmp_path / "projects"
        root.mkdir()
        return root

    def test_both_agents_share_one_corpus(
        self, tmp_path: Path, claude_source: Path, codex_source_root: Path
    ) -> None:
        corpus_root = tmp_path / "corpus"
        write_session(claude_source, SESSION_A, mtime_ns=STALE_NS)
        write_rollout(codex_source_root, SESSION_B, mtime_ns=STALE_NS)

        assert _materialize_claude(claude_source, corpus_root).materialized_count == 1
        codex = _materialize_codex(codex_source_root, corpus_root)
        assert codex.materialized_count == 1
        assert codex.removed_session_ids == ()
        assert codex.retained_count == 0

        claude = _materialize_claude(claude_source, corpus_root)
        assert claude.materialized_count == 0, "the Codex pass dropped the Claude watermark"
        assert claude.up_to_date_count == 1
        assert claude.removed_session_ids == ()
        assert _materialize_codex(codex_source_root, corpus_root).up_to_date_count == 1

        assert _meta(corpus_root, SESSION_A)["agent"] == "claude-code"
        assert _meta(corpus_root, SESSION_B)["agent"] == "codex"
        assert _meta(corpus_root, SESSION_A)["source_present"] is True
        assert _meta(corpus_root, SESSION_B)["source_present"] is True

    def test_a_pass_marks_only_its_own_agents_vanished_sessions(
        self, tmp_path: Path, claude_source: Path, codex_source_root: Path
    ) -> None:
        corpus_root = tmp_path / "corpus"
        claude_main = write_session(claude_source, SESSION_A, mtime_ns=STALE_NS)
        write_session(claude_source, KEPT_CLAUDE, mtime_ns=STALE_NS)
        rollout = write_rollout(codex_source_root, SESSION_B, mtime_ns=STALE_NS)
        write_rollout(codex_source_root, KEPT_CODEX, mtime_ns=STALE_NS)
        _materialize_claude(claude_source, corpus_root)
        _materialize_codex(codex_source_root, corpus_root)

        rollout.unlink()
        claude_main.unlink()
        codex = _materialize_codex(codex_source_root, corpus_root)
        assert codex.removed_session_ids == (SESSION_B,)
        assert _meta(corpus_root, SESSION_A)["source_present"] is True
        assert _materialize_claude(claude_source, corpus_root).removed_session_ids == (SESSION_A,)

    def test_status_retains_only_the_agents_own_sessions(
        self, tmp_path: Path, claude_source: Path, codex_source_root: Path
    ) -> None:
        corpus_root = tmp_path / "corpus"
        write_session(claude_source, SESSION_A, mtime_ns=STALE_NS)
        write_rollout(codex_source_root, SESSION_B, mtime_ns=STALE_NS)
        _materialize_claude(claude_source, corpus_root)
        _materialize_codex(codex_source_root, corpus_root)

        preview = preview_pass(
            source_root=codex_source_root,
            corpus_root=corpus_root,
            converter_version="0.1.0",
            source_layout=CODEX_LAYOUT,
            now_ns=NOW_NS,
        )
        assert preview.retained_session_ids == ()
        assert [s.session_id for s in preview.plan.up_to_date] == [SESSION_B]

    def test_a_session_written_before_the_agent_key_is_claude_codes(
        self, tmp_path: Path, claude_source: Path, codex_source_root: Path
    ) -> None:
        """No session predating Codex support is a Codex one, so a Codex pass leaves it be."""
        corpus_root = tmp_path / "corpus"
        write_session(claude_source, SESSION_A, mtime_ns=STALE_NS)
        _materialize_claude(claude_source, corpus_root)
        meta_path = CorpusLayout(corpus_root).meta_path(SESSION_A)
        meta = json.loads(meta_path.read_text())
        del meta["agent"]
        meta_path.write_text(json.dumps(meta))

        write_rollout(codex_source_root, SESSION_B, mtime_ns=STALE_NS)
        assert _materialize_codex(codex_source_root, corpus_root).removed_session_ids == ()
        assert _meta(corpus_root, SESSION_A)["source_present"] is True

    def test_an_empty_scan_is_suspicious_only_over_the_agents_own_sessions(
        self, tmp_path: Path, claude_source: Path, codex_source_root: Path
    ) -> None:
        corpus_root = tmp_path / "corpus"
        claude_main = write_session(claude_source, SESSION_A, mtime_ns=STALE_NS)
        _materialize_claude(claude_source, corpus_root)

        report = _materialize_codex(codex_source_root, corpus_root)
        assert report.materialized_count == 0
        assert report.removed_session_ids == ()

        claude_main.unlink()
        with pytest.raises(SuspiciousEmptyScanError):
            _materialize_claude(claude_source, corpus_root)

    def test_the_other_agents_empty_sessions_stay_recorded(
        self, tmp_path: Path, claude_source: Path, codex_source_root: Path
    ) -> None:
        corpus_root = tmp_path / "corpus"
        write_session(claude_source, SESSION_A, mtime_ns=STALE_NS)
        empty = FakeConverter(empty_sessions=frozenset({SESSION_A}))
        assert _materialize_claude(claude_source, corpus_root, empty).empty_session_ids == (
            SESSION_A,
        )
        layout = CorpusLayout(corpus_root)
        recorded = json.loads(layout.empty_sessions_path.read_text())
        assert recorded[SESSION_A]["agent"] == "claude-code"

        write_rollout(codex_source_root, SESSION_B, mtime_ns=STALE_NS)
        _materialize_codex(codex_source_root, corpus_root)
        assert SESSION_A in json.loads(layout.empty_sessions_path.read_text())

        again = _materialize_claude(claude_source, corpus_root, empty)
        assert again.empty_session_ids == (), "the empty session was converted again"
        assert again.up_to_date_count == 1


def write_rollout_file(
    sessions_root: Path,
    name: str,
    meta: dict[str, object],
    *,
    mtime_ns: int = STALE_NS,
    day: tuple[str, str, str] = ("2026", "09", "11"),
    compressed: bool = False,
) -> Path:
    """One rollout named ``name`` whose first record is ``session_meta`` with ``meta``."""
    import zstandard

    day_dir = sessions_root.joinpath(*day)
    day_dir.mkdir(parents=True, exist_ok=True)
    data = (json.dumps({"type": "session_meta", "payload": meta}) + "\n").encode()
    path = day_dir / (f"{name}.zst" if compressed else name)
    path.write_bytes(zstandard.ZstdCompressor().compress(data) if compressed else data)
    os.utime(path, ns=(mtime_ns, mtime_ns))
    return path


class TestCompressedRollouts:
    def test_the_plain_form_of_a_rollout_wins_over_its_compressed_copy(
        self, codex_source_root: Path
    ) -> None:
        plain = write_rollout(codex_source_root, SESSION_A, mtime_ns=STALE_NS)
        write_rollout_file(codex_source_root, plain.name, {"id": SESSION_A}, compressed=True)
        (session,) = scan_source_root(codex_source_root, CODEX_LAYOUT)
        assert session.source_files == (str(plain),)

    def test_a_compressed_rollout_alone_is_a_session(self, codex_source_root: Path) -> None:
        path = write_rollout_file(
            codex_source_root,
            f"rollout-2026-09-11T00-00-00-{SESSION_A}.jsonl",
            {"id": SESSION_A},
            compressed=True,
        )
        (session,) = scan_source_root(codex_source_root, CODEX_LAYOUT)
        assert session.session_jsonl == str(path)


SEGMENT = "01a158c3-25de-70a3-b159-33361d98a15a"
CHILD = "019ffb4f-6f3b-7873-b601-faed2e9a2316"


class TestCodexSessionGroups:
    def test_a_thread_and_its_segments_are_one_session(self, codex_source_root: Path) -> None:
        first = write_rollout(codex_source_root, SESSION_A, mtime_ns=STALE_NS)
        segment = write_rollout_file(
            codex_source_root,
            f"rollout-2026-09-12T00-00-00-{SESSION_A}_{SEGMENT}.jsonl",
            {"id": SESSION_A, "session_id": SESSION_A, "history_base": {"thread_id": SESSION_A}},
            day=("2026", "09", "12"),
            compressed=True,
        )
        (session,) = scan_source_root(codex_source_root, CODEX_LAYOUT)
        assert session.session_id == SESSION_A
        assert session.session_jsonl == str(first)
        assert set(session.source_files) == {str(first), str(segment)}

    def test_a_new_segment_makes_the_session_stale(
        self, codex_source_root: Path, tmp_path: Path
    ) -> None:
        corpus_root = tmp_path / "corpus"
        write_rollout(codex_source_root, SESSION_A, mtime_ns=STALE_NS)
        assert _materialize_codex(codex_source_root, corpus_root).materialized_count == 1
        assert _materialize_codex(codex_source_root, corpus_root).materialized_count == 0
        write_rollout_file(
            codex_source_root,
            f"rollout-2026-09-12T00-00-00-{SESSION_A}_{SEGMENT}.jsonl",
            {"id": SESSION_A, "history_base": {"thread_id": SESSION_A}},
            day=("2026", "09", "12"),
        )
        assert _materialize_codex(codex_source_root, corpus_root).materialized_count == 1
        assert _materialize_codex(codex_source_root, corpus_root).materialized_count == 0

    def test_the_converter_gets_the_group_and_the_archive_spans_it(
        self, codex_source_root: Path, tmp_path: Path
    ) -> None:
        corpus_root = tmp_path / "corpus"
        write_rollout(codex_source_root, SESSION_A, mtime_ns=STALE_NS)
        segment = write_rollout_file(
            codex_source_root,
            f"rollout-2026-10-01T00-00-00-{SESSION_A}_{SEGMENT}.jsonl",
            {"id": SESSION_A, "history_base": {"thread_id": SESSION_A}},
            day=("2026", "10", "01"),
        )
        _materialize_codex(codex_source_root, corpus_root)
        meta = json.loads(CorpusLayout(corpus_root=corpus_root).meta_path(SESSION_A).read_text())
        manifest = meta["source_archive"]
        # Two months apart: the deepest directory holding both is the year.
        assert manifest["source_dir"] == "2026"
        assert manifest["main"] == f"09/11/rollout-2026-09-11T00-00-00-{SESSION_A}.jsonl"
        assert {entry["path"] for entry in manifest["files"]} == {
            manifest["main"],
            f"10/01/{segment.name}",
        }


class TestSpawnedThreads:
    def test_spawned_threads_group_under_the_root(self, codex_source_root: Path) -> None:
        first = write_rollout(codex_source_root, SESSION_A, mtime_ns=STALE_NS)
        segment = write_rollout_file(
            codex_source_root,
            f"rollout-2026-09-12T00-00-00-{SESSION_A}_{SEGMENT}.jsonl",
            {"id": SESSION_A, "session_id": SESSION_A, "history_base": {"thread_id": SESSION_A}},
            day=("2026", "09", "12"),
            compressed=True,
        )
        child = write_rollout_file(
            codex_source_root,
            f"rollout-2026-09-11T01-00-00-{CHILD}.jsonl",
            {"id": CHILD, "session_id": SESSION_A, "parent_thread_id": SESSION_A},
        )
        (session,) = scan_source_root(codex_source_root, CODEX_LAYOUT)
        assert session.session_id == SESSION_A
        assert session.session_jsonl == str(first)
        assert set(session.source_files) == {str(first), str(segment), str(child)}

    def test_a_spawned_thread_of_a_spawned_thread_reaches_the_root(
        self, codex_source_root: Path
    ) -> None:
        write_rollout(codex_source_root, SESSION_A, mtime_ns=STALE_NS)
        write_rollout_file(
            codex_source_root,
            f"rollout-2026-09-11T01-00-00-{CHILD}.jsonl",
            {"id": CHILD, "session_id": SESSION_A},
        )
        grandchild = write_rollout_file(
            codex_source_root,
            f"rollout-2026-09-11T02-00-00-{SESSION_B}.jsonl",
            {"id": SESSION_B, "session_id": CHILD},
        )
        (session,) = scan_source_root(codex_source_root, CODEX_LAYOUT)
        assert str(grandchild) in session.source_files

    def test_spawned_rollouts_without_their_root_are_not_a_session(
        self, codex_source_root: Path
    ) -> None:
        """A child whose root thread is not under the source root has no main transcript."""
        write_rollout_file(
            codex_source_root,
            f"rollout-2026-09-11T01-00-00-{CHILD}.jsonl",
            {"id": CHILD, "session_id": SESSION_A},
        )
        assert scan_source_root(codex_source_root, CODEX_LAYOUT) == ()
