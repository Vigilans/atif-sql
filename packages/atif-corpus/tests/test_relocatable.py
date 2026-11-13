# SPDX-License-Identifier: Apache-2.0

"""A relocatable pass: a corpus several machines can share.

Each session a relocatable pass converts records the machine that converted
it as ``meta.host``, beside the same artifacts an ordinary pass writes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from corpus_fixtures import NOW_NS, SESSION_A, STALE_NS, write_session

from atif_corpus.application.materialize import MaterializationReport, materialize
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
