# SPDX-License-Identifier: Apache-2.0

"""``materialize --relocatable`` and ``ATIF_SQL_RELOCATABLE``: each session records ``meta.host``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from cli_fixtures import write_synthetic_session

from atif_cli.app import materialize

SESSION = "aaaaaaaa-0000-4000-8000-000000000001"
HOST = "ws.example.com"


def _session_dir(corpus: Path) -> Path:
    return corpus / "sessions" / SESSION


def _meta(corpus: Path) -> dict[str, Any]:
    meta = json.loads((_session_dir(corpus) / "meta.json").read_text())
    assert isinstance(meta, dict)
    return meta


@pytest.fixture
def source_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.delenv("ATIF_SQL_RELOCATABLE", raising=False)
    monkeypatch.setattr("atif_corpus.application.materialize.machine_hostname", lambda: HOST)
    root = tmp_path / "projects"
    write_synthetic_session(root, SESSION)
    return root


def _materialize(source: Path, corpus: Path, **kw: object) -> None:
    materialize(source_root=source, corpus_root=corpus, lake=False, fmt="json", **kw)  # type: ignore[arg-type]


def test_the_flag_records_the_host(source_root: Path, tmp_path: Path) -> None:
    _materialize(source_root, tmp_path / "plain")
    _materialize(source_root, tmp_path / "shared", relocatable=True)
    assert "host" not in _meta(tmp_path / "plain")
    assert _meta(tmp_path / "shared")["host"] == HOST
    assert (_session_dir(tmp_path / "shared") / "source").is_dir()


def test_the_environment_sets_the_default_and_the_flag_overrides_it(
    source_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ATIF_SQL_RELOCATABLE", "1")
    _materialize(source_root, tmp_path / "shared")
    _materialize(source_root, tmp_path / "plain", relocatable=False)
    assert _meta(tmp_path / "shared")["host"] == HOST
    assert "host" not in _meta(tmp_path / "plain")
