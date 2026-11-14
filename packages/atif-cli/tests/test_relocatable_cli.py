# SPDX-License-Identifier: Apache-2.0

"""``materialize --relocatable`` and ``ATIF_SQL_RELOCATABLE``.

Each session records ``meta.host``, and a pass re-converts from an archive
only the sessions that name its machine.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from cli_fixtures import write_synthetic_session

from atif_cli.app import materialize, status

SESSION = "aaaaaaaa-0000-4000-8000-000000000001"
LIVE_SESSION = "aaaaaaaa-0000-4000-8000-000000000002"
HOST = "ws.example.com"


def _session_dir(corpus: Path) -> Path:
    return corpus / "sessions" / SESSION


def _meta(corpus: Path) -> dict[str, Any]:
    meta = json.loads((_session_dir(corpus) / "meta.json").read_text())
    assert isinstance(meta, dict)
    return meta


def _write_meta(corpus: Path, **changes: object) -> None:
    (_session_dir(corpus) / "meta.json").write_text(json.dumps({**_meta(corpus), **changes}))


@pytest.fixture
def source_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.delenv("ATIF_SQL_RELOCATABLE", raising=False)
    monkeypatch.setattr("atif_corpus.application.materialize.machine_hostname", lambda: HOST)
    root = tmp_path / "projects"
    write_synthetic_session(root, SESSION)
    return root


def _materialize(source: Path, corpus: Path, **kw: object) -> None:
    materialize(source_root=source, corpus_root=corpus, lake=False, fmt="json", **kw)  # type: ignore[arg-type]


def _expire(source: Path, corpus: Path) -> None:
    """The transcript expires and the converter schema moves past the session's."""
    (source / "-tmp-proj" / f"{SESSION}.jsonl").unlink()
    _write_meta(corpus, converter_schema=-1)


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


def _from_archive(source: Path, corpus: Path, capsys: pytest.CaptureFixture[str]) -> list[int]:
    """``staleness.from_archive`` under an ordinary and a relocatable replay."""
    counts: list[int] = []
    for relocatable in (False, True):
        capsys.readouterr()
        status(source_root=source, corpus_root=corpus, relocatable=relocatable, fmt="json")  # type: ignore[arg-type]
        counts.append(json.loads(capsys.readouterr().out)["staleness"]["from_archive"])
    return counts


def test_status_replays_the_owning_pass(
    source_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plain, shared = tmp_path / "plain", tmp_path / "shared"
    _materialize(source_root, plain)
    _materialize(source_root, shared, relocatable=True)
    _expire(source_root, plain)
    _write_meta(shared, converter_schema=-1)
    assert _from_archive(source_root, plain, capsys) == [1, 0]
    assert _from_archive(source_root, shared, capsys) == [0, 1]


def test_a_relocatable_pass_reconverts_only_this_machines_sessions_from_their_archive(
    source_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = tmp_path / "shared"
    write_synthetic_session(source_root, LIVE_SESSION)
    _materialize(source_root, corpus, relocatable=True)
    _expire(source_root, corpus)
    capsys.readouterr()

    _materialize(source_root, corpus, relocatable=True)
    assert json.loads(capsys.readouterr().out)["from_archive_session_ids"] == [SESSION]
    assert _meta(corpus)["converter_schema"] != -1
    assert _meta(corpus)["host"] == HOST

    # The same session as another machine's pass recorded it.
    _write_meta(corpus, host="elsewhere.example.com", converter_schema=-1)
    _materialize(source_root, corpus, relocatable=True)
    assert json.loads(capsys.readouterr().out)["from_archive_session_ids"] == []
    assert _meta(corpus)["converter_schema"] == -1
