# SPDX-License-Identifier: Apache-2.0

"""``materialize --filter`` end to end: the real converter, the SQL filter stage, the real corpus.

Each test materializes the session in :mod:`filter_fixtures` through one or
more filter scripts and reads back what published: the stored artifacts, the
blob store, and the views over the filtered corpus.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import pickle
import time
from pathlib import Path
from typing import Any

import pytest
from cli_fixtures import read_artifact_text
from filter_fixtures import (
    AGENT_ID,
    CWD,
    HOME,
    PASTED_PNG,
    READ_PNG,
    SECRET,
    SESSION,
    write_filter_session,
)

from atif_cli.app import app, materialize, query
from atif_cli.output import OutputFormat
from atif_cli.sql_filter import SqlArtifactFilter
from atif_converter.domain.atif import Trajectory
from atif_duck.domain.examples import FILTER_EXAMPLES, ExampleQuery

PASTED_SHA = hashlib.sha256(PASTED_PNG).hexdigest()
READ_SHA = hashlib.sha256(READ_PNG).hexdigest()

#: Every action at once: reasoning out, the secret and the home directory
#: replaced, the pasted image dropped, the Bash result pruned, and the stop
#: hook's event dropped.
EVERY_ACTION = f"""
CREATE TEMP VIEW directives AS
SELECT 'drop_reasoning' AS action, trajectory_id, step_id
FROM steps WHERE reasoning_content IS NOT NULL
UNION ALL BY NAME
SELECT 'replace' AS action, '{SECRET}' AS find, '[REDACTED]' AS replacement
UNION ALL BY NAME
SELECT 'replace' AS action, '{HOME}' AS find, '~' AS replacement
UNION ALL BY NAME
SELECT 'drop_blob' AS action, sha256 AS blob_sha256 FROM images WHERE origin = 'user_message'
UNION ALL BY NAME
SELECT 'prune_tool_result' AS action, trajectory_id, step_id, tool_use_id
FROM tool_results WHERE tool_use_id = 'toolu_bash'
UNION ALL BY NAME
SELECT 'drop_event' AS action, seq AS event_seq
FROM session_events WHERE subtype = 'stop_hook_summary';
"""

#: No directive at all: only the hidden copies go.
NO_DIRECTIVES = "CREATE TEMP TABLE directives (action VARCHAR);"


def _scripts(tmp_path: Path, *texts: str) -> list[str]:
    paths: list[str] = []
    for number, text in enumerate(texts, start=1):
        path = tmp_path / f"filter-{number}.sql"
        path.write_text(text, encoding="utf-8")
        paths.append(str(path))
    return paths


def _run(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    scripts: list[str] | None,
    **kwargs: Any,
) -> dict[str, Any]:
    source = tmp_path / "projects"
    if not source.exists():
        write_filter_session(source)
    materialize(
        source_root=source,
        corpus_root=tmp_path / "corpus",
        filter_scripts=scripts,
        workers=1,
        fmt=OutputFormat.JSON,
        **kwargs,
    )
    report = json.loads(capsys.readouterr().out)
    assert isinstance(report, dict)
    return report


def _session_dir(tmp_path: Path) -> Path:
    return tmp_path / "corpus" / "sessions" / SESSION


def _trajectory(tmp_path: Path) -> dict[str, Any]:
    document = json.loads(read_artifact_text(_session_dir(tmp_path) / "trajectory.json.zst"))
    assert isinstance(document, dict)
    return document


def _query(tmp_path: Path, capsys: pytest.CaptureFixture[str], sql: str) -> list[dict[str, Any]]:
    query(sql, corpus_root=tmp_path / "corpus", lake=False, fmt=OutputFormat.JSON)
    rows = json.loads(capsys.readouterr().out)
    assert isinstance(rows, list)
    return rows


def _stored_blobs(tmp_path: Path) -> set[str]:
    return {path.stem for path in (tmp_path / "corpus" / "blobs").rglob("*") if path.is_file()}


def _walk_keys(value: object) -> set[str]:
    """Every object key anywhere in a parsed JSON value."""
    keys: set[str] = set()
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            keys.update(str(key) for key in item)
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return keys


@pytest.mark.integration
class TestEveryAction:
    def test_the_published_session_holds_only_what_the_directives_left(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        report = _run(tmp_path, capsys, _scripts(tmp_path, EVERY_ACTION))
        assert (report["materialized"], report["failed"]) == (1, 0)
        session_dir = _session_dir(tmp_path)
        meta = json.loads((session_dir / "meta.json").read_text())
        assert meta["filtered"] is True
        assert "source_archive" not in meta
        assert not (session_dir / "source").exists()
        assert not (session_dir / "edges.jsonl.zst").exists()
        assert _query(tmp_path, capsys, "SELECT count(*) AS n FROM messages") == [{"n": 0}]

        # Neither the secret nor the home directory survives in any artifact.
        for name in ("trajectory.json.zst", "session_events.jsonl.zst"):
            text = read_artifact_text(session_dir / name)
            assert SECRET not in text, name
            assert HOME not in text, name
        assert "[REDACTED]" in read_artifact_text(session_dir / "trajectory.json.zst")
        trajectory = _trajectory(tmp_path)
        Trajectory.model_validate(trajectory)
        assert not {"raw_arguments", "raw_tool_result", "tool_use_result"} & _walk_keys(trajectory)

        assert _query(
            tmp_path, capsys, "SELECT count(*) AS n FROM steps WHERE reasoning_content IS NOT NULL"
        ) == [{"n": 0}]
        assert _query(
            tmp_path,
            capsys,
            "SELECT tool_use_id, content::VARCHAR = '\"\"' AS pruned FROM tool_results "
            "WHERE tool_use_id IN ('toolu_bash', 'toolu_agent') ORDER BY tool_use_id",
        ) == [
            {"tool_use_id": "toolu_agent", "pruned": False},
            {"tool_use_id": "toolu_bash", "pruned": True},
        ]
        assert _query(tmp_path, capsys, "SELECT origin, sha256 FROM images") == [
            {"origin": "tool_result", "sha256": READ_SHA}
        ]
        # The dropped image is not published; the one a result still names is.
        assert _stored_blobs(tmp_path) == {READ_SHA}
        assert _query(
            tmp_path, capsys, "SELECT event_type, subtype FROM session_events ORDER BY seq"
        ) == [{"event_type": "attachment", "subtype": "hook_success"}]
        assert _query(tmp_path, capsys, "SELECT cwd FROM sessions") == [{"cwd": "~/work"}]

    def test_the_subagent_failure_keeps_its_is_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        prune_sub = """
        CREATE TEMP VIEW directives AS
        SELECT 'prune_tool_result' AS action, trajectory_id, step_id, tool_use_id
        FROM tool_results WHERE tool_use_id = 'toolu_sub';
        """
        report = _run(tmp_path, capsys, _scripts(tmp_path, prune_sub))
        assert report["failed"] == 0
        child = next(
            child
            for child in _trajectory(tmp_path)["subagent_trajectories"]
            if child["trajectory_id"] == AGENT_ID
        )
        result = next(
            res
            for step in child["steps"]
            for res in (step.get("observation") or {}).get("results", [])
            if res.get("source_call_id") == "toolu_sub"
        )
        assert result["content"] == ""
        assert result["extra"]["tool_result_metadata"] == {"is_error": True}
        assert result["extra"]["is_error"] is True


@pytest.mark.integration
class TestHiddenCopies:
    def test_a_pass_without_directives_strips_only_the_hidden_copies(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        report = _run(tmp_path, capsys, _scripts(tmp_path, NO_DIRECTIVES))
        assert report["failed"] == 0
        trajectory = _trajectory(tmp_path)
        assert not {"raw_arguments", "raw_tool_result", "tool_use_result"} & _walk_keys(trajectory)
        Trajectory.model_validate(trajectory)
        # Nothing the views show was touched.
        text = read_artifact_text(_session_dir(tmp_path) / "trajectory.json.zst")
        assert SECRET in text
        assert _query(
            tmp_path, capsys, "SELECT count(*) AS n FROM steps WHERE reasoning_content IS NOT NULL"
        ) == [{"n": 2}]
        assert _stored_blobs(tmp_path) == {READ_SHA, PASTED_SHA}


@pytest.mark.integration
class TestComposition:
    def test_each_script_sees_the_unmodified_views(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Both scripts name a helper `hits`, and both find the secret: the
        # second sees the session as converted, not as the first filtered it.
        first = f"""
        CREATE TEMP VIEW hits AS SELECT tool_use_id FROM tool_calls
        WHERE tool_input::VARCHAR LIKE '%{SECRET}%';
        CREATE TEMP VIEW directives AS
        SELECT 'replace' AS action, '{SECRET}' AS find, '[REDACTED]' AS replacement
        FROM hits LIMIT 1;
        """
        second = f"""
        CREATE TEMP TABLE hits AS SELECT trajectory_id, step_id, tool_use_id FROM tool_results
        WHERE tool_use_id IN (SELECT tool_use_id FROM tool_calls
                              WHERE tool_input::VARCHAR LIKE '%{SECRET}%');
        CREATE TEMP VIEW directives AS
        SELECT 'prune_tool_result' AS action, trajectory_id, step_id, tool_use_id FROM hits;
        """
        report = _run(tmp_path, capsys, _scripts(tmp_path, first, second))
        assert (report["materialized"], report["failed"]) == (1, 0)
        assert SECRET not in read_artifact_text(_session_dir(tmp_path) / "trajectory.json.zst")
        assert _query(
            tmp_path,
            capsys,
            "SELECT content::VARCHAR AS c FROM tool_results WHERE tool_use_id = 'toolu_bash'",
        ) == [{"c": '""'}]

    def test_a_keep_only_policy_converges(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        keep_only = f"""
        CREATE TEMP VIEW directives AS
        WITH keep(trajectory_id, tool_use_id) AS (VALUES ('{SESSION}', 'toolu_agent'))
        SELECT 'prune_tool_result' AS action, r.trajectory_id, r.step_id, r.tool_use_id
        FROM tool_results r
        WHERE NOT EXISTS (
            SELECT 1 FROM keep k
            WHERE k.trajectory_id = r.trajectory_id AND k.tool_use_id = r.tool_use_id
        );
        """
        report = _run(tmp_path, capsys, _scripts(tmp_path, keep_only))
        assert report["failed"] == 0
        assert _query(
            tmp_path,
            capsys,
            "SELECT tool_use_id, content::VARCHAR = '\"\"' AS pruned FROM tool_results "
            "ORDER BY tool_use_id",
        ) == [
            {"tool_use_id": "toolu_agent", "pruned": False},
            {"tool_use_id": "toolu_bash", "pruned": True},
            {"tool_use_id": "toolu_read", "pruned": True},
            {"tool_use_id": "toolu_sub", "pruned": True},
        ]
        # Pruning the Read result orphaned its image, so neither blob publishes.
        assert _stored_blobs(tmp_path) == {PASTED_SHA}

    def test_both_passes_see_the_empty_messages_of_the_published_session(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # A filtered session publishes no edges, so neither pass sees any: the
        # guard would fail either, and the replacement names the count both saw.
        counted = f"""
        CREATE TEMP VIEW directives AS
        SELECT 'replace' AS action, '{SECRET}' AS find,
               CASE WHEN n = 0 THEN 'messages=' || n ELSE error('messages=' || n) END
                   AS replacement
        FROM (SELECT count(*) AS n FROM messages);
        """
        report = _run(tmp_path, capsys, _scripts(tmp_path, counted))
        assert (report["materialized"], report["failed"]) == (1, 0)
        text = read_artifact_text(_session_dir(tmp_path) / "trajectory.json.zst")
        assert SECRET not in text
        assert "messages=0" in text
        assert _query(tmp_path, capsys, "SELECT count(*) AS n FROM messages") == [{"n": 0}]

    def test_identical_replacements_from_two_scripts_merge(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        same = f"CREATE TEMP VIEW directives AS SELECT 'replace' AS action, '{SECRET}' AS find, 'x' AS replacement;"
        report = _run(tmp_path, capsys, _scripts(tmp_path, same, same))
        assert report["failed"] == 0

    def test_the_longer_find_wins_on_overlap(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # The work dir matches both finds; the longer one replaces it whole.
        chain = f"""
        CREATE TEMP VIEW directives AS
        SELECT * FROM (VALUES
            ('replace', '{CWD}', '/srv/work'),
            ('replace', '{HOME}', '/srv/home')
        ) t(action, find, replacement);
        """
        report = _run(tmp_path, capsys, _scripts(tmp_path, chain))
        assert report["failed"] == 0
        assert _query(tmp_path, capsys, "SELECT cwd FROM sessions") == [{"cwd": "/srv/work"}]


#: Scripts that must fail the session, with what the failure says.
FAILING_SCRIPTS: dict[str, tuple[str, str]] = {
    "unknown_action": (
        "CREATE TEMP VIEW directives AS SELECT 'keep' AS action;",
        "unknown action 'keep'",
    ),
    "null_action": (
        "CREATE TEMP VIEW directives AS SELECT NULL::VARCHAR AS action;",
        "unknown action None",
    ),
    "unknown_column": (
        "CREATE TEMP VIEW directives AS SELECT 'drop_event' AS action, 1 AS seq;",
        "unknown column(s) ['seq']",
    ),
    "mistyped_parameter": (
        "CREATE TEMP VIEW directives AS SELECT 'drop_event' AS action, '1' AS event_seq;",
        "needs event_seq as a non-null int",
    ),
    "foreign_parameter": (
        (
            "CREATE TEMP VIEW directives AS "
            "SELECT 'drop_event' AS action, 0 AS event_seq, 'x' AS find;"
        ),
        "sets ['find'], which drop_event does not take",
    ),
    "empty_find": (
        "CREATE TEMP VIEW directives AS SELECT 'replace' AS action, '' AS find, 'x' AS replacement;",
        "has an empty find",
    ),
    "invalid_target": (
        (
            f"CREATE TEMP VIEW directives AS SELECT 'prune_tool_result' AS action, "
            f"'{SESSION}' AS trajectory_id, 2::BIGINT AS step_id, 'toolu_missing' AS tool_use_id;"
        ),
        "address no tool result",
    ),
    "missing_event": (
        "CREATE TEMP VIEW directives AS SELECT 'drop_event' AS action, 999::BIGINT AS event_seq;",
        "address no session event",
    ),
    "no_directives": ("SELECT 1;", "left no relation named directives"),
    "policy_error": (
        "SELECT error('credential found');",
        "credential found",
    ),
    "refused_statement": (
        "COPY (SELECT 1) TO 'out.csv';",
        "does not run COPY statements",
    ),
    "host_file": (
        (
            "CREATE TEMP VIEW directives AS SELECT 'replace' AS action, content AS find, "
            "'x' AS replacement FROM read_text('/etc/hostname');"
        ),
        "Permission Error",
    ),
}


@pytest.mark.integration
class TestFailures:
    @pytest.mark.parametrize("case", sorted(FAILING_SCRIPTS))
    def test_a_bad_script_fails_the_session_before_publication(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], case: str
    ) -> None:
        script, message = FAILING_SCRIPTS[case]
        report = _run(tmp_path, capsys, _scripts(tmp_path, script))
        assert (report["materialized"], report["failed"]) == (0, 1)
        assert message in report["failures"][0]["error"]
        assert not _session_dir(tmp_path).exists()
        assert _stored_blobs(tmp_path) == set()

    def test_conflicting_replacements_fail(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def pair(replacement: str) -> str:
            return (
                f"CREATE TEMP VIEW directives AS SELECT 'replace' AS action, '{SECRET}' AS find, "
                f"'{replacement}' AS replacement;"
            )

        report = _run(tmp_path, capsys, _scripts(tmp_path, pair("a"), pair("b")))
        assert report["failed"] == 1
        assert "with both 'a' and 'b'" in report["failures"][0]["error"]

    def test_a_filter_that_does_not_converge_fails(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        growing = """
        CREATE TEMP VIEW directives AS
        SELECT 'replace' AS action, cwd AS find, cwd || '/x' AS replacement FROM sessions;
        """
        report = _run(tmp_path, capsys, _scripts(tmp_path, growing))
        assert report["failed"] == 1
        assert "did not converge" in report["failures"][0]["error"]


@pytest.mark.integration
class TestOverwriteGuard:
    def test_an_unfiltered_pass_fails_the_filtered_session(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _run(tmp_path, capsys, _scripts(tmp_path, EVERY_ACTION))
        before = read_artifact_text(_session_dir(tmp_path) / "trajectory.json.zst")
        report = _run(tmp_path, capsys, None, force=True)
        assert (report["materialized"], report["failed"]) == (0, 1)
        assert report["failures"][0]["error"].startswith("FilteredSessionError: ")
        assert read_artifact_text(_session_dir(tmp_path) / "trajectory.json.zst") == before
        # Re-running the policy explicitly re-materializes it.
        report = _run(
            tmp_path, capsys, _scripts(tmp_path, EVERY_ACTION), force=True, sessions=SESSION
        )
        assert (report["materialized"], report["failed"]) == (1, 0)


class TestScriptInput:
    def test_stdin_is_read_once_in_the_parent(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("sys.stdin", io.StringIO(NO_DIRECTIVES))
        source = tmp_path / "projects"
        write_filter_session(source)
        app(
            [
                "materialize",
                "--source-root",
                str(source),
                "--corpus-root",
                str(tmp_path / "corpus"),
                "--filter",
                "-",
                "--no-lake",
                "--workers",
                "1",
                "--format",
                "json",
            ],
            exit_on_error=False,
            result_action="return_value",
        )
        report = json.loads(capsys.readouterr().out)
        assert (report["materialized"], report["failed"]) == (1, 0)

    @pytest.mark.parametrize(
        ("arguments", "message"),
        [
            (["-", "-"], "--filter - is given twice"),
            (["missing.sql"], "cannot read filter script missing.sql"),
            (["empty.sql"], "filter script empty.sql is empty"),
        ],
    )
    def test_an_unusable_script_exits_64_before_the_pass(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
        arguments: list[str],
        message: str,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "empty.sql").write_text(" \n", encoding="utf-8")
        monkeypatch.setattr("sys.stdin", io.StringIO(NO_DIRECTIVES))
        source = tmp_path / "projects"
        write_filter_session(source)
        with pytest.raises(SystemExit) as exit_info:
            materialize(
                source_root=source,
                corpus_root=tmp_path / "corpus",
                filter_scripts=arguments,
                fmt=OutputFormat.JSON,
            )
        assert exit_info.value.code == 64
        assert message in capsys.readouterr().err
        assert not (tmp_path / "corpus").exists()

    def test_the_filter_pickles_with_its_scripts(self) -> None:
        restored = pickle.loads(pickle.dumps(SqlArtifactFilter([NO_DIRECTIVES])))  # noqa: S301 - our own object
        assert restored.scripts == (NO_DIRECTIVES,)


def _effect_queries(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    """What each filter example changes, read back from the filtered corpus."""
    session_dir = _session_dir(tmp_path)
    text = "".join(
        read_artifact_text(session_dir / name)
        for name in ("trajectory.json.zst", "session_events.jsonl.zst")
    )
    return {
        "unpruned": _query(
            tmp_path,
            capsys,
            "SELECT count(*) AS n FROM tool_results WHERE content::VARCHAR <> '\"\"'",
        )[0]["n"],
        "reasoning": _query(
            tmp_path, capsys, "SELECT count(*) AS n FROM steps WHERE reasoning_content IS NOT NULL"
        )[0]["n"],
        "images": _query(tmp_path, capsys, "SELECT count(*) AS n FROM images")[0]["n"],
        "blobs": _stored_blobs(tmp_path),
        "secret": SECRET in text,
        "home": HOME in text,
    }


#: Each filter example's effect on the fixture session, over what it starts with.
EXAMPLE_EFFECTS: dict[str, dict[str, Any]] = {
    "prune_uncited_tool_results": {"unpruned": 1, "images": 1, "blobs": {PASTED_SHA}},
    "prune_large_tool_results": {},
    "redact_credentials": {"secret": False},
    "replace_home_paths": {"home": False},
    "drop_reasoning_content": {"reasoning": 0},
    "drop_images_and_blobs": {"images": 0, "blobs": set()},
}


@pytest.mark.integration
class TestFilterExamples:
    def test_every_example_has_an_effect_check(self) -> None:
        assert set(EXAMPLE_EFFECTS) == {example.name for example in FILTER_EXAMPLES}

    @pytest.mark.parametrize("example", FILTER_EXAMPLES, ids=[e.name for e in FILTER_EXAMPLES])
    def test_the_example_filters_the_session(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], example: ExampleQuery
    ) -> None:
        baseline = tmp_path / "baseline"
        baseline.mkdir()
        _run(baseline, capsys, _scripts(baseline, NO_DIRECTIVES))
        before = _effect_queries(baseline, capsys)
        assert before == {
            "unpruned": 4,
            "reasoning": 2,
            "images": 2,
            "blobs": {READ_SHA, PASTED_SHA},
            "secret": True,
            "home": True,
        }
        report = _run(tmp_path, capsys, _scripts(tmp_path, example.sql))
        assert (report["materialized"], report["failed"]) == (1, 0), report["failures"]
        Trajectory.model_validate(_trajectory(tmp_path))
        assert _effect_queries(tmp_path, capsys) == {**before, **EXAMPLE_EFFECTS[example.name]}


CODEX_SESSION = "01a11e48-ecd5-7e10-8d09-43926dcfdcca"
#: A spoken user turn and a tool's spoken answer; the blob store never parses audio.
WAV = b"RIFF\x24\x00\x00\x00WAVEfmt "
MP3 = b"ID3\x04\x00\x00\x00\x00\x00\x00\xff\xfb"
WAV_SHA = hashlib.sha256(WAV).hexdigest()
MP3_SHA = hashlib.sha256(MP3).hexdigest()


def _codex_record(record_type: str, payload: dict[str, Any], n: int) -> dict[str, Any]:
    return {"timestamp": f"2026-10-09T01:31:{n:02d}.000Z", "type": record_type, "payload": payload}


def _write_codex_audio_rollout(sessions: Path) -> None:
    wav_url = f"data:audio/wav;base64,{base64.b64encode(WAV).decode()}"
    spoken = {"type": "audio", "data": base64.b64encode(MP3).decode(), "mimeType": "audio/mp3"}
    records = [
        _codex_record(
            "session_meta",
            {"id": CODEX_SESSION, "cwd": CWD, "cli_version": "0.160.1"},
            0,
        ),
        _codex_record("event_msg", {"type": "task_started", "turn_id": "turn-1"}, 1),
        _codex_record("turn_context", {"turn_id": "turn-1", "model": "gpt-6-astra"}, 2),
        _codex_record(
            "response_item",
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "transcribe "},
                    {"type": "input_audio", "audio_url": wav_url},
                ],
            },
            3,
        ),
        _codex_record(
            "response_item",
            {
                "type": "function_call",
                "call_id": "call_tts",
                "name": "mcp__voice__speak",
                "arguments": "{}",
            },
            4,
        ),
        _codex_record(
            "response_item",
            {
                "type": "function_call_output",
                "call_id": "call_tts",
                "output": [{"type": "input_text", "text": "spoken:"}, spoken],
            },
            5,
        ),
        _codex_record(
            "response_item",
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "done"}],
            },
            6,
        ),
        _codex_record("event_msg", {"type": "task_complete", "turn_id": "turn-1"}, 7),
    ]
    day_dir = sessions / "2026" / "10" / "09"
    day_dir.mkdir(parents=True)
    rollout = day_dir / f"rollout-2026-10-09T01-31-00-{CODEX_SESSION}.jsonl"
    rollout.write_text("".join(json.dumps(record) + "\n" for record in records))
    stale = time.time_ns() - 3_600 * 1_000_000_000
    os.utime(rollout, ns=(stale, stale))


def _materialize_codex(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], scripts: list[str]
) -> dict[str, Any]:
    _write_codex_audio_rollout(tmp_path / "sessions")
    materialize(
        agent="codex",
        source_root=tmp_path / "sessions",
        corpus_root=tmp_path / "corpus",
        filter_scripts=scripts,
        workers=1,
        fmt=OutputFormat.JSON,
    )
    report = json.loads(capsys.readouterr().out)
    assert isinstance(report, dict)
    return report


def _stored_blob_files(tmp_path: Path) -> set[str]:
    return {path.name for path in (tmp_path / "corpus" / "blobs").rglob("*") if path.is_file()}


@pytest.mark.integration
class TestCodexAudio:
    def test_audio_blobs_publish_under_their_audio_extensions(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        report = _materialize_codex(tmp_path, capsys, _scripts(tmp_path, NO_DIRECTIVES))
        assert (report["materialized"], report["failed"]) == (1, 0), report["failures"]
        assert _stored_blob_files(tmp_path) == {f"{WAV_SHA}.wav", f"{MP3_SHA}.mp3"}
        assert _query(tmp_path, capsys, "SELECT count(*) AS n FROM images") == [{"n": 0}]
        assert _query(
            tmp_path,
            capsys,
            "SELECT content::VARCHAR AS content FROM tool_results WHERE tool_use_id = 'call_tts'",
        ) == [
            {
                "content": json.dumps(
                    f"spoken:\n[audio sha256:{MP3_SHA} audio/mpeg {len(MP3)} bytes]"
                )
            }
        ]

    def test_drop_blob_drops_the_audio_its_placeholders_name(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The placeholder-scanning example finds the audio, drops it, and converges."""
        example = next(e for e in FILTER_EXAMPLES if e.name == "drop_images_and_blobs")
        report = _materialize_codex(tmp_path, capsys, _scripts(tmp_path, example.sql))
        assert (report["materialized"], report["failed"]) == (1, 0), report["failures"]
        assert _stored_blob_files(tmp_path) == set()
        Trajectory.model_validate(
            json.loads(
                read_artifact_text(
                    tmp_path / "corpus" / "sessions" / CODEX_SESSION / "trajectory.json.zst"
                )
            )
        )
