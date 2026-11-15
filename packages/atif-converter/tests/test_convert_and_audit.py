# SPDX-License-Identifier: Apache-2.0

"""Pin the Claude Code conversion behavior our converter inherited from harbor 0.22.0.

Every assertion here encodes a behavior EMPIRICALLY OBSERVED in harbor's
conversion at 0.22.0 (probed 2026-08-22) and carried into our port
(``atif_converter.domain.claude_code_conversion``) on purpose. The parity
oracle (``harbor_oracle.py``, the goldens, ``test_parity_*``) is what holds the
port to that behavior; these tests say what the behavior IS, in terms of the
fidelity policy in ``atif_converter/domain/fidelity.py``. Changing one of them
is a decision to diverge from harbor, and the policy is where that is recorded.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from atif_converter.application.convert_and_audit import convert_and_audit
from atif_converter.domain.errors import InvalidSessionInput
from atif_converter.domain.fidelity import FidelityGap, LossReport, RecordType
from atif_converter.domain.trajectory_tree import iter_steps
from atif_converter.infrastructure.harbor_adapter import ConversionResult

Converted = tuple[ConversionResult, LossReport]

#: Typed stand-in for a step's absent ``extra``. A bare ``{}`` literal infers as
#: ``dict[Unknown, Unknown]``, which leaks Unknown into every ``.get`` chained
#: onto the ``or`` result.
_NO_EXTRA: Mapping[str, Any] = {}


@pytest.fixture
def converted(synthetic_session: Path) -> Converted:
    return convert_and_audit(synthetic_session)


class TestConversion:
    def test_conversion_validates_clean(self, converted: Converted) -> None:
        result, _ = converted
        assert result.is_valid, f"validator errors: {result.validation_errors}"
        assert result.trajectory["schema_version"].startswith("ATIF-")
        assert result.trajectory["session_id"] == "11111111-1111-1111-1111-111111111111"

    def test_tool_call_bundled_with_result(self, converted: Converted) -> None:
        result, _ = converted
        agent_steps = [s for s in result.trajectory["steps"] if s["source"] == "agent"]
        tool_steps = [s for s in agent_steps if s.get("tool_calls")]
        assert len(tool_steps) == 1
        step = tool_steps[0]
        assert step["tool_calls"][0]["function_name"] == "Bash"
        assert step["observation"]["results"][0]["source_call_id"] == "toolu_01"
        assert step["reasoning_content"] == "let me think"

    def test_cache_split_partial_gap6(self, converted: Converted) -> None:
        """cached_tokens carries cache_read only; cache_creation survives only in metrics.extra."""
        result, _ = converted
        step = next(s for s in result.trajectory["steps"] if s.get("tool_calls"))
        metrics = step["metrics"]
        assert metrics["cached_tokens"] == 100  # cache_read_input_tokens
        # prompt tokens are the sum of input, cache read, and cache creation
        assert metrics["prompt_tokens"] == 10 + 100 + 50
        assert metrics["extra"]["cache_creation_input_tokens"] == 50

    def test_side_files_nest_as_subagent_trajectories(self, converted: Converted) -> None:
        """Each side-file's agent converts into an embedded trajectory named by its agent id."""
        result, _ = converted
        assert not any(
            (s.get("extra") or _NO_EXTRA).get("is_sidechain") for s in result.trajectory["steps"]
        )
        children = result.trajectory["subagent_trajectories"]
        # flat side-file (2 steps) + workflow-nested file (1 step)
        assert [(c["trajectory_id"], len(c["steps"])) for c in children] == [("abc", 2), ("def", 1)]
        assert {c["session_id"] for c in children} == {result.trajectory["session_id"]}

    def test_workflow_nested_side_files_convert(self, converted: Converted) -> None:
        """Our converter discovers workflow-nested side-files itself, so their
        records convert (harbor's own discovery cannot see them)."""
        result, _ = converted
        workflow = result.trajectory["subagent_trajectories"][1]
        assert [s["message"] for s in workflow["steps"]] == ["workflow subagent prompt"]

    def test_uuid_not_preserved_gap7(self, synthetic_session: Path) -> None:
        """DOCUMENTS gap 7: harbor ALONE never puts the event uuid in the step.

        harbor 0.22.0 reads event["uuid"] only for dedup (seen_event_uuids in
        _convert_events_to_trajectory) and never copies it into Step.extra.
        The enrichment pass (domain/enrichment.py, tests in
        test_enrichment.py) restores the identity as extra["source_uuids"];
        this canary pins the RAW harbor output (convert_session, pre-
        enrichment) so an upstream fix is noticed.
        """
        from atif_converter.infrastructure.harbor_adapter import convert_session

        result = convert_session(synthetic_session)
        fixture_uuids = {"u-1", "a-1", "u-2", "a-2", "su-1", "sa-1"}
        for step in result.trajectory["steps"]:
            extra: Mapping[str, Any] = step.get("extra") or {}
            assert not (set(extra.values()) & fixture_uuids), (
                "harbor started preserving event uuids — gap 7 closed upstream; "
                "update FidelityGap.UUID_NOT_PRESERVED and this test"
            )
            assert "uuid" not in extra


class TestLossReport:
    def test_census_counts_by_record_type(self, converted: Converted) -> None:
        _, report = converted
        # main: 2 user + 2 assistant + 1 attachment + 1 queue-operation
        # subagent side-file: 1 user + 1 assistant; workflow file: 1 user
        assert report.record_counts[RecordType.USER] == 4
        assert report.record_counts[RecordType.ASSISTANT] == 3
        assert report.record_counts[RecordType.ATTACHMENT] == 1
        assert report.record_counts[RecordType.QUEUE_OPERATION] == 1
        assert report.records_total == 9

    def test_attachment_and_queue_operation_dropped_gap2(self, converted: Converted) -> None:
        _, report = converted
        assert report.records_converted == 7  # user + assistant records
        assert report.records_dropped == 2  # attachment + queue-operation
        assert FidelityGap.NON_MESSAGE_RECORDS_DROPPED in report.gaps_observed

    def test_workflow_files_are_convertible_and_gap1_is_retired(self, converted: Converted) -> None:
        """Every side file is convertible through our converter, and the retired
        ``workflow_subagents_missed`` value is never emitted again."""
        _, report = converted
        assert report.subagent_files_found == 2
        assert report.subagent_files_convertible == 2
        assert report.workflow_subagent_files_found == 1
        assert "workflow_subagents_missed" not in {gap.value for gap in report.gaps_observed}

    def test_raw_harbor_is_still_blind_to_workflow_files(self, synthetic_session: Path) -> None:
        """An ORACLE fact, not a gap: harbor's own discovery misses workflow-nested
        side files when handed the raw layout, which is why our converter does its
        own discovery. Skips the day harbor drops the private method; if it FAILS,
        upstream learned to see them and the oracle's flat staging should be
        re-checked against ours."""
        from harbor_oracle import require_harbor_private_api

        require_harbor_private_api("claude-code")
        from harbor.agents.installed.claude_code import ClaudeCode  # type: ignore[import-untyped]

        session_dir = synthetic_session.parent
        trajectory = ClaudeCode(logs_dir=session_dir.parent)._convert_events_to_trajectory(
            session_dir
        )
        assert trajectory is not None
        texts = [(s.message or "") for s in trajectory.steps if s.source == "user"]
        assert not any("workflow subagent prompt" in t for t in texts)

    def test_structural_gaps_always_observed(self, converted: Converted) -> None:
        _, report = converted
        assert FidelityGap.PARENT_CHAIN_FLATTENED in report.gaps_observed
        # Every subagent step sits in its own embedded trajectory.
        assert FidelityGap.SUBAGENTS_INLINED not in report.gaps_observed

    def test_a_sidechain_with_no_side_file_is_reported_inlined(
        self, synthetic_session: Path
    ) -> None:
        side_dir = synthetic_session.parent / synthetic_session.stem
        records = [
            json.loads(line)
            for path in sorted(side_dir.rglob("*.jsonl"))
            for line in path.read_text().splitlines()
        ]
        with synthetic_session.open("a") as handle:
            handle.writelines(json.dumps(record) + "\n" for record in records)
        shutil.rmtree(side_dir)
        result, report = convert_and_audit(synthetic_session)
        assert any(s["extra"].get("is_sidechain") for s in result.trajectory["steps"])
        assert FidelityGap.SUBAGENTS_INLINED in report.gaps_observed

    def test_gaps_enrichment_repaired_are_not_reported(self, converted: Converted) -> None:
        """Every step attributed and cache_creation_total surfaced: the three repairable gaps stay out."""
        result, report = converted
        extra = result.trajectory.get("extra") or {}
        assert "enrichment_unattributed_steps" not in extra
        assert "cache_creation_total" in extra
        assert (
            not {
                FidelityGap.CACHE_SPLIT_PARTIAL,
                FidelityGap.UUID_NOT_PRESERVED,
                FidelityGap.COMPACT_SUMMARY_UNHANDLED,
            }
            & report.gaps_observed
        )

    def test_workflow_records_now_reach_the_trajectory(self, converted: Converted) -> None:
        """With our own side-file discovery the workflow file's user record becomes a
        step: 1 main user + 2 agent turns in the root (the tool_result user
        event attaches to its pending call in place), 1 subagent user + 1
        subagent assistant, and 1 workflow user = 6."""
        result, _report = converted
        assert len(list(iter_steps(result.trajectory))) == 6


class TestEnrichedWiring:
    def test_result_trajectory_is_enriched(self, converted: Converted) -> None:
        result, _ = converted
        assert result.is_valid  # validator re-ran post-enrichment
        agent_steps = [s for s in result.trajectory["steps"] if s["source"] == "agent"]
        assert all((s.get("extra") or _NO_EXTRA).get("source_uuids") for s in agent_steps)
        assert result.trajectory["extra"]["cache_creation_total"] == 50

    def test_edges_lines_cover_every_raw_record(self, converted: Converted) -> None:
        result, report = converted
        assert len(result.edges_lines) == report.records_total == 9
        uuids = {json.loads(line)["uuid"] for line in result.edges_lines}
        assert {"u-1", "a-1", "u-2", "att-1", "q-1", "a-2", "su-1", "sa-1", "wu-1"} == uuids


_SYSTEM_PROMPT_SID = "33333333-3333-3333-3333-333333333333"


def _record(uuid: str, ts: str, record_type: str, **fields: Any) -> dict[str, Any]:
    return {
        "type": record_type,
        "uuid": uuid,
        "sessionId": _SYSTEM_PROMPT_SID,
        "timestamp": f"2026-11-01T00:00:{ts}Z",
        "isSidechain": False,
        **fields,
    }


def _snapshot_record(uuid: str, ts: str, prompt: list[str], **fields: Any) -> dict[str, Any]:
    attachment = {"type": "prompt_snapshot", "systemPrompt": prompt}
    return _record(uuid, ts, "attachment", attachment=attachment, **fields)


def _spawn(uuid: str, ts: str, msg_id: str, call_id: str, prompt: str) -> dict[str, Any]:
    task = {"type": "tool_use", "id": call_id, "name": "Task", "input": {"prompt": prompt}}
    message = {"id": msg_id, "role": "assistant", "content": [task]}
    return _record(uuid, ts, "assistant", message=message)


def _spawned(uuid: str, ts: str, call_id: str, agent_id: str) -> dict[str, Any]:
    result = {"type": "tool_result", "tool_use_id": call_id, "content": "done"}
    return _record(
        uuid,
        ts,
        "user",
        message={"role": "user", "content": [result]},
        toolUseResult={"status": "completed", "agentId": agent_id},
    )


def _child_records(agent_id: str, *prompts: list[str]) -> list[dict[str, Any]]:
    """``agent_id``'s transcript: a snapshot per prompt, then its prompt and its answer."""
    side = {"isSidechain": True, "agentId": agent_id}
    return [
        *(
            _snapshot_record(f"{agent_id}-s{index}", f"0{index + 3}", prompt, **side)
            for index, prompt in enumerate(prompts)
        ),
        _record(
            f"{agent_id}-u", "10", "user", message={"role": "user", "content": agent_id}, **side
        ),
        _record(
            f"{agent_id}-a",
            "11",
            "assistant",
            message={"id": f"m-{agent_id}", "role": "assistant", "content": "ok"},
            **side,
        ),
    ]


def _write_system_prompt_session(root: Path, custom_prompts: list[list[str]]) -> Path:
    """A root that spawns custom subagent ``c1`` and built-in ``c2``; each writes snapshots."""
    main = root / "-p" / f"{_SYSTEM_PROMPT_SID}.jsonl"
    side = main.parent / _SYSTEM_PROMPT_SID / "subagents"
    side.mkdir(parents=True)
    records = {
        main: [
            _snapshot_record("root-s", "00", ["Main prompt."]),
            _record("u1", "01", "user", message={"role": "user", "content": "go"}),
            _spawn("a1", "02", "m1", "t1", "c1"),
            _spawn("a2", "02", "m2", "t2", "c2"),
            _spawned("r1", "20", "t1", "c1"),
            _spawned("r2", "21", "t2", "c2"),
        ],
        side / "agent-c1.jsonl": _child_records("c1", *custom_prompts),
        side / "agent-c2.jsonl": _child_records("c2", ["Built-in prompt."]),
    }
    for path, lines in records.items():
        path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    (side / "agent-c1.meta.json").write_text(
        json.dumps({"agentType": "reviewer", "isBuiltIn": False}), encoding="utf-8"
    )
    (side / "agent-c2.meta.json").write_text(
        json.dumps({"agentType": "Explore", "isBuiltIn": True}), encoding="utf-8"
    )
    return main


class TestSubagentSystemPrompt:
    """A custom subagent's emitted snapshot is a converted record; skipped ones stay dropped."""

    @staticmethod
    def _children(result: ConversionResult) -> dict[str, dict[str, Any]]:
        return {
            child["trajectory_id"]: child for child in result.trajectory["subagent_trajectories"]
        }

    def test_the_emitted_snapshot_is_converted_and_attributed(self, tmp_path: Path) -> None:
        result, report = convert_and_audit(_write_system_prompt_session(tmp_path, [["A", "B"]]))
        assert result.is_valid, result.validation_errors
        children = self._children(result)
        system, *_turns = children["c1"]["steps"]
        assert (system["step_id"], system["source"], system["message"]) == (1, "system", "A\n\nB")
        assert system["extra"] == {
            "is_sidechain": True,
            "agent_id": "c1",
            "source_uuids": ["c1-s0"],
        }
        assert [s["source"] for s in children["c2"]["steps"]] == ["user", "agent"]
        assert all(s["source"] != "system" for s in result.trajectory["steps"])
        # 9 user/assistant records in all, plus c1's snapshot; the root's and c2's are dropped.
        assert report.record_counts[RecordType.ATTACHMENT] == 3
        assert report.records_converted == 9 + 1
        assert report.records_captured == 0
        assert report.records_dropped == 2
        assert report.records_total == 12
        assert result.events_lines == ()
        assert FidelityGap.UUID_NOT_PRESERVED not in report.gaps_observed
        assert "enrichment_unattributed_steps" not in (children["c1"].get("extra") or _NO_EXTRA)

    def test_a_skipped_repeat_stays_dropped(self, tmp_path: Path) -> None:
        result, report = convert_and_audit(
            _write_system_prompt_session(tmp_path, [["A"], ["A"], ["B"]])
        )
        c1 = self._children(result)["c1"]
        assert [(s["source"], s["extra"].get("source_uuids")) for s in c1["steps"]][:3] == [
            ("system", ["c1-s0"]),
            ("system", ["c1-s2"]),
            ("user", ["c1-u"]),
        ]
        assert report.records_converted == 9 + 2
        assert report.records_dropped == 3  # the root's, c2's, and c1's repeat
        assert (
            report.records_converted + report.records_captured + report.records_dropped
            == report.records_total
        )


class TestRecordTypeTaxonomy:
    @pytest.mark.parametrize(
        ("raw", "member"),
        [
            ("pr-link", RecordType.PR_LINK),
            ("started", RecordType.STARTED),
            ("result", RecordType.RESULT),
        ],
    )
    def test_new_members_map_in_census(self, tmp_path: Path, raw: str, member: RecordType) -> None:
        from atif_converter.infrastructure.census import census_from_snapshot
        from atif_converter.infrastructure.raw_records import (
            read_snapshot_records,
            take_session_snapshot,
        )

        session = tmp_path / "22222222-2222-2222-2222-222222222222.jsonl"
        session.write_text(json.dumps({"type": raw, "uuid": "x-1"}) + "\n")
        snapshot = take_session_snapshot(session)
        census = census_from_snapshot(snapshot, read_snapshot_records(snapshot))
        assert census.record_counts == {member: 1}


class TestInputValidation:
    def test_rejects_non_jsonl(self, tmp_path: Path) -> None:
        bogus = tmp_path / "not-a-session.txt"
        bogus.write_text("nope")
        with pytest.raises(InvalidSessionInput):
            convert_and_audit(bogus)


class TestReaderErrors:
    def test_unreadable_side_file_is_a_conversion_error_with_the_cause(
        self, synthetic_session: Path
    ) -> None:
        """A side file the process cannot open fails the SESSION, classified.

        The seam wraps the converter in ``ConversionError`` so materialize
        records one per-session failure and continues; the original ``OSError``
        rides along as ``__cause__`` so the log says which file and why.
        """
        import os
        import stat

        from atif_converter.domain.errors import ConversionError
        from atif_converter.infrastructure.harbor_adapter import convert_session

        if os.geteuid() == 0:  # pragma: no cover — root reads a 000 file anyway
            pytest.skip("permission bits do not bind root")
        side_dir = synthetic_session.parent / synthetic_session.stem
        side_file = next(side_dir.rglob("*.jsonl"))
        side_file.chmod(0)
        try:
            with pytest.raises(ConversionError) as excinfo:
                convert_session(synthetic_session)
        finally:
            side_file.chmod(stat.S_IRUSR | stat.S_IWUSR)
        assert isinstance(excinfo.value.__cause__, OSError)
