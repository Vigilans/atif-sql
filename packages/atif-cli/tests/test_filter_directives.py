# SPDX-License-Identifier: Apache-2.0

"""The directive executor over hand-built trajectories, Codex's hidden copies included."""

from __future__ import annotations

import json
from typing import Any

import pytest

from atif_cli.filter_directives import (
    Directives,
    FilterError,
    Targets,
    apply_directives,
    merge_directives,
    serialized,
)
from atif_corpus.domain.ports import BlobOutput, ConversionOutput

SESSION = "eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee"
SHARED, OWN = "a" * 64, "b" * 64


def _image(sha256: str) -> dict[str, Any]:
    return {"sha256": sha256, "media_type": "image/png", "bytes": 1, "extension": "png"}


def _output() -> ConversionOutput:
    """A Codex-shaped root step and an embedded step with its own result."""
    codex_step: dict[str, Any] = {
        "step_id": 1,
        "source": "agent",
        "message": "",
        "reasoning_content": "plan",
        "tool_calls": [{"tool_call_id": "c1", "function_name": "exec", "arguments": {"cmd": "ls"}}],
        "observation": {
            "results": [
                {
                    "source_call_id": "c1",
                    "content": f"[image sha256:{OWN} image/png 1 bytes]",
                    "extra": {"images": [_image(OWN)], "exit_code": 0},
                }
            ]
        },
        "extra": {
            "raw_arguments": '{"cmd":"ls"}',
            "tool_call_details": {
                "c1": {"raw_arguments": '{"cmd":"ls"}', "status": "completed"},
                "c2": {"raw_arguments": "{}"},
            },
            "images": [_image(SHARED)],
        },
    }
    child_step: dict[str, Any] = {
        "step_id": 1,
        "source": "agent",
        "message": f"see [image sha256:{SHARED} image/png 1 bytes]",
        "tool_calls": [
            {
                "tool_call_id": "c3",
                "function_name": "Read",
                "arguments": {"path": "/home/alice/x"},
                "extra": {"raw_arguments": {"path": "/home/alice/x"}},
            }
        ],
        "observation": {
            "results": [
                {
                    "source_call_id": "c3",
                    "content": "body",
                    "extra": {
                        "tool_result_metadata": {
                            "tool_use_result": {"stdout": "body"},
                            "raw_tool_result": {"content": "body"},
                            "is_error": True,
                        }
                    },
                }
            ]
        },
    }
    trajectory = {
        "schema_version": "ATIF-v1.7",
        "session_id": SESSION,
        "steps": [codex_step],
        "subagent_trajectories": [{"trajectory_id": "child", "steps": [child_step]}],
    }
    events = [
        json.dumps({"seq": 0, "event_type": "mode", "payload": {"note": "keep"}}),
        json.dumps({"seq": 1, "event_type": "mode", "payload": {"note": "drop"}}),
    ]
    return ConversionOutput(
        trajectory_dict=trajectory,
        loss_report_dict={"gaps_observed": []},
        edges_lines=[json.dumps({"uuid": "u1", "source_file": "/home/alice/s.jsonl"})],
        events_lines=events,
        blobs=(
            BlobOutput(sha256=SHARED, extension="png", data=b"s"),
            BlobOutput(sha256=OWN, extension="png", data=b"o"),
        ),
    )


def _steps(output: ConversionOutput) -> tuple[dict[str, Any], dict[str, Any]]:
    trajectory = output.trajectory_dict
    return trajectory["steps"][0], trajectory["subagent_trajectories"][0]["steps"][0]


def test_hidden_copies_go_and_everything_else_stays() -> None:
    original = _output()
    before = serialized(original)
    filtered = apply_directives(original, Directives(), session_id=SESSION)
    root, child = _steps(filtered)
    assert root["extra"] == {
        "tool_call_details": {"c1": {"status": "completed"}},
        "images": [_image(SHARED)],
    }
    assert "extra" not in child["tool_calls"][0]
    assert child["observation"]["results"][0]["extra"] == {
        "tool_result_metadata": {"is_error": True}
    }
    assert filtered.blobs == original.blobs
    assert filtered.events_lines == original.events_lines
    # A filtered session publishes no edges.
    assert filtered.edges_lines == []
    # The input is never mutated.
    assert serialized(original) == before


def test_prune_and_drop_act_on_their_targets_only() -> None:
    directives = merge_directives(
        [
            [
                ("prune_tool_result", (SESSION, 1, "c1")),
                ("drop_reasoning", (SESSION, 1)),
                ("drop_event", (1,)),
            ]
        ]
    )
    filtered = apply_directives(_output(), directives, session_id=SESSION)
    root, child = _steps(filtered)
    assert "reasoning_content" not in root
    assert root["observation"]["results"][0] == {
        "source_call_id": "c1",
        "content": "",
        "extra": {"exit_code": 0},
    }
    assert child["observation"]["results"][0]["content"] == "body"
    assert [json.loads(line)["seq"] for line in filtered.events_lines] == [0]
    # The pruned result owned OWN alone; SHARED is still named elsewhere.
    assert [blob.sha256 for blob in filtered.blobs] == [SHARED]


def test_drop_blob_removes_every_reference_entry_and_the_blob() -> None:
    directives = merge_directives([[("drop_blob", (SHARED,))]])
    filtered = apply_directives(_output(), directives, session_id=SESSION)
    root, _ = _steps(filtered)
    assert "images" not in root["extra"]
    assert [blob.sha256 for blob in filtered.blobs] == [OWN]


def test_replace_reaches_every_artifact_but_no_key() -> None:
    directives = merge_directives(
        [[("replace", ("/home/alice", "~")), ("replace", ("drop", "gone"))]]
    )
    filtered = apply_directives(_output(), directives, session_id=SESSION)
    _, child = _steps(filtered)
    assert child["tool_calls"][0]["arguments"] == {"path": "~/x"}
    assert json.loads(filtered.events_lines[1])["payload"] == {"note": "gone"}
    # An untouched line keeps its bytes.
    assert filtered.events_lines[0] == _output().events_lines[0]
    keys = merge_directives([[("replace", ("path", "x"))]])
    _, child = _steps(apply_directives(_output(), keys, session_id=SESSION))
    assert child["tool_calls"][0]["arguments"] == {"path": "/home/alice/x"}


def test_targets_are_the_converted_sessions() -> None:
    targets = Targets.of(_output(), SESSION)
    assert targets.results == {(SESSION, 1, "c1"), ("child", 1, "c3")}
    assert targets.steps == {(SESSION, 1), ("child", 1)}
    assert targets.events == {0, 1}
    assert targets.blobs == {SHARED, OWN}
    targets.check(merge_directives([[("drop_event", (1,)), ("drop_blob", (OWN,))]]))
    with pytest.raises(FilterError, match="address no step"):
        targets.check(merge_directives([[("drop_reasoning", ("child", 2))]]))


def test_a_second_application_changes_nothing() -> None:
    directives = merge_directives(
        [
            [
                ("prune_tool_result", ("child", 1, "c3")),
                ("drop_event", (1,)),
                ("drop_blob", (SHARED,)),
                ("replace", ("body", "text")),
            ]
        ]
    )
    once = apply_directives(_output(), directives, session_id=SESSION)
    twice = apply_directives(once, directives, session_id=SESSION)
    assert serialized(twice) == serialized(once)
