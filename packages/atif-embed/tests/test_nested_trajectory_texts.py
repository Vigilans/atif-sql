# SPDX-License-Identifier: Apache-2.0

"""Embedded ``subagent_trajectories`` steps are embeddable texts, after the root's."""

from __future__ import annotations

import json
from pathlib import Path

from atif_embed.infrastructure.corpus_text_rows import DuckDbTextRows

_TEXT = "x" * 40


def _step(step_id: int, uuid: str) -> dict[str, object]:
    return {
        "step_id": step_id,
        "timestamp": "2026-09-20T10:00:00Z",
        "source": "agent",
        "message": f"{uuid} {_TEXT}",
        "extra": {"source_uuids": [uuid]},
    }


def test_embedded_steps_are_read_after_the_root(tmp_path: Path) -> None:
    session_dir = tmp_path / "sessions" / "33333333-3333-3333-3333-333333333333"
    session_dir.mkdir(parents=True)
    child = {
        "trajectory_id": "c",
        "steps": [_step(1, "u-c1")],
        "subagent_trajectories": [{"trajectory_id": "g", "steps": [_step(1, "u-g1")]}],
    }
    (session_dir / "trajectory.json").write_text(
        json.dumps(
            {"steps": [_step(1, "u-r1"), _step(2, "u-r2")], "subagent_trajectories": [child]}
        )
    )
    (session_dir / "meta.json").write_text(json.dumps({"session_id": session_dir.name}))
    rows = list(DuckDbTextRows().iter_unembedded(tmp_path))
    assert [row.uuid for row in rows] == ["u-r1", "u-r2", "u-c1", "u-g1"]
