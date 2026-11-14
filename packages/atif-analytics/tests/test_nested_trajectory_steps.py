# SPDX-License-Identifier: Apache-2.0

"""The file source reads the steps of embedded ``subagent_trajectories`` as sidechain steps."""

from __future__ import annotations

import json
from pathlib import Path

from atif_analytics.infrastructure.corpus_reader import read_trajectory_steps


def _step(step_id: int, message: str, uuid: str) -> dict[str, object]:
    return {
        "step_id": step_id,
        "timestamp": f"2026-09-20T10:00:0{step_id}Z",
        "source": "user",
        "message": message,
        "extra": {"source_uuids": [uuid]},
    }


def test_embedded_steps_follow_the_root_as_sidechain_steps(tmp_path: Path) -> None:
    grandchild = {"trajectory_id": "g", "steps": [_step(1, "grandchild", "u-g1")]}
    child = {
        "trajectory_id": "c",
        "steps": [_step(1, "child", "u-c1")],
        "subagent_trajectories": [grandchild],
    }
    trajectory = {"steps": [_step(1, "root", "u-r1")], "subagent_trajectories": [child]}
    session_dir = tmp_path / "s"
    session_dir.mkdir()
    (session_dir / "trajectory.json").write_text(json.dumps(trajectory))
    steps = read_trajectory_steps(session_dir)
    assert [(step.text, step.is_sidechain) for step in steps] == [
        ("root", False),
        ("child", True),
        ("grandchild", True),
    ]
