# SPDX-License-Identifier: Apache-2.0

"""Embedded ``subagent_trajectories``: every step reaches the step-level views.

A session's root, child and grandchild each number their steps from 1, so the
views key a step by ``(session_id, trajectory_id, step_id)`` and mark every
embedded step a sidechain step of its trajectory's agent. Each read path the
registry can take (``trajectory.json``, compressed, the lake) must agree, and
so must the parquet artifacts the producer writes. ``images``, projected over
``steps`` and ``tool_results``, keys each image by the same triple.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
from duck_fixtures import NESTED_SESSION_ID, register_via, write_nested_session

from atif_duck.infrastructure.columnar import ColumnarArtifactProducer
from atif_duck.infrastructure.registry import register

#: Every step of the nested fixture: ``(trajectory_id, step_id, is_sidechain, agent_id)``.
EXPECTED_STEPS = [
    (NESTED_SESSION_ID, 1, False, None),
    (NESTED_SESSION_ID, 2, False, None),
    ("child-a", 1, True, "child-a"),
    ("child-a", 2, True, "child-a"),
    ("grand-b", 1, True, "grand-b"),
]


#: The images the corpus adds to the root's and ``child-a``'s steps, which share
#: step ids 1 and 2.
ROOT_PASTED, ROOT_READ, CHILD_PASTED, CHILD_READ = ("a" * 64, "b" * 64, "c" * 64, "d" * 64)

#: ``(trajectory_id, step_id, origin, sha256, second of the joined step's ts)``.
EXPECTED_IMAGES = [
    (NESTED_SESSION_ID, 1, "user_message", ROOT_PASTED, "01"),
    (NESTED_SESSION_ID, 2, "tool_result", ROOT_READ, "04"),
    ("child-a", 1, "user_message", CHILD_PASTED, "02"),
    ("child-a", 2, "tool_result", CHILD_READ, "03"),
]


def _rows(con: duckdb.DuckDBPyConnection, sql: str) -> list[tuple[Any, ...]]:
    return con.execute(sql).fetchall()


def _image(sha256: str) -> dict[str, Any]:
    return {
        "sha256": sha256,
        "media_type": "image/png",
        "bytes": 8,
        "width": 2,
        "height": 2,
        "extension": "png",
    }


def _corpus(tmp_path: Path) -> Path:
    """The nested session, with an image on each user step and tool result of the
    root and ``child-a``."""
    root = tmp_path / "corpus"
    path = write_nested_session(root) / "trajectory.json"
    document = json.loads(path.read_text())
    child = document["subagent_trajectories"][0]
    for trajectory, pasted, read in (
        (document, ROOT_PASTED, ROOT_READ),
        (child, CHILD_PASTED, CHILD_READ),
    ):
        user, agent = trajectory["steps"]
        user["extra"]["images"] = [_image(pasted)]
        agent["observation"]["results"][0]["extra"] = {"images": [_image(read)]}
    path.write_text(json.dumps(document, separators=(",", ":")))
    return root


def _assert_nested_views(con: duckdb.DuckDBPyConnection) -> None:
    assert (
        _rows(
            con,
            "SELECT trajectory_id, step_id, is_sidechain, agent_id FROM steps ORDER BY ALL",
        )
        == EXPECTED_STEPS
    )
    assert _rows(
        con, "SELECT trajectory_id, step_id, tool_use_id FROM tool_calls ORDER BY ALL"
    ) == [(NESTED_SESSION_ID, 2, "toolu_outer"), ("child-a", 2, "toolu_inner")]
    assert _rows(
        con, "SELECT trajectory_id, step_id, tool_use_id FROM tool_results ORDER BY ALL"
    ) == [(NESTED_SESSION_ID, 2, "toolu_outer"), ("child-a", 2, "toolu_inner")]
    assert _rows(con, "SELECT agent_id, count(*) FROM subagent_steps GROUP BY 1 ORDER BY 1") == [
        ("child-a", 2),
        ("grand-b", 1),
    ]
    # Usage summed over steps counts the subagents' steps too.
    assert _rows(con, "SELECT sum(prompt_tokens) FROM steps") == [(30,)]
    # Each image joins the step it came from, never another trajectory's step
    # of the same number.
    assert (
        _rows(
            con,
            "SELECT i.trajectory_id, i.step_id, i.origin, i.sha256, strftime(s.ts, '%S') "
            "FROM images i JOIN steps s USING (session_id, trajectory_id, step_id) ORDER BY ALL",
        )
        == EXPECTED_IMAGES
    )


def test_every_read_path_reaches_the_embedded_steps(tmp_path: Path, read_path: str) -> None:
    root = _corpus(tmp_path)
    con = duckdb.connect()
    register_via(con, root, read_path)
    _assert_nested_views(con)


def test_the_parquet_artifacts_hold_the_embedded_steps(tmp_path: Path) -> None:
    root = _corpus(tmp_path)
    session_dir = root / "sessions" / NESTED_SESSION_ID
    trajectory = json.loads((session_dir / "trajectory.json").read_text())
    extras = ColumnarArtifactProducer().produce(
        session_dir, session_id=NESTED_SESSION_ID, trajectory=trajectory
    )
    meta = json.loads((session_dir / "meta.json").read_text())
    (session_dir / "meta.json").write_text(json.dumps({**meta, **extras}))
    # The registry must read the parquet, not the document.
    (session_dir / "trajectory.json").unlink()
    con = duckdb.connect()
    sources = register(con, root)
    assert len(sources.columnar_session_ids) == 1
    _assert_nested_views(con)
    # Source order: the root's steps, then each embedded trajectory's, depth first.
    assert _rows(
        con,
        f"SELECT trajectory_id, step_id FROM read_parquet('{session_dir / 'steps.parquet'}')",
    ) == [
        (NESTED_SESSION_ID, 1),
        (NESTED_SESSION_ID, 2),
        ("child-a", 1),
        ("child-a", 2),
        ("grand-b", 1),
    ]
