# SPDX-License-Identifier: Apache-2.0

"""``content_text(content)`` and ``steps.message`` render ATIF content alike.

ATIF content is a string or a ``ContentPart`` array. The macro returns a
string unquoted and joins an array's parts with blank lines, an image or
audio part as ``[<kind> <path> <media_type>]`` with its session-relative
``../../blobs/`` path read from the corpus root. ``steps.message`` applies the
same expression, on every read path and in the parquet artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import pytest
from duck_fixtures import NESTED_SESSION_ID, register_via, write_nested_session

from atif_duck.infrastructure.columnar import ColumnarArtifactProducer
from atif_duck.infrastructure.projections import CONTENT_TEXT_SQL
from atif_duck.infrastructure.registry import register

PNG, WAV = "a" * 64, "b" * 64
PNG_PATH = f"blobs/sha256/aa/{PNG}.png"
WAV_PATH = f"blobs/sha256/bb/{WAV}.wav"


def _image(path: str = f"../../{PNG_PATH}") -> dict[str, Any]:
    return {"type": "image", "source": {"media_type": "image/png", "path": path}}


def _audio() -> dict[str, Any]:
    return {"type": "audio", "source": {"media_type": "audio/wav", "path": f"../../{WAV_PATH}"}}


def _text(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def _meta(sha256: str, size: int) -> dict[str, Any]:
    return {"sha256": sha256, "media_type": "image/png", "bytes": size, "width": 1, "height": 1}


@pytest.fixture(scope="module")
def macro() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"CREATE MACRO content_text(content) AS ({CONTENT_TEXT_SQL});")
    return con


def _render(con: duckdb.DuckDBPyConnection, content: Any) -> str | None:
    row = con.execute(
        "SELECT content_text(?::JSON)", [None if content is None else json.dumps(content)]
    ).fetchone()
    assert row is not None
    return row[0]


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("hello\nworld", "hello\nworld"),
        ("  padded  ", "  padded  "),
        (None, None),
        ([], ""),
        ([_text("one"), _text("two")], "one\n\ntwo"),
        ([_text("see"), _image()], f"see\n\n[image {PNG_PATH} image/png]"),
        ([_audio(), _text("said")], f"[audio {WAV_PATH} audio/wav]\n\nsaid"),
        # A path outside the corpus blob store is shown as written.
        ([_image("https://example.com/x.png")], "[image https://example.com/x.png image/png]"),
        # A part of another shape renders nothing.
        ([{"type": "video"}, _text("kept"), {"type": "text"}], "kept"),
    ],
)
def test_the_macro_renders_content(
    macro: duckdb.DuckDBPyConnection, content: Any, expected: str | None
) -> None:
    assert _render(macro, content) == expected


def test_the_macro_takes_an_untyped_null(macro: duckdb.DuckDBPyConnection) -> None:
    assert macro.execute("SELECT content_text(NULL)").fetchone() == (None,)


def _corpus(tmp_path: Path) -> Path:
    """The nested session with a pasted image on the root's user step, an image in
    the root's tool result, and audio in ``child-a``'s."""
    root = tmp_path / "corpus"
    path = write_nested_session(root) / "trajectory.json"
    document = json.loads(path.read_text())
    user, agent = document["steps"]
    user["message"] = [_text("survey it"), _image()]
    user["extra"]["images"] = [_meta(PNG, 90)]
    result = agent["observation"]["results"][0]
    result["content"] = [_image(), _text("inspected")]
    result["extra"] = {"images": [_meta(PNG, 90)]}
    child_result = document["subagent_trajectories"][0]["steps"][1]["observation"]["results"][0]
    child_result["content"] = [_text("deep done"), _audio()]
    path.write_text(json.dumps(document, separators=(",", ":")))
    return root


EXPECTED_MESSAGE = (NESTED_SESSION_ID, 1, f"survey it\n\n[image {PNG_PATH} image/png]")

EXPECTED_RESULTS = [
    (NESTED_SESSION_ID, "toolu_outer", f"[image {PNG_PATH} image/png]\n\ninspected"),
    ("child-a", "toolu_inner", f"deep done\n\n[audio {WAV_PATH} audio/wav]"),
]


def _assert_rendered(con: duckdb.DuckDBPyConnection) -> None:
    assert (
        con.execute(
            "SELECT trajectory_id, step_id, message FROM steps WHERE source = 'user' "
            "ORDER BY ALL LIMIT 1"
        ).fetchone()
        == EXPECTED_MESSAGE
    )
    assert (
        con.execute(
            "SELECT trajectory_id, tool_use_id, content_text(content) "
            "FROM tool_results ORDER BY ALL"
        ).fetchall()
        == EXPECTED_RESULTS
    )
    # The column itself stays the stored ContentPart array.
    content = con.execute(
        "SELECT content FROM tool_results WHERE tool_use_id = 'toolu_inner'"
    ).fetchone()
    assert content is not None
    assert json.loads(content[0]) == [_text("deep done"), _audio()]


def test_every_read_path_renders_content_parts(tmp_path: Path, read_path: str) -> None:
    root = _corpus(tmp_path)
    con = duckdb.connect()
    register_via(con, root, read_path)
    _assert_rendered(con)


def test_the_parquet_artifacts_render_content_parts(tmp_path: Path) -> None:
    root = _corpus(tmp_path)
    session_dir = root / "sessions" / NESTED_SESSION_ID
    trajectory = json.loads((session_dir / "trajectory.json").read_text())
    extras = ColumnarArtifactProducer().produce(
        session_dir, session_id=NESTED_SESSION_ID, trajectory=trajectory
    )
    meta = json.loads((session_dir / "meta.json").read_text())
    (session_dir / "meta.json").write_text(json.dumps({**meta, **extras}))
    (session_dir / "trajectory.json").unlink()
    con = duckdb.connect()
    sources = register(con, root)
    assert len(sources.columnar_session_ids) == 1
    _assert_rendered(con)
