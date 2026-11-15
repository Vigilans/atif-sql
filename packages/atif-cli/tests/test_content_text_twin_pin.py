# SPDX-License-Identifier: Apache-2.0

"""Drift pin for the ContentPart-to-text rule every package spells on its own.

atif-duck's ``content_text(content)`` macro renders ATIF content for
the views and the columnar artifacts; atif-embed renders a step's text with
its own copy of the SQL and atif-analytics with a Python twin, because the
independence contract keeps each from importing atif-duck. atif-cli may
import them all, so the pin lives here: every case below must read the same
through all three, or an embedding or an analysis prompt would see a
different text than a query does.
"""

from __future__ import annotations

import json
from typing import Any

import duckdb
import pytest

from atif_analytics.domain.transcript import content_text
from atif_duck.infrastructure.projections import CONTENT_TEXT_SQL
from atif_embed.infrastructure.corpus_text_rows import content_text_sql

PNG, WAV = "a" * 64, "b" * 64
PNG_PATH = f"../../blobs/sha256/aa/{PNG}.png"


def _image(path: str = PNG_PATH, media_type: str = "image/png") -> dict[str, Any]:
    return {"type": "image", "source": {"media_type": media_type, "path": path}}


def _audio() -> dict[str, Any]:
    return {
        "type": "audio",
        "source": {"media_type": "audio/wav", "path": f"../../blobs/sha256/bb/{WAV}.wav"},
    }


def _text(text: object) -> dict[str, Any]:
    return {"type": "text", "text": text}


CASES: list[Any] = [
    "plain",
    "  padded \n",
    "",
    None,
    [],
    [_text("one"), _text("two")],
    [_text(""), _text("after an empty part")],
    [_text("see"), _image()],
    [_image()],
    [_audio(), _text("said")],
    [_image("https://example.com/x.png")],
    [_image("blobs/sha256/aa/" + PNG + ".png")],
    [_image(media_type="image/heic")],
    [{"type": "video"}, _text("kept"), {"type": "text"}, _text(5)],
    [{"type": "image"}, {"type": "audio", "source": {"path": PNG_PATH}}],
    ["a string item", 3, None, _text("kept")],
]


@pytest.fixture(scope="module")
def con() -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect()
    connection.execute(f"CREATE MACRO duck_text(content) AS ({CONTENT_TEXT_SQL});")
    connection.execute(f"CREATE MACRO embed_text(content) AS ({content_text_sql('content')});")
    return connection


@pytest.mark.parametrize("content", CASES)
def test_every_package_renders_content_alike(con: duckdb.DuckDBPyConnection, content: Any) -> None:
    param = None if content is None else json.dumps(content)
    rendered = con.execute("SELECT duck_text(?::JSON), embed_text(?::JSON)", [param, param])
    row = rendered.fetchone()
    assert row is not None
    duck, embed = row
    assert embed == duck
    assert content_text(content) == duck


def test_the_cases_reach_every_branch() -> None:
    """The pin compares rendered text, so it is only as strong as the texts differ."""
    rendered = {content_text(content) for content in CASES}
    assert f"see\n\n[image blobs/sha256/aa/{PNG}.png image/png]" in rendered
    assert f"[image blobs/sha256/aa/{PNG}.png image/png]" in rendered
    assert f"[audio blobs/sha256/bb/{WAV}.wav audio/wav]\n\nsaid" in rendered
    assert "[image https://example.com/x.png image/png]" in rendered
    assert "\n\nafter an empty part" in rendered
    assert "kept" in rendered
