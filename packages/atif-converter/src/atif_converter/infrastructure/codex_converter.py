# SPDX-License-Identifier: Apache-2.0
# Ported from harbor 0.22.0, src/harbor/agents/installed/codex.py (Apache-2.0,
# Copyright the Harbor authors), so that atif-converter depends on harbor's
# PUBLIC trajectory models only.

"""Read one Codex rollout and convert it — the file half of the Codex port.

harbor's ``Codex._convert_events_to_trajectory`` takes a DIRECTORY: it globs
``*.jsonl`` inside it, converts ``max(...)`` of the matches (one file, chosen
by name, the rest dropped) and reads that file line by line. The adapter this
replaces satisfied that shape by symlinking one rollout alone into a scratch
directory. This module reads the rollout directly instead, and the two are
the same thing by construction: with exactly one file in the directory,
``max`` is that file, so "the directory's one file" and "this file" are one
read.

The line loop is harbor's, kept exact rather than replaced with
``read_text().splitlines()``: iterating the handle splits on newlines only,
where ``str.splitlines`` also splits on U+2028 / U+2029 / form feed, which are
legal unescaped inside a JSON string — a rollout carrying one would parse into
different records under the two loops.

A rollout whose ``session_meta`` names no thread id takes the rollout's stem
as its ``session_id``, as harbor 0.26.0 does; a ``.jsonl.zst`` rollout's stem
drops both suffixes, so it reads as the plain file it compresses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from atif_converter.domain.atif import Trajectory
from atif_converter.domain.codex_conversion import convert_codex_records
from atif_converter.infrastructure.raw_records import (
    LoadedSession,
    parse_jsonl_records,
    text_stream,
)


def rollout_stem(rollout_jsonl: Path) -> str:
    """The rollout's name without ``.jsonl`` (or ``.jsonl.zst``): harbor's ``session_file.stem``."""
    return rollout_jsonl.name.removesuffix(".zst").removesuffix(".jsonl")


def read_codex_rollout(rollout_jsonl: Path) -> list[dict[str, Any]]:
    """The rollout's JSON records in file order, malformed lines skipped.

    harbor's read loop: blank lines are ignored, a line that is not JSON is
    logged at debug level and dropped, and everything that parses is kept
    (harbor does not check that a parsed line is an object).

    Raises
    ------
        OSError: the file cannot be opened or read.
    """
    with rollout_jsonl.open("rb") as raw, text_stream(raw, rollout_jsonl) as handle:
        return parse_jsonl_records(handle, rollout_jsonl)


def convert_codex_rollout(rollout_jsonl: Path) -> Trajectory | None:
    """Convert one Codex rollout JSONL into an ATIF trajectory.

    ``None`` exactly when harbor's converter returns ``None`` for the same
    file staged alone: the rollout has no records, or none that becomes a
    step. Validation and the domain error taxonomy are the application
    layer's business, as they were for the adapter this replaces.

    Raises
    ------
        OSError: the file cannot be opened or read.
    """
    raw_events = read_codex_rollout(rollout_jsonl)
    return convert_codex_records(raw_events, fallback_session_id=rollout_stem(rollout_jsonl))


def convert_loaded_codex_rollout(loaded: LoadedSession) -> Trajectory | None:
    """Convert an already-read rollout: the records of :func:`~atif_converter.infrastructure.raw_records.load_session`.

    Same result as :func:`convert_codex_rollout` on the same bytes, with no
    second read. Only the main file's records are converted; a rollout has no
    side-files, so anything else the loader discovered is the audit's
    business, as it was before.
    """
    rollout_jsonl = loaded.snapshot.session_jsonl
    return convert_codex_records(
        loaded.main_records, fallback_session_id=rollout_stem(rollout_jsonl)
    )


def convert_codex_record_list(records: list[Any], rollout_jsonl: Path) -> Trajectory | None:
    """Convert a record list read from ``rollout_jsonl`` and its sibling rollouts.

    The same conversion as :func:`convert_loaded_codex_rollout`, over records the
    caller assembled (a session's rollouts in conversion order).
    """
    return convert_codex_records(records, fallback_session_id=rollout_stem(rollout_jsonl))


__all__ = [
    "convert_codex_record_list",
    "convert_codex_rollout",
    "convert_loaded_codex_rollout",
    "read_codex_rollout",
    "rollout_stem",
]
