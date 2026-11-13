# SPDX-License-Identifier: Apache-2.0

"""A Codex session of several rollouts: a thread and its continuation segment.

The session below is the shape Codex writes when a thread is continued from an
earlier point:

* the thread's first rollout holds two turns;
* a continuation segment (written later, compressed, in another day directory)
  continues from before the second turn, so that turn in the first rollout is
  a branch the thread left.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import zstandard

from atif_converter.application.convert_codex import convert_codex_and_audit
from atif_converter.domain.codex_threads import (
    RolloutFile,
    assemble_codex_session,
    rollout_ids,
)

if TYPE_CHECKING:
    from collections.abc import Callable

ROOT = "01a149e0-1f11-71d3-989c-2f36b98a004a"
SEGMENT = "01a158c3-25de-70a3-b159-33361d98a15a"
MODEL = "openai/gpt-5.1-codex"


def _token_count() -> dict[str, Any]:
    usage = {"input_tokens": 10, "cached_input_tokens": 0, "output_tokens": 5, "total_tokens": 15}
    return {
        "type": "token_count",
        "info": {"total_token_usage": usage, "last_token_usage": usage},
    }


def _message(role: str, text: str, item_id: str) -> dict[str, Any]:
    kind = "output_text" if role == "assistant" else "input_text"
    return {
        "type": "message",
        "id": item_id,
        "role": role,
        "content": [{"type": kind, "text": text}],
    }


def _records(meta: dict[str, Any], start: int, items: list[tuple[str, dict[str, Any]]]):
    """``session_meta`` at ordinal ``start - 1`` (or 0), then ``items`` from ``start`` on."""
    records: list[dict[str, Any]] = [
        {
            "timestamp": "2026-10-17T12:00:00.000Z",
            "ordinal": max(start - 1, 0),
            "type": "session_meta",
            "payload": meta,
        },
        {
            "timestamp": "2026-10-17T12:00:00.100Z",
            "type": "turn_context",
            "payload": {"turn_id": "t", "model": MODEL},
        },
    ]
    for offset, (record_type, payload) in enumerate(items):
        records.append(
            {
                "timestamp": f"2026-10-17T12:00:{offset + 1:02d}.000Z",
                "ordinal": start + offset,
                "type": record_type,
                "payload": payload,
            }
        )
    return records


def _turn(n: str) -> list[tuple[str, dict[str, Any]]]:
    return [
        ("response_item", _message("user", f"question {n}", f"msg_u{n}")),
        ("response_item", _message("assistant", f"answer {n}", f"msg_a{n}")),
        ("event_msg", _token_count()),
    ]


def first_rollout() -> list[dict[str, Any]]:
    return _records({"id": ROOT, "session_id": ROOT, "cwd": "/w"}, 1, [*_turn("1"), *_turn("2x")])


def segment_rollout() -> list[dict[str, Any]]:
    # Continues the first rollout before its second turn (ordinal 4).
    meta = {
        "id": ROOT,
        "session_id": ROOT,
        "history_base": {"thread_id": ROOT, "end_ordinal_exclusive": 4},
    }
    return _records(meta, 4, _turn("2"))


def _write(path: Path, records: list[dict[str, Any]], *, compressed: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = ("\n".join(json.dumps(r) for r in records) + "\n").encode()
    path.write_bytes(zstandard.ZstdCompressor().compress(data) if compressed else data)
    return path


def write_session(root: Path) -> tuple[Path, list[Path]]:
    first = _write(
        root / "2026/10/17" / f"rollout-2026-10-17T12-00-00-{ROOT}.jsonl", first_rollout()
    )
    segment = _write(
        root / "2026/10/20" / f"rollout-2026-10-20T12-00-00-{ROOT}_{SEGMENT}.jsonl.zst",
        segment_rollout(),
        compressed=True,
    )
    return first, [segment]


class TestRolloutIds:
    def test_first_rollout_and_segment(self) -> None:
        assert rollout_ids(f"rollout-2026-10-17T12-00-00-{ROOT}.jsonl") == (ROOT, ROOT)
        assert rollout_ids(f"rollout-2026-10-20T12-00-00-{ROOT}_{SEGMENT}.jsonl.zst") == (
            ROOT,
            SEGMENT,
        )
        assert rollout_ids("notes.jsonl") is None


class TestAssembly:
    def _stream(self):
        return assemble_codex_session(
            [
                RolloutFile(f"10/17/rollout-2026-10-17T12-00-00-{ROOT}.jsonl", first_rollout()),
                RolloutFile(
                    f"10/20/rollout-2026-10-20T12-00-00-{ROOT}_{SEGMENT}.jsonl.zst",
                    segment_rollout(),
                ),
            ]
        )

    def test_the_first_rollout_comes_before_its_segment(self) -> None:
        stream = self._stream()
        files = list(dict.fromkeys(source for _record, source in stream.records))
        assert [rollout_ids(Path(f).name) for f in files] == [(ROOT, ROOT), (ROOT, SEGMENT)]

    def test_the_branch_past_the_history_base_cut_is_reverted(self) -> None:
        stream = self._stream()
        reverted_texts = [
            record["payload"]["content"][0]["text"]
            for (record, _source), is_reverted in zip(stream.records, stream.reverted, strict=True)
            if is_reverted and record["payload"].get("type") == "message"
        ]
        assert reverted_texts == ["question 2x", "answer 2x"]

    def test_a_rollout_off_the_chain_is_reverted_in_full(self) -> None:
        # A second segment that continues the FIRST rollout, abandoned for the newest one.
        abandoned_id = "01a15000-0000-7000-8000-000000000000"
        abandoned = _records(
            {"id": ROOT, "history_base": {"thread_id": ROOT, "end_ordinal_exclusive": 4}},
            4,
            _turn("abandoned"),
        )
        stream = assemble_codex_session(
            [
                RolloutFile(f"rollout-2026-10-17T12-00-00-{ROOT}.jsonl", first_rollout()),
                RolloutFile(f"rollout-2026-10-18T12-00-00-{ROOT}_{abandoned_id}.jsonl", abandoned),
                RolloutFile(
                    f"rollout-2026-10-20T12-00-00-{ROOT}_{SEGMENT}.jsonl", segment_rollout()
                ),
            ]
        )
        by_file = {
            source: all(
                r
                for (_rec, s), r in zip(stream.records, stream.reverted, strict=True)
                if s == source
            )
            for _rec, source in stream.records
        }
        assert by_file[f"rollout-2026-10-18T12-00-00-{ROOT}_{abandoned_id}.jsonl"]

    def test_one_rollout_passes_through_unchanged(self) -> None:
        records = first_rollout()
        stream = assemble_codex_session([RolloutFile("r.jsonl", records)])
        assert [record for record, _source in stream.records] == records
        assert not any(stream.reverted)


class TestConvertSession:
    def test_the_session_converts_with_reverted_steps(self, tmp_path: Path) -> None:
        first, others = write_session(tmp_path / "sessions")
        result, report = convert_codex_and_audit(first, source_files=others)
        assert result.validation_errors == ()
        trajectory = result.trajectory
        assert trajectory["session_id"] == ROOT
        steps = trajectory["steps"]
        assert [s["step_id"] for s in steps] == list(range(1, len(steps) + 1))

        def texts(predicate: Callable[[dict[str, Any]], object]) -> list[str]:
            return [s.get("message") for s in steps if predicate(s.get("extra") or {})]

        assert texts(lambda e: e.get("reverted")) == ["question 2x", "answer 2x"]
        assert texts(lambda e: not e.get("reverted")) == [
            "question 1",
            "answer 1",
            "question 2",
            "answer 2",
        ]
        assert report.subagent_files_found == 0

        edges = [json.loads(line) for line in result.edges_lines]
        assert len({edge["uuid"] for edge in edges}) == len(edges)
        assert {edge["source_file"] for edge in edges} == {
            f"17/{first.name}",
            f"20/{others[0].name}",
        }

    def test_archive_names_are_relative_to_the_common_base(self, tmp_path: Path) -> None:
        from atif_converter.infrastructure.source_archive import SourceArchiveWriter

        first, others = write_session(tmp_path / "sessions")
        archive = SourceArchiveWriter(tmp_path / "archive", base=first.parents[1])
        convert_codex_and_audit(first, source_files=others, archive=archive)
        assert {f.relative_path for f in archive.files} == {
            f"17/{first.name}",
            f"20/{others[0].name}",
        }
