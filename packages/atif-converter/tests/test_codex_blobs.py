# SPDX-License-Identifier: Apache-2.0

"""Every Codex tool output image the converter names by an image part is a stored blob.

The converter's harbor 0.24.0 port turns an inline image in a tool output list
into an ATIF image part whose ``source.path`` is
``../../blobs/sha256/<ab>/<sha256>.<ext>``, and the pre-pass
(:func:`~atif_converter.domain.blobs.extract_codex_blobs`) is what hands those
bytes to the corpus blob store. Both read the image by
:func:`~atif_converter.domain.blobs.codex_tool_image_ref`, so these tests hold
the two to one rule: whatever part the port would write, the collector holds
the bytes it names, the index maps them to the output's ``call_id``, and
whatever the port would call ``[image omitted]`` stays in the record untouched.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pytest
from codex_fixtures import codex_rollout_records, write_codex_rollout
from subagent_fixtures import PASTED_PNG, READ_PNG, blob_part

from atif_converter.application.convert_codex import convert_codex_and_audit
from atif_converter.domain.blobs import BlobCollector, BlobIndex, extract_codex_blobs
from atif_converter.domain.codex_conversion import convert_codex_records

_TS = "2026-09-11T17:27:02.{:03d}Z"

_READ_B64 = base64.b64encode(READ_PNG).decode()
_PASTED_B64 = base64.b64encode(PASTED_PNG).decode()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _item(payload: dict[str, Any], n: int) -> dict[str, Any]:
    return {"timestamp": _TS.format(n), "type": "response_item", "payload": payload}


def _output(kind: str, call_id: str, output: Any) -> dict[str, Any]:
    return _item({"type": kind, "call_id": call_id, "output": output}, 2)


def _mcp(data: str, media_type: str | None = "image/png", key: str = "mimeType") -> dict[str, Any]:
    block: dict[str, Any] = {"type": "image", "data": data}
    if media_type is not None:
        block[key] = media_type
    return block


def _extract(records: list[dict[str, Any]]) -> tuple[BlobCollector, BlobIndex]:
    collector = BlobCollector()
    return collector, extract_codex_blobs(records, collector)


class TestMcpImageInAToolOutput:
    @pytest.mark.parametrize("kind", ["function_call_output", "custom_tool_call_output"])
    def test_is_lifted_stored_and_indexed_by_call_id(self, kind: str) -> None:
        records = [_output(kind, "c1", [{"type": "text", "text": "shot:"}, _mcp(_READ_B64)])]
        collector, index = _extract(records)
        output = records[0]["payload"]["output"]
        assert output == [
            {"type": "text", "text": "shot:"},
            blob_part(READ_PNG),
        ]
        assert [(blob.ref.sha256, blob.data) for blob in collector.blobs] == [
            (_sha(READ_PNG), READ_PNG)
        ]
        ((ref,),) = index.by_tool_call_id.values()
        assert list(index.by_tool_call_id) == ["c1"]
        assert (ref.sha256, ref.media_type, ref.width, ref.height) == (
            _sha(READ_PNG),
            "image/png",
            3,
            2,
        )
        assert index.by_record_key == {}

    @pytest.mark.parametrize(
        ("block", "media_type"),
        [
            (_mcp(_READ_B64, "image/png", key="mime_type"), "image/png"),
            (_mcp(_READ_B64, " IMAGE/JPG "), "image/jpeg"),
            (_mcp(f"data:image/webp;base64,{_READ_B64}", None), "image/webp"),
            (_mcp(f"data:image/gif;name=x.gif;base64,{_READ_B64}", "image/bmp"), "image/gif"),
            (_mcp(_READ_B64[:20] + "\n" + _READ_B64[20:]), "image/png"),
        ],
        ids=["mime_type-key", "jpg-alias", "data-url", "data-url-params-win", "wrapped-base64"],
    )
    def test_reads_the_media_type_and_payload_the_way_the_port_does(
        self, block: dict[str, Any], media_type: str
    ) -> None:
        records = [_output("function_call_output", "c1", [block])]
        collector, index = _extract(records)
        assert records[0]["payload"]["output"] == [blob_part(READ_PNG, media_type)]
        assert [blob.ref.media_type for blob in collector.blobs] == [media_type]
        assert [ref.sha256 for ref in index.by_tool_call_id["c1"]] == [_sha(READ_PNG)]

    @pytest.mark.parametrize(
        "block",
        [
            _mcp(_READ_B64, "image/bmp"),
            _mcp(_READ_B64, "text/plain"),
            _mcp("not base64!"),
            _mcp(""),
            _mcp(_READ_B64, None),
            _mcp("https://example.com/shot.png"),
            {"type": "image", "data": 7, "mimeType": "image/png"},
            {"type": "image", "mimeType": "image/png"},
        ],
        ids=[
            "unsupported-bmp",
            "not-an-image",
            "bad-base64",
            "empty",
            "no-media-type",
            "remote-url",
            "data-not-a-string",
            "no-data",
        ],
    )
    def test_an_image_the_port_omits_stays_unlifted(self, block: dict[str, Any]) -> None:
        records = [_output("function_call_output", "c1", [{"type": "text", "text": "a"}, block])]
        before = copy.deepcopy(records)
        collector, index = _extract(records)
        assert records == before
        assert collector.blobs == ()
        assert collector.references == 0
        assert index.by_tool_call_id == {}

    def test_an_image_block_in_message_content_stays(self) -> None:
        """Message content is not a tool output: the port reads only its ``text`` fields."""
        records = [
            _item(
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "q"}, _mcp(_READ_B64)],
                },
                1,
            )
        ]
        before = copy.deepcopy(records)
        collector, index = _extract(records)
        assert records == before
        assert collector.blobs == ()
        assert index.by_record_key == {}


class TestInputImageInAToolOutput:
    @pytest.mark.parametrize(
        "url",
        [
            f"data:image/png;charset=binary;base64,{_READ_B64}",
            f"data:image/png;base64,{_READ_B64[:20]}\n{_READ_B64[20:]}",
        ],
        ids=["data-url-params", "wrapped-base64"],
    )
    def test_the_ports_wider_data_urls_are_lifted_too(self, url: str) -> None:
        records = [
            _output("function_call_output", "c1", [{"type": "input_image", "image_url": url}])
        ]
        collector, index = _extract(records)
        assert records[0]["payload"]["output"] == [blob_part(READ_PNG)]
        assert [blob.data for blob in collector.blobs] == [READ_PNG]
        assert [ref.sha256 for ref in index.by_tool_call_id["c1"]] == [_sha(READ_PNG)]

    def test_in_message_content_only_the_strict_data_url_is_lifted(self) -> None:
        url = f"data:image/png;charset=binary;base64,{_READ_B64}"
        records = [
            _item(
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_image", "image_url": url}],
                },
                1,
            )
        ]
        before = copy.deepcopy(records)
        collector, _ = _extract(records)
        assert records == before
        assert collector.blobs == ()


def _tool_call_records(output: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = codex_rollout_records()
    extra = [
        _item(
            {
                "type": "function_call",
                "call_id": "call_mcp",
                "name": "mcp__browser__screenshot",
                "arguments": "{}",
            },
            310,
        ),
        _item({"type": "function_call_output", "call_id": "call_mcp", "output": output}, 320),
    ]
    return records[:10] + extra + records[10:]


def _result(trajectory: dict[str, Any], call_id: str) -> dict[str, Any]:
    for step in trajectory["steps"]:
        for result in (step.get("observation") or {}).get("results") or []:
            if result.get("source_call_id") == call_id:
                return result
    raise AssertionError(call_id)


_MIXED_OUTPUT: list[dict[str, Any]] = [
    {"type": "text", "text": "two shots:"},
    _mcp(_READ_B64),
    _mcp(_PASTED_B64, "image/jpg"),
    _mcp(_READ_B64, "image/bmp"),
]


class TestEndToEnd:
    def test_every_image_part_names_a_stored_blob(self, tmp_path: Path) -> None:
        """The production use case: each image part's path names a blob the result carries."""
        rollout = write_codex_rollout(tmp_path / "sessions", _tool_call_records(_MIXED_OUTPUT))
        result, _ = convert_codex_and_audit(rollout)
        content = _result(result.trajectory, "call_mcp")["content"]
        assert content == [
            {"type": "text", "text": "two shots:"},
            blob_part(READ_PNG),
            blob_part(PASTED_PNG, "image/jpeg"),
            {"type": "text", "text": "[image omitted]"},
        ]
        stored = {blob.ref.sha256: blob for blob in result.blobs}
        named = [
            re.findall(r"[0-9a-f]{64}", part["source"]["path"])[0]
            for part in content
            if part["type"] == "image"
        ]
        assert named == [_sha(READ_PNG), _sha(PASTED_PNG)]
        for sha in named:
            assert hashlib.sha256(stored[sha].data).hexdigest() == sha
        assert stored[_sha(PASTED_PNG)].data == PASTED_PNG
        images = _result(result.trajectory, "call_mcp")["extra"]["images"]
        assert [(i["sha256"], i["media_type"], i["width"], i["height"]) for i in images] == [
            (_sha(READ_PNG), "image/png", 3, 2),
            (_sha(PASTED_PNG), "image/jpeg", 5, 4),
        ]
        text = json.dumps(result.trajectory)
        assert _PASTED_B64 not in text
        assert result.validation_errors == ()

    def test_the_pre_pass_changes_no_converted_byte(self) -> None:
        """Raw and pre-passed records convert to the same trajectory: image parts are identical."""
        records = _tool_call_records(copy.deepcopy(_MIXED_OUTPUT))
        raw = convert_codex_records(copy.deepcopy(records), fallback_session_id="s")
        lifted = copy.deepcopy(records)
        _, index = _extract(lifted)
        assert list(index.by_tool_call_id) == ["call_mcp"]
        pre_passed = convert_codex_records(lifted, fallback_session_id="s")
        assert raw is not None
        assert pre_passed is not None
        assert raw.to_json_dict() == pre_passed.to_json_dict()

    def test_a_list_the_port_dumps_as_json_keeps_no_base64(self, tmp_path: Path) -> None:
        """One unknown block makes the port dump the list: the dump holds the image part."""
        output = [_mcp(_PASTED_B64), {"type": "mystery"}]
        rollout = write_codex_rollout(tmp_path / "sessions", _tool_call_records(output))
        result, _ = convert_codex_and_audit(rollout)
        content = _result(result.trajectory, "call_mcp")["content"]
        assert json.loads(content) == [
            blob_part(PASTED_PNG),
            {"type": "mystery"},
        ]
        assert [blob.data for blob in result.blobs] == [PASTED_PNG]


#: Two audio payloads; the blob store neither parses nor checks the bytes.
_WAV = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00"
_MP3 = b"ID3\x04\x00\x00\x00\x00\x00\x00\xff\xfb\x90\x00"
_WAV_B64 = base64.b64encode(_WAV).decode()
_MP3_B64 = base64.b64encode(_MP3).decode()


def _audio_url(data: str = _WAV_B64, media_type: str = "audio/wav") -> dict[str, Any]:
    return {"type": "input_audio", "audio_url": f"data:{media_type};base64,{data}"}


def _openai_audio(data: str = _WAV_B64, audio_format: str = "wav") -> dict[str, Any]:
    return {"type": "input_audio", "input_audio": {"data": data, "format": audio_format}}


def _mcp_audio(
    data: str = _WAV_B64, media_type: str = "audio/wav", key: str = "mimeType"
) -> dict[str, Any]:
    return {"type": "audio", "data": data, key: media_type}


def _user_message(content: list[dict[str, Any]], n: int = 1) -> dict[str, Any]:
    return _item({"type": "message", "role": "user", "content": content}, n)


class TestAudioInMessageContent:
    def test_is_lifted_between_its_verbatim_wrapper_and_indexed_by_record(self) -> None:
        """A ``local_audio`` item: the ``<audio>`` wrapper text stays, the data URL goes."""
        opening = {"type": "input_text", "text": '<audio name=[Audio #1] path="/tmp/a.wav">'}
        closing = {"type": "input_text", "text": "</audio>"}
        records = [_user_message([opening, _audio_url(), closing])]
        collector = BlobCollector()
        index = extract_codex_blobs(records, collector, record_keys=["u1"])
        assert records[0]["payload"]["content"] == [
            opening,
            blob_part(_WAV, "audio/wav"),
            closing,
        ]
        (blob,) = collector.blobs
        assert blob.data == _WAV
        assert (blob.ref.extension, blob.ref.is_audio, blob.ref.is_image) == ("wav", True, False)
        assert (blob.ref.width, blob.ref.height) == (None, None)
        assert index.by_record_key == {"u1": [blob.ref]}
        assert index.by_tool_call_id == {}

    def test_the_openai_form_is_lifted_too(self) -> None:
        records = [_user_message([_openai_audio(_MP3_B64, "mp3")])]
        collector, _ = _extract(records)
        assert records[0]["payload"]["content"] == [blob_part(_MP3, "audio/mpeg")]
        assert [blob.ref.extension for blob in collector.blobs] == ["mp3"]

    @pytest.mark.parametrize(
        "block",
        [
            _mcp_audio(),
            {"type": "input_audio", "file_id": "file-1"},
            {"type": "input_audio", "audio_url": "https://example.com/a.wav"},
            _audio_url("not base64!"),
        ],
        ids=["mcp-audio", "file-id", "remote-url", "bad-base64"],
    )
    def test_audio_with_no_liftable_bytes_stays(self, block: dict[str, Any]) -> None:
        records = [_user_message([{"type": "input_text", "text": "q"}, block])]
        before = copy.deepcopy(records)
        collector, index = _extract(records)
        assert records == before
        assert collector.blobs == ()
        assert index.by_record_key == {}


class TestAudioInAToolOutput:
    @pytest.mark.parametrize("kind", ["function_call_output", "custom_tool_call_output"])
    @pytest.mark.parametrize(
        "block",
        [_audio_url(), _openai_audio(), _mcp_audio(), _mcp_audio(key="mime_type")],
        ids=["audio_url", "openai", "mcp-mimeType", "mcp-mime_type"],
    )
    def test_is_lifted_stored_and_indexed_by_call_id(
        self, kind: str, block: dict[str, Any]
    ) -> None:
        records = [_output(kind, "c1", [{"type": "input_text", "text": "heard:"}, block])]
        collector, index = _extract(records)
        assert records[0]["payload"]["output"] == [
            {"type": "input_text", "text": "heard:"},
            blob_part(_WAV, "audio/wav"),
        ]
        assert [(blob.ref.sha256, blob.data) for blob in collector.blobs] == [(_sha(_WAV), _WAV)]
        assert [ref.sha256 for ref in index.by_tool_call_id["c1"]] == [_sha(_WAV)]

    @pytest.mark.parametrize(
        ("block", "media_type", "extension"),
        [
            (_audio_url(media_type="audio/x-wav"), "audio/wav", "wav"),
            (_audio_url(media_type="audio/wave"), "audio/wav", "wav"),
            (_audio_url(media_type="audio/vnd.wave"), "audio/wav", "wav"),
            (_audio_url(media_type=" Audio/MP3 "), "audio/mpeg", "mp3"),
            (_audio_url(media_type="audio/mpga"), "audio/mpeg", "mp3"),
            (_mcp_audio(media_type="audio/x-mpeg"), "audio/mpeg", "mp3"),
            (_mcp_audio(media_type="audio/m4a"), "audio/mp4", "m4a"),
            (_mcp_audio(media_type="audio/x-m4a"), "audio/mp4", "m4a"),
            (_mcp_audio(media_type="audio/x-aac"), "audio/aac", "aac"),
            (_mcp_audio(media_type="audio/ogg"), "audio/ogg", "ogg"),
            (_mcp_audio(media_type="audio/x-flac"), "audio/flac", "flac"),
            (_mcp_audio(media_type="audio/webm"), "audio/webm", "webm"),
            (_mcp_audio(media_type="audio/x-aiff"), "audio/aiff", "aiff"),
            (_openai_audio(audio_format="mp3"), "audio/mpeg", "mp3"),
            (_openai_audio(audio_format="m4a"), "audio/mp4", "m4a"),
            (_openai_audio(f"data:audio/flac;base64,{_WAV_B64}", "wav"), "audio/flac", "flac"),
            (
                _audio_url(f"{_WAV_B64[:8]}\n{_WAV_B64[8:]}", "audio/wav;rate=16000"),
                "audio/wav",
                "wav",
            ),
            (_mcp_audio(f"data:audio/ogg;base64,{_WAV_B64}", "audio/wav"), "audio/ogg", "ogg"),
        ],
        ids=[
            "x-wav",
            "wave",
            "vnd.wave",
            "mp3-alias-cased",
            "mpga",
            "x-mpeg",
            "m4a",
            "x-m4a",
            "x-aac",
            "ogg",
            "x-flac",
            "webm",
            "x-aiff",
            "format-mp3",
            "format-m4a",
            "data-url-in-openai-data-wins",
            "data-url-params-and-wrapped-base64",
            "data-url-in-mcp-data-wins",
        ],
    )
    def test_media_types_normalize_through_atif_audio_types(
        self, block: dict[str, Any], media_type: str, extension: str
    ) -> None:
        records = [_output("function_call_output", "c1", [block])]
        collector, _ = _extract(records)
        assert records[0]["payload"]["output"] == [blob_part(_WAV, media_type)]
        assert [(blob.ref.media_type, blob.ref.extension) for blob in collector.blobs] == [
            (media_type, extension)
        ]

    @pytest.mark.parametrize(
        "block",
        [
            _audio_url("not base64!"),
            _audio_url(""),
            _mcp_audio("@@@@"),
            _audio_url(media_type="audio/basic"),
            _mcp_audio(media_type="text/plain"),
            _openai_audio(audio_format="pcm16"),
            {"type": "input_audio", "input_audio": {"data": _WAV_B64}},
            {"type": "audio", "data": _WAV_B64},
            {"type": "audio", "data": 7, "mimeType": "audio/wav"},
            {"type": "input_audio", "file_id": "file-1"},
            {"type": "input_audio", "input_audio": {"file_id": "file-2"}},
            {"type": "input_audio", "audio_url": "https://example.com/a.wav"},
        ],
        ids=[
            "bad-base64",
            "empty",
            "mcp-bad-base64",
            "unsupported-type",
            "not-audio",
            "unsupported-format",
            "no-format",
            "no-media-type",
            "data-not-a-string",
            "file-id",
            "nested-file-id",
            "remote-url",
        ],
    )
    def test_audio_with_no_liftable_bytes_stays(self, block: dict[str, Any]) -> None:
        records = [_output("function_call_output", "c1", [{"type": "text", "text": "a"}, block])]
        before = copy.deepcopy(records)
        collector, index = _extract(records)
        assert records == before
        assert collector.blobs == ()
        assert collector.references == 0
        assert index.by_tool_call_id == {}

    def test_a_nested_mcp_results_audio_is_indexed_by_the_item_id(self) -> None:
        records: list[dict[str, Any]] = [
            {
                "timestamp": _TS.format(3),
                "type": "event_msg",
                "payload": {
                    "type": "item_completed",
                    "item": {
                        "type": "McpToolCall",
                        "id": "exec-m1",
                        "result": {"content": [_mcp_audio(_MP3_B64, "audio/mpeg")]},
                    },
                },
            }
        ]
        collector, index = _extract(records)
        content = records[0]["payload"]["item"]["result"]["content"]
        assert content == [blob_part(_MP3, "audio/mpeg")]
        assert [blob.data for blob in collector.blobs] == [_MP3]
        assert [ref.sha256 for ref in index.by_tool_call_id["exec-m1"]] == [_sha(_MP3)]


def _audio_records() -> list[dict[str, Any]]:
    """The fixture rollout with a spoken user turn and a tool that answers with audio."""
    records = _tool_call_records(
        [
            {"type": "input_text", "text": "recorded:"},
            _mcp_audio(_MP3_B64, "audio/mp3"),
            {"type": "input_audio", "file_id": "file-9"},
        ]
    )
    spoken = _user_message(
        [
            {"type": "input_text", "text": "transcribe "},
            _audio_url(),
            {"type": "input_audio", "audio_url": "https://example.com/b.wav"},
        ],
        5,
    )
    # Right after the fixture's own user message, so the spoken one is a step of its own.
    return [*records[:6], spoken, *records[6:]]


class TestAudioEndToEnd:
    def test_every_audio_part_names_a_stored_blob(self, tmp_path: Path) -> None:
        rollout = write_codex_rollout(tmp_path / "sessions", _audio_records())
        result, _ = convert_codex_and_audit(rollout)
        tool_result = _result(result.trajectory, "call_mcp")
        assert tool_result["content"] == [
            {"type": "text", "text": "recorded:"},
            blob_part(_MP3, "audio/mpeg"),
            {"type": "text", "text": "[audio: file-9]"},
        ]
        assert "images" not in tool_result.get("extra", {})
        messages = [step["message"] for step in result.trajectory["steps"]]
        spoken = [
            {"type": "text", "text": "transcribe "},
            blob_part(_WAV, "audio/wav"),
            {"type": "text", "text": "[audio: https://example.com/b.wav]"},
        ]
        assert spoken in messages
        spoken_step = result.trajectory["steps"][messages.index(spoken)]
        assert spoken_step["extra"]["source_uuids"]
        assert "images" not in spoken_step["extra"]
        stored = {blob.ref.sha256: blob for blob in result.blobs}
        assert {sha: (blob.data, blob.ref.extension) for sha, blob in stored.items()} == {
            _sha(_WAV): (_WAV, "wav"),
            _sha(_MP3): (_MP3, "mp3"),
        }
        text = json.dumps(result.trajectory)
        assert _WAV_B64 not in text
        assert _MP3_B64 not in text
        assert "enrichment_truncated_at_step" not in result.trajectory.get("extra", {})
        assert result.validation_errors == ()

    def test_the_pre_pass_changes_no_converted_byte(self) -> None:
        """Raw and pre-passed records convert alike: one rule names each audio item."""
        records = _audio_records()
        raw = convert_codex_records(copy.deepcopy(records), fallback_session_id="s")
        lifted = copy.deepcopy(records)
        _, index = _extract(lifted)
        assert list(index.by_tool_call_id) == ["call_mcp"]
        pre_passed = convert_codex_records(lifted, fallback_session_id="s")
        assert raw is not None
        assert pre_passed is not None
        assert raw.to_json_dict() == pre_passed.to_json_dict()
