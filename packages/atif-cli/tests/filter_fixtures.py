# SPDX-License-Identifier: Apache-2.0

"""A Claude Code session that touches every surface a ``materialize --filter`` script acts on.

The main chain, under ``/home/alice/work``, has a step with reasoning, a
``Bash`` call whose command and output carry a synthetic credential, a
``Read`` of a PNG (the result's image block and the ``toolUseResult.file``
copy), a pasted PNG beside text, a hook attachment and a ``stop_hook_summary``
record (two session events), and one ``Agent`` call whose subagent ``aaa``
reasons and runs a failing ``Bash`` of its own (an embedded trajectory).
"""

from __future__ import annotations

import base64
import json
import os
import struct
import time
import zlib
from pathlib import Path
from typing import Any

SESSION = "dddddddd-dddd-dddd-dddd-dddddddddddd"
HOME = "/home/alice"
CWD = f"{HOME}/work"
#: A synthetic credential, shaped like a provider key.
SECRET = "sk-test-0000filterfixture0000"
#: The subagent's id, which is also its embedded trajectory's ``trajectory_id``.
AGENT_ID = "aaa"


def png_bytes(width: int, height: int) -> bytes:
    """A real, minimal RGB PNG of ``width`` x ``height`` black pixels."""

    def chunk(kind: bytes, body: bytes) -> bytes:
        return (
            struct.pack(">I", len(body))
            + kind
            + body
            + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + b"\x00\x00\x00" * width for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


#: The PNG the main chain reads (3 x 2).
READ_PNG = png_bytes(3, 2)
#: The PNG the user pastes (5 x 4).
PASTED_PNG = png_bytes(5, 4)


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _base(uuid: str, ts: str, **extra: Any) -> dict[str, Any]:
    return {
        "uuid": uuid,
        "sessionId": SESSION,
        "timestamp": f"2026-09-27T00:00:{ts}Z",
        "version": "2.1.300",
        "cwd": CWD,
        "gitBranch": "main",
        **extra,
    }


def _assistant(
    uuid: str, ts: str, msg_id: str, content: list[dict[str, Any]], **extra: Any
) -> dict[str, Any]:
    return {
        **_base(uuid, ts, **extra),
        "type": "assistant",
        "message": {
            "id": msg_id,
            "role": "assistant",
            "model": "claude-test-1",
            "content": content,
            "usage": {"input_tokens": 3, "output_tokens": 2},
        },
    }


def _user(uuid: str, ts: str, content: Any, **extra: Any) -> dict[str, Any]:
    return {
        **_base(uuid, ts, **extra),
        "type": "user",
        "message": {"role": "user", "content": content},
    }


def _tool_use(call_id: str, name: str, **arguments: Any) -> dict[str, Any]:
    return {"type": "tool_use", "id": call_id, "name": name, "input": arguments}


def _image_block(data: bytes) -> dict[str, Any]:
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": _b64(data)},
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def write_filter_session(source_root: Path) -> Path:
    """Write the session under ``source_root``; return its main transcript, quiescent."""
    project = source_root / "-home-alice-work"
    main = project / f"{SESSION}.jsonl"
    side = project / SESSION / "subagents"
    _write_jsonl(
        main,
        [
            _user("u-1", "01", f"deploy from {CWD} with token {SECRET}"),
            _assistant(
                "a-1",
                "02",
                "m-1",
                [
                    {
                        "type": "thinking",
                        "thinking": "the token goes in the header",
                        "signature": "s",
                    },
                    _tool_use("toolu_bash", "Bash", command=f"curl -H 'Authorization: {SECRET}' x"),
                ],
            ),
            _user(
                "u-2",
                "03",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_bash",
                        "content": f"ok {CWD}/out.log",
                    }
                ],
                toolUseResult={"stdout": f"ok {CWD}/out.log", "stderr": "", "interrupted": False},
            ),
            {
                **_base("h-1", "04", parentUuid="u-2"),
                "type": "attachment",
                "attachment": {
                    "type": "hook_success",
                    "hookName": "PostToolUse:Bash",
                    "hookEvent": "PostToolUse",
                    "content": f"logged to {CWD}/hook.log",
                },
            },
            _assistant(
                "a-2", "05", "m-2", [_tool_use("toolu_read", "Read", file_path=f"{CWD}/s.png")]
            ),
            _user(
                "u-3",
                "06",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_read",
                        "content": [_image_block(READ_PNG)],
                    }
                ],
                toolUseResult={
                    "type": "image",
                    "file": {
                        "base64": _b64(READ_PNG),
                        "type": "image/png",
                        "originalSize": len(READ_PNG),
                    },
                },
            ),
            _user("u-4", "07", [{"type": "text", "text": "see this"}, _image_block(PASTED_PNG)]),
            _assistant(
                "a-3",
                "08",
                "m-3",
                [
                    _tool_use(
                        "toolu_agent",
                        "Agent",
                        subagent_type="Explore",
                        description="scout",
                        prompt="look",
                    )
                ],
            ),
            _user(
                "u-5",
                "20",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_agent",
                        "content": [{"type": "text", "text": "scouted"}],
                    }
                ],
                toolUseResult={"status": "completed", "agentId": AGENT_ID, "agentType": "Explore"},
            ),
            _assistant("a-4", "21", "m-4", [{"type": "text", "text": "done"}]),
            {
                **_base("s-1", "22", parentUuid="a-4"),
                "type": "system",
                "subtype": "stop_hook_summary",
                "hookCount": 1,
                "preventedContinuation": False,
            },
        ],
    )
    _write_jsonl(
        side / f"agent-{AGENT_ID}.jsonl",
        [
            _user("sa-u1", "09", "look", isSidechain=True, agentId=AGENT_ID),
            _assistant(
                "sa-a1",
                "10",
                "m-a1",
                [
                    {"type": "thinking", "thinking": "check the logs", "signature": "s"},
                    _tool_use("toolu_sub", "Bash", command="exit 2"),
                ],
                isSidechain=True,
                agentId=AGENT_ID,
            ),
            _user(
                "sa-u2",
                "11",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_sub",
                        "content": "Exit code 2\nboom",
                        "is_error": True,
                    }
                ],
                isSidechain=True,
                agentId=AGENT_ID,
                toolUseResult="Error: Exit code 2\nboom",
            ),
            _assistant(
                "sa-a2",
                "12",
                "m-a2",
                [{"type": "text", "text": "scouted"}],
                isSidechain=True,
                agentId=AGENT_ID,
            ),
        ],
    )
    (side / f"agent-{AGENT_ID}.meta.json").write_text(
        json.dumps({"agentType": "Explore", "description": "scout", "toolUseId": "toolu_agent"}),
        encoding="utf-8",
    )
    stale = time.time_ns() - 3_600 * 1_000_000_000
    for path in (main, *side.iterdir()):
        os.utime(path, ns=(stale, stale))
    return main
