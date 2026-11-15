# SPDX-License-Identifier: Apache-2.0
# Ported from harbor 0.26.0, src/harbor/agents/installed/claude_code.py
# (Apache-2.0, Copyright the Harbor authors), onto the ATIF models vendored in
# atif_converter.domain.atif, so atif-converter doesn't depend on harbor.

"""Claude Code session records -> ATIF trajectory, as a pure function.

This is ``ClaudeCode._convert_events_to_trajectory`` and its helpers, lifted
out of harbor's agent class so the conversion runs over ALREADY-PARSED records
with no file I/O and no harbor import. The ATIF data classes (harbor's
``harbor.models.trajectories``, vendored as :mod:`atif_converter.domain.atif`)
are the contract the output is expressed in, which is why this domain module
imports them. harbor's structure and names
are kept recognizable on purpose, so a future upstream diff can be re-applied
by hand; the one structural change is that the body of
``_convert_events_to_trajectory`` is split into helpers around the
:class:`_NormalizationState` it shared, because the repo's function-size
ratchets are set at today's worst function and this one was larger.

INPUT SHAPE. harbor takes a session DIRECTORY and reads
``sorted(glob("*.jsonl")) + sorted(rglob("subagents/*.jsonl"))``, one main
transcript followed by every staged side-file, in file-name order. Nothing in
the conversion uses a file name after that: the side-file names decide only
the READ ORDER, which matters because uuid de-duplication keeps the first
occurrence and the timestamp sort is stable. :func:`convert_claude_code_records`
therefore takes the main transcript's records in file order plus a mapping of
staged side-file NAME (``agent-abc.jsonl``,
``workflows__wf_123__agent-def.jsonl`` for a workflow-nested file, exactly as
the old adapter staged them flat) to that file's records, and reads the
side-files in sorted-name order.

SUBAGENT TRAJECTORIES. harbor 0.26.0 converts the main transcript and every
side-file as one stream. This module carries harbor #4076's nesting on top of
that port (:func:`split_claude_code_records`,
:func:`nest_claude_code_subagents`): each side-file agent's own transcript
converts on its own and embeds under the call that spawned it. #4076 names a
child by its side-file's path below the session directory; this module names
it by the native agent id, the key ``toolUseResult.agentId``, the
``agent-<id>.meta.json`` sidecar and the steps' ``extra.agent_id`` already
share, so a trajectory's identity does not depend on where its file sits.

OMITTED: ``_parse_total_cost_from_stream_json``. It reads
``<logs_dir>/claude-code.txt``, the stdout tee of a harbor-driven run. Our
adapter always handed harbor an EMPTY temp ``logs_dir``, so ``read_text``
raised ``OSError`` and the method returned ``None`` on every call, which sent
every trajectory down the ``litellm`` estimate path. That fallback ordering is
kept verbatim: ``total_cost_usd`` is the litellm estimate or ``None``, and
``final_metrics.extra["cost_source"] == "litellm_estimate"`` whenever the
estimate priced the session. Since harbor 0.23.0 every priced step also carries
its own ``metrics.cost_usd`` and ``metrics.extra["cost_source"]``. The estimate itself now comes from
:mod:`atif_converter.domain.pricing`, which repeats ``litellm.cost_per_token``'s
arithmetic over a vendored copy of litellm's price data without litellm being
installed; the floats are identical, so the label stays. Also not ported: ``_session_dirs`` (log-dir
discovery, replaced by the infrastructure reader), and ``_session_text`` /
``_session_tool_result_content`` (they serve ``atif_to_native_trajectory``, the
REVERSE conversion, not this one).

``self.logger`` is loguru's ``logger``; ``self.model_name`` is the
``model_name`` parameter (our adapter never set one, so the default ``None``
reproduces the behavior we had).
"""

from __future__ import annotations

import base64
import contextlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from atif_converter.domain import pricing
from atif_converter.domain.agents import AgentSource
from atif_converter.domain.atif import (
    Agent,
    ContentPart,
    FinalMetrics,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    SubagentTrajectoryRef,
    ToolCall,
    Trajectory,
)
from atif_converter.domain.blobs import lifted_part

#: The schema version harbor 0.23.0 stamps on a Claude Code trajectory.
SCHEMA_VERSION = "ATIF-v1.7"


def _convert_event_to_step(event: dict[str, Any], step_id: int, model_name: str | None) -> Step:
    """Convert a normalized Claude Code event dictionary into an ATIF step."""
    kind = event.get("kind")
    timestamp = event.get("timestamp")

    if kind == "message":
        role = event.get("role", "user")
        text = event.get("text", "")
        reasoning = event.get("reasoning")
        metrics = event.get("metrics")
        extra = event.get("extra")
        event_model_name = event.get("model_name")

        if role == "assistant":
            source = "agent"
        elif role == "user":
            source = "user"
        else:
            source = "system"

        step = Step(
            step_id=step_id,
            timestamp=timestamp,
            source=source,
            message=text,
            llm_call_count=1 if source == "agent" else None,
        )

        if source == "agent":
            if reasoning:
                step.reasoning_content = reasoning
            if event_model_name:
                step.model_name = event_model_name
            elif model_name:
                step.model_name = model_name

        if metrics:
            step.metrics = metrics
        if extra:
            step.extra = extra

        return step

    if kind == "agent_step":
        text = event.get("text") or ""
        reasoning = event.get("reasoning")
        metrics = event.get("metrics")
        extra = event.get("extra")
        step_model_name = event.get("model_name") or model_name
        tool_specs = event.get("tool_calls") or []

        tool_calls: list[ToolCall] = []
        results: list[ObservationResult] = []
        for spec in tool_specs:
            spec_call_id = spec.get("call_id")
            if not spec_call_id:
                continue
            tool_calls.append(
                ToolCall(
                    tool_call_id=spec_call_id,
                    function_name=spec.get("tool_name") or "",
                    arguments=spec.get("arguments") or {},
                    extra=spec.get("extra"),
                )
            )
            if spec.get("output") is not None:
                results.append(
                    ObservationResult(
                        source_call_id=spec_call_id,
                        content=spec.get("output"),
                        subagent_trajectory_ref=None,
                        extra=spec.get("result_extra"),
                    )
                )

        step = Step(
            step_id=step_id,
            timestamp=timestamp,
            source="agent",
            message=text,
            tool_calls=tool_calls or None,
            observation=Observation(results=results) if results else None,
            llm_call_count=1,
        )
        if reasoning:
            step.reasoning_content = reasoning
        if step_model_name:
            step.model_name = step_model_name
        if metrics:
            step.metrics = metrics
        if extra:
            step.extra = extra

        return step

    if kind == "tool_call":
        call_id = event.get("call_id")
        tool_name = event.get("tool_name")
        if not call_id or not tool_name:
            msg = "Tool call event missing call_id or tool_name"
            raise ValueError(msg)

        arguments = event.get("arguments") or {}
        raw_arguments = event.get("raw_arguments")
        reasoning = event.get("reasoning")
        metrics = event.get("metrics")
        extra = event.get("extra")
        status = event.get("status")
        message = event.get("message")
        output = event.get("output")
        metadata = event.get("metadata")
        step_model_name = event.get("model_name") or model_name

        tool_call = ToolCall(
            tool_call_id=call_id,
            function_name=tool_name,
            arguments=arguments,
        )

        observation_result = ObservationResult(
            source_call_id=call_id,
            content=output,
            subagent_trajectory_ref=None,
        )

        observation = Observation(results=[observation_result]) if output is not None else None

        extra = extra or {}
        updates = {
            "metadata": metadata,
            "raw_arguments": raw_arguments,
            "status": status,
        }
        for key, value in updates.items():
            if value is not None:
                extra.setdefault(key, value)

        step = Step(
            step_id=step_id,
            timestamp=timestamp,
            source="agent",
            message=message or "",
            tool_calls=[tool_call],
            observation=observation,
            llm_call_count=1,
        )

        if step_model_name:
            step.model_name = step_model_name
        if reasoning:
            step.reasoning_content = reasoning
        if metrics:
            step.metrics = metrics
        if extra:
            step.extra = extra

        return step

    msg = f"Unsupported event kind '{kind}'"
    raise ValueError(msg)


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        dumped = json.dumps(value, ensure_ascii=False)
    except TypeError:
        return str(value)
    return dumped


def _extract_text_reasoning_tool_uses(
    content: Any,
) -> tuple[str, str | None, list[dict[str, Any]]]:
    if isinstance(content, str):
        text = content.strip()
        return text, None, []

    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_blocks: list[dict[str, Any]] = []

    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                text_parts.append(_stringify(block))
                continue

            block_type = block.get("type")
            if block_type == "tool_use":
                tool_blocks.append(block)
                continue

            if block_type in {"thinking", "reasoning", "analysis"}:
                text_value = (
                    block.get("text") if block.get("text") is not None else block.get("thinking")
                )
                if isinstance(text_value, str):
                    reasoning_parts.append(text_value.strip())
                else:
                    reasoning_parts.append(_stringify(text_value))
                continue

            if block_type == "redacted_thinking":
                # Anthropic redacted_thinking blocks carry encrypted reasoning
                # in `data` that clients cannot decrypt. OpenRouter mis-uses
                # this envelope to pass through PLAIN reasoning from
                # non-Anthropic models: `data` is `openrouter.reasoning:<b64>`
                # whose base64 decodes to JSON with a `text` field. Surface
                # that text as reasoning; drop genuine ciphertext rather than
                # dumping the raw envelope into the human-readable message.
                data = block.get("data")
                if isinstance(data, str) and data.startswith("openrouter.reasoning:"):
                    with contextlib.suppress(ValueError, json.JSONDecodeError):
                        payload = data[len("openrouter.reasoning:") :]
                        decoded = base64.b64decode(payload + "==").decode("utf-8", "replace")
                        inner = json.loads(decoded)
                        inner_text = inner.get("text")
                        if isinstance(inner_text, str):
                            reasoning_parts.append(inner_text.strip())
                continue

            if block_type == "code" and isinstance(block.get("code"), str):
                text_parts.append(block["code"])
                continue

            text_value = block.get("text")
            if isinstance(text_value, str):
                text_parts.append(text_value)
            else:
                text_parts.append(_stringify(block))
    elif content is not None:
        text_parts.append(_stringify(content))

    text = "\n\n".join(part.strip() for part in text_parts if part and str(part).strip())
    reasoning = "\n\n".join(part.strip() for part in reasoning_parts if part and str(part).strip())

    return text, (reasoning or None), tool_blocks


def _build_metrics(usage: Any) -> Metrics | None:
    if not isinstance(usage, dict):
        return None

    # ``or 0`` rather than a ``get`` default: an interrupted streaming response
    # can leave a usage field present but ``None``, and ``get(key, 0)`` only
    # falls back when the key is absent. Without this guard the arithmetic
    # below raises ``TypeError`` and aborts the whole conversion.
    cached_tokens = usage.get("cache_read_input_tokens") or 0
    creation = usage.get("cache_creation_input_tokens") or 0
    input_tokens = usage.get("input_tokens") or 0
    # Align with Anthropic session totals: input + cache read + cache creation.
    prompt_tokens = input_tokens + cached_tokens + creation
    completion_tokens = usage.get("output_tokens") or 0

    extra: dict[str, Any] = {}
    for key, value in usage.items():
        if key in {"input_tokens", "output_tokens"}:
            continue
        extra[key] = value

    # harbor then tests ``prompt_tokens is None and completion_tokens is None
    # and cached_tokens is None and not extra`` and returns None; every one of
    # those is an int after the ``or 0`` above, so the branch is unreachable
    # and is left out here (ty rejects an always-false condition).

    return Metrics(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_tokens=cached_tokens,
        cost_usd=None,
        extra=extra or None,
    )


def _is_text_block(block: Any) -> bool:
    """A ``{"type": "text", "text": <str>}`` block, the one harbor reads verbatim."""
    if not isinstance(block, dict):
        return False
    block_type: object = block.get("type")
    return block_type == "text" and isinstance(block.get("text"), str)


def _tool_result_parts(content: list[Any]) -> list[ContentPart] | None:
    """A block-list tool result as ATIF parts, when it holds an image or audio part.

    Not harbor's: harbor writes every item's JSON text. An item the pre-pass
    lifted (:func:`~atif_converter.domain.blobs.lifted_part`) is its part, a
    ``text`` item its text, and any other item its JSON text; whitespace-only
    texts are left out, as harbor leaves out an empty item. ``None`` when no
    item is an image or audio part, and the result stays harbor's text.
    """
    parts: list[ContentPart] = []
    has_media = False
    for item in content:
        media = lifted_part(item)
        if media is not None:
            parts.append(media)
            has_media = True
            continue
        text_value = item["text"] if _is_text_block(item) else _stringify(item).strip()
        if text_value.strip():
            parts.append(ContentPart(type="text", text=text_value))
    return parts if has_media else None


def _format_tool_result(
    block: dict[str, Any], tool_use_result: Any
) -> tuple[str | list[ContentPart] | None, dict[str, Any] | None]:
    """A ``tool_result`` block's observation content and harbor's metadata for it.

    harbor's ``_format_tool_result``, with two departures. A string result is
    kept verbatim where harbor strips it (it strips the joined text too, so
    the strip is skipped there as well; every other part has no outer
    whitespace to lose). A block-list result holding an image or audio part
    is a list of ATIF parts (:func:`_tool_result_parts`), harbor's suffix
    chunks and error line each one more text part.
    """
    parts: list[str] = []

    content = block.get("content")
    content_parts = _tool_result_parts(content) if isinstance(content, list) else None
    if isinstance(content, str):
        if content.strip():
            parts.append(content)
    elif isinstance(content, list):
        for item in content:
            text_value = _stringify(item)
            if text_value.strip():
                parts.append(text_value.strip())
    elif content not in (None, ""):
        parts.append(_stringify(content))

    metadata: dict[str, Any] | None = None
    if tool_use_result and isinstance(tool_use_result, dict):
        metadata = {"tool_use_result": tool_use_result}
        stdout = tool_use_result.get("stdout")
        stderr = tool_use_result.get("stderr")
        exit_code = tool_use_result.get("exitCode") or tool_use_result.get("exit_code")
        interrupted = tool_use_result.get("interrupted")
        is_image = tool_use_result.get("isImage")

        formatted_chunks: list[str] = []
        if stdout:
            formatted_chunks.append(f"[stdout]\n{stdout}".rstrip())
        if stderr:
            formatted_chunks.append(f"[stderr]\n{stderr}".rstrip())
        if exit_code not in (None, 0):
            formatted_chunks.append(f"[exit_code] {exit_code}")
        if interrupted:
            formatted_chunks.append(f"[interrupted] {interrupted}")
        if is_image:
            formatted_chunks.append(f"[is_image] {is_image}")

        remaining_meta = {
            key: value
            for key, value in tool_use_result.items()
            if key
            not in {
                "stdout",
                "stderr",
                "exitCode",
                "exit_code",
                "interrupted",
                "isImage",
            }
        }
        if remaining_meta:
            formatted_chunks.append(f"[metadata] {json.dumps(remaining_meta, ensure_ascii=False)}")

        if formatted_chunks:
            suffix = "\n".join(chunk for chunk in formatted_chunks if chunk)
            parts.append(suffix)
            if content_parts is not None:
                content_parts.append(ContentPart(type="text", text=suffix))

    if block.get("is_error") is True:
        parts.append("[error] tool reported failure")
        if content_parts is not None:
            content_parts.append(ContentPart(type="text", text="[error] tool reported failure"))
        metadata = metadata or {}
        metadata["is_error"] = True

    if metadata is not None:
        metadata.setdefault("raw_tool_result", block)

    if content_parts is not None:
        return content_parts, metadata
    result_text = "\n\n".join(part for part in parts if part)
    return (result_text or None), metadata


def _estimate_step_costs(steps: list[Step]) -> float | None:
    """Price each step, writing ``metrics.cost_usd`` on it, and return the sum.

    harbor 0.23.0's ``_estimate_step_costs``: every step with token usage that
    can be priced gets ``metrics.cost_usd`` and ``metrics.extra["cost_source"]``,
    and the sum is the session estimate used when Claude omits its result event.
    The sum is ``None`` when any step with usage could not be priced, since a
    partial sum would understate the total. harbor priced each step through
    ``litellm.cost_per_token``; the call goes through
    :mod:`atif_converter.domain.pricing`, which returns the same floats from a
    vendored copy of litellm's price data. A pricing failure leaves that step
    unpriced, as a litellm error did in harbor.
    """
    total_cost = 0.0
    priced_any_step = False
    priced_all_steps = True
    for step in steps:
        metrics = step.metrics
        if metrics is None:
            continue

        prompt_tokens = metrics.prompt_tokens or 0
        completion_tokens = metrics.completion_tokens or 0
        if prompt_tokens <= 0 and completion_tokens <= 0:
            continue
        if not step.model_name:
            logger.debug("Cannot estimate Claude cost without a step model")
            priced_all_steps = False
            continue

        extra = metrics.extra or {}
        cache_creation_tokens = extra.get("cache_creation_input_tokens")
        if not isinstance(cache_creation_tokens, int):
            cache_creation_tokens = 0
        cache_read_tokens = extra.get("cache_read_input_tokens")
        if not isinstance(cache_read_tokens, int):
            cache_read_tokens = metrics.cached_tokens or 0
        service_tier = extra.get("service_tier")
        if not isinstance(service_tier, str):
            service_tier = None

        try:
            priced = pricing.priced_cost_per_token(
                model=step.model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cache_creation_input_tokens=cache_creation_tokens,
                cache_read_input_tokens=cache_read_tokens,
                service_tier=service_tier,
            )
        except Exception as exc:  # noqa: BLE001 — harbor swallowed every pricing error into "unpriced"
            logger.debug(
                "Cannot estimate Claude cost for model '{}': {}",
                step.model_name,
                exc,
            )
            priced_all_steps = False
            continue
        if priced is None:
            # DELIBERATE DIVERGENCE from harbor, which writes litellm's (0.0, 0.0)
            # for a model with no rates as a $0 step and sums it into a $0
            # session. The step stays unpriced (no ``cost_usd``, no label) and
            # the session estimate is unknown, as a litellm error makes it.
            logger.debug("No price for Claude model '{}'; step left unpriced", step.model_name)
            priced_all_steps = False
            continue

        prompt_cost, completion_cost = priced
        metrics.cost_usd = prompt_cost + completion_cost
        # harbor labels every priced step "litellm_estimate"; a step priced from a
        # local override says so, as the session label does.
        metrics.extra = {**extra, "cost_source": pricing.cost_source_label([step.model_name])}
        total_cost += metrics.cost_usd
        priced_any_step = True

    return total_cost if priced_any_step and priced_all_steps else None


def _first_event_model(events: list[dict[str, Any]], *, include_sidechain: bool) -> str | None:
    for event in events:
        if not include_sidechain and event.get("isSidechain"):
            continue
        message = event.get("message")
        if not isinstance(message, dict):
            continue
        model_name = message.get("model")
        if isinstance(model_name, str) and model_name:
            return model_name
    return None


@dataclass(slots=True)
class _NormalizationState:
    """The locals ``_convert_events_to_trajectory`` shared across its event loop.

    One object rather than seven positional arguments, so the per-event-type
    helpers below read like the branches they were cut from.
    """

    default_model_name: str | None
    #: Per message id, the last usage seen (streaming updates it on each chunk).
    last_usage_by_msg_id: dict[str, Any]
    normalized_events: list[dict[str, Any]] = field(default_factory=list)
    pending_calls: dict[str, dict[str, Any]] = field(default_factory=dict)
    completed_call_ids: set[str] = field(default_factory=set)
    seen_message_ids: set[str] = field(default_factory=set)
    #: Maps an assistant ``message.id`` to the single agent_step it bundles, so
    #: text / reasoning / every tool_use from one LLM inference land on one
    #: ATIF step even if the session log splits them across events.
    turn_by_msgid: dict[str, dict[str, Any]] = field(default_factory=dict)


def _collect_events(
    main_records: Sequence[dict[str, Any]],
    side_records: Mapping[str, Sequence[dict[str, Any]]] | None,
) -> list[dict[str, Any]]:
    """Concatenate in harbor's read order, de-duplicate by uuid, sort by timestamp."""
    raw_events: list[dict[str, Any]] = list(main_records)
    if side_records:
        for name in sorted(side_records):
            raw_events.extend(side_records[name])

    seen_event_uuids: set[str] = set()
    deduped_raw_events: list[dict[str, Any]] = []
    for event in raw_events:
        uuid = event.get("uuid")
        if isinstance(uuid, str) and uuid:
            if uuid in seen_event_uuids:
                logger.debug("Skipping duplicate Claude Code session event {}", uuid)
                continue
            seen_event_uuids.add(uuid)
        deduped_raw_events.append(event)

    # Keep events in chronological order across the main chain and any
    # subagent sidechains, so the first user step remains the instruction
    # (downstream byte-identity checks rely on this) and step timestamps
    # stay monotonic; sidechain steps carry `extra.is_sidechain` and
    # `extra.agent_id` so consumers can distinguish concurrent subagents.
    deduped_raw_events.sort(key=lambda e: e.get("timestamp", ""))
    return deduped_raw_events


def _first_seen(events: list[dict[str, Any]], key: str) -> list[str]:
    """Every distinct non-empty string ``event[key]``, in the order events carry it."""
    values = (event.get(key) for event in events)
    return list(dict.fromkeys(value for value in values if isinstance(value, str) and value))


def _agent_extra(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    """``agent.extra``: the session's working dirs, branches and subagent ids.

    harbor collects each into a ``set`` and dumps it as a list, so its order
    follows the process hash seed and two conversions of one session disagree.
    Here each list keeps FIRST-SEEN order over ``events``, which are already in
    timestamp order (:func:`_collect_events`): the first ``cwds`` entry is where
    the session started, later ones are where it moved, and the same holds for
    branch switches and for the order subagents first spoke. The parity oracle
    forgives the reordering only when the two lists hold the same values.
    """
    cwds = _first_seen(events, "cwd")
    git_branches = _first_seen(events, "gitBranch")
    agent_ids = _first_seen(events, "agentId")

    agent_extra: dict[str, Any] = {}
    if cwds:
        agent_extra["cwds"] = cwds
    if git_branches:
        agent_extra["git_branches"] = git_branches
    if agent_ids:
        agent_extra["agent_ids"] = agent_ids
    return agent_extra or None


def _event_identity_extra(event: dict[str, Any]) -> dict[str, Any]:
    """Keep each Claude session event's agent identity on its ATIF step.

    harbor 0.24.0's ``ClaudeCode._event_identity_extra`` (harbor #3434): every
    step a session event becomes, the agent turn, a user message and a tool
    result alike, carries ``is_sidechain`` and, when the event names one, the
    ``agentId`` of the subagent that wrote it.
    """
    extra: dict[str, Any] = {"is_sidechain": event.get("isSidechain", False)}
    if agent_id := event.get("agentId"):
        extra["agent_id"] = agent_id
    return extra


def _last_usage_by_msg_id(events: list[dict[str, Any]]) -> dict[str, Any]:
    last_usage_by_msg_id: dict[str, Any] = {}
    for ev in events:
        if ev.get("type") != "assistant":
            continue
        msg = ev.get("message")
        if not isinstance(msg, dict):
            continue
        mid = msg.get("id")
        usage = msg.get("usage")
        if mid and usage is not None:
            last_usage_by_msg_id[mid] = usage
    return last_usage_by_msg_id


def _normalize_assistant_event(
    event: dict[str, Any], message: dict[str, Any], state: _NormalizationState
) -> None:
    timestamp = event.get("timestamp")
    text, reasoning, tool_blocks = _extract_text_reasoning_tool_uses(message.get("content"))

    msg_id = message.get("id")
    if msg_id and msg_id in state.seen_message_ids:
        metrics = None
    else:
        # Claude Code accumulates usage, so the "real" usage is the last
        # message in the chain. We still report this usage on the first
        # shown in the trajectory.
        usage = (
            state.last_usage_by_msg_id.get(msg_id, message.get("usage"))
            if msg_id
            else message.get("usage")
        )
        metrics = _build_metrics(usage)
        if msg_id:
            state.seen_message_ids.add(msg_id)
    extra: dict[str, Any] = {}
    for key in ("stop_reason", "stop_sequence", "requestId"):
        value = message.get(key)
        if value is not None:
            extra[key] = value
    if event.get("id"):
        extra["id"] = event["id"]
    extra.update(_event_identity_extra(event))
    if event.get("cwd"):
        extra.setdefault("cwd", event["cwd"])
    if event.get("userType") and event.get("userType") != "external":
        extra["user_type"] = event["userType"]

    model_name = message.get("model") or state.default_model_name

    # Bundle one LLM inference (text, reasoning, and all tool_use calls share
    # a `message.id`) into a single ATIF step, per RFC-0001 (`step` == one
    # turn; `tool_calls` is multi-valued). Reuse the turn when the same
    # `message.id` is split across several session-log events.
    turn = state.turn_by_msgid.get(msg_id) if msg_id else None
    if turn is None:
        turn = {
            "kind": "agent_step",
            "timestamp": timestamp,
            "role": message.get("role", "assistant"),
            "text": "",
            "reasoning": None,
            "metrics": None,
            "extra": extra or None,
            "model_name": model_name,
            "tool_calls": [],
        }
        state.normalized_events.append(turn)
        if msg_id:
            state.turn_by_msgid[msg_id] = turn

    if text:
        turn["text"] = f"{turn['text']}\n\n{text}".strip() if turn["text"] else text
    if reasoning and message.get("role") == "assistant":
        turn["reasoning"] = (
            f"{turn['reasoning']}\n\n{reasoning}" if turn["reasoning"] else reasoning
        )
    if turn["metrics"] is None and metrics is not None:
        turn["metrics"] = metrics
        metrics = None

    turn_calls = turn["tool_calls"]
    if not isinstance(turn_calls, list):
        turn_calls = []
        turn["tool_calls"] = turn_calls
    for tool_block in tool_blocks:
        call_id = tool_block.get("id") or tool_block.get("tool_use_id")
        if not call_id:
            continue
        # A call is keyed by call_id; skip a tool_use already seen (e.g. an
        # event replayed after compaction) so it is not bundled twice.
        if call_id in state.pending_calls or call_id in state.completed_call_ids:
            continue

        raw_arguments = tool_block.get("input")
        arguments = raw_arguments if isinstance(raw_arguments, dict) else {"input": raw_arguments}

        call_extra: dict[str, Any] = {}
        if raw_arguments is not None:
            call_extra["raw_arguments"] = raw_arguments
        if tool_block.get("status") is not None:
            call_extra["status"] = tool_block.get("status")
        if tool_block.get("is_error") is not None:
            call_extra["tool_use_is_error"] = tool_block.get("is_error")
        if tool_block.get("name"):
            call_extra.setdefault("tool_use_name", tool_block.get("name"))

        tool_call_spec: dict[str, Any] = {
            "call_id": call_id,
            "tool_name": tool_block.get("name") or "",
            "arguments": arguments or {},
            "extra": call_extra or None,
            "output": None,
            "result_extra": None,
        }
        turn_calls.append(tool_call_spec)
        state.pending_calls[call_id] = tool_call_spec


def _normalize_tool_result_block(
    block: dict[str, Any], event: dict[str, Any], state: _NormalizationState
) -> None:
    timestamp = event.get("timestamp")
    call_id = block.get("tool_use_id")
    formatted_output, metadata = _format_tool_result(block, event.get("toolUseResult"))
    call_info = state.pending_calls.pop(call_id, None) if call_id else None
    if call_info is not None:
        # Matched a pending tool call: attach the result in place on its
        # bundled turn (already appended to normalized_events as one
        # agent_step).
        result_extra: dict[str, Any] = {}
        if metadata:
            result_extra["tool_result_metadata"] = metadata
        if block.get("is_error") is not None:
            result_extra["tool_result_is_error"] = block.get("is_error")
        call_info["output"] = formatted_output
        call_info["result_extra"] = result_extra or None
        if call_id:
            state.completed_call_ids.add(call_id)
        return

    # Orphan tool_result with no matching tool_use in this window (e.g.
    # replayed after compaction): keep the legacy single-call handling so its
    # output is not lost. A duplicate of an already-completed call is
    # dropped; a result with no tool name is skipped.
    if call_id and call_id in state.completed_call_ids:
        logger.debug("Skipping duplicate Claude Code tool result {}", call_id)
        return
    tool_name = block.get("name") or block.get("tool_name") or ""
    if not tool_name:
        logger.debug(
            "Skipping orphan Claude Code tool result {} without tool name",
            call_id or "<missing>",
        )
        return
    call_info = {
        "kind": "tool_call",
        "timestamp": timestamp,
        "call_id": call_id or "",
        "tool_name": tool_name,
        "arguments": {},
        "raw_arguments": None,
        "reasoning": None,
        "status": None,
        "message": None,
        "extra": None,
        "metrics": None,
        "model_name": state.default_model_name,
    }

    extra_val = call_info.get("extra")
    extra = extra_val if isinstance(extra_val, dict) else {}
    extra.update(_event_identity_extra(event))
    if metadata:
        extra.setdefault("tool_result_metadata", metadata)
    if block.get("is_error") is not None:
        extra.setdefault("tool_result_is_error", block.get("is_error"))

    call_info["extra"] = extra or None
    call_info["output"] = formatted_output
    call_info["metadata"] = metadata
    call_info["timestamp"] = call_info.get("timestamp") or timestamp
    call_info.setdefault("model_name", state.default_model_name)

    state.normalized_events.append(call_info)
    if call_id:
        state.completed_call_ids.add(call_id)


def user_message_content(content: list[Any]) -> str | list[ContentPart]:
    """A user record's block-list content as its step message (``""`` when it shows none).

    harbor's rule: a ``text`` block contributes its string verbatim (no
    strip, so byte spans inside each part round-trip), a ``tool_result``
    block nothing (it becomes an observation), and any other block its JSON
    text; empty or whitespace-only parts are dropped and the rest joined with
    blank lines. One departure: when a block is an image or audio part the
    pre-pass lifted (:func:`~atif_converter.domain.blobs.lifted_part`), the
    message is those parts in order, each kept text a ``text`` part.
    """
    parts: list[ContentPart] = []
    has_media = False
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_result":
            continue
        media = lifted_part(block)
        if media is not None:
            parts.append(media)
            has_media = True
            continue
        text = block["text"] if _is_text_block(block) else _stringify(block)
        if text.strip():
            parts.append(ContentPart(type="text", text=text))
    if has_media:
        return parts
    return "\n\n".join(part.text or "" for part in parts)


def _normalize_user_event(
    event: dict[str, Any], message: dict[str, Any], state: _NormalizationState
) -> None:
    timestamp = event.get("timestamp")
    content = message.get("content")
    if isinstance(content, str):
        # Preserve the raw bytes of the user message so downstream
        # byte-identity checks (sha256 of the canonical instruction.md vs the
        # first user step) hold; the strip is only the empty-skip filter.
        text = content
        if text.strip():
            extra = _event_identity_extra(event)
            state.normalized_events.append(
                {
                    "kind": "message",
                    "timestamp": timestamp,
                    "role": "user",
                    "text": text,
                    "extra": extra,
                }
            )
        return

    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                _normalize_tool_result_block(block, event, state)
        text_message = user_message_content(content)
        if text_message:
            state.normalized_events.append(
                {
                    "kind": "message",
                    "timestamp": timestamp,
                    "role": "user",
                    "text": text_message,
                    "extra": _event_identity_extra(event),
                }
            )
        return

    if content not in (None, ""):
        # Same byte-faithful rule: keep the stringified content's raw bytes
        # and use strip only for the empty-skip filter.
        text = _stringify(content)
        if text.strip():
            state.normalized_events.append(
                {
                    "kind": "message",
                    "timestamp": timestamp,
                    "role": "user",
                    "text": text,
                    "extra": _event_identity_extra(event),
                }
            )


def _normalize_events(
    events: list[dict[str, Any]], default_model_name: str | None
) -> list[dict[str, Any]]:
    state = _NormalizationState(
        default_model_name=default_model_name,
        last_usage_by_msg_id=_last_usage_by_msg_id(events),
    )
    for event in events:
        message = event.get("message")
        if not isinstance(message, dict):
            continue

        event_type = event.get("type")
        if event_type == "assistant":
            _normalize_assistant_event(event, message, state)
        elif event_type == "user":
            _normalize_user_event(event, message, state)

    # Leftover pending calls (a tool_use that never received a result) are
    # already embedded in their bundled turn's `tool_calls`, so there is
    # nothing to flush here; they render as a ToolCall with no observation.
    return state.normalized_events


def _build_final_metrics(steps: list[Step]) -> FinalMetrics:
    prompt_values = [
        step.metrics.prompt_tokens
        for step in steps
        if step.metrics and step.metrics.prompt_tokens is not None
    ]
    completion_values = [
        step.metrics.completion_tokens
        for step in steps
        if step.metrics and step.metrics.completion_tokens is not None
    ]
    cached_values = [
        step.metrics.cached_tokens
        for step in steps
        if step.metrics and step.metrics.cached_tokens is not None
    ]

    total_prompt_tokens = sum(prompt_values) if prompt_values else None
    total_completion_tokens = sum(completion_values) if completion_values else None
    total_cached_tokens = sum(cached_values) if cached_values else None

    service_tiers: set[str] = set()
    cache_creation_total, cache_read_total = 0, 0
    cache_creation_seen, cache_read_seen = False, False
    for step in steps:
        if not step.metrics or not step.metrics.extra:
            continue
        extra = step.metrics.extra
        tier = extra.get("service_tier")
        if isinstance(tier, str):
            service_tiers.add(tier)
        cache_creation = extra.get("cache_creation_input_tokens")
        if isinstance(cache_creation, int):
            cache_creation_total += cache_creation
            cache_creation_seen = True
        cache_read = extra.get("cache_read_input_tokens")
        if isinstance(cache_read, int):
            cache_read_total += cache_read
            cache_read_seen = True

    final_extra: dict[str, Any] = {}
    if service_tiers:
        final_extra["service_tiers"] = sorted(service_tiers)
    if cache_creation_seen:
        final_extra["total_cache_creation_input_tokens"] = cache_creation_total
    if cache_read_seen:
        final_extra["total_cache_read_input_tokens"] = cache_read_total

    # harbor consults ``_parse_total_cost_from_stream_json`` first; in our
    # usage it always returned ``None`` (see the module docstring), so the
    # estimate is the only cost source.
    total_cost_usd = _estimate_step_costs(steps)
    if total_cost_usd is not None:
        final_extra["cost_source"] = pricing.cost_source_label(
            step.model_name for step in steps if step.metrics is not None
        )

    return FinalMetrics(
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        total_cached_tokens=total_cached_tokens,
        total_cost_usd=total_cost_usd,
        total_steps=len(steps),
        extra=final_extra or None,
    )


def _convert_events(
    main_records: Sequence[dict[str, Any]],
    side_records: Mapping[str, Sequence[dict[str, Any]]] | None,
    *,
    fallback_session_id: str,
    model_name: str | None,
) -> Trajectory | None:
    """Harbor 0.26.0's conversion of one event stream, from the point the files were read.

    harbor 0.26.0 reads the main transcript and every side-file into this one
    stream; :func:`convert_claude_code_records` hands it one transcript's own
    events at a time instead.

    Returns ``None`` exactly when harbor returns ``None``: no records at all,
    or no record produced a step.
    """
    # harbor builds agent.extra's lists from SETS, so their order follows the
    # process hash seed. Ours keep first-seen order (``_agent_extra``), a
    # deliberate divergence the oracle forgives only for a reordering.
    events = _collect_events(main_records, side_records)
    if not events:
        return None

    session_id: str = fallback_session_id
    for event in events:
        sid = event.get("sessionId")
        if isinstance(sid, str):
            session_id = sid
            break

    agent_version: str = "unknown"
    for event in events:
        ver = event.get("version")
        if isinstance(ver, str) and ver:
            agent_version = ver
            break

    agent_extra = _agent_extra(events)

    # Prefer the main chain's model so a subagent running a different model
    # (e.g. a small/fast model) cannot become the trajectory's
    # `agent.model_name` or the fallback for model-less steps.
    default_model_name = (
        _first_event_model(events, include_sidechain=False)
        or _first_event_model(events, include_sidechain=True)
        or model_name
    )

    normalized_events = _normalize_events(events, default_model_name)

    steps: list[Step] = []
    for norm_event in normalized_events:
        try:
            step = _convert_event_to_step(norm_event, len(steps) + 1, model_name)
        except ValueError as exc:
            logger.debug("Skipping event during step conversion: {}", exc)
            continue

        if step.source == "agent" and not step.model_name and default_model_name:
            step.model_name = default_model_name

        steps.append(step)

    if not steps:
        logger.debug("No valid steps produced from Claude Code session")
        return None

    return Trajectory(
        schema_version=SCHEMA_VERSION,
        session_id=session_id,
        agent=Agent(
            name=AgentSource.CLAUDE_CODE.value,
            version=agent_version,
            model_name=default_model_name,
            extra=agent_extra,
        ),
        steps=steps,
        final_metrics=_build_final_metrics(steps),
    )


# ---------------------------------------------------------------------------
# Subagent transcripts as embedded trajectories (harbor #4076)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ClaudeCodeTranscripts:
    """A session's records split into the root's own transcript and each subagent's."""

    #: The root's own events: every main record, minus the events of an agent
    #: that has a side-file.
    root: list[dict[str, Any]]
    #: Native agent id -> that agent's own events, in sorted side-file order.
    children: dict[str, list[dict[str, Any]]]


def _child_agent_id(name: str, records: Sequence[dict[str, Any]]) -> str:
    """The agent a side-file belongs to: its ``agent-<id>`` file name when a sidechain event carries that id.

    Otherwise the first sidechain ``agentId`` in the file, else the id the
    file name gives. The name is the file's own, without the directories a
    workflow-nested file's staged name joins in front of it. The first
    sidechain event alone is not trusted, since a file can open with a copy of
    its parent's sidechain context.
    """
    file_id = name.rsplit("__", 1)[-1].removesuffix(".jsonl").removeprefix("agent-")
    sidechain_ids: list[str] = [
        agent_id
        for event in records
        if isinstance(agent_id := event.get("agentId"), str) and event.get("isSidechain")
    ]
    return file_id if file_id in sidechain_ids else next(iter(sidechain_ids), file_id)


def split_claude_code_records(
    main_records: Sequence[dict[str, Any]],
    side_records: Mapping[str, Sequence[dict[str, Any]]] | None,
) -> ClaudeCodeTranscripts:
    """Keep each native turn only in the transcript that owns it (harbor #4076).

    A side-file belongs to one agent (:func:`_child_agent_id`), so that agent's
    events leave the root, where an older Claude Code also inlined them. A
    side-file can open with a copy of its parent's history, which may carry a
    new uuid, timestamp or usage; the first event carrying the file's own agent
    id starts its own transcript, an event of another agent is dropped, and a
    uuid already kept anywhere is kept once. Two side-files of one agent join
    into one transcript, in sorted-name order.
    """
    if not side_records:
        return ClaudeCodeTranscripts(root=list(main_records), children={})
    names = sorted(side_records)
    child_ids = {name: _child_agent_id(name, side_records[name]) for name in names}
    own_ids = set(child_ids.values())
    root = [event for event in main_records if event.get("agentId") not in own_ids]
    seen_uuids = {event["uuid"] for event in root if isinstance(event.get("uuid"), str)}
    children: dict[str, list[dict[str, Any]]] = {}
    for name in names:
        own_id = child_ids[name]
        events = side_records[name]
        first_own = next(
            (index for index, event in enumerate(events) if event.get("agentId") == own_id), 0
        )
        kept = children.setdefault(own_id, [])
        for event in events[first_own:]:
            event_uuid = event.get("uuid")
            if event.get("agentId") not in (None, own_id):
                continue
            if isinstance(event_uuid, str) and event_uuid in seen_uuids:
                continue
            kept.append(event)
            if isinstance(event_uuid, str):
                seen_uuids.add(event_uuid)
    return ClaudeCodeTranscripts(root=root, children=children)


def _task_agent_id(result: ObservationResult | None) -> str | None:
    metadata = (result.extra or {}).get("tool_result_metadata") if result else None
    tool_result = metadata.get("tool_use_result") if isinstance(metadata, dict) else None
    if isinstance(tool_result, dict):
        for key in ("agentId", "agent_id"):
            agent_id = tool_result.get(key)
            if isinstance(agent_id, str) and agent_id:
                return agent_id
    return None


_TaskCall = tuple[Trajectory, Step, ToolCall, "ObservationResult | None"]


def _task_calls(trajectory: Trajectory) -> list[tuple[Step, ToolCall, ObservationResult | None]]:
    """Every ``Task`` / ``Agent`` call of ``trajectory`` with its result, when it has one."""
    calls: list[tuple[Step, ToolCall, ObservationResult | None]] = []
    for step in trajectory.steps:
        for call in step.tool_calls or []:
            if call.function_name not in {"Task", "Agent"}:
                continue
            result = next(
                (
                    item
                    for item in (step.observation.results if step.observation else [])
                    if item.source_call_id == call.tool_call_id
                ),
                None,
            )
            calls.append((step, call, result))
    return calls


def _attach_subagent(
    parent: Trajectory,
    step: Step,
    call: ToolCall,
    result: ObservationResult | None,
    child: Trajectory,
) -> None:
    if step.observation is None:
        step.observation = Observation(results=[])
    if result is None:
        result = ObservationResult(source_call_id=call.tool_call_id)
        step.observation.results.append(result)
    refs = result.subagent_trajectory_ref or []
    if not any(ref.trajectory_id == child.trajectory_id for ref in refs):
        refs.append(
            SubagentTrajectoryRef(trajectory_id=child.trajectory_id, session_id=child.session_id)
        )
    result.subagent_trajectory_ref = refs
    if parent.subagent_trajectories is None:
        parent.subagent_trajectories = []
    if not any(existing is child for existing in parent.subagent_trajectories):
        parent.subagent_trajectories.append(child)


def _spawning_calls(
    child_id: str,
    child: Trajectory,
    prompt: str | None,
    prompts: Sequence[Any],
    task_calls: Sequence[_TaskCall],
) -> list[_TaskCall]:
    """The calls that started or resumed ``child``: by its agent id, else by a unique prompt."""
    candidates = [
        (parent, step, call, result)
        for parent, step, call, result in task_calls
        if parent is not child
        and (_task_agent_id(result) == child_id or call.arguments.get("resume") == child_id)
        and _task_agent_id(result) in (None, child_id)
        and call.arguments.get("resume") in (None, child_id)
    ]
    if candidates:
        return candidates
    prompt_candidates = [
        (parent, step, call, result)
        for parent, step, call, result in task_calls
        if parent is not child
        if prompt
        and call.arguments.get("prompt") == prompt
        and _task_agent_id(result) is None
        and call.arguments.get("resume") is None
    ]
    # Prompt text identifies a child only when one eligible call and one child
    # share it. Repeated prompts have no stable ID.
    if len(prompt_candidates) == 1 and sum(other == prompt for other in prompts) == 1:
        return prompt_candidates
    return []


def _roll_up(parent: Trajectory) -> None:
    """Add each embedded trajectory's usage and cost to its parent's totals, deepest first."""
    for child in parent.subagent_trajectories or []:
        _roll_up(child)
        if parent.final_metrics is None or child.final_metrics is None:
            continue
        for field_name in ("total_prompt_tokens", "total_completion_tokens", "total_cached_tokens"):
            value = getattr(child.final_metrics, field_name)
            if value is not None:
                current = getattr(parent.final_metrics, field_name) or 0
                setattr(parent.final_metrics, field_name, current + value)
        parent_extra = parent.final_metrics.extra or {}
        child_extra = child.final_metrics.extra or {}
        for key in ("total_cache_creation_input_tokens", "total_cache_read_input_tokens"):
            child_value = child_extra.get(key)
            if isinstance(child_value, int):
                parent_extra[key] = parent_extra.get(key, 0) + child_value
        child_tiers = child_extra.get("service_tiers") or []
        if child_tiers:
            parent_extra["service_tiers"] = sorted(
                set(parent_extra.get("service_tiers", [])) | set(child_tiers)
            )
        parent.final_metrics.extra = parent_extra or None
        # harbor skips the root's cost when Claude's own stream result priced the
        # whole run; this port never reads that result, so every cost adds up.
        child_cost = child.final_metrics.total_cost_usd
        current_cost = parent.final_metrics.total_cost_usd
        parent.final_metrics.total_cost_usd = (
            current_cost + child_cost
            if current_cost is not None and child_cost is not None
            else None
        )


def _first_prompt(child: Trajectory) -> str | None:
    """The text of ``child``'s first user step, when that step's message is text."""
    first = next((step for step in child.steps if step.source == "user"), None)
    return first.message if first is not None and isinstance(first.message, str) else None


def nest_claude_code_subagents(root: Trajectory, children: Mapping[str, Trajectory]) -> None:
    """Embed each subagent trajectory under the trajectory whose call spawned it (harbor #4076).

    ``children`` maps each native agent id to its converted trajectory, whose
    ``trajectory_id`` is that id. A child links to the ``Task`` / ``Agent``
    calls whose result names its agent id or that resume it, else to the one
    call sharing its first prompt when no other child shares that prompt; it
    is embedded under the trajectory that owns the spawning call (a resumed
    call's result is linked too), unless that would close a cycle. A child
    with no such call is embedded under the root. Each parent's totals then
    include its children's.
    """
    task_calls: list[_TaskCall] = [
        (parent, step, call, result)
        for parent in (root, *children.values())
        for step, call, result in _task_calls(parent)
    ]
    child_prompts = {child_id: _first_prompt(child) for child_id, child in children.items()}
    prompts = list(child_prompts.values())
    assigned: set[str] = set()
    parent_of: dict[str, str | None] = {}
    for child_id, child in children.items():
        candidates = _spawning_calls(child_id, child, child_prompts[child_id], prompts, task_calls)
        # A resumed Task reuses its agentId. Link each matching result to one
        # embedded transcript under the parent that owns the spawn.
        spawn_candidates = [
            candidate
            for candidate in candidates
            if candidate[2].arguments.get("resume") != child_id
        ]
        owner_candidates = spawn_candidates or candidates
        parent = owner_candidates[0][0] if owner_candidates else None
        if parent is None or not all(candidate[0] is parent for candidate in owner_candidates):
            continue
        ancestor = parent.trajectory_id
        while ancestor is not None and ancestor != child_id:
            ancestor = parent_of.get(ancestor)
        if ancestor == child_id:
            continue
        for candidate in candidates:
            if candidate[0] is parent:
                _attach_subagent(*candidate, child)
        parent_of[child_id] = parent.trajectory_id
        assigned.add(child_id)

    # Preserve transcripts whose spawning Task is missing from the log.
    for child_id, child in children.items():
        if child_id not in assigned:
            root.subagent_trajectories = [*(root.subagent_trajectories or []), child]

    _roll_up(root)


def convert_claude_code_records(
    main_records: Sequence[dict[str, Any]],
    side_records: Mapping[str, Sequence[dict[str, Any]]] | None = None,
    *,
    fallback_session_id: str = "-unknown",
    model_name: str | None = None,
) -> Trajectory | None:
    """Convert one Claude Code session's parsed records into an ATIF trajectory.

    This is harbor's ``_convert_events_to_trajectory`` from the point where the
    session files have been read, with harbor #4076's subagent nesting: each
    side-file agent's own transcript (:func:`split_claude_code_records`)
    converts into a trajectory of its own, embedded in ``subagent_trajectories``
    (:func:`nest_claude_code_subagents`) with the native agent id as its
    ``trajectory_id``. When the root's own transcript yields no step, every
    transcript converts as one stream instead, as harbor 0.26.0 converts it.

    Parameters
    ----------
    main_records
        Every JSON record of the main ``<session-id>.jsonl``, in file order.
    side_records
        Staged side-file NAME -> that file's records (see the module docstring
        for the name shape), in sorted-name order, exactly as harbor's
        ``sorted(rglob("subagents/*.jsonl"))``.
    fallback_session_id
        harbor falls back to the session directory's name when no record
        carries a ``sessionId``; the old adapter staged under
        ``<parent-dir-name or "-unknown">``, so callers pass that.
    model_name
        harbor's ``self.model_name``: the run-configured model, used only when
        no record names one. Our adapter never set it.

    Returns
    -------
    Trajectory | None
        ``None`` exactly when harbor returns ``None``: no records at all, or no
        record produced a step.
    """
    if not side_records:
        return _convert_events(
            main_records, None, fallback_session_id=fallback_session_id, model_name=model_name
        )
    transcripts = split_claude_code_records(main_records, side_records)
    root = (
        _convert_events(
            transcripts.root, None, fallback_session_id=fallback_session_id, model_name=model_name
        )
        if transcripts.root
        else None
    )
    if root is None:
        # Nesting needs a root step. Without one, keep the child transcripts in
        # a single stream rather than drop the run.
        flat = [*transcripts.root, *(e for events in transcripts.children.values() for e in events)]
        return _convert_events(
            flat, None, fallback_session_id=fallback_session_id, model_name=model_name
        )
    children: dict[str, Trajectory] = {}
    for agent_id, events in transcripts.children.items():
        child = _convert_events(
            events, None, fallback_session_id=fallback_session_id, model_name=model_name
        )
        if child is None:
            continue
        child.trajectory_id = agent_id
        children[agent_id] = child
    nest_claude_code_subagents(root, children)
    return root
