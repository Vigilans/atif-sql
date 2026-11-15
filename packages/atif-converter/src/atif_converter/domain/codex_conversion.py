# SPDX-License-Identifier: Apache-2.0
# Ported from harbor 0.22.0, src/harbor/agents/installed/codex.py (Apache-2.0,
# Copyright the Harbor authors), brought to harbor 0.26.0, onto the ATIF models vendored in
# atif_converter.domain.atif, so atif-converter doesn't depend on harbor.

"""Codex rollout records -> ATIF trajectory, as one pure function.

This is harbor's ``Codex._convert_session_events`` and the helpers it
calls, ported line for line onto harbor's ``harbor.models.trajectories``
data classes, which are vendored as :mod:`atif_converter.domain.atif`. harbor
documents those classes and the validator and nothing else, so the converter
itself has to live here. The parity oracle in ``tests/harbor_oracle.py`` measures this port
against harbor's private method for as long as harbor still ships it.

FAITHFUL, NOT IMPROVED. Every quirk below is harbor's, kept so the two agree
byte for byte; a behavior change is a decision for a later commit, made with
the golden re-frozen on purpose. Names and structure follow harbor's so an
upstream diff can be re-applied by hand. harbor 0.24.0's two Codex changes
are applied: a ``web_search_call`` keeps its item id or gets
``<api call>_web_search_<n>`` (harbor #2972), and a structured tool output (a
list of MCP content blocks or content items) becomes text rather than Python's
``str()`` of the list (harbor #3467). So is harbor 0.26.0's conversion of one
thread: each model request takes the model its own ``turn_context`` names, and
the final token totals keep a zero. The departures are an inline image in
such an output, which harbor saves to a file and this pure function names by
its corpus blob path (:func:`_tool_output_content` says why), and Codex
media in messages and audio in tool outputs: harbor's
``_extract_message_text`` drops a user or developer message's ``input_image``
and ``input_audio`` and its tool output keeps an audio block as JSON text,
where here each is an ATIF image or audio part, or ``[audio:
<file_id-or-url>]``
(:func:`~atif_converter.domain.codex_enrichment.message_content`,
:func:`~atif_converter.domain.blobs.codex_audio_part`):

=================================  =========================================
harbor 0.26.0                       here
=================================  =========================================
``_extract_message_text``           :func:`~atif_converter.domain.codex_enrichment.message_text`,
                                    and
                                    :func:`~atif_converter.domain.codex_enrichment.message_content`
                                    for a user or developer message
``_parse_output_blob``              :func:`_parse_output_blob`
``_group_events_by_api_call_id``    :func:`_group_events_by_api_call_id`
``_metrics_from_token_count_payload``  :func:`_metrics_from_token_count_payload`
``_convert_event_to_step``          :func:`_convert_event_to_step`
``_compute_cost_from_pricing``      :func:`_compute_cost_from_pricing`
``_convert_session_events``         :func:`convert_codex_records` (+ the three
                                    ``_normalize_events`` /
                                    ``_attach_api_call_metrics`` /
                                    ``_final_metrics`` slices it was cut into
                                    for the size gates)
=================================  =========================================

The three instance attributes the method read are replaced as follows:

``self.logger``
    loguru's ``logger``; the message texts are harbor's.
``self.model_name``
    the ``model_name`` parameter, default ``None`` — the adapter this replaces
    never set one, so every step's model comes from a ``turn_context``.
``self._version``
    ``None``. harbor only consults it when ``session_meta`` carries no
    ``cli_version``, and the adapter never set it either, so ``agent.version``
    is the rollout's ``cli_version`` or the literal ``"unknown"``.

One substitution under the hood: ``_compute_cost_from_pricing`` reads the
price table and prices each call through :mod:`atif_converter.domain.pricing`
rather than ``litellm`` directly. The floats are identical (the module
repeats ``litellm.cost_per_token``'s arithmetic over a vendored copy of
litellm's price data); what changed is that litellm isn't installed at all.

Pure over already-parsed records in FILE order: no file I/O, no
``harbor.agents`` import. Reading the rollout is
:mod:`atif_converter.infrastructure.codex_converter`'s job.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

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
    ToolCall,
    Trajectory,
)
from atif_converter.domain.blobs import (
    BlobCollector,
    codex_audio_part,
    codex_tool_image_ref,
    lifted_part,
)
from atif_converter.domain.codex_enrichment import message_content, message_text

#: The schema version harbor 0.22.0 stamps on a Codex trajectory.
_SCHEMA_VERSION = "ATIF-v1.7"

#: ``agent.version`` when neither ``session_meta.cli_version`` nor a runtime
#: version is known — harbor's literal.
_UNKNOWN_VERSION = "unknown"


#: harbor 0.24.0's tool output content types (``codex.py``, harbor #3467).
_TOOL_OUTPUT_TEXT_TYPES = frozenset({"text", "input_text"})
_TOOL_OUTPUT_IMAGE_TYPES = frozenset({"image", "input_image"})
_TOOL_OUTPUT_AUDIO_TYPES = frozenset({"audio", "input_audio"})
_TOOL_OUTPUT_CONTENT_TYPES = (
    _TOOL_OUTPUT_TEXT_TYPES
    | _TOOL_OUTPUT_IMAGE_TYPES
    | _TOOL_OUTPUT_AUDIO_TYPES
    | frozenset({"resource", "resource_link", "encrypted_content"})
)
#: harbor's text for an image it cannot reference.
_IMAGE_OMITTED = "[image omitted]"


def _parse_output_blob(raw: Any) -> tuple[str | list[ContentPart] | None, dict[str, Any] | None]:
    """Extract the observation content and metadata from a Codex tool output.

    harbor 0.24.0's ``Codex._parse_output_blob`` (harbor #3467). A JSON object
    yields its ``output`` (JSON text when that is not a string, or, when the key
    is missing, the whole object re-serialized) plus its ``metadata`` when that
    is an object; a list of MCP content blocks or content items becomes the
    text :func:`_tool_output_content` renders; any other JSON value keeps the
    raw string, or its JSON text when the output was not a string; a non-JSON
    string is the output itself.
    """
    if raw is None:
        return None, None

    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return raw, None
    else:
        parsed = raw

    if isinstance(parsed, dict):
        output = parsed.get("output")
        if output is None and parsed:
            # dumping remaining structure if output missing
            output = json.dumps(parsed, ensure_ascii=False)
        elif output is not None and not isinstance(output, str):
            output = json.dumps(output, ensure_ascii=False)
        metadata = parsed.get("metadata")
        return output, metadata if isinstance(metadata, dict) else None

    return _non_object_output(raw, parsed), None


def _non_object_output(raw: Any, parsed: Any) -> str | list[ContentPart]:
    """The content of a tool output that is not a JSON object (harbor's tail branches)."""
    if _is_tool_output_content(parsed):
        return _tool_output_content(parsed)
    if isinstance(raw, str):
        return raw
    return json.dumps(parsed, ensure_ascii=False)


def _is_tool_output_content(value: Any) -> bool:
    """Whether a tool output is a list of MCP content blocks or content items.

    harbor's ``Codex._is_tool_output_content``: a non-empty list whose every
    element is an object with a known content ``type``.
    """
    if not isinstance(value, list) or not value:
        return False
    for block in value:
        if not isinstance(block, dict):
            return False
        block_type = block.get("type")
        if not isinstance(block_type, str) or block_type not in _TOOL_OUTPUT_CONTENT_TYPES:
            return False
    return True


def _tool_output_content(blocks: list[dict[str, Any]]) -> str | list[ContentPart]:
    """Tool output content blocks -> the observation's content.

    harbor's ``Codex._tool_output_content``: text blocks keep their text, an
    image becomes :func:`_tool_output_image_part`, an audio block or item
    becomes :func:`~atif_converter.domain.blobs.codex_audio_part`, and any
    other block (a resource, encrypted content) is kept as its JSON text. An
    output holding an image or audio part is the list of parts; any other is
    the parts' texts joined by newlines.

    DELIBERATE DIVERGENCE from harbor 0.24.0 for an inline image harbor can
    decode: harbor writes it to ``<logs_dir>/images/codex_<sha256[:16]>.<ext>``
    and points its image part there. This converter is pure and the corpus
    keeps attachments in its blob store, so the image part points at the blob
    the pre-pass (:func:`~atif_converter.domain.blobs.extract_codex_blobs`)
    stores for the same item, ``../../blobs/sha256/<ab>/<sha256>.<ext>`` from
    the session directory. harbor keeps an audio block as its JSON text, base64
    and all; here inline audio is an audio part naming its blob, and an
    ``input_audio`` naming only a ``file_id`` or a URL is
    ``[audio: <file_id-or-url>]``. Audio neither rule reads stays harbor's
    JSON text. The parity oracle forgives exactly these
    (``tests/harbor_oracle.py::_media_content_forgiven``).
    """
    parts = [_tool_output_block_part(block) for block in blocks]
    if any(part.type != "text" for part in parts):
        return parts
    return "\n".join(part.text or "" for part in parts)


def structured_output_content(value: Any) -> str | list[ContentPart] | None:
    """:func:`_tool_output_content` for a list of content blocks or items, else ``None``.

    Not part of harbor's converter: the result-signals pass renders a code-mode
    script's nested MCP result through it
    (:mod:`atif_converter.domain.codex_nested_calls`), so the blocks of a
    nested call read the way the same blocks read in a top-level tool output.
    """
    return _tool_output_content(value) if _is_tool_output_content(value) else None


def _tool_output_block_part(block: dict[str, Any]) -> ContentPart:
    lifted = lifted_part(block)
    if lifted is not None:
        return lifted
    block_type = block["type"]
    text = block.get("text")
    if block_type in _TOOL_OUTPUT_TEXT_TYPES and isinstance(text, str):
        return ContentPart(type="text", text=text)
    if block_type in _TOOL_OUTPUT_IMAGE_TYPES:
        return _tool_output_image_part(block)
    if block_type in _TOOL_OUTPUT_AUDIO_TYPES:
        audio = codex_audio_part(block)
        if audio is not None:
            return audio
    return ContentPart(type="text", text=json.dumps(block, ensure_ascii=False))


def _tool_output_image_part(block: dict[str, Any]) -> ContentPart:
    """The part for an MCP image block or an ``input_image`` item.

    harbor's ``Codex._tool_output_image_part`` and ``_save_tool_output_image``,
    with the same texts for an image nobody can reference (``[image omitted]``)
    and for a remote URL (``[image: <url>]``); an inline image harbor would save
    is an image part naming its blob (see :func:`_tool_output_content`).
    """
    if block["type"] == "image":
        data = block.get("data")
        media_type = block.get("mimeType") or block.get("mime_type")
    else:
        data = block.get("image_url")
        media_type = None
    if not isinstance(data, str):
        return ContentPart(type="text", text=_IMAGE_OMITTED)
    if "://" in data and not data.startswith("data:"):
        return ContentPart(type="text", text=f"[image: {data}]")

    return _inline_image_part(data, media_type)


def _inline_image_part(data: str, media_type: Any) -> ContentPart:
    """An inline image's part, or harbor's text when harbor would not save it.

    The rule is :func:`~atif_converter.domain.blobs.codex_tool_image_ref`, the
    one the pre-pass lifts the same image by, so in production the pre-pass has
    already replaced the image with this part and stored its bytes; the
    throwaway collector here only names the bytes when the records were not
    pre-passed.
    """
    ref = codex_tool_image_ref(data, media_type, BlobCollector())
    part = None if ref is None else ref.content_part()
    if part is None:
        logger.debug("Unsupported or undecodable Codex tool image: {}", media_type)
        return ContentPart(type="text", text=_IMAGE_OMITTED)
    return part


def _group_events_by_api_call_id(
    normalized_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge assistant events from the same Codex model request into one step.

    harbor's ``Codex._group_events_by_api_call_id``. Every event carrying a
    string ``api_call_id`` joins the group for that request; a non-assistant
    message flushes the open groups first, which is what keeps a user turn
    between two agent steps rather than after both.
    """
    result: list[dict[str, Any]] = []
    groups: dict[str, dict[str, Any]] = {}
    group_order: list[str] = []

    def flush() -> None:
        for group_id in group_order:
            group = groups.pop(group_id, None)
            if group is None:
                continue
            tool_calls: list[dict[str, Any]] = group["tool_calls"]
            tool_calls.sort(key=lambda tc: tc.get("tool_order", 0))
            message_parts = [
                part for part in group.pop("message_parts") if isinstance(part, str) and part
            ]
            group["text"] = "\n\n".join(message_parts)
            result.append(group)
        group_order.clear()

    for event in normalized_events:
        api_call_id = event.get("api_call_id")
        kind = event.get("kind")
        role = event.get("role")

        if kind == "message" and role != "assistant":
            flush()
            result.append(event)
            continue

        if not isinstance(api_call_id, str):
            flush()
            result.append(event)
            continue

        if api_call_id not in groups:
            groups[api_call_id] = {
                "kind": "bundled",
                "api_call_id": api_call_id,
                "codex_turn_id": event.get("codex_turn_id"),
                "timestamp": event.get("timestamp"),
                "message_parts": [],
                "reasoning": None,
                "tool_calls": [],
                "metrics": event.get("metrics"),
            }
            group_order.append(api_call_id)

        group = groups[api_call_id]
        if kind == "message":
            text = event.get("text")
            if isinstance(text, str) and text:
                group["message_parts"].append(text)
            if event.get("reasoning"):
                group["reasoning"] = event["reasoning"]
            if event.get("timestamp"):
                group["timestamp"] = event["timestamp"]
        elif kind == "tool_call":
            group["tool_calls"].append(event)
            if not group["reasoning"] and event.get("reasoning"):
                group["reasoning"] = event["reasoning"]
            if not group.get("metrics") and event.get("metrics"):
                group["metrics"] = event["metrics"]

    flush()
    return result


def _metrics_from_token_count_payload(
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    """Per-call metrics from a ``token_count`` event's ``last_token_usage``.

    harbor's ``Codex._metrics_from_token_count_payload``. Zero counts become
    ``None`` (``or None``), so a call with no cached tokens reports no
    ``cached_tokens`` key rather than ``0``.
    """
    info = payload.get("info")
    if not isinstance(info, dict):
        return None

    last_usage = info.get("last_token_usage")
    if not isinstance(last_usage, dict):
        return None

    prompt_tokens = last_usage.get("input_tokens")
    completion_tokens = last_usage.get("output_tokens")
    cached_tokens = last_usage.get("cached_input_tokens")
    cache_write_tokens = last_usage.get("cache_write_input_tokens")
    reasoning_tokens = last_usage.get("reasoning_output_tokens")
    total_tokens = last_usage.get("total_tokens")

    extra = {
        "reasoning_output_tokens": reasoning_tokens,
        "total_tokens": total_tokens,
    }
    if cache_write_tokens is not None:
        extra["cache_write_input_tokens"] = cache_write_tokens

    return {
        "prompt_tokens": prompt_tokens or None,
        "completion_tokens": completion_tokens or None,
        "cached_tokens": cached_tokens or None,
        "extra": extra,
    }


def _convert_event_to_step(event: dict[str, Any], step_id: int, model_name: str | None) -> Step:
    """Convert a normalized Codex event dictionary into an ATIF step.

    harbor's ``Codex._convert_event_to_step`` with ``self.model_name`` as the
    ``model_name`` parameter. Three kinds: ``message`` (in practice only
    non-assistant ones reach here — every assistant event carries an
    ``api_call_id`` and is bundled first), ``tool_call`` (kept for the same
    reason harbor keeps it; unreachable through the grouping above) and
    ``bundled``.

    Raises
    ------
        ValueError: an unsupported kind, or a pydantic validation failure
        (``ValidationError`` is a ``ValueError``), which the caller skips.
    """
    kind = event.get("kind")
    timestamp = event.get("timestamp")

    if kind == "message":
        role = event.get("role", "user")
        text = event.get("text", "")
        reasoning = event.get("reasoning")
        source: Literal["system", "user", "agent"]
        if role == "assistant":
            source = "agent"
        elif role == "user":
            source = "user"
        else:
            source = "system"

        message_extra = event.get("extra")

        return Step(
            step_id=step_id,
            timestamp=timestamp,
            source=source,
            message=text,
            reasoning_content=reasoning if source == "agent" and reasoning else None,
            model_name=model_name if source == "agent" and model_name else None,
            llm_call_count=1 if source == "agent" else None,
            extra=message_extra or None,
        )

    if kind == "tool_call":
        return _tool_call_step(event, step_id, model_name)

    if kind == "bundled":
        return _bundled_step(event, step_id, model_name)

    msg = f"Unsupported event kind '{kind}'"
    raise ValueError(msg)


def _tool_call_step(event: dict[str, Any], step_id: int, model_name: str | None) -> Step:
    """The ``kind == "tool_call"`` branch of harbor's ``_convert_event_to_step``."""
    timestamp = event.get("timestamp")
    call_id = event.get("call_id", "")
    tool_name = event.get("tool_name", "")
    reasoning = event.get("reasoning")
    arguments = event.get("arguments") or {}
    if not isinstance(arguments, dict):
        arguments = {"value": arguments}

    tool_call = ToolCall(
        tool_call_id=call_id,
        function_name=tool_name,
        arguments=arguments,
    )

    observation: Observation | None = None
    output_text = event.get("output")
    if output_text is not None:
        observation = Observation(
            results=[
                ObservationResult(
                    source_call_id=call_id or None,
                    content=output_text,
                )
            ]
        )

    metrics_payload = event.get("metrics")
    metrics: Metrics | None = None
    if isinstance(metrics_payload, dict):
        metrics = Metrics(**metrics_payload)

    extra: dict[str, Any] | None = None
    metadata = event.get("metadata")
    if metadata:
        extra = {"tool_metadata": metadata}
    raw_arguments = event.get("raw_arguments")
    if raw_arguments:
        extra = extra or {}
        extra["raw_arguments"] = raw_arguments
    status = event.get("status")
    if status:
        extra = extra or {}
        extra["status"] = status
    api_call_id = event.get("api_call_id")
    if api_call_id:
        extra = extra or {}
        extra["api_call_id"] = api_call_id
    codex_turn_id = event.get("codex_turn_id")
    if codex_turn_id:
        extra = extra or {}
        extra["codex_turn_id"] = codex_turn_id

    message = event.get("message") or ""

    return Step(
        step_id=step_id,
        timestamp=timestamp,
        source="agent",
        message=message,
        tool_calls=[tool_call],
        observation=observation,
        model_name=model_name or None,
        reasoning_content=reasoning or None,
        metrics=metrics,
        llm_call_count=1,
        extra=extra,
    )


def _bundled_step(event: dict[str, Any], step_id: int, model_name: str | None) -> Step:
    """The ``kind == "bundled"`` branch of harbor's ``_convert_event_to_step``.

    One step per model request: every tool call in the group becomes a
    ``ToolCall`` AND an ``ObservationResult`` (content ``None`` when the call
    produced no output, as ``web_search_call`` never does), and the per-call
    details harbor cannot fit on the public ``ToolCall`` land in
    ``extra["tool_call_details"]`` keyed by call id.
    """
    text = event.get("text", "")
    reasoning = event.get("reasoning")

    tool_calls: list[ToolCall] = []
    observation_results: list[ObservationResult] = []
    for tc in event.get("tool_calls", []):
        call_id = tc.get("call_id", "")
        arguments = tc.get("arguments") or {}
        if not isinstance(arguments, dict):
            arguments = {"value": arguments}

        tool_calls.append(
            ToolCall(
                tool_call_id=call_id,
                function_name=tc.get("tool_name", ""),
                arguments=arguments,
            )
        )
        observation_results.append(
            ObservationResult(
                source_call_id=call_id or None,
                content=tc.get("output"),
            )
        )

    extra: dict[str, Any] | None = None
    api_call_id = event.get("api_call_id")
    if api_call_id:
        extra = {"api_call_id": api_call_id}
    codex_turn_id = event.get("codex_turn_id")
    if codex_turn_id:
        extra = extra or {}
        extra["codex_turn_id"] = codex_turn_id

    tool_details: dict[str, Any] = {}
    for tc in event.get("tool_calls", []):
        call_id = tc.get("call_id", "")
        details: dict[str, Any] = {}
        for source_key, target_key in (
            ("metadata", "metadata"),
            ("raw_arguments", "raw_arguments"),
            ("item_type", "item_type"),
            ("status", "status"),
        ):
            value = tc.get(source_key)
            if value:
                details[target_key] = value
        if details:
            tool_details[call_id] = details
    if tool_details:
        extra = extra or {}
        extra["tool_call_details"] = tool_details

    observation = Observation(results=observation_results) if observation_results else None

    return Step(
        step_id=step_id,
        timestamp=event.get("timestamp"),
        source="agent",
        message=text,
        model_name=model_name or None,
        reasoning_content=reasoning or None,
        tool_calls=tool_calls or None,
        observation=observation,
        metrics=Metrics(**event["metrics"]) if event.get("metrics") else None,
        llm_call_count=1,
        extra=extra,
    )


def _compute_cost_from_pricing(
    prompt_tokens: int | None,
    completion_tokens: int | None,
    cached_tokens: int | None,
    cache_write_tokens: int | None,
    model_name: str | None = None,
    fallback_model_name: str | None = None,
) -> float | None:
    """Compute one API call's cost in USD via LiteLLM's pricing logic.

    harbor's ``Codex._compute_cost_from_pricing``; ``fallback_model_name``
    stands in for ``self.model_name``. LiteLLM selects context-dependent rates
    from the token count for this individual request. Returns None when the
    pricing table is unavailable, the model is missing from it (tried as given
    and with any ``provider/`` prefix stripped) or the calculation fails.

    The table lookup and the arithmetic go through
    :mod:`atif_converter.domain.pricing`, which reads the vendored copy of
    litellm's price data and returns the same floats litellm would.
    """
    resolved_model_name = model_name or fallback_model_name
    if not resolved_model_name:
        return None

    pricing_model_name = next(
        (
            key
            for key in (resolved_model_name, resolved_model_name.split("/", 1)[-1])
            if pricing.has_pricing_entry(key)
        ),
        None,
    )

    if pricing_model_name is None:
        logger.debug(
            "No LiteLLM pricing entry for model '{}'; leaving codex cost_usd as None",
            resolved_model_name,
        )
        return None

    try:
        input_cost, output_cost = pricing.cost_per_token(
            model=pricing_model_name,
            prompt_tokens=prompt_tokens or 0,
            completion_tokens=completion_tokens or 0,
            cache_creation_input_tokens=cache_write_tokens or 0,
            cache_read_input_tokens=cached_tokens or 0,
        )
    except Exception:  # noqa: BLE001 — harbor swallows every pricing failure into None
        logger.opt(exception=True).debug(
            "Failed to calculate Codex cost for model '{}'", resolved_model_name
        )
        return None

    return float(input_cost + output_cost)


def _reasoning_summary(payload: dict[str, Any]) -> str | None:
    """A ``reasoning`` item's summary texts joined by newlines, or ``None`` when it has none."""
    summary = payload.get("summary")
    if not isinstance(summary, list) or not summary:
        return None
    reasoning_parts: list[str] = []
    for item in summary:
        if isinstance(item, str):
            reasoning_parts.append(item)
        elif isinstance(item, dict):
            text = item.get("text")
            if isinstance(text, str):
                reasoning_parts.append(text)
    return "\n".join(reasoning_parts) if reasoning_parts else None


@dataclass(slots=True)
class _NormalizedSession:
    """What the "normalize events" pass hands the step builder."""

    events: list[dict[str, Any]]
    #: Per-API-call metrics, keyed ``api_call_<n>``.
    api_call_metrics: dict[str, dict[str, Any]]
    #: Per-API-call model: the one the latest ``turn_context`` named when the
    #: request first produced output (``None`` before any did).
    api_call_models: dict[str, str | None]
    #: The model of the first request that produced output, else ``model_name``.
    default_model_name: str | None


def _normalize_events(
    raw_events: list[dict[str, Any]],
    model_name: str | None,
) -> _NormalizedSession:
    """The "normalize events" pass of harbor's ``_convert_session_events``.

    Walks the raw records once and returns the normalized event list plus the
    per-API-call metrics and models, keyed ``api_call_<n>``. The counter
    advances on a ``token_count`` event only when the request it closes
    produced model output (an assistant message, a web search or a tool call);
    a tool call's normalized event is emitted when its OUTPUT arrives but
    carries the request id current when the CALL was seen. A ``turn_context``
    names the model of the requests after it, and of the current request too
    while that one has produced no output yet; each request is priced at its
    own model.
    """
    normalized_events: list[dict[str, Any]] = []
    pending_calls: dict[str, dict[str, Any]] = {}
    pending_reasoning: str | None = None
    codex_turn_id: str | None = None
    api_call_index = 1
    current_api_call_id = f"api_call_{api_call_index}"
    api_call_metrics: dict[str, dict[str, Any]] = {}
    api_call_models: dict[str, str | None] = {}
    active_model_name = model_name
    default_model_name: str | None = None
    saw_model_output_in_api_call = False
    tool_order_counter = 0

    def record_model_output() -> None:
        nonlocal saw_model_output_in_api_call, default_model_name
        saw_model_output_in_api_call = True
        api_call_models.setdefault(current_api_call_id, active_model_name)
        if default_model_name is None:
            default_model_name = active_model_name

    def finish_api_call(token_count_payload: dict[str, Any]) -> None:
        nonlocal api_call_index, current_api_call_id, saw_model_output_in_api_call
        nonlocal tool_order_counter

        if not saw_model_output_in_api_call:
            return

        metrics = _metrics_from_token_count_payload(token_count_payload)
        if metrics:
            cache_write_tokens = metrics["extra"].get("cache_write_input_tokens")
            metrics["cost_usd"] = _compute_cost_from_pricing(
                prompt_tokens=metrics.get("prompt_tokens"),
                completion_tokens=metrics.get("completion_tokens"),
                cached_tokens=metrics.get("cached_tokens"),
                cache_write_tokens=cache_write_tokens,
                model_name=api_call_models.get(current_api_call_id),
                fallback_model_name=model_name,
            )
            api_call_metrics[current_api_call_id] = metrics

        api_call_index += 1
        current_api_call_id = f"api_call_{api_call_index}"
        saw_model_output_in_api_call = False
        tool_order_counter = 0

    for event in raw_events:
        etype = event.get("type")
        payload = event.get("payload", {})
        timestamp = event.get("timestamp")

        if etype == "event_msg" and isinstance(payload, dict):
            event_type = payload.get("type")
            if event_type in {"task_started", "turn_started"}:
                turn_id = payload.get("turn_id")
                codex_turn_id = turn_id if isinstance(turn_id, str) else None
            elif event_type in {"task_complete", "turn_complete", "turn_aborted"}:
                codex_turn_id = None
            elif event_type == "token_count":
                # A token_count event closes one model API call.
                finish_api_call(payload)
            continue

        if etype == "turn_context":
            turn_id = payload.get("turn_id") if isinstance(payload, dict) else None
            if isinstance(turn_id, str) and codex_turn_id is None:
                codex_turn_id = turn_id
            turn_model_name = payload.get("model") if isinstance(payload, dict) else None
            if isinstance(turn_model_name, str):
                active_model_name = turn_model_name
                if not saw_model_output_in_api_call:
                    api_call_models[current_api_call_id] = turn_model_name
            continue

        if etype != "response_item":
            continue

        payload_type = payload.get("type")
        if payload_type == "reasoning":
            pending_reasoning = _reasoning_summary(payload)
            continue

        if payload_type == "message":
            normalized_events.append(
                {
                    "kind": "message",
                    "api_call_id": current_api_call_id,
                    "codex_turn_id": codex_turn_id,
                    "timestamp": timestamp,
                    "role": payload.get("role", "user"),
                    # harbor: ``_extract_message_text(content)`` when content
                    # is a list, else "" — which is ``message_text``, audio
                    # aside; a user or developer message keeps its media.
                    "text": (
                        message_text(payload)
                        if payload.get("role") == "assistant"
                        else message_content(payload)
                    ),
                    "reasoning": pending_reasoning if payload.get("role") == "assistant" else None,
                }
            )
            if payload.get("role") == "assistant":
                record_model_output()
            pending_reasoning = None
            continue

        if payload_type == "web_search_call":
            action = payload.get("action") or {}
            action_type = action.get("type", "")
            arguments: dict[str, Any] = {"action_type": action_type}
            if "query" in action:
                arguments["query"] = action["query"]
            if "queries" in action:
                arguments["queries"] = action["queries"]
            if "url" in action:
                arguments["url"] = action["url"]

            # harbor 0.24.0 (harbor #2972): the call keeps Codex's own item id,
            # or gets one unique within its request.
            native_call_id = payload.get("id")
            web_search_call_id = (
                native_call_id
                if isinstance(native_call_id, str) and native_call_id.strip()
                else f"{current_api_call_id}_web_search_{tool_order_counter}"
            )

            normalized_events.append(
                {
                    "kind": "tool_call",
                    "api_call_id": current_api_call_id,
                    "codex_turn_id": codex_turn_id,
                    "tool_order": tool_order_counter,
                    "timestamp": timestamp,
                    "call_id": web_search_call_id,
                    "tool_name": "web_search_call",
                    "arguments": arguments,
                    "raw_arguments": None,
                    "reasoning": pending_reasoning,
                    "status": payload.get("status"),
                    "message": None,
                }
            )
            tool_order_counter += 1
            record_model_output()
            pending_reasoning = None
            continue

        if payload_type in {"function_call", "custom_tool_call"}:
            call_id = payload.get("call_id")
            if not call_id:
                continue

            raw_args_key = "arguments" if payload_type == "function_call" else "input"
            raw_arguments = payload.get(raw_args_key)
            try:
                parsed_args = json.loads(raw_arguments)
            except (json.JSONDecodeError, TypeError):
                if isinstance(raw_arguments, str):
                    parsed_args = {"input": raw_arguments}
                elif raw_arguments is None:
                    parsed_args = {}
                else:
                    parsed_args = {"value": raw_arguments}

            pending_calls[call_id] = {
                "kind": "tool_call",
                "api_call_id": current_api_call_id,
                "codex_turn_id": codex_turn_id,
                "tool_order": tool_order_counter,
                "timestamp": timestamp,
                "call_id": call_id,
                "tool_name": payload.get("name") or "",
                "arguments": parsed_args,
                "raw_arguments": raw_arguments,
                "item_type": payload_type,
                "reasoning": pending_reasoning,
                "status": payload.get("status"),
                "message": None,
            }
            tool_order_counter += 1
            record_model_output()
            pending_reasoning = None
            continue

        if payload_type in {"function_call_output", "custom_tool_call_output"}:
            call_id = payload.get("call_id")
            output_text, metadata = _parse_output_blob(payload.get("output"))

            call_info = pending_calls.pop(call_id, None) if call_id else None

            if call_info is None:
                call_info = {
                    "kind": "tool_call",
                    "api_call_id": current_api_call_id,
                    "codex_turn_id": codex_turn_id,
                    "tool_order": tool_order_counter,
                    "timestamp": timestamp,
                    "call_id": call_id or "",
                    "tool_name": payload.get("name", "") or "",
                    "arguments": {},
                    "raw_arguments": None,
                    "reasoning": pending_reasoning,
                    "status": None,
                    "message": None,
                }
                tool_order_counter += 1

            call_info["output"] = output_text
            call_info["metadata"] = metadata
            call_info["timestamp"] = call_info.get("timestamp") or timestamp
            normalized_events.append(call_info)
            pending_reasoning = None
            continue

    return _NormalizedSession(
        events=normalized_events,
        api_call_metrics=api_call_metrics,
        api_call_models=api_call_models,
        default_model_name=default_model_name if default_model_name is not None else model_name,
    )


def _attach_api_call_metrics(
    normalized_events: list[dict[str, Any]],
    api_call_metrics: dict[str, dict[str, Any]],
) -> None:
    """Stamp each normalized event with its request's metrics, in place.

    The same metrics dict is shared by every event of one request; grouping
    then keeps the first non-empty one per step.
    """
    for norm_event in normalized_events:
        event_api_call_id = norm_event.get("api_call_id")
        if isinstance(event_api_call_id, str) and event_api_call_id in api_call_metrics:
            norm_event["metrics"] = api_call_metrics[event_api_call_id]


def _final_metrics(
    raw_events: list[dict[str, Any]],
    total_steps: int,
    estimated_total_cost_usd: float | None,
) -> FinalMetrics | None:
    """Final metrics from the LAST ``token_count`` event carrying totals.

    The "Extract final metrics" pass of harbor's
    ``_convert_session_events``. Codex CLI does not include cost in
    ``token_count`` events, so ``total_cost`` / ``cost_usd`` in ``info`` win
    when present and the LiteLLM estimate fills in otherwise — explicit
    ``is None`` checks so a legitimate $0 is not mistaken for a missing field.
    """
    for event in reversed(raw_events):
        if event.get("type") != "event_msg":
            continue
        payload = event.get("payload", {})
        if not isinstance(payload, dict) or payload.get("type") != "token_count":
            continue

        info = payload.get("info")
        if not isinstance(info, dict):
            continue

        total_usage = info.get("total_token_usage")
        if not isinstance(total_usage, dict):
            continue

        prompt_tokens = total_usage.get("input_tokens")
        completion_tokens = total_usage.get("output_tokens")
        reasoning_tokens = total_usage.get("reasoning_output_tokens")
        cached_tokens = total_usage.get("cached_input_tokens")
        cache_write_tokens = total_usage.get("cache_write_input_tokens")
        overall_tokens = total_usage.get("total_tokens")

        total_cost_usd = info.get("total_cost")
        if total_cost_usd is None:
            total_cost_usd = info.get("cost_usd")
        if total_cost_usd is None:
            total_cost_usd = estimated_total_cost_usd

        final_extra: dict[str, Any] = {
            "reasoning_output_tokens": reasoning_tokens,
            "total_tokens": overall_tokens,
            "last_token_usage": info.get("last_token_usage"),
        }
        if cache_write_tokens is not None:
            final_extra["total_cache_write_input_tokens"] = cache_write_tokens

        return FinalMetrics(
            total_prompt_tokens=prompt_tokens,
            total_completion_tokens=completion_tokens,
            total_cached_tokens=cached_tokens,
            total_cost_usd=total_cost_usd,
            total_steps=total_steps,
            extra=final_extra,
        )

    return None


def convert_codex_records(
    raw_events: list[dict[str, Any]],
    *,
    fallback_session_id: str,
    model_name: str | None = None,
) -> Trajectory | None:
    """Convert one rollout's parsed records into an ATIF trajectory.

    harbor's ``Codex._convert_session_events`` from the point where the
    file has been read. Returns ``None`` exactly when harbor does: no records
    at all, or no record that becomes a step.

    Parameters
    ----------
    raw_events
        The rollout's JSON records in FILE order, malformed lines already
        dropped (see :mod:`atif_converter.infrastructure.codex_converter`).
    fallback_session_id
        ``session_id`` when the rollout's ``session_meta`` names no thread id.
        harbor 0.26.0 uses the rollout file's stem, so the reader passes it.
    model_name
        harbor's ``self.model_name``: the fallback model when no
        ``turn_context`` names one, and the pricing fallback. ``None`` in
        every production call, which is what the adapter this replaces did.
    """
    if not raw_events:
        return None

    session_meta = next((e for e in raw_events if e.get("type") == "session_meta"), None)
    payload = session_meta.get("payload", {}) if session_meta else {}
    # harbor 0.26.0's ``_read_rollout``: the thread id is the metadata's
    # non-empty ``id``, else the rollout's stem.
    meta_id = payload.get("id") if isinstance(payload, dict) else None
    session_id = meta_id if isinstance(meta_id, str) and meta_id else fallback_session_id

    agent_version = _UNKNOWN_VERSION
    agent_extra: dict[str, Any] | None = None

    if session_meta and isinstance(payload, dict):
        agent_version = payload.get("cli_version") or agent_version
        extra: dict[str, Any] = {}
        for key in ("originator", "cwd", "git", "instructions"):
            value = payload.get(key)
            if value is not None:
                extra[key] = value
        agent_extra = extra or None

    # harbor: ``if agent_version == "unknown" and self._version:`` — the CLI
    # version probed at install time. Never set in this port (see the module
    # docstring), so "unknown" stands.

    normalized = _normalize_events(raw_events, model_name)
    default_model_name = normalized.default_model_name
    api_call_metrics = normalized.api_call_metrics
    _attach_api_call_metrics(normalized.events, api_call_metrics)
    grouped_events = _group_events_by_api_call_id(normalized.events)

    steps: list[Step] = []
    for idx, norm_event in enumerate(grouped_events, start=1):
        # harbor 0.26.0: a step takes its request's model, else the default.
        api_call_id = norm_event.get("api_call_id")
        step_model_name = (
            normalized.api_call_models.get(api_call_id, default_model_name)
            if isinstance(api_call_id, str)
            else default_model_name
        )
        try:
            step = _convert_event_to_step(norm_event, idx, step_model_name)
        except ValueError as exc:
            logger.debug("Skipping event during step conversion: {}", exc)
            continue

        steps.append(step)

    if not steps:
        logger.debug("No valid steps produced from Codex session")
        return None

    estimated_total_cost_usd: float | None = 0.0 if api_call_metrics else None
    for metrics in api_call_metrics.values():
        call_cost = metrics["cost_usd"]
        if call_cost is None:
            estimated_total_cost_usd = None
            break
        if estimated_total_cost_usd is not None:
            estimated_total_cost_usd += call_cost

    total_metrics = _final_metrics(raw_events, len(steps), estimated_total_cost_usd)

    return Trajectory(
        schema_version=_SCHEMA_VERSION,
        session_id=session_id,
        agent=Agent(
            name=AgentSource.CODEX.value,
            version=agent_version,
            model_name=default_model_name,
            extra=agent_extra,
        ),
        steps=steps,
        final_metrics=total_metrics,
    )


__all__ = ["convert_codex_records", "structured_output_content"]
