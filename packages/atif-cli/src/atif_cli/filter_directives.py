# SPDX-License-Identifier: Apache-2.0

"""The directives a ``materialize --filter`` script emits, and the executor that applies them.

A filter script leaves a relation named ``directives``; each row is one
action, with the parameter columns that action names (a script combining
actions writes ``UNION ALL BY NAME``, so a row's other columns read NULL):

==================== ==========================================
``action``           parameter columns
==================== ==========================================
``prune_tool_result`` ``trajectory_id``, ``step_id``, ``tool_use_id``
``drop_reasoning``    ``trajectory_id``, ``step_id``
``drop_event``        ``event_seq``
``drop_blob``         ``blob_sha256``
``replace``           ``find``, ``replacement``
==================== ==========================================

:func:`parse_directives` checks one script's rows: an unknown column or
action, a missing or mistyped parameter, a parameter another action owns, or
an empty ``find`` fails the session. :func:`merge_directives` unions the
scripts' rows, deduplicating per target, and fails on a ``find`` paired with
two different replacements. :func:`apply_directives` then rewrites a copy of
the converted session in a fixed order, leaving out its edges, which a
filtered session never publishes:

1. Strip the converter's hidden copies, which no public view shows and
   which would carry what the scripts pruned or redacted: ``raw_arguments``
   (``step.extra``, ``step.extra.tool_call_details.*``, ``tool_calls[].extra``)
   and ``tool_result_metadata``'s ``raw_tool_result`` and ``tool_use_result``.
   This runs on every filtered pass, whatever the scripts said.
2. Prune and drop: a pruned result's ``content`` becomes ``""``, its
   ``images`` go, and its ``tool_result_metadata`` keeps only ``is_error``
   (atif-analytics reads it as a fallback); a dropped step loses its
   ``reasoning_content``; a dropped event leaves ``session_events``; a dropped
   blob leaves every ``images`` list and every ``ContentPart`` list (a step's
   ``message``, a result's ``content``) whose image or audio part names it,
   a list left empty becoming ``""``, and is not published.
3. Replace: every ``find`` is replaced with its ``replacement`` in every
   string value of the session (the trajectory, embedded trajectories and
   ``extra`` included, the session events and the loss report);
   object keys are left alone. One alternation, longest ``find`` first, makes
   it a single pass: a longer match wins on overlap and replaced text is
   never matched again.
4. Publish only the blobs the session still references: a ``sha256`` field
   (an ``images`` entry) or a blob path (``blobs/sha256/<ab>/<hex>``, in a
   part's ``source.path`` or a text placeholder) in any string.

Addresses are checked against the session as converted, before any filtering:
``(trajectory_id, step_id, tool_use_id)`` must name an observation result,
``(trajectory_id, step_id)`` a step, ``event_seq`` a session event and
``blob_sha256`` a blob or an ``images`` entry. A trajectory is addressed the
way atif-duck's step views address it: an embedded trajectory by its own
``trajectory_id``, the root by its ``trajectory_id``, else the session id.
Applying a directive whose target an earlier pass already removed changes
nothing, so a second pass over a filtered session can only change it through
what its scripts newly emit.

Pure Python over contract-shaped artifacts: no DuckDB, no I/O.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence

    from atif_corpus.domain.ports import ConversionOutput

#: Each action's parameter columns, with the Python type DuckDB hands back.
ACTION_PARAMETERS: dict[str, tuple[tuple[str, type], ...]] = {
    "prune_tool_result": (("trajectory_id", str), ("step_id", int), ("tool_use_id", str)),
    "drop_reasoning": (("trajectory_id", str), ("step_id", int)),
    "drop_event": (("event_seq", int),),
    "drop_blob": (("blob_sha256", str),),
    "replace": (("find", str), ("replacement", str)),
}

#: Every column a ``directives`` relation may have.
DIRECTIVE_COLUMNS: frozenset[str] = frozenset(
    {"action", *(name for params in ACTION_PARAMETERS.values() for name, _ in params)}
)

#: The hash a blob path names, as the converter writes it in a ``ContentPart``'s
#: ``source.path`` (``../../blobs/sha256/<ab>/<hex>.<ext>``) and in a text
#: placeholder (``[file blobs/sha256/<ab>/<hex>.<ext> ...]``).
_BLOB_PATH_SHA = re.compile(r"blobs/sha256/[0-9a-f]{2}/([0-9a-f]{64})")

_COMPACT: tuple[str, str] = (",", ":")


class FilterError(ValueError):
    """A filter script's directives cannot be applied; the session fails before publication."""


@dataclass(frozen=True, slots=True)
class Directives:
    """Every script's directives for one session, merged and deduplicated."""

    #: ``(trajectory_id, step_id, tool_use_id)`` of each result to prune.
    prune_tool_results: frozenset[tuple[str, int, str]] = frozenset()
    #: ``(trajectory_id, step_id)`` of each step to drop reasoning from.
    drop_reasoning: frozenset[tuple[str, int]] = frozenset()
    #: ``seq`` of each session event to drop.
    drop_events: frozenset[int] = frozenset()
    #: Hash of each blob to drop.
    drop_blobs: frozenset[str] = frozenset()
    #: ``find`` -> ``replacement``.
    replacements: Mapping[str, str] = field(default_factory=dict)


def _typed(value: object, expected: type) -> bool:
    # bool is an int to Python, never a step id or an event seq.
    return isinstance(value, expected) and not isinstance(value, bool)


def parse_directives(
    columns: Sequence[str], rows: Iterable[Sequence[object]], *, script: int
) -> list[tuple[str, tuple[Any, ...]]]:
    """``(action, parameters)`` for each row of one script's ``directives`` relation.

    ``script`` is the script's 1-based position on the command line, named in
    every error.
    """
    names = list(columns)
    unknown = sorted(set(names) - DIRECTIVE_COLUMNS)
    if unknown:
        msg = f"filter script {script}: directives has unknown column(s) {unknown}"
        raise FilterError(msg)
    if "action" not in names:
        msg = f"filter script {script}: directives has no action column"
        raise FilterError(msg)
    parsed: list[tuple[str, tuple[Any, ...]]] = []
    for number, row in enumerate(rows, start=1):
        values = dict(zip(names, row, strict=True))
        action = values.get("action")
        if not isinstance(action, str) or action not in ACTION_PARAMETERS:
            msg = f"filter script {script}: directives row {number} has unknown action {action!r}"
            raise FilterError(msg)
        params = ACTION_PARAMETERS[action]
        owned = {name for name, _ in params}
        foreign = sorted(
            name
            for name, value in values.items()
            if name != "action" and name not in owned and value is not None
        )
        if foreign:
            msg = (
                f"filter script {script}: directives row {number} ({action}) sets "
                f"{foreign}, which {action} does not take"
            )
            raise FilterError(msg)
        arguments: list[Any] = []
        for name, expected in params:
            value = values.get(name)
            if not _typed(value, expected):
                msg = (
                    f"filter script {script}: directives row {number} ({action}) needs "
                    f"{name} as a non-null {expected.__name__}, got {value!r}"
                )
                raise FilterError(msg)
            arguments.append(value)
        if action == "replace" and arguments[0] == "":
            msg = f"filter script {script}: directives row {number} (replace) has an empty find"
            raise FilterError(msg)
        parsed.append((action, tuple(arguments)))
    return parsed


def merge_directives(scripts: Iterable[Sequence[tuple[str, tuple[Any, ...]]]]) -> Directives:
    """Union every script's directives; a ``find`` with two replacements fails."""
    prune: set[tuple[str, int, str]] = set()
    reasoning: set[tuple[str, int]] = set()
    events: set[int] = set()
    blobs: set[str] = set()
    replacements: dict[str, str] = {}
    for directives in scripts:
        for action, arguments in directives:
            if action == "prune_tool_result":
                prune.add((arguments[0], arguments[1], arguments[2]))
            elif action == "drop_reasoning":
                reasoning.add((arguments[0], arguments[1]))
            elif action == "drop_event":
                events.add(arguments[0])
            elif action == "drop_blob":
                blobs.add(arguments[0])
            else:
                find, replacement = arguments
                if replacements.get(find, replacement) != replacement:
                    msg = (
                        f"replace pairs {find!r} with both {replacements[find]!r} and "
                        f"{replacement!r}"
                    )
                    raise FilterError(msg)
                replacements[find] = replacement
    return Directives(
        prune_tool_results=frozenset(prune),
        drop_reasoning=frozenset(reasoning),
        drop_events=frozenset(events),
        drop_blobs=frozenset(blobs),
        replacements=dict(sorted(replacements.items())),
    )


# ---------------------------------------------------------------------------
# Walking the session
# ---------------------------------------------------------------------------


def _trajectories(
    trajectory: dict[str, Any], session_id: str
) -> Iterator[tuple[str | None, dict[str, Any]]]:
    """``(key, trajectory)`` for the root and every embedded trajectory, depth first."""
    root_id = trajectory.get("trajectory_id")
    stack: list[tuple[str | None, dict[str, Any]]] = [
        (root_id if isinstance(root_id, str) else session_id, trajectory)
    ]
    while stack:
        key, node = stack.pop()
        yield key, node
        children = node.get("subagent_trajectories")
        embedded = (
            [child for child in children if isinstance(child, dict)]
            if isinstance(children, list)
            else []
        )
        for child in reversed(embedded):
            child_id = child.get("trajectory_id")
            stack.append((child_id if isinstance(child_id, str) else None, child))


def _steps(
    trajectory: dict[str, Any], session_id: str
) -> Iterator[tuple[str | None, object, dict[str, Any]]]:
    """``(trajectory key, step_id, step)`` for every step of the session."""
    for key, node in _trajectories(trajectory, session_id):
        steps = node.get("steps")
        for step in steps if isinstance(steps, list) else []:
            if isinstance(step, dict):
                yield key, step.get("step_id"), step


def _results(step: dict[str, Any]) -> list[dict[str, Any]]:
    observation = step.get("observation")
    results = observation.get("results") if isinstance(observation, dict) else None
    return [res for res in results if isinstance(res, dict)] if isinstance(results, list) else []


def _dict_at(holder: dict[str, Any], key: str) -> dict[str, Any] | None:
    value = holder.get(key)
    return value if isinstance(value, dict) else None


def _drop_empty(holder: dict[str, Any], key: str) -> None:
    """Remove ``holder[key]`` when it is an empty dict or list, the converter's "said nothing"."""
    value = holder.get(key)
    if isinstance(value, (dict, list)) and not value:
        del holder[key]


def _image_hashes(holder: dict[str, Any]) -> set[str]:
    """The ``sha256`` of every ``extra.images`` entry on a step or a result."""
    extra = _dict_at(holder, "extra")
    images = extra.get("images") if extra is not None else None
    if not isinstance(images, list):
        return set()
    return {
        image["sha256"]
        for image in images
        if isinstance(image, dict) and isinstance(image.get("sha256"), str)
    }


def _event_seq(line: str) -> object:
    try:
        event = json.loads(line)
    except ValueError:
        return None
    return event.get("seq") if isinstance(event, dict) else None


@dataclass(frozen=True, slots=True)
class Targets:
    """What a directive may address in one converted session."""

    results: frozenset[tuple[str, int, str]]
    steps: frozenset[tuple[str, int]]
    events: frozenset[int]
    blobs: frozenset[str]

    @classmethod
    def of(cls, output: ConversionOutput, session_id: str) -> Targets:
        """The addresses ``output`` (converted, not yet filtered) holds."""
        results: set[tuple[str, int, str]] = set()
        steps: set[tuple[str, int]] = set()
        blobs = {blob.sha256 for blob in output.blobs}
        for key, step_id, step in _steps(output.trajectory_dict, session_id):
            if key is None or not isinstance(step_id, int) or isinstance(step_id, bool):
                continue
            steps.add((key, step_id))
            blobs.update(_image_hashes(step))
            for res in _results(step):
                call_id = res.get("source_call_id")
                if isinstance(call_id, str):
                    results.add((key, step_id, call_id))
                blobs.update(_image_hashes(res))
        events: set[int] = set()
        for line in output.events_lines:
            seq = _event_seq(line)
            if isinstance(seq, int) and not isinstance(seq, bool):
                events.add(seq)
        return cls(
            results=frozenset(results),
            steps=frozenset(steps),
            events=frozenset(events),
            blobs=frozenset(blobs),
        )

    def check(self, directives: Directives) -> None:
        """Fail on the first directive whose address names nothing in the session."""
        for missing, what in (
            (directives.prune_tool_results - self.results, "tool result"),
            (directives.drop_reasoning - self.steps, "step"),
            (directives.drop_events - self.events, "session event"),
            (directives.drop_blobs - self.blobs, "blob"),
        ):
            if missing:
                msg = f"directives address no {what} of the session: {min(missing)!r}"
                raise FilterError(msg)


# ---------------------------------------------------------------------------
# The executor
# ---------------------------------------------------------------------------


def _strip_hidden_copies(step: dict[str, Any]) -> None:
    """Remove the converter's duplicates of call arguments and raw tool results from ``step``."""
    extra = _dict_at(step, "extra")
    if extra is not None:
        extra.pop("raw_arguments", None)
        details = _dict_at(extra, "tool_call_details")
        if details is not None:
            for call_id in list(details):
                detail = details[call_id]
                if isinstance(detail, dict):
                    detail.pop("raw_arguments", None)
                    if not detail:
                        del details[call_id]
            _drop_empty(extra, "tool_call_details")
        _strip_result_metadata(extra)
        _drop_empty(step, "extra")
    calls = step.get("tool_calls")
    for call in calls if isinstance(calls, list) else []:
        call_extra = _dict_at(call, "extra") if isinstance(call, dict) else None
        if call_extra is not None:
            call_extra.pop("raw_arguments", None)
            _drop_empty(call, "extra")
    for res in _results(step):
        res_extra = _dict_at(res, "extra")
        if res_extra is not None:
            _strip_result_metadata(res_extra)
            _drop_empty(res, "extra")


def _strip_result_metadata(extra: dict[str, Any]) -> None:
    metadata = _dict_at(extra, "tool_result_metadata")
    if metadata is not None:
        metadata.pop("raw_tool_result", None)
        metadata.pop("tool_use_result", None)
        _drop_empty(extra, "tool_result_metadata")


def _prune_result(res: dict[str, Any]) -> None:
    res["content"] = ""
    extra = _dict_at(res, "extra")
    if extra is None:
        return
    extra.pop("images", None)
    metadata = _dict_at(extra, "tool_result_metadata")
    if metadata is not None:
        kept = {"is_error": metadata["is_error"]} if "is_error" in metadata else {}
        extra["tool_result_metadata"] = kept
        _drop_empty(extra, "tool_result_metadata")
    _drop_empty(res, "extra")


def _drop_images(holder: dict[str, Any], hashes: frozenset[str]) -> None:
    extra = _dict_at(holder, "extra")
    images = extra.get("images") if extra is not None else None
    if extra is None or not isinstance(images, list):
        return
    extra["images"] = [
        image for image in images if not (isinstance(image, dict) and image.get("sha256") in hashes)
    ]
    _drop_empty(extra, "images")
    _drop_empty(holder, "extra")


def _drop_media_parts(holder: dict[str, Any], key: str, hashes: frozenset[str]) -> None:
    """Remove the image and audio parts of ``holder[key]`` whose path names a dropped blob."""
    parts = holder.get(key)
    if not isinstance(parts, list):
        return
    kept = [part for part in parts if not _names_blob(part, hashes)]
    if len(kept) != len(parts):
        holder[key] = kept or ""


def _names_blob(part: object, hashes: frozenset[str]) -> bool:
    if not isinstance(part, dict) or part.get("type") not in {"image", "audio"}:
        return False
    source = part.get("source")
    path = source.get("path") if isinstance(source, dict) else None
    return isinstance(path, str) and bool(set(_BLOB_PATH_SHA.findall(path)) & hashes)


def _replacer(replacements: Mapping[str, str]) -> Callable[[str], str] | None:
    """One-pass literal replacement of every ``find``, longest first; ``None`` when there is none."""
    if not replacements:
        return None
    finds = sorted(replacements, key=lambda find: (-len(find), find))
    pattern = re.compile("|".join(re.escape(find) for find in finds))
    return lambda text: pattern.sub(lambda match: replacements[match.group(0)], text)


def _replace_strings(value: Any, substitute: Callable[[str], str]) -> Any:
    """``value`` with ``substitute`` applied to every string in it; dict keys are kept."""
    if isinstance(value, str):
        return substitute(value)
    if isinstance(value, dict):
        for key, item in value.items():
            value[key] = _replace_strings(item, substitute)
        return value
    if isinstance(value, list):
        for index, item in enumerate(value):
            value[index] = _replace_strings(item, substitute)
        return value
    return value


def _rewrite_lines(
    lines: Sequence[str], substitute: Callable[[str], str] | None, drop_seqs: frozenset[int]
) -> list[str]:
    """JSON lines minus dropped events, replaced; an untouched line keeps its bytes."""
    rewritten: list[str] = []
    for line in lines:
        record = json.loads(line)
        if drop_seqs and isinstance(record, dict) and record.get("seq") in drop_seqs:
            continue
        if substitute is None:
            rewritten.append(line)
            continue
        replaced = _replace_strings(copy.deepcopy(record), substitute)
        rewritten.append(
            line
            if replaced == record
            else json.dumps(replaced, ensure_ascii=False, separators=_COMPACT)
        )
    return rewritten


def _referenced_hashes(*values: Any) -> set[str]:
    """Every blob hash ``values`` reference: a ``sha256`` field or a blob path in a string."""
    found: set[str] = set()
    stack: list[Any] = list(values)
    while stack:
        value = stack.pop()
        if isinstance(value, str):
            if "blobs/sha256/" in value:
                found.update(_BLOB_PATH_SHA.findall(value))
        elif isinstance(value, dict):
            digest = value.get("sha256")
            if isinstance(digest, str):
                found.add(digest)
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
    return found


def apply_directives(
    output: ConversionOutput, directives: Directives, *, session_id: str
) -> ConversionOutput:
    """A filtered copy of ``output``; ``output`` itself is left as it is."""
    trajectory = copy.deepcopy(output.trajectory_dict)
    for key, step_id, step in _steps(trajectory, session_id):
        _strip_hidden_copies(step)
        if (key, step_id) in directives.drop_reasoning:
            step.pop("reasoning_content", None)
        for res in _results(step):
            if (key, step_id, res.get("source_call_id")) in directives.prune_tool_results:
                _prune_result(res)
            if directives.drop_blobs:
                _drop_images(res, directives.drop_blobs)
                _drop_media_parts(res, "content", directives.drop_blobs)
        if directives.drop_blobs:
            _drop_images(step, directives.drop_blobs)
            _drop_media_parts(step, "message", directives.drop_blobs)
    substitute = _replacer(directives.replacements)
    loss_report = copy.deepcopy(output.loss_report_dict)
    if substitute is not None:
        _replace_strings(trajectory, substitute)
        _replace_strings(loss_report, substitute)
    events = _rewrite_lines(output.events_lines, substitute, directives.drop_events)
    referenced = _referenced_hashes(trajectory, [json.loads(line) for line in events])
    return replace(
        output,
        trajectory_dict=trajectory,
        loss_report_dict=loss_report,
        edges_lines=[],
        events_lines=events,
        blobs=tuple(
            blob
            for blob in output.blobs
            if blob.sha256 in referenced and blob.sha256 not in directives.drop_blobs
        ),
    )


def serialized(output: ConversionOutput) -> tuple[object, ...]:
    """What publishing ``output`` writes, for comparing two filter passes byte for byte."""
    return (
        json.dumps(output.trajectory_dict, separators=_COMPACT),
        json.dumps(output.loss_report_dict, separators=_COMPACT),
        tuple(output.events_lines),
        tuple(sorted((blob.sha256, blob.extension) for blob in output.blobs)),
    )


__all__ = [
    "ACTION_PARAMETERS",
    "DIRECTIVE_COLUMNS",
    "Directives",
    "FilterError",
    "Targets",
    "apply_directives",
    "merge_directives",
    "parse_directives",
    "serialized",
]
