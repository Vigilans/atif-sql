# SPDX-License-Identifier: Apache-2.0

"""The parity oracle: harbor's PRIVATE converters, reachable from tests only.

atif-converter used to call ``ClaudeCode._convert_events_to_trajectory`` and
``Codex._convert_events_to_trajectory`` in production. Those are not public API
(harbor documents the ``harbor.models.trajectories`` data classes and the
validator, and nothing that converts a native session log), and the two files
that hold them took 26 and 18 upstream commits between June and September 2026.
The converters now live in this package, built on the public models; harbor's
private ones survive HERE, as the oracle a port is measured against.

Two ways to consult the oracle:

* LIVE, through :func:`harbor_claude_code_trajectory` and
  :func:`harbor_codex_trajectory`, while harbor still ships the private method.
  A test that needs it calls :func:`require_harbor_private_api` first and is
  SKIPPED, loudly, the day the method is gone — a skip here is the signal that
  the frozen goldens are now the only oracle, never a failure.
* FROZEN, through the ``goldens/`` directory beside this module: harbor's
  output for each synthetic fixture, captured while the private method
  existed, so parity stays checkable after it is removed. ``test_goldens``
  asserts the live oracle still agrees with the frozen one, which is how an
  upstream behavior change becomes a failing test instead of silent drift.

Staging is replicated here rather than imported from the adapter, because the
adapter no longer stages anything: the private methods take a DIRECTORY and
discover the transcript inside it, which is the shape this module builds.
"""

from __future__ import annotations

import json
import re
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from atif_converter.domain.trajectory_tree import iter_steps

if TYPE_CHECKING:
    from atif_converter.domain.atif import Trajectory

GOLDENS_DIR = Path(__file__).parent / "goldens"

_CONVERT_METHOD = "_convert_events_to_trajectory"

#: Where the port departs from harbor ON PURPOSE, one pattern per decision,
#: matched against :func:`diff_paths` lines. Anything not named here is still
#: a parity failure. Keep each entry narrow (a path AND a direction), so a
#: divergence the decision did not cover still fails.
#:
#: * Claude Code subagent side-files nest as ``subagent_trajectories``, the
#:   shape harbor #4076 proposes and harbor 0.26.0 does not ship: 0.26.0 reads
#:   every side-file into the root's step list. A pattern cannot forgive that
#:   narrowly (every step after the first subagent event moves), so the oracle
#:   itself carries the divergence: :func:`harbor_claude_code_trajectory`
#:   converts each transcript :func:`split_claude_code_records` keeps apart
#:   with harbor, then nests harbor's trajectories with
#:   :func:`nest_claude_code_subagents`. Every step, metric and field of every
#:   trajectory is still harbor's; only the split and the nesting are ours.
#: * A Codex spawned thread harbor's ``spawn_agent`` / ``task_name`` rule does
#:   not link is linked from the call that reported it
#:   (``atif_converter.domain.codex_nesting``), so the spawning result carries
#:   a ``subagent_trajectory_ref`` only in ours.
DELIBERATE_DIVERGENCES: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"^\$(?:\.subagent_trajectories\[\d+\])*\.steps\[\d+\]\.observation\.results\[\d+\]"
        r"\.subagent_trajectory_ref: only in ours = "
    ),
)

#: The blob placeholder :class:`atif_converter.domain.blobs.BlobRef` writes for an
#: image, as ``_media_content_forgiven`` reads it.
_IMAGE_PLACEHOLDER = re.compile(r"\[image sha256:(?P<sha>[0-9a-f]{64}) (?P<media>\S+) \d+ bytes\]")
#: harbor's path for a Codex tool output image it saved (harbor #3467).
_HARBOR_IMAGE_PATH = re.compile(r"images/codex_(?P<sha16>[0-9a-f]{16})\.(?:png|jpg|gif|webp)")
#: What the converter writes for a Codex audio item (``blobs.codex_audio_text``): its
#: blob placeholder, or ``[audio: <file_id-or-url>]``.
_AUDIO_TEXT = re.compile(r"\[audio sha256:[0-9a-f]{64} audio/\S+ \d+ bytes\]|\[audio: [^\]]+\]")
#: The audio block types harbor keeps as their JSON text in a tool output.
_AUDIO_BLOCK_TYPES = frozenset({"audio", "input_audio"})

#: ``agent.extra`` lists harbor builds from a Python ``set``, so its order
#: follows the process hash seed; ours keep first-seen order (see
#: ``_agent_extra`` in ``domain/claude_code_conversion.py``). Their element
#: paths are forgiven only when the two lists hold the same values, so a
#: missing, extra or changed value still fails parity.
SET_ORDER_DIVERGENCE_KEYS: tuple[str, ...] = ("cwds", "git_branches", "agent_ids")


def harbor_has_private_api(agent: str) -> bool:
    """Whether harbor still exposes the private converter for ``agent``."""
    if agent == "claude-code":
        from harbor.agents.installed.claude_code import ClaudeCode  # type: ignore[import-untyped]

        return callable(getattr(ClaudeCode, _CONVERT_METHOD, None))
    if agent == "codex":
        from harbor.agents.installed.codex import Codex  # type: ignore[import-untyped]

        return callable(getattr(Codex, _CONVERT_METHOD, None))
    msg = f"unknown agent {agent!r}"
    raise ValueError(msg)


def require_harbor_private_api(agent: str) -> None:
    """Skip the calling test when the live oracle is gone; never fail on it."""
    if not harbor_has_private_api(agent):
        pytest.skip(
            f"harbor no longer exposes {agent}'s private converter; the frozen "
            f"goldens under {GOLDENS_DIR} are now the only parity oracle"
        )


def _stage_claude_code_session(
    session_jsonl: Path, staging: Path, *, include_subagents: bool
) -> Path:
    """Symlink one session into the ``<projects>/<slug>/`` layout harbor discovers.

    Per-FILE symlinks under REAL directories (Python 3.13's ``rglob`` does not
    descend symlinked directories), and every side-file — including the
    workflow-nested ones harbor's own discovery cannot see — staged FLAT with
    ``__``-joined names. This is the staging the production adapter used to do.
    """
    session_dir = staging / "sessions" / "projects" / (session_jsonl.parent.name or "-unknown")
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / session_jsonl.name).symlink_to(session_jsonl.resolve())
    side_dir = session_jsonl.parent / session_jsonl.stem
    if include_subagents and side_dir.is_dir():
        staged_subagents = session_dir / side_dir.name / "subagents"
        for side_file in sorted(side_dir.rglob("*.jsonl")):
            rel_parts = side_file.relative_to(side_dir).parts
            if rel_parts and rel_parts[0] == "subagents":
                rel_parts = rel_parts[1:]
            staged = staged_subagents / "__".join(rel_parts)
            staged.parent.mkdir(parents=True, exist_ok=True)
            staged.symlink_to(side_file.resolve())
    return session_dir


def _harbor_claude_code_session(
    session_jsonl: Path, *, include_subagents: bool
) -> dict[str, Any] | None:
    """harbor's conversion of one Claude Code session staged as it is on disk."""
    from harbor.agents.installed.claude_code import ClaudeCode  # type: ignore[import-untyped]

    with tempfile.TemporaryDirectory(prefix="atif-oracle-") as scratch:
        staging = Path(scratch)
        session_dir = _stage_claude_code_session(
            session_jsonl, staging, include_subagents=include_subagents
        )
        logs_dir = staging / "logs"
        logs_dir.mkdir()
        trajectory = getattr(ClaudeCode(logs_dir=logs_dir), _CONVERT_METHOD)(session_dir)
    return None if trajectory is None else trajectory.to_json_dict()


def _harbor_claude_code_events(session_jsonl: Path, events: list[Any]) -> Trajectory | None:
    """harbor's conversion of one transcript's events, staged alone under the session's name."""
    from atif_converter.domain.atif import Trajectory

    with tempfile.TemporaryDirectory(prefix="atif-oracle-part-") as scratch:
        part = Path(scratch) / (session_jsonl.parent.name or "-unknown") / session_jsonl.name
        part.parent.mkdir(parents=True)
        part.write_text(
            "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in events),
            encoding="utf-8",
        )
        trajectory = _harbor_claude_code_session(part, include_subagents=False)
    return None if trajectory is None else Trajectory.model_validate(trajectory)


def harbor_claude_code_trajectory(
    session_jsonl: Path, *, include_subagents: bool = True
) -> dict[str, Any] | None:
    """harbor's own conversion of one Claude Code session, as ``to_json_dict()``.

    A session with side-files is split and nested the way
    :data:`DELIBERATE_DIVERGENCES` describes: each transcript is harbor's
    conversion of that transcript alone.
    """
    from atif_converter.domain.claude_code_conversion import (
        nest_claude_code_subagents,
        split_claude_code_records,
    )
    from atif_converter.infrastructure.claude_code_converter import (
        discover_side_files,
        read_session_records,
    )

    side_files = discover_side_files(session_jsonl, include_subagents=include_subagents)
    if not side_files:
        return _harbor_claude_code_session(session_jsonl, include_subagents=include_subagents)
    transcripts = split_claude_code_records(
        read_session_records(session_jsonl),
        {name: read_session_records(path) for name, path in side_files.items()},
    )
    flat = [*transcripts.root, *(e for events in transcripts.children.values() for e in events)]
    root = _harbor_claude_code_events(session_jsonl, transcripts.root) if transcripts.root else None
    if root is None:
        flat_trajectory = _harbor_claude_code_events(session_jsonl, flat)
        return None if flat_trajectory is None else flat_trajectory.to_json_dict()
    children: dict[str, Trajectory] = {}
    for agent_id, events in transcripts.children.items():
        child = _harbor_claude_code_events(session_jsonl, events)
        if child is not None:
            child.trajectory_id = agent_id
            children[agent_id] = child
    nest_claude_code_subagents(root, children)
    return root.to_json_dict()


def harbor_codex_trajectory(rollout_jsonl: Path) -> dict[str, Any] | None:
    """harbor's own conversion of one Codex rollout, staged alone, as ``to_json_dict()``."""
    from harbor.agents.installed.codex import Codex  # type: ignore[import-untyped]

    with tempfile.TemporaryDirectory(prefix="atif-oracle-") as scratch:
        staging = Path(scratch)
        session_dir = staging / "sessions"
        session_dir.mkdir()
        (session_dir / rollout_jsonl.name).symlink_to(rollout_jsonl.resolve())
        logs_dir = staging / "logs"
        logs_dir.mkdir()
        trajectory = getattr(Codex(logs_dir=logs_dir), _CONVERT_METHOD)(session_dir)
    return None if trajectory is None else trajectory.to_json_dict()


def harbor_codex_session_trajectory(rollouts: list[Path]) -> dict[str, Any] | None:
    """harbor's own conversion of a Codex session's rollouts staged together, as ``to_json_dict()``.

    harbor 0.26.0 reads every rollout under the session directory, converts
    the root thread and nests each spawned thread in its spawner.
    """
    from harbor.agents.installed.codex import Codex  # type: ignore[import-untyped]

    with tempfile.TemporaryDirectory(prefix="atif-oracle-") as scratch:
        staging = Path(scratch)
        session_dir = staging / "sessions"
        session_dir.mkdir()
        for rollout in rollouts:
            (session_dir / rollout.name).symlink_to(rollout.resolve())
        logs_dir = staging / "logs"
        logs_dir.mkdir()
        trajectory = getattr(Codex(logs_dir=logs_dir), _CONVERT_METHOD)(session_dir)
    return None if trajectory is None else trajectory.to_json_dict()


def golden_path(name: str) -> Path:
    """Where the frozen oracle output for one named fixture lives."""
    return GOLDENS_DIR / f"{name}.trajectory.json"


def load_golden(name: str) -> dict[str, Any]:
    return json.loads(golden_path(name).read_text(encoding="utf-8"))


def write_golden(name: str, trajectory: dict[str, Any]) -> None:
    golden_path(name).write_text(
        json.dumps(trajectory, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def diff_paths(expected: Any, actual: Any, path: str = "$") -> list[str]:
    """Every JSON path where two trajectories disagree, with both values.

    A flat list rather than an assertion, so a parity failure names ALL the
    divergent fields at once instead of one per test run.
    """
    if isinstance(expected, dict) and isinstance(actual, dict):
        out: list[str] = []
        for key in sorted(set(expected) | set(actual)):
            if key not in expected:
                out.append(f"{path}.{key}: only in ours = {actual[key]!r}")
            elif key not in actual:
                out.append(f"{path}.{key}: only in harbor = {expected[key]!r}")
            else:
                out.extend(diff_paths(expected[key], actual[key], f"{path}.{key}"))
        return out
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{path}: length harbor={len(expected)} ours={len(actual)}"]
        out = []
        for index, (e, a) in enumerate(zip(expected, actual, strict=True)):
            out.extend(diff_paths(e, a, f"{path}[{index}]"))
        return out
    if expected != actual:
        return [f"{path}: harbor={expected!r} ours={actual!r}"]
    return []


#: JSON paths where our converter departs from harbor ON PURPOSE when a
#: session holds a model harbor prices at a fabricated $0 (a rate-less litellm
#: entry) or one we price from a local override (``pricing.override_models()``). harbor sums
#: litellm's ``(0.0, 0.0)``; we report no estimate, or the override's price.
PRICING_DIVERGENCE_PATHS: frozenset[str] = frozenset(
    {"$.final_metrics.total_cost_usd", "$.final_metrics.extra.cost_source"}
)


#: The same policy at the step level (harbor 0.23.0 prices every step): a step
#: whose model harbor prices at a fabricated $0 carries no ``cost_usd`` and no
#: label in ours, and a step priced from a local override carries the override's
#: price and label. ``{i}`` is the step index.
STEP_PRICING_DIVERGENCE_PATHS: tuple[str, ...] = (
    "$.steps[{i}].metrics.cost_usd",
    "$.steps[{i}].metrics.extra.cost_source",
)


def _diff_path(diff: str) -> str:
    return diff.split(":", 1)[0]


def _policy_prices_differently(model: object) -> bool:
    """Whether the pricing policy prices ``model`` differently from harbor's litellm.

    True for a model priced from a local override, and for one the vendored
    table can't price (harbor writes litellm's $0 there, we write nothing).
    """
    from atif_converter.domain import pricing

    if not isinstance(model, str):
        return False
    if pricing.is_local_override(model):
        return True
    try:
        priced = pricing.priced_cost_per_token(
            model=model,
            prompt_tokens=1,
            completion_tokens=1,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )
    except Exception:  # noqa: BLE001 - a model nobody prices is not a policy divergence
        return False
    return priced is None


def _step_pricing_forgiven(expected: dict[str, Any], ours: dict[str, Any]) -> frozenset[str]:
    """The step-level :data:`STEP_PRICING_DIVERGENCE_PATHS` the pricing policy explains.

    Checked per step against that step's own model, so a priced step that
    regressed still fails parity even in a session holding an unpriced one.
    """
    forgiven: set[str] = set()
    harbor_steps = expected.get("steps") or []
    for index, step in enumerate(ours.get("steps") or []):
        if not _policy_prices_differently(step.get("model_name")):
            continue
        forgiven.update(path.format(i=index) for path in STEP_PRICING_DIVERGENCE_PATHS)
        # When the label was harbor's only metrics.extra key, ours has no extra
        # at all, and the diff names the parent path instead.
        harbor_step = harbor_steps[index] if index < len(harbor_steps) else {}
        harbor_extra = (harbor_step.get("metrics") or {}).get("extra")
        if isinstance(harbor_extra, dict) and set(harbor_extra) == {"cost_source"}:
            forgiven.add(f"$.steps[{index}].metrics.extra")
    return frozenset(forgiven)


def _pricing_divergence_expected(ours: dict[str, Any]) -> bool:
    """Whether ``ours`` is a trajectory the pricing policy SHOULD price differently from harbor.

    Two cases, each checked against the policy itself so a priced session that
    regressed to ``None`` still fails parity: an estimate labeled as using the
    local overrides, or no estimate while some agent step's model has no price.
    """
    from atif_converter.domain import pricing

    final_metrics = ours.get("final_metrics") or {}
    extra = final_metrics.get("extra") or {}
    if extra.get("cost_source") == pricing.COST_SOURCE_WITH_OVERRIDES:
        return True
    if final_metrics.get("total_cost_usd") is not None:
        return False
    # A total includes its embedded trajectories' costs, so their steps count.
    models = {
        model
        for model in (
            step.get("model_name") for step in iter_steps(ours) if step.get("source") == "agent"
        )
        if isinstance(model, str)
    }
    for model in models:
        try:
            priced = pricing.priced_cost_per_token(
                model=model,
                prompt_tokens=1,
                completion_tokens=1,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
            )
        except Exception:  # noqa: BLE001, S112 - a model nobody prices is not a policy divergence
            continue
        if priced is None:
            return True
    return False


def _set_order_forgiven(expected: dict[str, Any], ours: dict[str, Any]) -> tuple[str, ...]:
    """Path prefixes of the :data:`SET_ORDER_DIVERGENCE_KEYS` lists that differ only in order."""
    harbor_extra = (expected.get("agent") or {}).get("extra") or {}
    our_extra = (ours.get("agent") or {}).get("extra") or {}
    forgiven: list[str] = []
    for key in SET_ORDER_DIVERGENCE_KEYS:
        theirs, mine = harbor_extra.get(key), our_extra.get(key)
        if isinstance(theirs, list) and isinstance(mine, list) and sorted(theirs) == sorted(mine):
            forgiven.append(f"$.agent.extra.{key}[")
    return tuple(forgiven)


def _our_audio_line(line: str) -> str:
    """``line`` as ours renders it, when it is the JSON text of an audio block harbor kept.

    harbor 0.26.0 keeps a Codex tool output's audio block as its JSON text, one
    line per block; the converter writes ``blobs.codex_audio_text`` of the same
    block (its blob placeholder, or ``[audio: <file_id-or-url>]``) and keeps the
    JSON text of a block that function cannot read.
    """
    from atif_converter.domain.blobs import codex_audio_text

    try:
        block = json.loads(line)
    except json.JSONDecodeError:
        return line
    if not isinstance(block, dict) or block.get("type") not in _AUDIO_BLOCK_TYPES:
        return line
    audio_text = codex_audio_text(block)
    return line if audio_text is None else audio_text


def _harbor_image_parts_text(theirs: list[Any]) -> str | None:
    """harbor's image-bearing parts joined by newlines, each image in marker form, else ``None``."""
    if not any(isinstance(part, dict) and part.get("type") == "image" for part in theirs):
        return None
    texts: list[str] = []
    for part in theirs:
        if not isinstance(part, dict):
            return None
        if part.get("type") == "text" and isinstance(part.get("text"), str):
            texts.append(part["text"])
            continue
        source = part.get("source") or {}
        path = _HARBOR_IMAGE_PATH.fullmatch(str(source.get("path", "")))
        if part.get("type") != "image" or path is None:
            return None
        texts.append(f"\0{path['sha16']}\0{source.get('media_type')}\0")
    return "\n".join(texts)


def _media_content_matches(theirs: Any, ours: Any) -> bool:
    """Whether ``ours`` is harbor's tool output, each image and audio block in our text form.

    harbor 0.24.0 saves a Codex tool output's inline image to a file and returns
    the content as parts; our pure converter writes the blob placeholder text
    instead and joins the parts with newlines, as harbor joins a text-only list
    (see ``_tool_output_content`` in ``domain/codex_conversion.py``). Each
    image placeholder must name the same bytes harbor saved (its SHA-256 starts
    with the 16 hex digits of harbor's file name) and the same media type. Each
    audio block harbor kept as JSON text must be what the converter renders for
    that block, and every other text part must match exactly.
    """
    if not isinstance(ours, str):
        return False
    harbor_text = (
        theirs
        if isinstance(theirs, str)
        else _harbor_image_parts_text(theirs)
        if isinstance(theirs, list)
        else None
    )
    if harbor_text is None:
        return False
    # A JSON text never holds a raw newline, so each audio block is one line.
    expected = "\n".join(_our_audio_line(line) for line in harbor_text.split("\n"))
    # Rebuild ours with each image placeholder in the same marker form, then compare whole.
    rebuilt = _IMAGE_PLACEHOLDER.sub(lambda m: f"\0{m['sha'][:16]}\0{m['media']}\0", ours)
    return rebuilt == expected


def _audio_message_forgiven(expected: dict[str, Any], ours: dict[str, Any]) -> frozenset[str]:
    """The ``steps[i].message`` paths where ours adds only the text of ``input_audio`` items.

    harbor's ``_extract_message_text`` drops a message's ``input_audio`` items;
    ours writes each as its blob placeholder or ``[audio: <file_id-or-url>]``
    (``codex_enrichment.message_text``). With those texts removed, ours must be
    harbor's message exactly.
    """
    forgiven: set[str] = set()
    harbor_steps = expected.get("steps") or []
    for index, (theirs, mine) in enumerate(
        zip(harbor_steps, ours.get("steps") or [], strict=False)
    ):
        harbor_message, our_message = theirs.get("message"), mine.get("message")
        if not isinstance(harbor_message, str) or not isinstance(our_message, str):
            continue
        stripped, removed = _AUDIO_TEXT.subn("", our_message)
        if removed and stripped == harbor_message:
            forgiven.add(f"$.steps[{index}].message")
    return frozenset(forgiven)


def _media_content_forgiven(expected: dict[str, Any], ours: dict[str, Any]) -> frozenset[str]:
    """The ``observation.results[j].content`` paths :func:`_media_content_matches` explains."""
    forgiven: set[str] = set()
    harbor_steps = expected.get("steps") or []
    for index, step in enumerate(ours.get("steps") or []):
        if index >= len(harbor_steps):
            break
        our_results = (step.get("observation") or {}).get("results") or []
        harbor_results = (harbor_steps[index].get("observation") or {}).get("results") or []
        for position, (theirs, mine) in enumerate(zip(harbor_results, our_results, strict=False)):
            if _media_content_matches(theirs.get("content"), mine.get("content")):
                forgiven.add(f"$.steps[{index}].observation.results[{position}].content")
    return frozenset(forgiven)


def _trajectory_pairs(
    expected: dict[str, Any], ours: dict[str, Any], prefix: str = "$"
) -> Iterator[tuple[str, dict[str, Any], dict[str, Any]]]:
    """Each trajectory of the two trees with its counterpart and JSON path, parents first."""
    yield prefix, expected, ours
    pairs = zip(
        expected.get("subagent_trajectories") or [],
        ours.get("subagent_trajectories") or [],
        strict=False,
    )
    for index, (harbor_child, our_child) in enumerate(pairs):
        if isinstance(harbor_child, dict) and isinstance(our_child, dict):
            yield from _trajectory_pairs(
                harbor_child, our_child, f"{prefix}.subagent_trajectories[{index}]"
            )


def _rebase(paths: Iterable[str], prefix: str) -> set[str]:
    """``paths``, written from a trajectory's ``$``, as paths below ``prefix``."""
    return {prefix + path.removeprefix("$") for path in paths}


def parity_diffs(expected: dict[str, Any], ours: dict[str, Any]) -> list[str]:
    """:func:`diff_paths` minus the documented deliberate divergences.

    Six kinds are forgiven. Every line a :data:`DELIBERATE_DIVERGENCES`
    pattern matches, always; a Codex tool output's content where harbor saved
    an image or kept an audio block's JSON text and ours holds the converter's
    text for it, only when :func:`_media_content_matches` holds; a step message
    that differs only by the text of the ``input_audio`` items harbor drops,
    only when :func:`_audio_message_forgiven` says so; the element lines of a
    :data:`SET_ORDER_DIVERGENCE_KEYS` list, only when both sides hold the same
    values; the :data:`PRICING_DIVERGENCE_PATHS`, only when
    :func:`_pricing_divergence_expected` says the pricing policy explains them;
    and a step's :data:`STEP_PRICING_DIVERGENCE_PATHS`, only when that step's
    model is one the policy prices differently. All but the patterns are
    judged per trajectory, for the root and each embedded trajectory alike.
    """
    step_forgiven: set[str] = set()
    order_forgiven: list[str] = []
    forgiven: set[str] = set()
    for prefix, harbor_node, our_node in _trajectory_pairs(expected, ours):
        step_forgiven |= _rebase(
            _step_pricing_forgiven(harbor_node, our_node)
            | _media_content_forgiven(harbor_node, our_node)
            | _audio_message_forgiven(harbor_node, our_node),
            prefix,
        )
        order_forgiven.extend(_rebase(_set_order_forgiven(harbor_node, our_node), prefix))
        if _pricing_divergence_expected(our_node):
            # When the label was harbor's only final_metrics.extra key, ours has
            # no extra at all, and the diff names the parent path instead.
            harbor_extra = (harbor_node.get("final_metrics") or {}).get("extra")
            label_only = isinstance(harbor_extra, dict) and set(harbor_extra) == {"cost_source"}
            forgiven |= _rebase(
                PRICING_DIVERGENCE_PATHS
                | ({"$.final_metrics.extra"} if label_only else frozenset()),
                prefix,
            )
    order_prefixes = tuple(order_forgiven)
    return [
        line
        for line in diff_paths(expected, ours)
        if not any(pattern.match(line) for pattern in DELIBERATE_DIVERGENCES)
        and _diff_path(line) not in step_forgiven
        and not (order_prefixes and _diff_path(line).startswith(order_prefixes))
        and _diff_path(line) not in forgiven
    ]
