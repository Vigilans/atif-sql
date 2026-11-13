# SPDX-License-Identifier: Apache-2.0

"""The Codex conversion seam: our converter in, a validated ATIF dict out.

The raw rollout -> ``Trajectory`` conversion is ours, in
:mod:`atif_converter.domain.codex_conversion` via
:mod:`atif_converter.infrastructure.codex_converter`, ported from harbor 0.22.0
and held to parity with it by the oracle in this package's tests. harbor itself
supplies only the public data classes and the validator. One rollout in, one
trajectory out, by construction: the converter reads the file it is given, so
the directory-globbing hazard the old wrapper staged around no longer exists.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from atif_converter.domain.codex_threads import rollout_ids
from atif_converter.domain.errors import (
    ConversionError,
    EmptySessionError,
    InvalidSessionInput,
)
from atif_converter.infrastructure.harbor_adapter import (
    ConversionResult,
    validate_trajectory,
)
from atif_converter.infrastructure.raw_records import LoadedSession

if TYPE_CHECKING:
    from atif_converter.domain.atif import Trajectory


def codex_session_id(rollout_jsonl: Path) -> str:
    """The session id a rollout filename carries: its thread uuid.

    ``rollout-<ISO-ish timestamp>-<uuid>[_<segment>].jsonl[.zst]`` — the
    thread uuid equals the ``id`` inside the file's ``session_meta`` record
    (verified 2026-09-11 over the local rollouts), and a continuation segment
    carries the thread it continues. Read from the NAME rather than the
    contents because discovery needs a session id before anything parses the
    file.

    Raises
    ------
        InvalidSessionInput: the name is not a rollout filename.
    """
    ids = rollout_ids(rollout_jsonl.name)
    if ids is None:
        msg = f"not a Codex rollout filename (expected rollout-<ts>-<uuid>.jsonl): {rollout_jsonl}"
        raise InvalidSessionInput(msg)
    return ids[0]


def convert_codex_session(rollout_jsonl: Path) -> ConversionResult:
    """Convert one Codex rollout JSONL into a validated ATIF trajectory.

    Raises
    ------
        InvalidSessionInput: not an existing ``.jsonl`` file.
        EmptySessionError: the converter found no convertible events.
        ConversionError: any unexpected failure inside the converter.
    """
    from atif_converter.infrastructure.codex_converter import convert_codex_rollout

    if not rollout_jsonl.name.endswith((".jsonl", ".jsonl.zst")) or not rollout_jsonl.is_file():
        msg = f"not a rollout JSONL file: {rollout_jsonl}"
        raise InvalidSessionInput(msg)

    try:
        trajectory = convert_codex_rollout(rollout_jsonl)
    except (
        Exception
    ) as exc:  # the converter is a port of untyped upstream code; classify at the seam
        msg = f"Codex conversion failed for {rollout_jsonl}"
        raise ConversionError(msg) from exc

    if trajectory is None:
        msg = f"no convertible events in {rollout_jsonl}"
        raise EmptySessionError(msg)

    return _result_from(trajectory, rollout_jsonl)


def convert_loaded_codex_session(loaded: LoadedSession) -> ConversionResult:
    """Convert an already-read rollout into a validated ATIF trajectory.

    The records are :func:`~atif_converter.infrastructure.harbor_adapter.read_session`'s;
    nothing is read from disk here.

    Raises
    ------
        EmptySessionError: the converter found no convertible events.
        ConversionError: any unexpected failure inside the converter.
    """
    from atif_converter.infrastructure.codex_converter import convert_loaded_codex_rollout

    rollout_jsonl = loaded.snapshot.session_jsonl
    try:
        trajectory = convert_loaded_codex_rollout(loaded)
    except (
        Exception
    ) as exc:  # the converter is a port of untyped upstream code; classify at the seam
        msg = f"Codex conversion failed for {rollout_jsonl}"
        raise ConversionError(msg) from exc
    return _result_from(trajectory, rollout_jsonl)


def convert_codex_stream(records: list[dict[str, Any]], rollout_jsonl: Path) -> ConversionResult:
    """Convert a session's assembled record stream into a validated ATIF trajectory.

    ``records`` are :func:`~atif_converter.domain.codex_threads.assemble_codex_session`'s
    stream, read from ``rollout_jsonl`` (the session's first rollout) and the
    session's other rollouts. A session of one rollout passes that rollout's
    records, which converts exactly as :func:`convert_loaded_codex_session` does.

    Raises
    ------
        EmptySessionError: the converter found no convertible events.
        ConversionError: any unexpected failure inside the converter.
    """
    from atif_converter.infrastructure.codex_converter import convert_codex_record_list

    try:
        trajectory = convert_codex_record_list(records, rollout_jsonl)
    except (
        Exception
    ) as exc:  # the converter is a port of untyped upstream code; classify at the seam
        msg = f"Codex conversion failed for {rollout_jsonl}"
        raise ConversionError(msg) from exc
    return _result_from(trajectory, rollout_jsonl)


def _result_from(trajectory: Trajectory | None, rollout_jsonl: Path) -> ConversionResult:
    if trajectory is None:
        msg = f"no convertible events in {rollout_jsonl}"
        raise EmptySessionError(msg)

    trajectory_dict: dict[str, Any] = trajectory.model_dump(mode="json", exclude_none=True)

    errors = validate_trajectory(trajectory_dict)
    if errors:
        logger.warning("codex trajectory for {} failed validation: {}", rollout_jsonl, errors)

    return ConversionResult(trajectory=trajectory_dict, validation_errors=errors)
