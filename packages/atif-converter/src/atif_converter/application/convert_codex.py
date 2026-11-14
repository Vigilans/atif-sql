# SPDX-License-Identifier: Apache-2.0

"""Use case: convert one Codex rollout AND account for what conversion lost.

The Codex counterpart of
:mod:`atif_converter.application.convert_and_audit`, composed the same way and
under the same discipline: the rollout is read once, fingerprinted and parsed
in the same pass, the converter and the audit consume that one list of
records, and the fingerprints are re-checked once the artifacts are built, so
the census, the trajectory and ``edges.jsonl`` all describe the same bytes of a
rollout that Codex may still be appending to.

What differs is only what a rollout IS. No side-file tree, a record taxonomy
of its own, and a set of gaps of its own — so the census, the edges and the
enrichment come from the Codex modules while the snapshot machinery, the error
taxonomy and the returned ``(ConversionResult, LossReport)`` pair are shared
with the Claude Code path.

A session can span several rollouts: a thread's continuation segments and the
rollouts of the sub-agent threads it spawned, passed as ``source_files``.
:mod:`atif_converter.domain.codex_threads` orders them into one record stream,
and every artifact is built from that stream. Each thread converts into a
trajectory of its own, and each spawned thread's trajectory is embedded in the
``subagent_trajectories`` of the thread that spawned it; each trajectory is
enriched from its own thread's records, and its steps are then marked
reverted from the records behind them.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from atif_converter.domain.blobs import BlobCollector, extract_codex_blobs
from atif_converter.domain.codex_edges import build_codex_edges, codex_edges_jsonl_lines
from atif_converter.domain.codex_enrichment import enrich_codex_trajectory
from atif_converter.domain.codex_fidelity import (
    CODEX_STRUCTURAL_GAPS,
    CodexFidelityGap,
    CodexRecordType,
)
from atif_converter.domain.codex_threads import (
    CodexSessionStream,
    RolloutFile,
    assemble_codex_session,
    mark_codex_session,
    thread_records,
)
from atif_converter.domain.errors import SourceMutatedDuringConversion
from atif_converter.domain.fidelity import AnyRecordType, LossReport
from atif_converter.domain.result_signals import annotate_codex_trajectory
from atif_converter.domain.session_events import (
    codex_session_events,
    session_events_jsonl_lines,
)
from atif_converter.domain.trajectory_tree import iter_trajectories
from atif_converter.infrastructure.codex_adapter import convert_codex_stream
from atif_converter.infrastructure.codex_census import (
    CodexSessionCensus,
    codex_census_from_records,
)
from atif_converter.infrastructure.harbor_adapter import (
    ConversionResult,
    read_session,
    validate_trajectory,
)
from atif_converter.infrastructure.raw_records import SessionSnapshot, mutated_files
from atif_converter.infrastructure.source_archive import SourceArchiveWriter

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

#: Response-item payload types whose presence in a rollout means a specific
#: content-shaped gap was actually exercised, rather than merely being possible.
_ITEM_TYPE_GAPS: dict[str, CodexFidelityGap] = {
    "reasoning": CodexFidelityGap.REASONING_ENCRYPTED_DROPPED,
    "tool_search_call": CodexFidelityGap.TOOL_SEARCH_CALLS_DROPPED,
    "tool_search_output": CodexFidelityGap.TOOL_SEARCH_CALLS_DROPPED,
}


def _loss_report(
    census: CodexSessionCensus,
    *,
    developer_messages: int,
    captured: int = 0,
    spawned_files: int = 0,
) -> LossReport:
    """Loss accounting for one rollout: total records vs items that convert.

    ``records_converted`` counts the CONVERTIBLE RESPONSE ITEMS, not every
    response item and not every record: ``event_msg`` and ``turn_context``
    records are read for token counts and turn ids while producing no step of
    their own, so counting them as converted would claim the trajectory
    carries content it does not.
    """
    total = sum(census.record_counts.values())
    converted = census.convertible_items
    dropped = total - converted - captured

    gaps = set(CODEX_STRUCTURAL_GAPS)
    if dropped:
        gaps.add(CodexFidelityGap.NON_ITEM_RECORDS_DROPPED)
    if census.record_counts.get(CodexRecordType.COMPACTED):
        gaps.add(CodexFidelityGap.COMPACTION_UNHANDLED)
    if developer_messages:
        gaps.add(CodexFidelityGap.DEVELOPER_ROLE_FLATTENED_TO_SYSTEM)
    for item_type, gap in _ITEM_TYPE_GAPS.items():
        if census.item_type_counts.get(item_type):
            gaps.add(gap)

    # Named rather than inlined: LossReport is shared by both agents, so its
    # key type is the UNION of the two record taxonomies and a same-typed dict
    # of one agent's members does not match the constructor overload without
    # being widened here.
    record_counts: dict[AnyRecordType, int] = {**census.record_counts}
    return LossReport(
        record_counts=record_counts,
        records_converted=converted,
        records_captured=captured,
        records_dropped=dropped,
        gaps_observed=frozenset(gaps),
        # A Codex sub-agent writes rollouts of its own; the ones read with
        # this session are its subagent files, and each one converts.
        subagent_files_found=spawned_files,
        subagent_files_convertible=spawned_files,
        workflow_subagent_files_found=0,
    )


def _developer_message_count(records: list[tuple[dict[str, object], str]]) -> int:
    """How many ``developer``-role messages the rollout carried."""
    count = 0
    for record, _source_file in records:
        if record.get("type") != "response_item":
            continue
        payload = record.get("payload")
        if not isinstance(payload, dict):
            continue
        if payload.get("type") == "message" and payload.get("role") == "developer":
            count += 1
    return count


def _enrich_threads(
    trajectory: dict[str, Any], stream: CodexSessionStream, record_keys: Sequence[str]
) -> None:
    """Enrich each thread's trajectory from that thread's records, in place.

    A trajectory names its thread in ``trajectory_id``; the root names it only
    when it has children, so a trajectory without one is the root thread's.
    The record ids are the session's edge ids, so ``source_uuids`` join
    ``edges.jsonl`` whichever trajectory they sit in.
    """
    positions = thread_records(stream)
    for thread_trajectory in iter_trajectories(trajectory):
        thread_id = thread_trajectory.get("trajectory_id") or stream.root_thread_id
        thread_positions = positions.get(str(thread_id), [])
        enrich_codex_trajectory(
            thread_trajectory,
            [stream.records[position] for position in thread_positions],
            copy_input=False,
            record_ids=[record_keys[position] for position in thread_positions],
        )


def _refuse_if_mutated(
    snapshot: SessionSnapshot, archive: SourceArchiveWriter | None = None
) -> None:
    mutated = mutated_files(snapshot, archive=archive)
    if not mutated:
        if archive is not None:
            # The verifying read already archived the transcripts AND the
            # parsed sidecars; the sweep takes only files nothing parsed.
            archive.archive_side_files(
                snapshot.session_jsonl,
                already=(*snapshot.files, *snapshot.sidecar_fingerprints),
            )
        return
    logger.warning(
        "convert_codex: {} source file(s) changed while converting {}; refusing",
        len(mutated),
        snapshot.session_jsonl,
    )
    raise SourceMutatedDuringConversion(snapshot.session_jsonl, mutated)


def convert_codex_and_audit(
    rollout_jsonl: Path,
    *,
    source_files: Sequence[Path] = (),
    archive: SourceArchiveWriter | None = None,
) -> tuple[ConversionResult, LossReport]:
    """Convert ``rollout_jsonl`` to ATIF and produce its loss accounting.

    ``rollout_jsonl`` is the session's first rollout; ``source_files`` are its
    other rollouts (continuation segments and spawned threads' rollouts).

    The returned :class:`ConversionResult` carries the ENRICHED trajectory
    (``source_uuids`` on steps, ``cache_creation_total`` and the compaction
    count on the trajectory — see
    :func:`~atif_converter.domain.codex_enrichment.enrich_codex_trajectory`)
    plus the ready-to-write ``edges.jsonl`` lines derived from the RAW records.
    Validation is re-run post-enrichment.

    With ``archive``, the verifying re-read also streams every source file
    into that writer (see
    :mod:`atif_converter.infrastructure.source_archive`): the archive costs
    no extra open of a transcript and holds exactly the bytes that were
    parsed. It is complete only when this function returns; on any raise the
    caller must discard it.

    Raises
    ------
        InvalidSessionInput: ``rollout_jsonl`` is not an existing ``.jsonl`` file.
        SourceMutatedDuringConversion: the rollout changed between the
            read and the end of the audit, so the artifacts would not
            describe the rollout as it stands.
    """
    loaded = read_session(rollout_jsonl, source_files)
    snapshot = loaded.snapshot
    base = snapshot.base
    stream = assemble_codex_session(
        [
            RolloutFile(
                source_file=path.relative_to(base).as_posix(),
                records=[record for record in records if isinstance(record, dict)],
            )
            for path, records in loaded.records_by_file.items()
        ]
    )
    records = stream.records
    sidechain_files = stream.sidechain_files
    # Attachments out BEFORE conversion, named by the same record ids the
    # enrichment pass writes into ``source_uuids`` (an edge's uuid depends on
    # the record's ids and position, never on the content being rewritten).
    record_keys = [
        str(edge["uuid"]) for edge in build_codex_edges(records, sidechain_files=sidechain_files)
    ]
    collector = BlobCollector()
    blob_index = extract_codex_blobs(
        [record for record, _src in records], collector, record_keys=record_keys
    )
    result = convert_codex_stream(stream, rollout_jsonl)

    census = codex_census_from_records(rollout_jsonl, records)
    events = codex_session_events(records, record_ids=record_keys, sidechain_files=sidechain_files)
    report = _loss_report(
        census,
        developer_messages=_developer_message_count(records),
        captured=len(events),
        spawned_files=len(sidechain_files),
    )
    edges_lines = tuple(codex_edges_jsonl_lines(records, sidechain_files=sidechain_files))
    events_lines = tuple(session_events_jsonl_lines(events))
    # The harbor trajectory has no other consumer, so enrich in place rather
    # than deep-copying a whole rollout's worth of steps.
    enriched = result.trajectory
    _enrich_threads(enriched, stream, record_keys)
    annotate_codex_trajectory(enriched, [record for record, _src in records], blob_index)
    mark_codex_session(enriched, stream, record_keys)
    del records, loaded, events, stream
    _refuse_if_mutated(snapshot, archive)

    return (
        ConversionResult(
            trajectory=enriched,
            validation_errors=validate_trajectory(enriched),
            edges_lines=edges_lines,
            events_lines=events_lines,
            blobs=collector.blobs,
        ),
        report,
    )
