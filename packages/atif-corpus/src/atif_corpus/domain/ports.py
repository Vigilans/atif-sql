# SPDX-License-Identifier: Apache-2.0

"""The ports materialization needs filled: a converter and, optionally, an artifact filter, an artifact producer and a session sink.

atif-corpus may never import atif-converter or atif-duck (import-linter
independence contract), so the Protocols are typed to CONTRACT.md's artifact
shapes rather than to either package's internals. atif-cli adapts the real
converter, its SQL filter stage and the lake writer to these ports; tests use
:class:`atif_corpus.infrastructure.fake_converter.FakeConverter` and a fake
producer of their own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path


class EmptySourceError(Exception):
    """The session holds nothing to convert (no user or assistant record).

    Not a failure: the transcript is well-formed and simply empty, and it stays
    empty until something writes to it. An adapter translates its converter's
    own "empty" verdict into this so the use case can record the session as
    empty, keyed on its source mtimes, instead of retrying it every pass. Any
    OTHER exception is a real failure.
    """


@dataclass(frozen=True, slots=True)
class ArchivedSource:
    """One file of a session's raw source archive, as the converter wrote it.

    The compressed copy lives at ``<archive_dir>/<relative_path>.zst``;
    ``size`` and ``sha256`` describe the UNCOMPRESSED bytes, so a restore can
    prove it got the original back.
    """

    #: POSIX path relative to the main transcript's parent directory.
    relative_path: str
    #: Uncompressed size in bytes.
    size: int
    #: Hex sha256 of the uncompressed bytes.
    sha256: str


@dataclass(frozen=True, slots=True)
class BlobOutput:
    """One attachment's bytes, content-addressed, for the corpus blob store.

    ``sha256`` is the lowercase hex digest of ``data`` and ``extension`` the
    file extension its media type maps to (``png``, ``jpg``, ``pdf``, ``bin``);
    the writer validates both before either becomes part of a path.
    """

    sha256: str
    extension: str
    data: bytes


@dataclass(frozen=True, slots=True)
class ConversionOutput:
    """Everything one conversion yields that the corpus writes to disk.

    Shapes are the contract's, pre-serialized by the adapter:

    * ``trajectory_dict`` — the ATIF-v1.7 trajectory, ready for compact
      ``json.dumps(separators=(",", ":"))``.
    * ``loss_report_dict`` — ``LossReport.to_json()``-shaped dict.
    * ``edges_lines`` — one already-serialized JSON line per RAW record
      (uuid, parent_uuid, message_id, type, ts, is_sidechain,
      is_compact_summary, source_file, tool_use_ids), WITHOUT trailing
      newlines; the writer owns line termination.
    * ``events_lines`` — one already-serialized JSON line per kept
      NON-MESSAGE record (``session_events.jsonl``: seq, ts, event_type,
      subtype, uuid, parent_uuid, tool_use_id, is_sidechain, source_file,
      payload, payload_bytes, payload_truncated), same termination rule.
      Empty by default, so a converter that has none writes an empty file.
    * ``source_archive`` — the files the converter archived into the
      ``archive_dir`` it was handed (empty when it was handed none, or when
      the adapter does not archive). Every entry must exist on disk as
      ``<archive_dir>/<relative_path>.zst``; the use case checks that.
    * ``blobs`` — the inline attachments the converter replaced with a part
      or a placeholder naming ``blobs/sha256/<ab>/<hash>.<ext>``, one per
      distinct hash. The writer stores each under ``<corpus_root>/blobs/``
      BEFORE it publishes the session that references it.
    """

    trajectory_dict: dict[str, Any]
    loss_report_dict: dict[str, Any]
    edges_lines: list[str]
    events_lines: list[str] = field(default_factory=list)
    source_archive: tuple[ArchivedSource, ...] = ()
    blobs: tuple[BlobOutput, ...] = ()


class ConverterPort(Protocol):
    """Anything that can turn one session JSONL into corpus artifacts.

    Implementations may raise any exception: the materialize use case
    records the failure against the session and continues — one broken
    transcript must never abort a corpus sync. :class:`EmptySourceError` is
    the one exception with its own meaning: the session is empty, not broken.
    """

    def convert(
        self,
        session_jsonl: Path,
        *,
        source_files: Sequence[Path] = (),
        archive_dir: Path | None = None,
    ) -> ConversionOutput:
        """Convert one session (main JSONL + its side-files) to artifacts.

        ``source_files`` are the session's other transcripts when they do not
        sit under the main transcript's side dir, as the scanner grouped them
        (a Codex session's other rollouts). A Claude Code session passes none.

        With ``archive_dir``, also write a zstd copy of every source file of the
        session under it, named by its path relative to
        :func:`source_base` plus ``.zst``, and list them in
        :attr:`ConversionOutput.source_archive`. The copy must be of the bytes
        the artifacts were built from; an adapter that cannot promise that
        should archive nothing rather than something else.
        """
        ...


def source_base(session_jsonl: Path, source_files: Sequence[Path] = ()) -> Path:
    """The directory a session's source files are named relative to.

    The main transcript's parent, which holds its side dir too; for a session
    whose ``source_files`` lie elsewhere (a Codex thread spanning several date
    directories), the deepest directory holding all of them. The archive
    manifest and the converter's ``source_file`` names both use it.
    """
    base = session_jsonl.parent
    for path in source_files:
        while base not in path.parents and base != base.parent:
            base = base.parent
    return base


class ArtifactFilter(Protocol):
    """Anything that rewrites a converted session before any of it is written.

    The materialize use case calls :meth:`apply` once per session, right after
    the conversion and before anything reaches the corpus: before the
    session's blobs enter the shared blob store and before its staged
    directory is written. Only what :meth:`apply` returns is published, so a
    blob it leaves out is never stored. atif-cli plugs in the SQL filter stage
    of ``atif-sql materialize --filter``.

    A pass with a filter writes no raw source archive or ``edges.jsonl``, since
    the raw bytes and the edges' line per raw record keep what a filter
    removes, and stamps ``filtered: true`` into every ``meta.json`` it writes.

    Implementations may raise any exception: the use case records the failure
    against the session and publishes nothing of it, exactly as it does for a
    converter failure. A pool worker holds its own copy of the filter, so an
    implementation has to stay picklable.
    """

    def apply(self, output: ConversionOutput, *, session_id: str) -> ConversionOutput:
        """Return what ``session_id``'s ``output`` publishes as."""
        ...


class ArtifactProducer(Protocol):
    """Anything that writes EXTRA per-session artifacts beside the contract ones.

    The materialize use case calls :meth:`produce` once per session, inside
    the staged session directory, after ``trajectory.json`` /
    ``loss_report.json`` / ``edges.jsonl`` / ``session_events.jsonl`` are
    written (so a producer may read those staged files) and before
    ``meta.json`` is. Whatever the producer writes therefore publishes
    atomically with the contract artifacts (the whole directory is
    swapped into place) and is covered by the same completeness marker.

    The producer returns extra keys for ``meta.json``. They must not collide
    with the contract's own keys; the use case treats a collision as a
    programming error and fails the session.

    Implementations may raise any exception: the use case records the
    failure against the session, never publishes the staged directory, and
    retries the session next pass, exactly as it does for a converter
    failure.
    """

    def produce(
        self,
        session_dir: Path,
        *,
        session_id: str,
        trajectory: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Write extra artifacts for ``session_id`` into ``session_dir``; return meta extras."""
        ...


class SessionSink(Protocol):
    """Anything that keeps a copy of a corpus's published sessions in step with it.

    atif-cli plugs atif-duck's DuckLake writer in here. The materialize use
    case calls :meth:`sync_corpus` once per pass, in the PARENT process only
    (never in a pool worker, so an implementation need not pickle), after
    every session the pass publishes has been swapped into ``sessions/<id>/``
    and every source-removed mark written.

    The sink finds the sessions to copy itself, by comparing the corpus with
    its copy: a pass hands over no list, so sessions an earlier pass
    published without the sink (a failed sync, a pass that ran without one)
    reach the copy the same way. Returning normally means the copy now
    matches the corpus (an implementation with no copy to keep, say because
    none has been created yet, may simply return 0). Raising means it may
    not; the next call compares again.
    """

    def sync_corpus(self, *, corpus_root: Path, agent: str) -> int:
        """Make the copy match the corpus's published sessions; return how many sessions it wrote."""
        ...
