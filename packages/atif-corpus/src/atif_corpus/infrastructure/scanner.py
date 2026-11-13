# SPDX-License-Identifier: Apache-2.0

"""Filesystem discovery: sessions + side-files with their ``st_mtime_ns``.

The one place atif-corpus stats the source corpus. WHERE it looks comes from a
:class:`~atif_corpus.domain.source_layout.SourceLayout` — one per agent, so
this module walks one algorithm instead of carrying a branch per transcript
shape. Discovery follows CONTRACT.md exactly:

* transcripts sit ``layout.transcript_depth`` directory levels under the source
  root: ``<source_root>/<project>/<session>.jsonl`` for Claude Code,
  ``<source_root>/<YYYY>/<MM>/<DD>/rollout-<ts>-<uuid>.jsonl`` for Codex;
* the session id comes from the FILENAME via ``layout.session_id`` — the whole
  stem for Claude Code, the trailing uuid for a Codex rollout — and must pass
  :func:`~atif_corpus.domain.session_id.session_id_rejection` before it is
  used for anything: it becomes a directory name in the corpus and, from
  there, part of the paths atif-duck opens. A name that fails is REJECTED:
  skipped this pass, logged with the reason, and reported in
  :attr:`SourceScan.rejected_session_ids` so the skip is visible;
* side-files are found by ``rglob`` over the session dir (the directory
  named after the session stem) filtered to ``*.jsonl`` and ``*.meta.json``
  — the rglob deliberately does NOT hardcode ``subagents/`` vs
  ``subagents/workflows/wf_*/`` so any deeper future nesting is still
  watermarked. The ``agent-*.meta.json`` sidecars are watched because the
  converter reads them (subagent type and description), so a sidecar that
  changes on its own has to make the session stale; the other files under the
  side dir (``tool-results/`` spills) are archived but not watched. Only layouts that HAVE side-files are walked for them: a Codex
  sub-agent writes its own rollout under its own session id, so a rollout is
  always alone and looking for siblings would be a stat per session for a
  directory that cannot exist.

A Codex session is every rollout that records it. A thread's continuation
segments (``rollout-<ts>-<thread>_<segment>.jsonl``) sit in the date
directories of the days they were written, and a spawned sub-agent thread
writes rollouts of its own whose ``session_meta`` names the root thread in
``session_id``. The scanner reads each rollout's first record to file it under
its root session; the session's main transcript is the root thread's first
rollout, and every rollout of the group is in its watermark. A rollout is
plain JSONL or zstd-compressed ``.jsonl.zst``; when both forms of one rollout
are present, the plain file is the one read.

A scan reports two kinds of absence and they must never be conflated. A
session whose main transcript is genuinely GONE is marked source-removed:
its corpus dir is kept and its meta says the source is gone. A session whose
``stat`` merely FAILED (EIO, ESTALE, an NFS or permission blip) is only
unreadable this pass and its source may well still be there — marking it
would record a live session as deleted over a transient error. :class:`SourceScan` keeps the two in
separate fields so a caller cannot accidentally treat one as the other.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import IO, TYPE_CHECKING, Any

import zstandard
from loguru import logger

from atif_corpus.domain.agents import AgentSource
from atif_corpus.domain.session_id import session_id_rejection
from atif_corpus.domain.sessions import SessionSource
from atif_corpus.domain.source_layout import (
    CLAUDE_CODE_LAYOUT,
    RolloutName,
    SourceLayout,
    parse_rollout_name,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

#: The side-dir files whose mtimes are part of a session's watermark: the
#: side-file transcripts, and the ``agent-*.meta.json`` sidecars the converter
#: reads beside them.
WATCHED_SIDE_FILE_GLOBS: tuple[str, ...] = ("*.jsonl", "*.meta.json")

#: Shared empty default; a frozen dataclass must not hand out a mutable dict.
_NO_UNREADABLE: Mapping[str, str] = MappingProxyType[str, str]({})


@dataclass(frozen=True, slots=True)
class SourceScan:
    """One scan of the source root: sessions found, plus sessions it could not read."""

    #: Sessions discovered this pass, sorted by session id.
    sessions: tuple[SessionSource, ...] = ()
    #: ``{session_id: main_jsonl}`` for sessions whose sources raised a
    #: non-``FileNotFoundError`` ``OSError``. Absent from ``sessions`` (nothing
    #: converts without an mtime), so callers MUST exclude them from source-removal
    #: removal and retain their watermark entries — the main path is carried
    #: because retention scopes entries by it. Wrapped read-only so the frozen
    #: dataclass is actually frozen rather than frozen-except-this-field.
    unreadable: Mapping[str, str] = field(default_factory=lambda: _NO_UNREADABLE)
    #: Directories whose CONTENTS could not be listed at all (a project dir at
    #: mode 000). Their sessions cannot appear in ``unreadable`` — that map is
    #: keyed by discovered session, and nothing was discovered — so the
    #: directory is reported instead and the caller resolves it to session ids
    #: through the watermark.
    unlistable_dirs: tuple[str, ...] = ()
    #: Transcript names whose derived session id failed the boundary check
    #: (:mod:`atif_corpus.domain.session_id`), sorted. Nothing from them is
    #: materialized, they are never marked source-removed, and they are reported so an
    #: operator can see why a transcript is missing from the corpus.
    rejected_session_ids: tuple[str, ...] = ()

    @property
    def unreadable_session_ids(self) -> frozenset[str]:
        """Ids of the sessions this scan could not read."""
        return frozenset(self.unreadable)

    def with_unreadable(self, extra: Mapping[str, str]) -> SourceScan:
        """A copy of this scan carrying more unreadable ``{session_id: main_jsonl}``.

        Lets a caller that resolved :attr:`unlistable_dirs` to session ids
        fold them in, so source-removal marking, watermark retention and the pass
        report all read one unreadable set instead of three.
        """
        if not extra:
            return self
        return SourceScan(
            sessions=self.sessions,
            unreadable=MappingProxyType({**self.unreadable, **extra}),
            unlistable_dirs=self.unlistable_dirs,
            rejected_session_ids=self.rejected_session_ids,
        )


def _mtime_ns(path: Path) -> int | None:
    """``st_mtime_ns`` for ``path``; ``None`` when it vanished mid-scan.

    Every other ``OSError`` propagates so the caller can tell "deleted" from
    "could not read".
    """
    try:
        return path.stat().st_mtime_ns
    except FileNotFoundError:
        # A transcript deleted between glob and stat is a normal race on a
        # live corpus, not an error: it simply isn't part of this scan.
        logger.debug("scan: {} vanished mid-scan; skipping", path)
        return None


def _session_mtimes(main_jsonl: Path, layout: SourceLayout) -> dict[str, int] | None:
    """``{path: mtime_ns}`` for one session, or ``None`` when its main file vanished.

    Side-files are the transcripts and ``*.meta.json`` sidecars under the
    session dir (:data:`WATCHED_SIDE_FILE_GLOBS`). A vanished SIDE-file is
    simply omitted: the session is still part of the scan, and the missing entry is exactly the staleness signal that makes it
    re-materialize. Raises ``OSError`` when any source could not be stat'd.
    """
    main_mtime = _mtime_ns(main_jsonl)
    if main_mtime is None:
        return None
    mtimes = {str(main_jsonl): main_mtime}

    if not layout.has_side_files:
        return mtimes
    side_dir = main_jsonl.parent / main_jsonl.stem
    if side_dir.is_dir():
        watched = {path for pattern in WATCHED_SIDE_FILE_GLOBS for path in side_dir.rglob(pattern)}
        for side_file in sorted(watched):
            side_mtime = _mtime_ns(side_file)
            if side_mtime is not None:
                mtimes[str(side_file)] = side_mtime
    return mtimes


def _list_dir(directory: Path) -> list[Path] | None:
    """Sorted entries of ``directory``; ``[]`` if it vanished, ``None`` if unlistable.

    ``Path.glob`` swallows a ``PermissionError`` on a directory it cannot
    open and simply yields nothing, which is indistinguishable from an empty
    directory — and "empty" makes every session under it look deleted. Listing each
    level explicitly is what keeps the two apart.
    """
    try:
        return sorted(directory.iterdir())
    except FileNotFoundError:
        logger.debug("scan: directory {} vanished mid-scan; skipping", directory)
        return []
    except OSError as error:
        logger.warning(
            "scan: cannot list directory {} ({}); its sessions are unreadable "
            "this pass and are NOT treated as deleted",
            directory,
            error,
        )
        return None


def _transcript_dirs(source_root: Path, layout: SourceLayout) -> tuple[list[Path], list[str]]:
    """The directories that hold transcripts, plus every dir that would not list.

    Descends exactly ``layout.transcript_depth`` levels, listing each one, so an
    unopenable directory at ANY level is reported as unlistable instead of
    reading as empty. Claude Code stops at the project dirs; Codex walks
    year -> month -> day.
    """
    level = [source_root]
    unlistable: list[str] = []
    for _ in range(layout.transcript_depth):
        next_level: list[Path] = []
        for directory in level:
            entries = _list_dir(directory)
            if entries is None:
                unlistable.append(str(directory))
                continue
            next_level.extend(entry for entry in entries if entry.is_dir())
        level = sorted(next_level)
    return level, unlistable


def _transcripts(transcript_dir: Path, layout: SourceLayout) -> list[Path] | None:
    """List one dir's transcript files for this agent, or ``None`` if unlistable."""
    entries = _list_dir(transcript_dir)
    if entries is None:
        return None
    return [entry for entry in entries if layout.is_transcript(entry.name)]


def scan_sources(
    source_root: Path,
    layout: SourceLayout = CLAUDE_CODE_LAYOUT,
) -> SourceScan:
    """Discover every session under ``source_root``, separating unreadable and rejected ones.

    Sessions come back sorted by session id so the scan itself is
    deterministic input to :func:`atif_corpus.domain.sessions.build_plan`.
    An absent or empty root yields an empty scan — a fresh machine is not an
    error condition.

    Absence is separated from unreadability at EVERY level: a session whose
    ``stat`` failed lands in :attr:`SourceScan.unreadable`, and any directory
    that could not be LISTED — the root, a Claude Code project dir, a Codex
    year/month/day dir — lands in :attr:`SourceScan.unlistable_dirs`. Its
    sessions were never discovered, so only the caller's watermark knows which
    ids they were.

    ``layout`` defaults to Claude Code's shape, which is what every caller
    predating Codex support meant.
    """
    if not source_root.is_dir():
        logger.debug("scan: source root {} does not exist", source_root)
        return SourceScan()

    transcript_dirs, unlistable = _transcript_dirs(source_root, layout)
    if layout.agent is AgentSource.CODEX:
        return _scan_codex(source_root, transcript_dirs, unlistable)

    sessions: list[SessionSource] = []
    unreadable: dict[str, str] = {}
    rejected: list[str] = []
    for transcript_dir in transcript_dirs:
        transcripts = _transcripts(transcript_dir, layout)
        if transcripts is None:
            unlistable.append(str(transcript_dir))
            continue
        for main_jsonl in transcripts:
            session_id = layout.session_id(main_jsonl.name)
            if session_id is None:
                logger.debug(
                    "scan: {} is not a {} transcript name; skipping",
                    main_jsonl,
                    layout.agent.value,
                )
                continue
            rejection = session_id_rejection(session_id)
            if rejection is not None:
                # The one outside influence on a corpus path. A name that fails
                # here never becomes a directory, a watermark key, or a SQL
                # parameter; it is reported instead of guessed at.
                logger.warning(
                    "scan: rejecting session name {!r} from {} ({}); nothing from it "
                    "is materialized",
                    session_id,
                    main_jsonl,
                    rejection,
                )
                rejected.append(session_id)
                continue
            try:
                mtimes = _session_mtimes(main_jsonl, layout)
            except OSError as error:
                logger.warning(
                    "scan: cannot stat sources of session {} ({}); skipping this pass "
                    "and NOT treating it as deleted",
                    session_id,
                    error,
                )
                unreadable[session_id] = str(main_jsonl)
                continue
            if mtimes is None:
                continue
            sessions.append(
                SessionSource(
                    session_id=session_id,
                    session_jsonl=str(main_jsonl),
                    source_mtimes=mtimes,
                )
            )
    sessions.sort(key=lambda s: s.session_id)
    logger.debug(
        "scan: {} sessions ({} unreadable, {} unlistable dirs, {} rejected names) under {} as {}",
        len(sessions),
        len(unreadable),
        len(unlistable),
        len(rejected),
        source_root,
        layout.agent.value,
    )
    return SourceScan(
        sessions=tuple(sessions),
        unreadable=MappingProxyType(dict(unreadable)),
        unlistable_dirs=tuple(unlistable),
        rejected_session_ids=tuple(sorted(rejected)),
    )


#: How much of a rollout's first line is read for its ``session_meta``. The
#: record carries the model instructions and tool list, so it runs to tens of
#: kilobytes; a first line longer than this is treated as carrying no ids.
_SESSION_META_MAX_BYTES = 4 * 1024 * 1024


def _first_line(stream: IO[bytes] | zstandard.ZstdDecompressionReader) -> bytes:
    """Bytes up to the first newline of a binary stream, at most :data:`_SESSION_META_MAX_BYTES`."""
    chunks: list[bytes] = []
    total = 0
    while total < _SESSION_META_MAX_BYTES:
        chunk = stream.read(64 * 1024)
        if not chunk:
            break
        newline = chunk.find(b"\n")
        if newline >= 0:
            chunks.append(chunk[:newline])
            break
        chunks.append(chunk)
        total += len(chunk)
    return b"".join(chunks)


def _first_record(path: Path) -> dict[str, Any] | None:
    """The rollout's first JSON record, or ``None`` when it does not parse as one.

    Raises ``OSError`` when the file cannot be read, so the caller can tell an
    unreadable rollout from one that records no ids.
    """
    with path.open("rb") as raw:
        try:
            if path.name.endswith(".zst"):
                line = _first_line(zstandard.ZstdDecompressor().stream_reader(raw))
            else:
                line = _first_line(raw)
        except zstandard.ZstdError:
            return None
    try:
        record = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return record if isinstance(record, dict) else None


def _recorded_ids(record: dict[str, Any] | None) -> tuple[str | None, str | None]:
    """``(thread id, root session id)`` from a ``session_meta`` record, each ``None`` if absent."""
    if record is None or record.get("type") != "session_meta":
        return None, None
    payload = record.get("payload")
    if not isinstance(payload, dict):
        return None, None
    thread_id = payload.get("id")
    session_id = payload.get("session_id")
    return (
        thread_id if isinstance(thread_id, str) and thread_id else None,
        session_id if isinstance(session_id, str) and session_id else None,
    )


@dataclass(frozen=True, slots=True)
class _Rollout:
    path: Path
    name: RolloutName
    #: The session the rollout's ``session_meta`` files it under (its own
    #: thread when it records none).
    parent_session: str


def _codex_rollouts(
    transcript_dirs: list[Path], unlistable: list[str]
) -> tuple[list[_Rollout], dict[str, str]]:
    """Every rollout under the day dirs, one per rollout id, plus the unreadable ones."""
    by_rollout: dict[str, _Rollout] = {}
    unreadable: dict[str, str] = {}
    for transcript_dir in transcript_dirs:
        entries = _list_dir(transcript_dir)
        if entries is None:
            unlistable.append(str(transcript_dir))
            continue
        for path in entries:
            name = parse_rollout_name(path.name)
            if name is None:
                logger.debug("scan: {} is not a codex transcript name; skipping", path)
                continue
            previous = by_rollout.get(name.rollout_id)
            if previous is not None and not previous.name.compressed:
                continue  # the plain form of a rollout wins over its compressed copy
            try:
                thread_id, session_id = _recorded_ids(_first_record(path))
            except OSError as error:
                logger.warning(
                    "scan: cannot read {} ({}); its session is unreadable this pass "
                    "and NOT treated as deleted",
                    path,
                    error,
                )
                unreadable.setdefault(name.thread_id, str(path))
                continue
            if thread_id is not None and thread_id != name.thread_id:
                logger.warning(
                    "scan: {} records thread {} but its name says {}; skipping",
                    path,
                    thread_id,
                    name.thread_id,
                )
                continue
            by_rollout[name.rollout_id] = _Rollout(
                path=path, name=name, parent_session=session_id or name.thread_id
            )
    return sorted(by_rollout.values(), key=lambda r: str(r.path)), unreadable


def _root_session(thread_id: str, parents: dict[str, str]) -> str:
    """Follow recorded sessions up to the thread that records itself as its own.

    A thread whose recorded session is not among the rollouts ends the walk
    there: the group is named after the absent root and holds no rollout of it.
    """
    seen: set[str] = set()
    current = thread_id
    while current in parents and parents[current] != current and current not in seen:
        seen.add(current)
        current = parents[current]
    return current


def _scan_codex(
    source_root: Path, transcript_dirs: list[Path], unlistable: list[str]
) -> SourceScan:
    """The Codex scan: rollouts grouped under the root session that records them."""
    rollouts, unreadable = _codex_rollouts(transcript_dirs, unlistable)
    parents = {r.name.thread_id: r.parent_session for r in rollouts}
    groups: dict[str, list[_Rollout]] = {}
    for rollout in rollouts:
        groups.setdefault(_root_session(rollout.name.thread_id, parents), []).append(rollout)

    sessions: list[SessionSource] = []
    rejected: list[str] = []
    for session_id, members in sorted(groups.items()):
        own = [r for r in members if r.name.thread_id == session_id]
        if not own:
            logger.debug(
                "scan: rollouts of session {} without its own thread's rollout; skipping",
                session_id,
            )
            continue
        rejection = session_id_rejection(session_id)
        if rejection is not None:
            logger.warning(
                "scan: rejecting session name {!r} ({}); nothing from it is materialized",
                session_id,
                rejection,
            )
            rejected.append(session_id)
            continue
        first = [r for r in own if r.name.rollout_id == session_id]
        main = (first or sorted(own, key=lambda r: r.path.name))[0].path
        if session_id in unreadable:
            unreadable[session_id] = str(main)
            continue
        mtimes: dict[str, int] = {}
        try:
            for rollout in members:
                mtime = _mtime_ns(rollout.path)
                if mtime is not None:
                    mtimes[str(rollout.path)] = mtime
        except OSError as error:
            logger.warning(
                "scan: cannot stat sources of session {} ({}); skipping this pass "
                "and NOT treating it as deleted",
                session_id,
                error,
            )
            unreadable[session_id] = str(main)
            continue
        if str(main) not in mtimes:
            continue
        sessions.append(
            SessionSource(session_id=session_id, session_jsonl=str(main), source_mtimes=mtimes)
        )
    logger.debug(
        "scan: {} sessions ({} unreadable, {} unlistable dirs, {} rejected names) under {} as codex",
        len(sessions),
        len(unreadable),
        len(unlistable),
        len(rejected),
        source_root,
    )
    return SourceScan(
        sessions=tuple(sessions),
        unreadable=MappingProxyType(dict(unreadable)),
        unlistable_dirs=tuple(unlistable),
        rejected_session_ids=tuple(sorted(rejected)),
    )


def scan_source_root(
    source_root: Path,
    layout: SourceLayout = CLAUDE_CODE_LAYOUT,
) -> tuple[SessionSource, ...]:
    """The sessions found under ``source_root`` — :func:`scan_sources` without diagnostics.

    For read-only callers that plan but never mark: only source-removal
    marking needs ``unreadable_session_ids``, and a caller that marks must
    reach for :func:`scan_sources` and handle it. Dropping the
    diagnostics loses no operator signal here — :func:`scan_sources` logs each
    unreadable session and unlistable directory at WARNING, and the CLI's sink
    is WARNING-and-up on stderr, so an unreadable source still announces
    itself alongside the status output.
    """
    return scan_sources(source_root, layout).sessions
