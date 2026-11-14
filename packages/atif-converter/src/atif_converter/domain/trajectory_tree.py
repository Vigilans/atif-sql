# SPDX-License-Identifier: Apache-2.0

"""Walk a trajectory dict and the subagent trajectories embedded in it.

A converted session is one root trajectory whose ``subagent_trajectories``
hold the trajectories of the subagents it delegated to, each of which may
embed its own. Every step-level pass after conversion (enrichment, typed
result signals, reverted marks) visits each of them, because each one numbers
its own steps from 1 and carries its own ``steps`` array.

Pure module: no filesystem access, no env, no logging.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator


def iter_trajectories(trajectory: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """``trajectory`` and every trajectory embedded below it, parents before children."""
    pending = [trajectory]
    while pending:
        current = pending.pop()
        yield current
        children = current.get("subagent_trajectories")
        if isinstance(children, list):
            pending.extend(child for child in reversed(children) if isinstance(child, dict))


def iter_steps(trajectory: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Every step dict of ``trajectory`` and of every trajectory embedded below it."""
    for current in iter_trajectories(trajectory):
        steps = current.get("steps")
        if isinstance(steps, list):
            yield from (step for step in steps if isinstance(step, dict))


__all__ = ["iter_steps", "iter_trajectories"]
