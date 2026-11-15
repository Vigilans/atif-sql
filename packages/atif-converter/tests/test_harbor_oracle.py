# SPDX-License-Identifier: Apache-2.0

"""The oracle itself must be trustworthy before a port is measured against it.

Three properties, each a precondition for the parity tests meaning anything:
the live oracle converts both synthetic fixtures; it is deterministic (two runs
agree byte for byte, so a diff is a real divergence and never noise); and it
still agrees with the frozen goldens, so an upstream behavior change in harbor
surfaces here as a named diff rather than as a port that "fails parity".
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from harbor_oracle import (
    diff_paths,
    golden_path,
    harbor_claude_code_trajectory,
    harbor_codex_trajectory,
    load_golden,
    parity_diffs,
    require_harbor_private_api,
    write_golden,
)

#: Set to re-freeze the goldens from the live oracle. Deliberately an env var
#: and not a default: a golden is re-frozen after a DECISION about an upstream
#: change, never as a side effect of a normal test run.
FREEZE_ENV = "ATIF_FREEZE_GOLDENS"


class TestLiveOracle:
    def test_claude_code_fixture_converts_and_is_deterministic(
        self, synthetic_session: Path
    ) -> None:
        require_harbor_private_api("claude-code")
        first = harbor_claude_code_trajectory(synthetic_session)
        second = harbor_claude_code_trajectory(synthetic_session)
        assert first is not None
        assert diff_paths(first, second) == []

    def test_codex_fixture_converts_and_is_deterministic(self, codex_rollout: Path) -> None:
        require_harbor_private_api("codex")
        first = harbor_codex_trajectory(codex_rollout)
        second = harbor_codex_trajectory(codex_rollout)
        assert first is not None
        assert diff_paths(first, second) == []


class TestFreeze:
    """Re-freeze the goldens from the live oracle; runs only under FREEZE_ENV=1."""

    def test_freeze(self, synthetic_session: Path, codex_rollout: Path) -> None:
        if os.environ.get(FREEZE_ENV) != "1":
            pytest.skip(f"set {FREEZE_ENV}=1 to re-freeze the goldens")
        require_harbor_private_api("claude-code")
        require_harbor_private_api("codex")
        claude = harbor_claude_code_trajectory(synthetic_session)
        codex = harbor_codex_trajectory(codex_rollout)
        assert claude is not None and codex is not None
        write_golden("claude_code.synthetic", claude)
        write_golden("codex.synthetic", codex)


class TestFrozenGoldens:
    """The live oracle and the frozen one must agree, or the freeze is stale.

    A diff here means harbor changed conversion behavior under the pin, OR
    litellm's bundled pricing table changed a rate the fixtures price through
    (``cost_usd`` paths only; conftest pins the table to the locked litellm, so
    that arrives with a lockfile bump). The response is a decision, not a
    re-freeze: either the port follows upstream (re-freeze and port the change)
    or it deliberately does not (record why).
    """

    @pytest.mark.parametrize("name", ["claude_code.synthetic", "codex.synthetic"])
    def test_golden_exists(self, name: str) -> None:
        assert golden_path(name).is_file(), f"freeze it: {FREEZE_ENV}=1 uv run pytest -k freeze"

    def test_claude_code_live_matches_frozen(self, synthetic_session: Path) -> None:
        require_harbor_private_api("claude-code")
        live = harbor_claude_code_trajectory(synthetic_session)
        assert live is not None
        assert diff_paths(load_golden("claude_code.synthetic"), live) == []

    def test_codex_live_matches_frozen(self, codex_rollout: Path) -> None:
        require_harbor_private_api("codex")
        live = harbor_codex_trajectory(codex_rollout)
        assert live is not None
        assert diff_paths(load_golden("codex.synthetic"), live) == []


def _one_result(content: object) -> dict[str, object]:
    return {"steps": [{"observation": {"results": [{"content": content}]}}]}


class TestVerbatimResultForgiveness:
    """A Claude Code string result ours keeps verbatim, where harbor strips it, and nothing more."""

    @pytest.mark.parametrize(
        ("harbor", "ours"),
        [
            ("out", "\n  out \n"),
            ("out\n\n[stdout]\no", "out \n\n\n[stdout]\no"),
            ("out\n\n[error] tool reported failure", " out\n\n[error] tool reported failure"),
        ],
        ids=["alone", "before-a-chunk", "before-the-error-line"],
    )
    def test_outer_whitespace_of_the_result_text_is_forgiven(self, harbor: str, ours: str) -> None:
        assert diff_paths(_one_result(harbor), _one_result(ours)) != []
        assert parity_diffs(_one_result(harbor), _one_result(ours)) == []

    @pytest.mark.parametrize(
        ("harbor", "ours"),
        [
            ("out", " outs "),
            ("out\n\n[stdout]\no", "out\n\n[stdout]\no "),
            ("out\n\n[stdout]\no", "out\n[stdout]\no"),
        ],
        ids=["other-text", "whitespace-in-a-chunk", "a-separator-lost"],
    )
    def test_anything_else_still_fails(self, harbor: str, ours: str) -> None:
        assert parity_diffs(_one_result(harbor), _one_result(ours)) != []


def _step(step_id: int, source: str, message: str) -> dict[str, object]:
    return {"step_id": step_id, "source": source, "message": message}


def _tree(
    root_steps: list[dict[str, object]],
    child_steps: list[dict[str, object]],
    child_total: int | None = None,
) -> dict[str, object]:
    return {
        "steps": root_steps,
        "final_metrics": {"total_steps": len(root_steps)},
        "subagent_trajectories": [
            {
                "trajectory_id": "c1",
                "steps": child_steps,
                "final_metrics": {
                    "total_steps": len(child_steps) if child_total is None else child_total
                },
            }
        ],
    }


class TestSystemPromptForgiveness:
    """A named subagent's system steps are taken out of ours, and nothing else is forgiven."""

    HARBOR = _tree([_step(1, "user", "go")], [_step(1, "user", "inspect"), _step(2, "agent", "x")])

    def test_a_named_subagent_s_system_steps_are_forgiven(self) -> None:
        ours = _tree(
            [_step(1, "user", "go")],
            [
                _step(1, "system", "A"),
                _step(2, "user", "inspect"),
                _step(3, "system", "B"),
                _step(4, "agent", "x"),
            ],
        )
        assert parity_diffs(self.HARBOR, ours) != []
        assert parity_diffs(self.HARBOR, ours, system_prompt_agent_ids={"c1"}) == []

    def test_another_subagent_s_system_step_still_fails(self) -> None:
        ours = _tree(
            [_step(1, "user", "go")],
            [_step(1, "system", "A"), _step(2, "user", "inspect"), _step(3, "agent", "x")],
        )
        assert parity_diffs(self.HARBOR, ours, system_prompt_agent_ids={"c2"}) != []

    def test_a_root_system_step_still_fails(self) -> None:
        ours = _tree(
            [_step(1, "system", "A"), _step(2, "user", "go")],
            [_step(1, "user", "inspect"), _step(2, "agent", "x")],
        )
        ours["trajectory_id"] = "c1"
        assert parity_diffs(self.HARBOR, ours, system_prompt_agent_ids={"c1"}) != []

    def test_the_other_steps_are_still_compared(self) -> None:
        wrong_turn = _tree(
            [_step(1, "user", "go")],
            [_step(1, "system", "A"), _step(2, "user", "inspect"), _step(3, "agent", "y")],
        )
        wrong_id = _tree(
            [_step(1, "user", "go")],
            [_step(1, "system", "A"), _step(3, "user", "inspect"), _step(4, "agent", "x")],
        )
        wrong_total = _tree(
            [_step(1, "user", "go")],
            [_step(1, "system", "A"), _step(2, "user", "inspect"), _step(3, "agent", "x")],
            child_total=2,
        )
        for ours in (wrong_turn, wrong_id, wrong_total):
            assert parity_diffs(self.HARBOR, ours, system_prompt_agent_ids={"c1"}) != []
