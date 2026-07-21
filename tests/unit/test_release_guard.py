"""Tests for the release guard's decision table.

`scripts/release/assert_release_state.py` is what makes a green Release run mean
something: it decides, after changesets/action, whether the run did what it was
supposed to. It got that wrong on its first real run (it failed the job for
Version PR #34, a perfectly healthy version run), so each state in its table is
pinned here directly -- with the inputs that produce it, not through the
workflow.

The four states:

| Version PR | pending changesets | tag on origin | verdict |
|------------|--------------------|---------------|---------|
| yes        | (irrelevant)       | (not queried) | ok      |
| no         | yes                | (not queried) | FAIL    |
| no         | no                 | no            | FAIL    |
| no         | no                 | yes           | ok      |
"""

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Callable, List

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GUARD_PATH = REPO_ROOT / "scripts" / "release" / "assert_release_state.py"


def _load_guard() -> ModuleType:
    """Import the guard by path -- scripts/ is not an importable package."""
    spec = importlib.util.spec_from_file_location("assert_release_state", GUARD_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


guard = _load_guard()


def _never_called(tag: str) -> bool:
    raise AssertionError(
        f"tag_exists({tag!r}) must not be consulted when a Version PR was opened"
    )


class TestVersionRun:
    """A Version PR was opened: nothing published, and that is success."""

    def test_version_pr_is_ok_even_though_tree_looks_like_a_failed_publish(
        self,
    ) -> None:
        """The regression from run 29839040289 / PR #34.

        changesets/action consumes the changeset files and bumps the version
        files *in the working tree* before this guard runs. So a healthy version
        run presents exactly the fingerprint of a broken publish: no pending
        changesets, and a version with no tag. Keying off tree state failed the
        job; only `version_pr` distinguishes the two.
        """
        decision = guard.decide(
            version="4.2.3",
            version_pr="34",
            pending=[],  # consumed by `changeset version`
            tag_exists=lambda _tag: False,  # 4.2.3 is not released yet
        )

        assert decision.ok is True
        assert "#34" in decision.message
        assert "by design" in decision.message

    def test_version_pr_does_not_query_the_remote(self) -> None:
        """A version run must not need the network to reach its verdict."""
        decision = guard.decide(
            version="4.2.3",
            version_pr="34",
            pending=[],
            tag_exists=_never_called,
        )

        assert decision.ok is True

    def test_version_pr_wins_even_with_changesets_still_on_disk(self) -> None:
        """Belt and braces: the PR decides regardless of leftover changesets."""
        decision = guard.decide(
            version="4.2.3",
            version_pr="34",
            pending=["some-fix.md"],
            tag_exists=_never_called,
        )

        assert decision.ok is True


class TestBrokenVersionStep:
    """Changesets exist but no Version PR was opened -> nothing will ship."""

    def test_pending_changesets_without_a_version_pr_fails(self) -> None:
        decision = guard.decide(
            version="4.2.2",
            version_pr="",
            pending=["sql-null-semantics.md", "window-functions.md"],
            tag_exists=_never_called,
        )

        assert decision.ok is False
        assert "2 pending changeset(s)" in decision.message
        assert "sql-null-semantics.md" in decision.message
        assert "Nothing was released" in decision.message


class TestPublishRun:
    """No Version PR: the run was a publish, so the version must be tagged."""

    def test_untagged_version_fails_loudly(self) -> None:
        """The case that actually matters: a publish that silently no-op'd.

        This is the exact shape the old semantic-release workflow reported as
        green -- publish job skipped, nothing on the feed, no tag.
        """
        decision = guard.decide(
            version="4.2.3",
            version_pr="",
            pending=[],
            tag_exists=lambda _tag: False,
        )

        assert decision.ok is False
        assert "v4.2.3" in decision.message
        assert "did not run to completion" in decision.message
        assert "NOT confirmed on the Azure Artifacts feed" in decision.message

    def test_tagged_version_is_ok(self) -> None:
        decision = guard.decide(
            version="4.2.3",
            version_pr="",
            pending=[],
            tag_exists=lambda _tag: True,
        )

        assert decision.ok is True
        assert "was published" in decision.message

    def test_the_tag_queried_is_the_v_prefixed_tree_version(self) -> None:
        """Guards against asking about the wrong tag and trusting the answer."""
        asked: List[str] = []

        def record(tag: str) -> bool:
            asked.append(tag)
            return True

        guard.decide(version="5.1.0", version_pr="", pending=[], tag_exists=record)

        assert asked == ["v5.1.0"]


class TestDecisionTable:
    """Every row of the table in one place, so the shape stays reviewable."""

    @pytest.mark.parametrize(  # type: ignore[misc,untyped-decorator]
        ("version_pr", "pending", "tagged", "expected_ok"),
        [
            pytest.param("34", [], False, True, id="version-pr-opened"),
            pytest.param("", ["a.md"], False, False, id="changesets-but-no-pr"),
            pytest.param("", [], False, False, id="publish-did-not-complete"),
            pytest.param("", [], True, True, id="published-and-tagged"),
        ],
    )
    def test_table(
        self,
        version_pr: str,
        pending: List[str],
        tagged: bool,
        expected_ok: bool,
    ) -> None:
        tag_lookup: Callable[[str], bool] = lambda _tag: tagged  # noqa: E731

        decision = guard.decide(
            version="4.2.3",
            version_pr=version_pr,
            pending=pending,
            tag_exists=tag_lookup,
        )

        assert decision.ok is expected_ok
        assert decision.message.strip()
