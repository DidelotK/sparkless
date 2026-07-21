#!/usr/bin/env python3
"""Fail the release workflow when nothing happened but something should have.

This is the guard that makes a green Release run mean something. It runs after
changesets/action and asserts that the run accomplished what it was supposed to.

Exactly two outcomes are acceptable:

1. A "Version Packages" PR was opened or updated -> nothing was published, by
   design. Merging that PR is what publishes.
2. No Version PR -> this was a publish run, so the version in the tree must be
   tagged on origin. The tag is only pushed (see scripts/release/publish.sh)
   after the wheel has been uploaded AND the feed confirmed to serve it.

Anything else means the machinery broke, and this exits non-zero.

The setup this replaces could not tell those apart: it decided whether to
publish by grepping semantic-release's console output for a sentence, and a
miss produced `new_release_published=false`, which *skipped* the publish job.
A skipped job is green. So "no release was due" and "the release broke" looked
identical from the outside. That ambiguity is the reason for this file.

IMPORTANT -- why the Version PR is checked FIRST.

By the time this runs, changesets/action has already executed the version
command *in the working tree*: the changeset files are consumed and the version
files carry the NEXT version. So on a version run the tree looks exactly like a
finished publish that forgot to tag -- no pending changesets, and a version with
no tag. Keying the decision off tree state therefore reports a perfectly healthy
version run as a broken publish, which is what happened on the first real run
(PR #34 / v4.2.3). The `--version-pr` argument is the only input that
distinguishes the two, so it decides first.

Usage::

    python3 scripts/release/assert_release_state.py --version-pr "$PR_NUMBER"
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Callable, List, NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[2]
CHANGESET_DIR = REPO_ROOT / ".changeset"

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_version_consistency import collect  # noqa: E402


class Decision(NamedTuple):
    """The verdict on a release run."""

    ok: bool
    message: str


def decide(
    version: str,
    version_pr: str,
    pending: List[str],
    tag_exists: Callable[[str], bool],
) -> Decision:
    """Classify a release run. Pure: all inputs are arguments.

    Args:
        version: the version the working tree carries *after* the release step.
        version_pr: changesets/action's ``pullRequestNumber`` output. Non-empty
            means a "Version Packages" PR was opened or updated.
        pending: names of changeset files still on disk.
        tag_exists: called with ``vX.Y.Z``; True if the tag is on origin. Only
            consulted when there is no Version PR, so a version run never needs
            to reach the network.

    Returns:
        A Decision whose ``ok`` drives the exit code.
    """
    tag = "v" + version

    # 1. A Version PR settles it. Do NOT look at `pending` or `version` here:
    #    the version command has already consumed the changesets and bumped the
    #    tree, so both describe the *next* release, not this run.
    if version_pr:
        return Decision(
            True,
            f"Version Packages PR #{version_pr} is open, proposing version "
            f"{version}.\nMerge it to publish. Nothing published by this run -- "
            "by design.",
        )

    # 2. No Version PR, yet changesets are still sitting on disk. The version
    #    command never ran (or ran and produced nothing), so nothing will ever
    #    be released from these changesets.
    if pending:
        return Decision(
            False,
            f"{len(pending)} pending changeset(s) ({', '.join(pending)}) but "
            "changesets/action did not open or update a 'Version Packages' PR. "
            "The version step did not do its job -- check the changesets/action "
            "logs above. Nothing was released.",
        )

    # 3. No Version PR and no pending changesets: this was a publish run, so the
    #    tree's version must now be tagged. publish.sh pushes the tag only after
    #    the feed has confirmed it serves the wheel, so a missing tag means the
    #    publish did not complete -- however green the steps above looked.
    if not tag_exists(tag):
        return Decision(
            False,
            f"no pending changesets and the tree is at version {version}, but "
            f"tag {tag} does not exist on origin. That means the publish step "
            "did not run to completion -- the wheel was NOT confirmed on the "
            "Azure Artifacts feed. Do not treat this run as a release. Re-run "
            "the workflow (workflow_dispatch) once the cause is fixed.",
        )

    return Decision(
        True,
        f"Version {version} is tagged {tag} and was published. Nothing further to do.",
    )


def pending_changesets() -> List[str]:
    """Markdown files in .changeset/ that are actual changesets."""
    if not CHANGESET_DIR.is_dir():
        return []
    return sorted(
        path.name
        for path in CHANGESET_DIR.glob("*.md")
        if path.name.lower() != "readme.md"
    )


def tag_exists(tag: str) -> bool:
    result = subprocess.run(
        ["git", "ls-remote", "--tags", "origin", "refs/tags/" + tag],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        sys.stderr.write(
            f"error: could not query remote tags: {result.stderr.strip()}\n"
        )
        raise SystemExit(1)
    return bool(result.stdout.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version-pr",
        default="",
        help="pullRequestNumber output from changesets/action (may be empty)",
    )
    args = parser.parse_args()

    versions = collect()
    if len(set(versions.values())) != 1:
        sys.stderr.write(
            f"::error::version files disagree after the release step: {versions}\n"
        )
        return 1

    decision = decide(
        version=versions["pyproject.toml"],
        version_pr=args.version_pr.strip(),
        pending=pending_changesets(),
        tag_exists=tag_exists,
    )

    if decision.ok:
        print(decision.message)
        return 0

    sys.stderr.write(f"::error::{decision.message}\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
