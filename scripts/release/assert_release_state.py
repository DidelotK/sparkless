#!/usr/bin/env python3
"""Fail the release workflow when nothing happened but something should have.

This is the guard that makes a green Release run mean something. It runs after
changesets/action and re-derives, from repository state alone, what the run was
supposed to accomplish -- then asserts it did.

Exactly two outcomes are acceptable:

1. There are pending changesets  -> a "Version Packages" PR must exist.
2. There are no pending changesets -> the version in the tree must be tagged,
   which (see scripts/release/publish.sh) only happens after the wheel has been
   uploaded AND the feed confirmed to serve it.

Anything else means the machinery broke, and this exits non-zero.

The setup this replaces could not tell those apart: it decided whether to
publish by grepping semantic-release's console output for a sentence, and a
miss produced `new_release_published=false`, which *skipped* the publish job.
A skipped job is green. So "no release was due" and "the release broke" looked
identical from the outside. That ambiguity is the reason for this file.

Usage::

    python3 scripts/release/assert_release_state.py --version-pr "$PR_NUMBER"
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[2]
CHANGESET_DIR = REPO_ROOT / ".changeset"

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_version_consistency import collect  # noqa: E402


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


def fail(message: str) -> int:
    sys.stderr.write(f"::error::{message}\n")
    return 1


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
        return fail(f"version files disagree after the release step: {versions}")
    version = versions["pyproject.toml"]
    tag = "v" + version

    pending = pending_changesets()
    version_pr = args.version_pr.strip()

    if pending:
        if not version_pr:
            return fail(
                "{} pending changeset(s) ({}) but changesets/action did not open "
                "or update a 'Version Packages' PR. The version step did not do "
                "its job -- check the changesets/action logs above. Nothing was "
                "released.".format(len(pending), ", ".join(pending))
            )
        print(
            f"{len(pending)} pending changeset(s); Version Packages PR #{version_pr} is open.\n"
            "Merge it to publish the next version. Nothing published by this "
            "run -- by design."
        )
        return 0

    # No pending changesets: the tree's version is the released version.
    if not tag_exists(tag):
        return fail(
            f"no pending changesets and the tree is at version {version}, but tag {tag} "
            "does not exist on origin. That means the publish step did not run "
            "to completion -- the wheel was NOT confirmed on the Azure "
            "Artifacts feed. Do not treat this run as a release. Re-run the "
            "workflow (workflow_dispatch) once the cause is fixed."
        )

    print(
        f"Version {version} is tagged {tag} and was published. Nothing further to do."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
