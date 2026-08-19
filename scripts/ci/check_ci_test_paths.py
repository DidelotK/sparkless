#!/usr/bin/env python3
"""Assert every test path named in CI actually holds collectible tests.

A CI job that names a directory which no longer exists collects nothing and,
depending on how the step is written, can still report success. That is exactly
what happened to the ``test-compatibility`` job: ``tests/compatibility/`` was
moved to ``tests/archive/compatibility/`` in December 2025 and the job kept
running -- green -- against a path that had not existed for months.

The defect is not "someone forgot to update a path". It is that nothing in the
repository connects the paths in ``.github/workflows/ci.yml`` to the paths on
disk, so the two can drift silently and forever. This closes that loop: every
``pytest <path>`` argument in the workflow must resolve to a directory that
contains at least one ``test_*.py`` file.

Usage::

    python3 scripts/ci/check_ci_test_paths.py
"""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"

# Matches the path argument of a pytest invocation, e.g.
#   python3 -m pytest tests/unit/ -v -n 8 ...
# Only the first positional argument is taken; flags are ignored. The `-m pytest`
# form is required so that prose mentioning pytest -- including the comments in
# ci.yml explaining this very check -- is not mistaken for an invocation.
PYTEST_INVOCATION = re.compile(r"python3?\s+-m\s+pytest\s+(?P<path>[^\s\-][^\s]*)")

TEST_FILE_GLOB = "test_*.py"

# pytest exits 5 when it collected nothing. Two jobs used to translate that into
# success, which is how an empty suite reported green. Treating 5 as a pass is
# never correct here: it converts "this job verified nothing" into "this job
# passed". If a suite is genuinely allowed to be empty, delete the job.
EXIT_CODE_5_TOLERANCE = re.compile(r"exit_?code\s*(?:-eq|==)\s*5|\$\?\s*-eq\s*5")


def _pytest_paths(workflow_text: str) -> "list[str]":
    """Every distinct path argument passed to pytest in the workflow."""
    seen: dict[str, None] = {}
    for line in workflow_text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        for match in PYTEST_INVOCATION.finditer(line):
            seen.setdefault(match.group("path"), None)
    return list(seen)


def main() -> int:
    if not WORKFLOW.is_file():
        sys.stderr.write(f"error: {WORKFLOW} is missing\n")
        return 1

    workflow_text = WORKFLOW.read_text(encoding="utf-8")

    tolerated = [
        line.strip()
        for line in workflow_text.splitlines()
        if not line.lstrip().startswith("#") and EXIT_CODE_5_TOLERANCE.search(line)
    ]
    if tolerated:
        sys.stderr.write(
            f"error: {WORKFLOW.name} treats pytest exit code 5 (no tests collected) "
            "as success.\n"
        )
        for line in tolerated:
            sys.stderr.write(f"  - {line}\n")
        return 1

    paths = _pytest_paths(workflow_text)
    if not paths:
        sys.stderr.write(
            f"error: no pytest invocation found in {WORKFLOW.name}; either the "
            "workflow stopped running tests or this check's regex is stale\n"
        )
        return 1

    failures = []
    for raw in paths:
        target = REPO_ROOT / raw
        if not target.exists():
            failures.append(f"{raw}: does not exist")
            continue
        if target.is_file():
            continue
        if not any(target.rglob(TEST_FILE_GLOB)):
            failures.append(f"{raw}: contains no {TEST_FILE_GLOB} file")

    if failures:
        sys.stderr.write(
            f"error: {WORKFLOW.name} runs pytest against paths that collect nothing.\n"
            "A suite that collects nothing is a failure, not a pass.\n"
        )
        for failure in failures:
            sys.stderr.write(f"  - {failure}\n")
        return 1

    print(f"ok: {len(paths)} CI test path(s) resolve to collectible tests")
    for raw in paths:
        print(f"  - {raw}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
