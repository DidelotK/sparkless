#!/usr/bin/env python3
"""Assert every test path named in CI actually holds collectible tests.

A CI job that names a directory which no longer exists collects nothing and,
depending on how the step is written, can still report success. That is exactly
what happened to the ``test-compatibility`` job: ``tests/compatibility/`` was
moved to ``tests/archive/compatibility/`` in December 2025 and the job kept
running -- green -- against a path that had not existed for months.

The defect is not "someone forgot to update a path". It is that nothing in the
repository connects the paths in ``.github/workflows/`` to the paths on disk, so
the two can drift silently and forever. This closes that loop: every
``pytest <path>`` argument in **every** workflow must resolve to a directory that
contains at least one ``test_*.py`` file.

A test path can also live in a script rather than in YAML -- the nightly archive
job runs ``scripts/ci/check_archive_baseline.py``, which invokes pytest itself.
Exempting it would recreate exactly the blind spot this check exists to close, so
its ``ARCHIVE_DIR`` is imported and held to the same rule.

Usage::

    python3 scripts/ci/check_ci_test_paths.py
"""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_archive_baseline import ARCHIVE_DIR  # noqa: E402

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


def _collects_tests(raw: str) -> "str | None":
    """Describe why ``raw`` collects nothing, or None if it is fine."""
    target = REPO_ROOT / raw
    if not target.exists():
        return f"{raw}: does not exist"
    if target.is_file():
        return None
    if not any(target.rglob(TEST_FILE_GLOB)):
        return f"{raw}: contains no {TEST_FILE_GLOB} file"
    return None


def main() -> int:
    workflows = sorted(WORKFLOW_DIR.glob("*.yml")) + sorted(WORKFLOW_DIR.glob("*.yaml"))
    if not workflows:
        sys.stderr.write(f"error: no workflow files found in {WORKFLOW_DIR}\n")
        return 1

    # scripts/ci/check_archive_baseline.py names its own pytest target, so it is
    # checked alongside the YAML rather than exempted from the rule.
    paths: dict[str, str] = {
        str(ARCHIVE_DIR.relative_to(REPO_ROOT)): "scripts/ci/check_archive_baseline.py"
    }
    tolerated: list[str] = []

    for workflow in workflows:
        text = workflow.read_text(encoding="utf-8")
        for line in text.splitlines():
            if line.lstrip().startswith("#"):
                continue
            if EXIT_CODE_5_TOLERANCE.search(line):
                tolerated.append(f"{workflow.name}: {line.strip()}")
        for raw in _pytest_paths(text):
            paths.setdefault(raw, workflow.name)

    if tolerated:
        sys.stderr.write(
            "error: a workflow treats pytest exit code 5 (no tests collected) "
            "as success.\n"
        )
        for line in tolerated:
            sys.stderr.write(f"  - {line}\n")
        return 1

    if len(paths) < 2:
        sys.stderr.write(
            "error: no pytest invocation found in any workflow; either CI stopped "
            "running tests or this check's regex is stale\n"
        )
        return 1

    failures = [
        f"{problem}  (named by {source})"
        for raw, source in paths.items()
        if (problem := _collects_tests(raw)) is not None
    ]

    if failures:
        sys.stderr.write(
            "error: CI runs pytest against paths that collect nothing.\n"
            "A suite that collects nothing is a failure, not a pass.\n"
        )
        for failure in failures:
            sys.stderr.write(f"  - {failure}\n")
        return 1

    print(f"ok: {len(paths)} CI test path(s) resolve to collectible tests")
    for raw, source in paths.items():
        print(f"  - {raw}  ({source})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
