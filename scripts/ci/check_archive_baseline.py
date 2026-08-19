#!/usr/bin/env python3
"""Run tests/archive/ and hold its failures to a shrink-only allowlist.

``tests/archive/`` holds 92 test files that no CI job ran between 2025-12-14 and
2026-08-19. Two measurements decide how it is wired in, and they are recorded
here so the next person does not have to re-derive them. Same flags for both,
``pytest <dir> -o addopts= --no-cov -q --tb=no -n 8 --dist loadfile``:

===================================================  ==========  =======
directory                                            tests       wall
===================================================  ==========  =======
``tests/archive``                                    1330        4.4 s
``tests/unit`` + ``tests/parity`` + ``documentation``  1710        9.3 s
===================================================  ==========  =======

- **The archive costs less than half of what CI already runs on every PR**
  (11.9 s serial). "It was excluded because it was too slow for PRs" was never
  true, and nothing in the commit that archived it mentions duration. Expect to
  meet that story again; these are the numbers that answer it.
- **92.3 % of its tests have no same-named equivalent** in tests/unit/ or
  tests/parity/ (114 of 1488 names overlap; 85 of 108 files overlapped by zero).
  The archive README's claim that everything was migrated to tests/parity/ is not
  supported by the data, so deleting the directory would drop real coverage.

What kept it out of CI is not cost, it is that it fails: 167 of 1330 tests are
red, and roughly 46 of those are pointing at genuine sparkless defects with no
live test anywhere (filed as solya-data-platform #2417-#2420). So it cannot go
into the PR path as-is, and it must not be deleted. It runs nightly instead,
against the allowlist in ``tests/archive/known-failures.txt``.

The allowlist is a ratchet, keyed by pytest node id so that line drift is a
no-op -- the same shape as data-platform's ``.linters/*-baseline`` files:

- a failure **in** the list is tolerated;
- a failure **outside** the list fails the job -- that is new breakage;
- a listed entry that **stops failing** fails the job as stale, so the list is
  trimmed rather than left to rot into a place where breakage can hide;
- **zero tests collected fails the job**. A suite that collects nothing is a
  failure, not a pass -- the whole reason this audit happened.

``--prune`` rewrites the file with the stale entries removed. There is
deliberately no flag that *adds* entries: growing the allowlist means editing a
checked-in file by hand, where it shows up in a pull request diff.

Usage::

    python3 scripts/ci/check_archive_baseline.py            # check (CI)
    python3 scripts/ci/check_archive_baseline.py --prune    # drop stale entries
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# The path the nightly job exercises. scripts/ci/check_ci_test_paths.py imports
# this constant and holds it to the same "must collect tests" rule as the paths
# named in the workflows, so a typo here fails the PR gate rather than producing
# a nightly that quietly tests nothing.
ARCHIVE_DIR = REPO_ROOT / "tests" / "archive"

BASELINE = ARCHIVE_DIR / "known-failures.txt"

# addopts is cleared deliberately. The project addopts carry -v, which cancels
# the -q this script needs to get one node id per line out of --collect-only, and
# a coverage config the nightly has no use for. Everything this run depends on is
# therefore stated explicitly below rather than inherited.
PYTEST_BASE = [
    sys.executable,
    "-m",
    "pytest",
    str(ARCHIVE_DIR.relative_to(REPO_ROOT)),
    "-o",
    "addopts=",
    "--no-cov",
    "-p",
    "no:cacheprovider",
    "--timeout=300",
    "--timeout-method=thread",
]

HEADER = """\
# Known failures in tests/archive/, tolerated by the nightly test-archive job.
#
# This list may only SHRINK. A failure outside it fails the nightly; an entry
# here that stops failing also fails the nightly, as stale. Run
# `python3 scripts/ci/check_archive_baseline.py --prune` to drop stale entries.
#
# Roughly 46 of these are real sparkless defects with no live test elsewhere,
# tracked on Solya-app/solya-data-platform:
#   #2417  F.struct renames aliased literals to col1; struct typed StringType
#   #2418  F.expr drops function arguments and mis-binds operator precedence
#   #2419  F.exists/forall/filter return NULL for every row
#   #2420  F.flatten returns NULL, so array_distinct(flatten(...)) keeps dupes
# Closing one of those issues is what shrinks this list.
#
# The rest are tests written against APIs that have since changed shape (the
# inferSchema reader contract, SparkSession(db_path=...), raise-on-invalid
# behaviour, type-inference expectations). Those need a decision per group, not
# a fix here.
"""


def _read_baseline() -> "list[str]":
    if not BASELINE.is_file():
        return []
    entries = []
    for line in BASELINE.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            entries.append(stripped)
    return entries


def _run(extra: "list[str]") -> str:
    completed = subprocess.run(  # noqa: S603
        PYTEST_BASE + extra,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return completed.stdout + completed.stderr


def _collected() -> "set[str]":
    output = _run(["--collect-only", "-q"])
    return {
        line.strip()
        for line in output.splitlines()
        if "::" in line and line.strip().startswith("tests/")
    }


def _failed() -> "set[str]":
    output = _run(["-q", "--tb=no", "-rfE", "-n", "8", "--dist", "loadfile"])
    failed = set()
    for line in output.splitlines():
        for prefix in ("FAILED ", "ERROR "):
            if line.startswith(prefix):
                failed.add(line[len(prefix) :].split(" - ")[0].strip())
    return failed


def _write(entries: "list[str]") -> None:
    BASELINE.write_text(
        HEADER + "\n" + "\n".join(sorted(entries)) + "\n", encoding="utf-8"
    )


def main() -> int:
    prune = "--prune" in sys.argv[1:]

    collected = _collected()
    if not collected:
        sys.stderr.write(
            f"error: {ARCHIVE_DIR.relative_to(REPO_ROOT)} collected no tests.\n"
            "A suite that collects nothing is a failure, not a pass.\n"
        )
        return 1

    baseline = _read_baseline()
    failed = _failed()

    unexpected = sorted(failed - set(baseline))
    stale = sorted(entry for entry in baseline if entry not in failed)

    if prune:
        kept = [entry for entry in baseline if entry in failed]
        _write(kept)
        print(
            f"pruned {len(stale)} stale entr{'y' if len(stale) == 1 else 'ies'}; "
            f"{len(kept)} remain"
        )
        for entry in stale:
            print(f"  - {entry}")
        return 0

    if unexpected:
        sys.stderr.write(
            f"error: {len(unexpected)} test(s) failed that are not in "
            f"{BASELINE.relative_to(REPO_ROOT)}.\n"
            "This is new breakage. Fix it -- do not add it to the allowlist "
            "unless a maintainer has agreed the failure is expected.\n"
        )
        for entry in unexpected:
            sys.stderr.write(f"  + {entry}\n")

    if stale:
        sys.stderr.write(
            f"error: {len(stale)} entr{'y' if len(stale) == 1 else 'ies'} in "
            f"{BASELINE.relative_to(REPO_ROOT)} no longer fail(s) "
            "(they now pass, are skipped, or no longer exist).\n"
            "The allowlist may only shrink. Run "
            "`python3 scripts/ci/check_archive_baseline.py --prune`.\n"
        )
        for entry in stale:
            sys.stderr.write(f"  - {entry}\n")

    if unexpected or stale:
        return 1

    print(
        f"ok: {len(collected)} archive tests collected, {len(failed)} failed, "
        f"all {len(baseline)} in the allowlist, none stale"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
