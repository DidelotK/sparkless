# Archived Tests

These tests are **not** dead. They run every night, in the `test-archive` job
(`.github/workflows/nightly-archive.yml`), held to the shrink-only allowlist in
[`known-failures.txt`](known-failures.txt).

Between **2025-12-14** and **2026-08-19** no CI job ran this directory at all.
That gap is how four real sparkless defects reached a consumer while the test
suite reported green — see "What this directory is for" below.

## Why it is not in the PR path — and why "too slow" is not the reason

Measured on the current tree, identical flags for both
(`pytest <dir> -o addopts= --no-cov -q --tb=no -n 8 --dist loadfile`):

| directory | tests | wall |
|---|---:|---:|
| `tests/archive` | 1330 | **4.4 s** |
| `tests/unit` + `tests/parity` + `tests/documentation` (all of PR CI) | 1710 | **9.3 s** |

The archive costs **less than half** of what CI already runs on every pull
request (11.9 s even serially). Nothing in `d608d2e`, the commit that archived
it, mentions duration. **"It was excluded because it would slow PRs down" is a
story, not a finding** — it is written down here because it is the explanation
the next person will reach for.

The actual reason is that **167 of the 1330 tests fail**. That is why the
directory runs nightly against an allowlist rather than on the PR path: a red
suite cannot gate merges until it is green, but it can still be watched.

## Why it is not deleted

**92.3 % of these tests have no same-named equivalent** anywhere in
`tests/unit/` or `tests/parity/` — 114 of 1488 test-function names overlap, and
85 of the 108 files overlapped by nothing at all. The claim in this file's
previous version, that everything here had been migrated to `tests/parity/`, is
not supported by the data.

Roughly **46 of the 167 failures point at genuine sparkless defects that no live
test covers** (`F.array_min`, `F.array_max` and `F.flatten` have zero hits in
`tests/unit/` and `tests/parity/` combined). Deleting the directory would delete
the only evidence of those gaps. They are filed on
`Solya-app/solya-data-platform`:

| issue | defect |
|---|---|
| [#2417](https://github.com/Solya-app/solya-data-platform/issues/2417) | `F.struct` renames aliased literals to `col1`; struct columns typed `StringType` |
| [#2418](https://github.com/Solya-app/solya-data-platform/issues/2418) | `F.expr` drops function arguments and mis-binds operator precedence |
| [#2419](https://github.com/Solya-app/solya-data-platform/issues/2419) | `F.exists`/`forall`/`filter` return NULL for every row |
| [#2420](https://github.com/Solya-app/solya-data-platform/issues/2420) | `F.flatten` returns NULL, so `array_distinct(flatten(...))` keeps duplicates |

Closing one of those is what shrinks the allowlist.

## The allowlist is a ratchet

`known-failures.txt` is keyed by pytest node id, so line drift is a no-op — the
same shape as data-platform's `.linters/*-baseline` files.

- a failure **in** the list is tolerated;
- a failure **outside** the list fails the nightly — that is new breakage;
- a listed entry that **stops failing** fails the nightly as stale, so the list
  is trimmed instead of rotting into a place where breakage can hide;
- **zero tests collected fails the nightly.** A suite that collects nothing is a
  failure, not a pass.

```bash
python3 scripts/ci/check_archive_baseline.py          # what the nightly runs
python3 scripts/ci/check_archive_baseline.py --prune  # drop stale entries
```

There is deliberately **no flag that adds entries**. Growing the allowlist means
hand-editing a checked-in file, where it shows up in a pull request diff.

## What was deleted, and why

16 files (239 test functions) covered modules that no longer exist in sparkless
— `sparkless.backend`, the logical plan, the plan adapter — and failed at import
rather than at assertion. Dead coverage of removed code; removed in the same
change that wired up this job.

## Running it locally

```bash
pytest tests/archive/                 # works; the exclusion is scoped to the default run
pytest                                # excludes tests/archive via norecursedirs
```

`--ignore=tests/archive` was removed from `addopts`: it was redundant with
`norecursedirs` for the default run, never affected an explicit path, and read
as though it locked the directory out entirely.

## Contents

| directory | files | origin |
|---|---:|---|
| `unit/` | 55 | original unit tests, superseded where a parity equivalent exists |
| `compatibility/` | 37 | original compatibility tests, run against recorded PySpark 3.5 outputs |

Archived by `d608d2e` (2025-12-14). Wired into CI 2026-08-19.
