# Contributing to Sparkless

## Setup

```bash
pip install -e ".[dev]"
pre-commit install && pre-commit install --hook-type pre-push
```

## Before you push

```bash
python3 -m ruff format .
python3 -m ruff check .
python3 -m mypy sparkless tests
python3 -m pytest tests/unit/ -q -o "addopts=" -m "not performance"
```

CI runs `mypy sparkless tests` with `disallow_untyped_decorators = true`, so
`@pytest.mark.parametrize` needs `# type: ignore[misc,untyped-decorator]`.

## Before you debug a "function X returns NULL" report

Read this first. It is the single most common shape of sparkless bug, it has
one mechanism, and the mechanism is not the function you are looking at.

### There are two expression evaluators, and their function tables have drifted

| evaluator | file | serves | answer when the function is missing |
|---|---|---|---|
| `ConditionEvaluator` | `sparkless/core/condition_evaluator.py` | the lazy `select` path | **silent NULL** — a `logger.debug` line, nothing else |
| `ExpressionEvaluator` | `sparkless/dataframe/evaluation/expression_evaluator.py` | the `withColumn` path | **its own first operand**, with a `UserWarning` — except for the boolean and predicate operations, which return NULL silently |

`ConditionEvaluator._evaluate_column_operation_value` dispatches on a
hand-maintained whitelist of operation names. Anything absent from it falls
through to `return None`. `ExpressionEvaluator` dispatches on
`self._function_registry`. Neither table is derived from the other, so a
function added to one is not added to the other, and the same expression
answers differently depending on how it was projected:

```python
df.select(F.flatten(x))            # None          <- ConditionEvaluator
df.withColumn("f", F.flatten(x))   # [1, 2, 3, 4]  <- ExpressionEvaluator
```

**How large the drift is, measured on `main` (`d3b0809`, 2026-08-19):** of 30
functions sampled from those `ExpressionEvaluator` implements, **26 return
NULL** when projected through `select` — `atan2`, `hypot`, `signum`, `cbrt`,
`log1p`, `expm1`, `pmod`, `bit_length`, `octet_length`, `btrim`, `left`,
`right`, `position`, `split_part`, `find_in_set`, `format_string`, `hash`,
`typeof`, `to_char`, `try_add`, `try_divide`, `equal_null`, `arrays_zip`,
`sequence`, `map_from_arrays`, `to_utc_timestamp`. Each of those is a future
"function X returns NULL" issue that has not been filed yet. This is a
backlog, not a list of exceptions.

### Why it stays invisible

NULL is a legitimate value, so nothing downstream can tell an unimplemented
function from a computed one. `F.exists(...)` answering NULL made a
data-integrity guard report "clean" for every input without ever raising; the
counting aggregates answering `0` made a validation counter report "nothing
wrong" because it had measured nothing. Both were green in the unit tier.

### Diagnosing one in thirty seconds

```python
import logging; logging.basicConfig(level=logging.DEBUG)
df.select(F.the_function(col)).collect()
```

```
DEBUG:sparkless.core.condition_evaluator:ConditionEvaluator has no value handler
for operation 'to_json'; returning NULL. This is an unimplemented operation,
not a SQL NULL.
```

If that line appears, the function is missing from the whitelist and the bug
is not in the function's builder in `sparkless/functions/`.

### Fixing one

Do **not** add a third implementation. Put the semantics in a canonical
module under `sparkless/core/` and have both evaluators call it. This is the
established pattern; the existing modules are:

| module | owns | since |
|---|---|---|
| `core/struct_builder.py` | `struct` / `named_struct` field naming and values | already on `main` |
| `core/array_values.py` | `flatten`, `array_min`, `array_max`, `slice`, `array_distinct` | #47 |
| `core/higher_order.py` | `exists`, `forall`, `filter`, `transform` | #48 |
| `core/aggregate_values.py` | reading an aggregate target that is an expression | #50 |
| `core/json_values.py` | `to_json` rendering | #51 |

Then wire `ConditionEvaluator` (whitelist entry **and** handler) and
`ExpressionEvaluator` (registry entry), and write the test so it passes under
`MOCK_SPARK_TEST_BACKEND=pyspark` as well. Where sparkless and PySpark
disagree, PySpark decides — measure the expected value against real PySpark
before asserting it, including the NULL and empty edge cases, which are where
these functions differ from each other.

A test asserting only `select` would not have caught any of the drift above.
Assert that `select` and `withColumn` agree.

### Not just `select` vs `withColumn`

The same "read it by name and hope" shape appears in the aggregate
dispatchers (`sparkless/dataframe/grouped/base.py` and its independent copy in
`grouped/pivot.py`): an aggregate whose target is an *expression* has no
column of that name to look up, so the lookup misses on every row and the
aggregate returns its empty default — `0` for the counting ones, `None` for
`avg`/`max`/`min`. Gate on `is_row_evaluatable_expression` and evaluate per
row, as `sum`/`avg`/`max` already do.

## Add a changeset

**Any PR that changes runtime behaviour needs a changeset in the same PR.** This
is what decides the next version number and what goes in `CHANGELOG.md`.

```bash
pnpm install        # once
pnpm changeset      # interactive: pick the bump, write a one-line summary
```

Or write the file by hand — a changeset is just markdown in `.changeset/`:

```markdown
---
"sparkless": patch
---

One-line summary of the user-visible change.
```

| Bump | When |
|------|------|
| `patch` | Bug fix, closer PySpark parity, no new API |
| `minor` | New function / new API surface, backwards compatible |
| `major` | Behaviour callers depend on changes, or API is removed |

Docs-only, CI-only and test-only PRs do not need one.

Commit messages no longer drive versioning — `fix:` / `feat:` prefixes are still
nice for readability, but a PR without a changeset ships nothing to the feed.

## How a release is cut

Merging to `main` opens a **"Version Packages"** PR that bumps the version and
writes the changelog. Merging *that* PR builds the wheel, uploads it to the
Azure Artifacts feed, verifies the feed serves it, and tags the release.

Full details, including how the workflow fails loudly when a publish does not
happen: [`docs/release-process.md`](docs/release-process.md).
