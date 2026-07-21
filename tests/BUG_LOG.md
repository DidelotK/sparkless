# Sparkless Bug Log

This file tracks bugs and issues discovered during test refactoring and development.

**Last Updated**: 2026-07-21  
**Context**: Unified PySpark Parity Testing Refactor  
**Total Bugs Logged**: 26
**Total Bugs Logged**: 25
**Total Bugs Logged**: 29
**Total Bugs Logged**: 42
**Total Bugs Logged**: 27
**Total Bugs Logged**: 48
**Total Bugs Logged**: 49

---

## Critical Issues

### BUG-051: A nested CASE WHEN was never evaluated
**Status**: Fixed
**Severity**: Critical
**Discovered**: 2026-07-21
**Files**: `sparkless/dataframe/evaluation/evaluators/conditional_evaluator.py`,
`sparkless/functions/conditional.py`

A `when` branch whose value is *itself* a `when(...).otherwise(...)` was not
recognised as an expression. `CaseWhen` is neither a `Column` nor a
`ColumnOperation`, and the two evaluation paths each mishandled it differently:

```python
inner  = F.when(F.col("a") <= F.col("ub"), F.lit(1)).otherwise(F.lit(0))
nested = F.when(F.col("lb").isNotNull(), inner).otherwise(F.lit(0))

df.select(nested)      # sparkless [None, None, 0]   PySpark [1, 0, 0]
df.agg(F.sum(nested))  # sparkless ColumnOperation   PySpark 1
```

* `ConditionalEvaluator.evaluate_case_when` fell through to its terminal
  `return value` and handed back the **unevaluated `CaseWhen` object**. `F.sum`
  then folded that object into its accumulator (`acc + CaseWhen`), yielding a
  `ColumnOperation` where a number belonged. Downstream this surfaced as
  `TypeError: int() argument must be ... not 'ColumnOperation'` — the only loud
  symptom of the whole family.
* `CaseWhen._evaluate_value` matched the `hasattr(value, "name")` fallback,
  because a `CaseWhen` carries a generated `.name` (`"CASE WHEN ... END"`). It
  was looked up as though it were a *column of that name*; no such column
  exists, so the branch silently evaluated to NULL.

Same family as BUG-038/BUG-046/BUG-050: an unhandled expression type reaching a
terminal `return value`, answering plausibly rather than failing.

**Fix**: both paths now dispatch a nested `CaseWhen` to a real evaluation —
`_resolve_branch_value` recurses via the base evaluator, and
`CaseWhen._evaluate_value` recurses via `value.evaluate(row)`. The
`ColumnOperation` branch also stopped falling through to an implicit `None`.

**Regression tests**: `tests/parity/functions/test_nested_case_when_parity.py`
(11 tests, passing under both engines).

### BUG-052: `last_day` and `trunc` returned NULL for every row
**Status**: Fixed
**Severity**: Critical
**Discovered**: 2026-07-21
**Files**: `sparkless/core/condition_evaluator.py`,
`sparkless/dataframe/evaluation/expression_evaluator.py`,
`sparkless/core/datetime_utils.py` (new)

Both functions are exported from the public API and marked supported in
`PYSPARK_FUNCTION_MATRIX.md`, but neither had an evaluator implementation. Each
built a `ColumnOperation` (`"last_day"`, `"trunc"`) that matched no dispatch
branch in either evaluator, so both answered NULL — for any input type,
including a correctly-typed `DateType` column:

```python
df.select(F.last_day(F.col("d")))        # sparkless None;  PySpark 2026-01-31
df.select(F.trunc(F.col("d"), "month"))  # sparkless None;  PySpark 2026-01-01
df.select(F.date_add(F.col("d"), 6))     # control: works
```

**Reference behaviour** (PySpark 4.0.0 / OpenJDK 21): both are DATE-valued even
for a TIMESTAMP operand. `trunc` accepts `year`/`yyyy`/`yy`, `month`/`mon`/`mm`,
`week` and `quarter`, case-insensitively; `week` truncates to **Monday**; an
unrecognised unit yields NULL rather than raising.

**Fix**: `spark_last_day()` / `spark_trunc()` in the new
`sparkless/core/datetime_utils.py`, registered in **both** evaluators so the
predicate path and the `withColumn` path cannot drift.

**Regression tests**: `tests/parity/functions/test_date_trunc_last_day_parity.py`
(passing under both engines).

### BUG-053: date predicates collapsed to NULL, silently emptying filters
**Status**: Fixed
**Severity**: Critical
**Discovered**: 2026-07-21
**Files**: `sparkless/core/condition_evaluator.py`, `sparkless/core/datetime_utils.py`

Found while fixing BUG-052: correct `last_day` values still produced an empty
frame, because the *comparison* was broken in two independent ways.

1. **The cast was dropped on the predicate path.** `_evaluate_column_operation`'s
   legacy string-type-name cast chain handled `long`/`int`/`double`/`string`/
   `boolean` and fell to `else: return value` for everything else — so
   `F.lit("2026-01-01").cast("date")` stayed a **`str`** when resolved as a
   comparison operand, while the projection path produced a real `date`.
2. **No temporal coercion in the comparison kernel.** `_coerce_for_comparison`
   reconciled string/numeric pairs only. `date >= str` therefore raised
   `TypeError`, which `_evaluate_comparison` converts to NULL by design.

Net effect, with no error anywhere:

```python
df.filter(F.last_day(F.col("d")) >= F.lit("2026-01-01").cast("date"))
# sparkless 0 rows;  PySpark 1 row
```

This is the worst failure shape for a mock: on a real cluster the code is
correct, on sparkless it quietly returns an empty result.

**Reference behaviour**: Spark implicitly casts across the temporal boundary —
`date_col >= '2026-01-01'` compares two DATEs, and a DATE beside a TIMESTAMP is
promoted to midnight.

**Fix**: `date`/`timestamp` now delegate to the same `TypeConverter` the
`DataType` branch already used, and `coerce_temporal_pair()` reconciles
temporal/string and date/datetime pairs before comparison.

**Regression tests**: `tests/parity/functions/test_date_trunc_last_day_parity.py`
(`TestDatePredicateParity`, `TestDateFilterKeepsRows`).

### BUG-046: Predicates and logical connectives returned their operand instead of a boolean
**Status**: Fixed
**Severity**: Critical
**Discovered**: 2026-07-21
**Files**: `sparkless/dataframe/evaluation/expression_evaluator.py`,
`sparkless/core/condition_evaluator.py`

> Numbering note: 001-043 were claimed on `main` or in open PRs, and two
> concurrent sweeps (ordering/NULL semantics, `least`/`greatest`) are taking
> 044/045. Renumber on merge if it still collides.

**Description**:
`ExpressionEvaluator._evaluate_column_operation` — the evaluator behind
`withColumn`, `when(...)` and aggregate projections — had no branch for the
null predicates, `isin`, `eqNullSafe`, or the logical connectives `&` / `|` /
`~`. All of them fell through to `_evaluate_function_call`, whose terminal
`return value` hands back the **operand**:

- `col.isNotNull()` evaluated to the column's own value.
- `a & b` evaluated to `a`; `a | b` evaluated to `a`; `~a` evaluated to `a`.

The leaked value is truthy for any non-null row, so an enclosing `when` /
`filter` matched everything. `a & b` returning `a` also means **every compound
condition in `when()` was silently reduced to its left operand**.

Separately, `ConditionEvaluator._evaluate_column_operation_value` — the
evaluator behind `select` — had no entry for `isin`, `between` or
`eqNullSafe`, so *projecting* those predicates yielded NULL while *filtering*
on them worked: a third dispatch table that had drifted from the other two.
`eqNullSafe` was wrong on the predicate path as well (`_evaluate_comparison`
collapses any NULL operand to `operation == "!="`, so `NULL <=> NULL` was
FALSE and `1.0 <=> 1.0` was NULL).

**Why it stayed hidden**: a downstream validation framework builds its rules
as `F.sum(F.when(F.col(c).isNotNull() & cond, 1).otherwise(0))`. Until
BUG-037/039 were fixed, `F.sum(F.when(...))` returned a constant `0`, so the
invalid-row count was structurally zero and every rule reported "pass". With
the aggregate fixed the arithmetic is finally correct — running on a condition
that was still wrong, which is what surfaced this.

**Reproduction**:
```python
df = spark.createDataFrame([{"price": 100.0}, {"price": 200.0}, {"price": 150.0}])

df.withColumn("c", F.col("price").isNotNull()).collect()
# sparkless -> [100.0, 200.0, 150.0]   PySpark 4.0.0 -> [True, True, True]

df.withColumn("c", (F.col("price") > 1) & (F.col("price") > 1000)).collect()
# sparkless -> [True, True, True]      PySpark 4.0.0 -> [False, False, False]

c = F.col("price").isNotNull() & (F.col("price") < 0)
df.agg(F.sum(F.when(c, 1).otherwise(0)).alias("v")).collect()[0]["v"]
# sparkless -> 3                       PySpark 4.0.0 -> 0
```

**Reference behaviour**: captured from PySpark 4.0.0 on OpenJDK 21
(`local[1]`). Divergences found, per evaluation path:

| expression | `withColumn` | `select` | `filter` | PySpark 4.0.0 |
|---|---|---|---|---|
| `isNotNull()` | operand | ok | ok | `True` |
| `isin([...])` | operand | `NULL` | ok | `True` |
| `eqNullSafe(v)` | operand | `NULL` | wrong | `True` |
| `between(a, b)` | ok | `NULL` | ok | `True` |
| `~pred` | operand | ok | ok | `False` |
| `a & b`, `a \| b` | left operand | ok | ok | combined |
| `isNull()`, `like`, `rlike`, `contains`, `startswith`, `endswith` | ok | ok | ok | ok |

Three-valued logic, also captured from 4.0.0: `isNull` / `isNotNull` /
`isNaN` / `eqNullSafe` are total (never NULL); `between` / `isin` / `like` /
`rlike` / `contains` / `startswith` / `endswith` are NULL over a NULL operand;
SQL `IN` is NULL when there is no match and the list holds a NULL.

**The fix**:
- `ExpressionEvaluator` gains `_evaluate_predicate_operation`, dispatched from
  `_evaluate_column_operation` *before* the function-registry lookup, covering
  the connectives (through the existing Kleene helpers) and the null /
  null-safe-equality / membership predicates.
- The identity fallback in `_evaluate_function_call` now returns NULL rather
  than the operand when the operation is BOOLEAN-typed, so this class of
  defect cannot recur silently for an operation added later.
- `ConditionEvaluator._evaluate_column_operation_value` delegates every
  predicate to `_evaluate_column_operation` instead of keeping a second,
  drifting table.
- `eqNullSafe` and `isnan` are implemented on the predicate path, and
  `isin` / `between` / `like` / `rlike` return NULL rather than FALSE over a
  NULL operand.
- `Column.isNaN()` / `ColumnOperation.isNaN()` / `Literal.isNaN()` added — the
  PySpark method was missing entirely (only `F.isnan` existed).

**Verified by mutation**: reverting the source changes makes 19 of the 46 new
tests in `tests/parity/functions/test_predicate_parity.py` fail, including the
decisive aggregate check. The remaining 27 are guards that must keep passing
(`isNull`, the string predicates, `filter` agreement) so the fix cannot be
"achieved" by making the predicates return NULL for everything. Reverting each
of the three source files independently fails a distinct subset.

---

### BUG-047: `<comparison>` against NULL is FALSE/TRUE on the select and filter paths
**Status**: Fixed — duplicate of BUG-033, fixed by the same change
**Severity**: High
**File**: `sparkless/core/condition_evaluator.py` (`_evaluate_comparison`)

> Filed independently while collapsing the predicate schism (BUG-046), and
> deferred to "the ordering/NULL-semantics sweep" — which is the change that
> fixed BUG-033. Both entries describe the same defect from the two directions
> it was found from; see BUG-033 for the fix and the test coverage.

`_evaluate_comparison` short-circuits `if col_value is None or condition_value
is None: return operation == "!="`. Spark returns NULL for *every* comparison
with a NULL operand, including `!=`. Two consequences, both verified against
PySpark 4.0.0:

```python
# one row, d = NULL
df.filter(F.col("d") != 1).count()      # sparkless 1; PySpark 0
df.select((~(F.col("d") > 1)).alias("c"))  # sparkless True; PySpark NULL
```

The second follows from the first: `_kleene_not` is correct, but it is handed
`False` where it should be handed `None`. `ExpressionEvaluator` already gets
this right (`withColumn` returns NULL), so this is the select/filter path only
— a fourth spelling of the same schism BUG-046 collapsed.

Not fixed by BUG-046's change: it moves rows under every `filter` in the suite
at once and belongs with the ordering/NULL-semantics sweep rather than with a
predicate fix. That sweep is BUG-033: `_evaluate_comparison` is now the single
comparison kernel for `ConditionEvaluator`, returns `None` for a NULL operand,
and `_evaluate_comparison_operation` delegates to it.

---

### BUG-043: `createDataFrame` bound the caller's schema object graph by reference
**Status**: Fixed
**Severity**: Critical
**Discovered**: 2026-07-21
**File**: `sparkless/session/services/dataframe_factory.py`, `sparkless/spark_types.py`

> Numbering note: 001-034 are claimed on `main` or in open PRs (#20/#22/#25 hold
> 026/028/029). 035 was the next free number at the time of writing; renumber on
> merge if it collides.

**Description**:
`spark.createDataFrame(data, schema=...)` stored the caller's `StructType`
**by reference** on the resulting DataFrame — not just the same `StructType`, but
the same `fields` list and the same `StructField` and `DataType` objects:

```python
df = spark.createDataFrame(rows, schema=my_schema)
df.schema is my_schema                       # True   (PySpark: False)
df.schema.fields is my_schema.fields         # True   (PySpark: False)
df.schema.fields[0] is my_schema.fields[0]   # True   (PySpark: False)
```

PySpark round-trips the schema through the JVM, so its DataFrame always owns a
freshly deserialised object graph. Because sparkless did not, **any** in-place
mutation of a bound schema wrote straight into the caller's object.

That matters far beyond the one DataFrame, because of an idiom that looks
completely innocuous:

```python
_SOURCE  = StructType([...])                                       # module level
_DERIVED = StructType([StructField("k", StringType()), *_SOURCE.fields])
```

Unpacking shares the `StructField` **objects** between the two schemas. Binding
`_DERIVED` therefore handed the DataFrame the very objects `_SOURCE` is made of,
so a mutation reached through the bound schema corrupted `_SOURCE` for every
later user in the same process — a module-level schema silently degrading
mid-run.

**Reproduction**:
```python
from sparkless import SparkSession
from sparkless.spark_types import StructType, StructField, StringType, DoubleType

SOURCE = StructType([StructField("a", DoubleType(), nullable=True)])
DERIVED = StructType([StructField("k", StringType(), nullable=False), *SOURCE.fields])

spark = SparkSession("repro")
df = spark.createDataFrame([("k1", 1.0)], schema=DERIVED)

df.schema.add_field(StructField("injected", StringType()))
df.schema.fields[1].nullable = False

len(SOURCE.fields)          # 2     -- PySpark-equivalent: 1
SOURCE.fields[0].nullable   # False -- PySpark-equivalent: True
```

**Reference behaviour**:
Captured from real PySpark 4.0.0 on OpenJDK 21.

| check | PySpark 4.0.0 | Sparkless (before) |
|---|---|---|
| `df.schema is caller_schema` | `False` | `True` |
| `df.schema.fields is caller.fields` | `False` | `True` |
| `df.schema.fields[0] is caller.fields[0]` | `False` | `True` |
| `df.schema == caller_schema` | `True` | `True` |
| `df.schema is df.schema` | `True` | `True` |
| caller field names after mutating `df.schema` | unchanged | **corrupted** |
| source schema after binding a schema derived from it | unchanged | **corrupted** |

Note that `StructType(other.fields)` shares the `fields` **list** in PySpark too
(`self.fields = fields`, no copy), and `StructType.add()` appends to it. That
aliasing is parity and is deliberately left alone; only the *bind* boundary is
changed.

**Impact**:
- Order-dependent, cross-test corruption. The symptom lands in whichever test
  runs after the polluter, so under `pytest -n auto` it reads as an unrelated
  flake and gets retried or quarantined instead of fixed. A downstream data
  platform hit exactly this: a shared module-level schema corrupted 10 other
  tests in the same file, caught only because a pre-push hook happened to run
  the parallel configuration.
- No amount of downstream discipline fixes it — short of forbidding the
  derived-schema idiom entirely, which is what the downstream repo ended up
  doing (an "independent literal" copy-pasted from the source schema).

**Fix**:
- New `copy_schema()` / `copy_struct_field()` / `copy_data_type()` helpers in
  `sparkless/spark_types.py`. `copy_schema` returns a `StructType` that is equal
  to its input but shares nothing with it: new `fields` list, new `StructField`
  objects, new `DataType` objects, copied `metadata` dicts, and recursion
  through `ArrayType` / `MapType` / nested `StructType`.
- `DataFrameFactory.create_dataframe` now binds `copy_schema(schema)`. This is
  the single user-facing bind boundary (`SparkSession.createDataFrame` is its
  only caller); internally constructed DataFrames already build fresh schemas.
- `copy_data_type` tests membership against `(DataType, PySparkDataType)` when
  PySpark is installed, because sparkless's type classes inherit from *PySpark's*
  `DataType` — `isinstance(x, DataType)` against the sparkless class alone
  silently misses genuine PySpark type objects. The PySpark class is only added
  when PySpark is really importable; the ImportError fallback aliases it to
  `object`, which would otherwise match everything.

**Cost**: `copy_schema` is 4.7 µs for a 3-field schema and 28.5 µs for 20 fields,
against ~400 µs for a `createDataFrame` call — a few percent, paid once per bind.
The full `tests/unit` + `tests/parity` suite runs in the same wall clock as
before (1349 passed, 14 skipped, both trees).

**Not the reported cause**: the downstream report attributed the corruption to
sparkless mutating shared `StructField` objects *during* a bind. That specific
mutation does not exist in 4.2.2 — an AST sweep finds only two in-place writes to
a `DataType` (`functions/base.py:87`, `dataframe/lazy.py:3306`, both on
freshly-constructed objects), a reflective sweep over all 97 public `DataFrame`
members leaves a shared source schema byte-identical, and the whole unit+parity
suite under a `__setattr__` watcher records zero post-construction mutations of
`StructField`/`DataType` state. What is real is the **aliasing that makes such a
mutation catastrophic**, and that is what this fix removes: after it, no
sparkless code path — present or future — can reach a caller-owned schema object.

**Regression tests**:
`tests/unit/spark_types/test_schema_binding_isolation.py` — 21 tests. 9 fail on
the unfixed tree (the ownership and leak assertions); the other 12 are guards
that must keep passing (schema equality, `df.schema is df.schema` stability,
value round-trip, empty-schema binding, nested-type copies).

---

### BUG-034: Negation of function results / CASE WHEN ignored three-valued logic
**Status**: Fixed
**Severity**: Critical
**Discovered**: 2026-07-21
**File**: `sparkless/core/condition_evaluator.py`

> Numbering note: BUG-023/024 are on `main`, and open PRs #19/#20/#22/#23/#24
> variously claim 023-033. 034 is the next free number at the time of writing;
> renumber on merge if it collides.

**Description**:
BUG-023 fixed three-valued logic for *bare boolean columns*, but only that operand
shape. Two other shapes of boolean expression remained wrong, and the BUG-023 fix
turned one of them from silently-compensated into visibly wrong:

1. **Function results.** The predicate path (`_evaluate_column_operation`) evaluated
   function operations with the scalar helper
   `_evaluate_function_operation(col_value, op)`, which receives only the *first*
   operand's value and short-circuits `if value is None: return None`. So
   `coalesce(NULL, False)` evaluated to NULL instead of FALSE. Before BUG-023 the row
   still survived `~` because `not None` is `True` — two bugs cancelling. After
   BUG-023, `_kleene_not(None)` correctly returns NULL, so the row was dropped and the
   latent defect became visible.
2. **CASE WHEN.** `CaseWhen.__invert__` emits the operation string `"~"`, which no
   evaluator branch matched (they check `["not", "!"]`). `~F.when(...)` therefore fell
   through to `return False` for every row and the filter returned an empty result.
   `CaseWhen` is also not a `Column` subclass, so as a bare predicate it hit the
   truthiness fallback and was `True` for every row.

**Reproduction**:
```python
df = spark.createDataFrame(
    [("obs", False), ("fc", True), ("legacy", None)],
    StructType([StructField("vid", StringType()), StructField("is_forecast", BooleanType())]),
)
df.filter(~F.coalesce(F.col("is_forecast"), F.lit(False))).collect()
# -> ['obs']; PySpark returns ['legacy', 'obs']
df.filter(~F.when(F.col("is_forecast"), F.lit(True)).otherwise(F.lit(False))).collect()
# -> []; PySpark returns ['legacy', 'obs']
```

**Reference behaviour**:
Captured from real PySpark 4.0.0 on Java 21. `coalesce(NULL, False)` is FALSE, so
`NOT coalesce(...)` is TRUE and the row is kept. For a `CASE WHEN` with no
`.otherwise`, unmatched rows are NULL and `NOT NULL` is NULL, so `~F.when(cond, x)`
legitimately filters everything out - that case was already correct and is pinned by a
test so the fix cannot "achieve" it by accident.

**Impact**:
- `~F.coalesce(col, F.lit(False))` is the idiomatic "treat NULL as false" gate. It
  silently dropped every NULL-derived row. Found by a downstream data platform whose
  historical weather-correlation filter uses exactly this expression to include legacy
  rows that predate a flag - real rows were being excluded from a real computation.
- `~F.when(...)` returned an empty result set in all cases.

**Fix**:
- Added `_to_sql_boolean()`: interprets an evaluated result as a SQL boolean, keeping
  NULL as NULL so enclosing NOT/AND/OR apply three-valued logic.
- The function-operation branch of the predicate path now delegates to the row-aware
  value evaluator (`_evaluate_column_operation_value`) instead of the lossy scalar
  helper - mirroring what the `udf` branch already did. The value path routes a strict
  superset of these operations (115 vs 62), so no coverage is lost.
- `evaluate_condition()` now recognises `CaseWhen` (via `.evaluate(row)`), and `"~"` is
  accepted alongside `"not"`/`"!"` at all three negation dispatch sites.

**Regression tests**:
`tests/unit/functions/test_negation_operand_shapes.py` - indexed by operand *shape*
(function result, CASE WHEN, compound expression, bare column) rather than by truth
value, which is the axis the BUG-023 truth-table file already covered and the axis on
which these bugs hid. 5 of the 10 tests fail on the unfixed tree; the other 5 guard
against fixing one shape by breaking another.

---

### BUG-023: Boolean columns evaluated as presence checks, breaking three-valued logic
**Status**: Fixed
**Severity**: Critical
**Discovered**: 2026-07-20
**File**: `sparkless/core/condition_evaluator.py`

**Description**:
`ConditionEvaluator.evaluate_condition()` evaluated a bare `Column` predicate as
`get_row_value(row, column.name) is not None` - a *presence* check rather than the
stored boolean value. Consequently `~F.col("flag")` meant "flag IS NULL", and
`filter(~col)` returned the NULL rows while dropping the explicitly-false ones: a
precise inversion of the correct result. `col == F.lit(False)` was unaffected, so the
two idioms silently disagreed.

Separately, the three logical connectives used Python's `not` / `and` / `or`, which do
not implement SQL three-valued (Kleene) logic:
`not None` is `True` (should be NULL), `None and False` is `None` (should be FALSE),
and `None or False` is `False` (should be NULL).

**Reproduction**:
```python
df = spark.createDataFrame(
    [("v-active", True), ("v-inactive", False), ("v-unknown", None)],
    StructType([StructField("vid", StringType()), StructField("flag", BooleanType())]),
)
df.filter(~F.col("flag")).collect()  # -> ['v-unknown']; PySpark returns ['v-inactive']
df.filter(F.col("flag")).collect()   # -> ['v-active','v-inactive']; PySpark: ['v-active']
```

**Reference behaviour**:
Captured from real PySpark 4.0.0 (Java 21, the DBR 17.3 runtime) and re-confirmed on
PySpark 3.5.3 (Java 17). `NOT TRUE = FALSE`, `NOT FALSE = TRUE`, `NOT NULL = NULL`;
`NULL AND FALSE = FALSE`; `NULL OR TRUE = TRUE`; a NULL predicate filters the row out.

**Impact**:
- Any `~F.col(bool_col)` predicate returned the complement of the correct rows whenever
  the column was nullable - silently, with no error.
- Downstream test suites that use sparkless as their unit-test engine would validate
  inverted production behaviour as correct. This was found when an eligibility gate
  written as `col == F.lit(False)` was proposed for replacement by the idiomatic
  `~F.col(...)`; under the old semantics the two selected disjoint row sets.
- Comparison-shaped operands (`~(F.col("n") == 1)`, `(a > 0) & (b < 3)`) were already
  correct - only *bare boolean columns* were affected, which is why it went unnoticed.

**Fix**:
- `evaluate_condition()` returns the stored boolean for boolean columns and `None`
  (SQL NULL) for NULL values; non-boolean non-null columns keep the previous truthy
  behaviour.
- Added `_kleene_not` / `_kleene_and` / `_kleene_or` helpers and routed both copies of
  the logical-operator block (`_evaluate_logical_operation` and the duplicate inside
  `_evaluate_column_operation`) through them.

**Regression tests**:
`tests/unit/functions/test_boolean_three_valued_logic.py` - full NOT/AND/OR truth
tables plus filter semantics. The file is backend-agnostic and passes against real
PySpark via `SPARKLESS_TEST_BACKEND=pyspark`.

---

### BUG-001: GroupedData.count() returns AggregateFunction instead of ColumnOperation
**Status**: Open  
**Severity**: High  
**Discovered**: 2025-01-15  
**File**: `sparkless/dataframe/grouped/base.py`

**Description**:
The `count()` method on `GroupedData` returns an `AggregateFunction` directly instead of wrapping it in a `ColumnOperation`. This causes strict validation errors in `agg()` which expects only `Column` or `ColumnOperation` objects.

**Error**:
```
AssertionError: all exprs should be Column, got AggregateFunction at argument 0. 
AggregateFunction objects should be converted to Column/ColumnOperation before passing to agg().
```

**Reproduction**:
```python
df = spark.createDataFrame([{"dept": "IT", "val": 1}])
result = df.groupBy("dept").count()  # Fails
result = df.groupBy("dept").agg(F.count("*"))  # Also fails
```

**Confirmed**: 2025-01-15 - Both patterns fail with same error

**Impact**:
- Prevents use of convenience methods like `groupBy().count()`
- Affects all aggregation convenience methods
- Breaks compatibility with PySpark API where `groupBy().count()` works

**Workaround**:
None currently - must use workarounds in tests or fix implementation

**Related Code**:
- `sparkless/dataframe/grouped/base.py:1586` - `count()` method
- `sparkless/dataframe/grouped/base.py:78` - strict validation in `agg()`

---

### BUG-002: Aggregate functions with string arguments return AggregateFunction instead of ColumnOperation
**Status**: Open  
**Severity**: High  
**Discovered**: 2025-01-15  
**Files**: `sparkless/functions/aggregate.py`

**Description**:
When calling aggregate functions like `F.sum("column_name")` or `F.avg("column_name")` with string arguments, they return `AggregateFunction` objects directly instead of `ColumnOperation` objects. This breaks the strict validation in `agg()` which only accepts `Column` or `ColumnOperation`.

**Error**:
```
AssertionError: all exprs should be Column, got AggregateFunction at argument 0.
AggregateFunction objects should be converted to Column/ColumnOperation before passing to agg().
```

**Reproduction**:
```python
df = spark.createDataFrame([{"dept": "IT", "salary": 50000}])
result = df.groupBy("dept").agg(F.sum("salary"))  # Fails
result = df.groupBy("dept").agg(F.sum(df.salary))  # Also fails
```

**Confirmed**: 2025-01-15 - Both string and column arguments fail with same error

**Impact**:
- All string-based aggregate function calls fail
- Inconsistent behavior: some aggregations work, others don't
- Breaks compatibility with PySpark where `F.sum("col")` works in `agg()`

**Workaround**:
None reliable - inconsistent behavior between string and column arguments

**Related Code**:
- `sparkless/functions/aggregate.py` - aggregate function implementations
- `sparkless/dataframe/grouped/base.py:78` - strict validation

**Notes**:
- **BUG CONFIRMED**: Compatibility tests also fail with same error
- Bug affects both string and Column arguments
- Bug affects all test suites (unit, compatibility, parity)
- This is a blocking issue for aggregation functionality

---

### BUG-003: Window functions with aggregations fail due to AggregateFunction issue
**Status**: Open  
**Severity**: High  
**Discovered**: 2025-01-15  
**Files**: `sparkless/dataframe/window.py`, `sparkless/dataframe/grouped/base.py`

**Description**:
Window functions that use aggregations (like `F.sum().over()`) fail with the same AggregateFunction validation error as BUG-002. This affects `sum_over_window`, `lag`, `lead` and other window operations with aggregations.

**Error**:
```
AssertionError: all exprs should be Column, got AggregateFunction at argument 0.
```

**Reproduction**:
```python
df = spark.createDataFrame([{"dept": "IT", "salary": 50000}])
window = Window.partitionBy("dept").orderBy("salary")
result = df.withColumn("running_total", F.sum("salary").over(window))  # Fails
```

**Impact**:
- All window functions with aggregations fail
- Affects running totals, moving averages, and other window aggregations

**Related Issues**:
- BUG-002 (root cause)

**Affected Tests**:
- `test_sum_over_window`
- `test_lag`
- `test_lead`

---

### BUG-031: `struct` evaluates to NULL in `select`, and `.cast(StructType)` is a silent no-op
**Status**: Fixed
**Severity**: Critical
**Discovered**: 2026-07-21
**Files**: `sparkless/core/condition_evaluator.py`,
`sparkless/dataframe/schema/schema_manager.py`,
`sparkless/dataframe/casting/type_converter.py`,
`sparkless/dataframe/evaluation/expression_evaluator.py`

**Description**:
`df.select(F.struct(a, b).cast(some_struct_type))` returned `NULL` for the whole
struct, typed `STRING`. Three independent gaps produced it, each an instance of
the *same* failure shape -- a hardcoded dispatch whose unmatched case falls
through to a silent default, making "unimplemented" indistinguishable from a
genuine SQL NULL:

1. `ConditionEvaluator._evaluate_column_operation_value` -- the evaluator the
   lazy `select` path uses -- dispatches on a whitelist of ~150 operation names
   and ends in `# Default fallback / return None`. `struct` was not in the list.
   The parallel `ExpressionEvaluator` (used by `withColumn`) *did* implement
   `struct`, so the two projection paths disagreed about the same expression.
2. `SchemaManager._infer_expression_type` (the `select` type-inference table)
   had neither a `struct` nor a `cast` case and defaulted to `StringType`. The
   parallel `_handle_withcolumn_operation` table did handle `cast`. Two
   hardcoded tables for the same question, drifted apart.
3. `TypeConverter.cast_to_type` handled `ArrayType` and `MapType` but not
   `StructType`, so a struct-to-struct cast fell through its `else: return
   value` -- a silent no-op instead of the positional rename/retype Spark
   performs.

Separately, `F.struct("a", "b")` treated every string after the first as a
*literal* (`{"a": 1.0, "col2": "b"}`) rather than a column reference.

**Reproduction**:
```python
df = spark.createDataFrame([(1.5, 2.5)], ["urgency", "risk"])
target = StructType([StructField("urgency", DoubleType()),
                     StructField("risk", DoubleType())])
df.select(F.struct(F.col("urgency"), F.col("risk")).cast(target).alias("scores")).collect()
# sparkless: [Row(scores=None)]        schema: struct<scores:string>
# PySpark 4.0.0: [Row(scores=Row(urgency=1.5, risk=2.5))]
#                                      schema: struct<scores:struct<urgency:double,risk:double>>
```

**PySpark 4.0.0 reference** (DBR 17.3 runtime):

| Expression | Result | Field names |
|---|---|---|
| `F.struct(F.col("a"), F.col("b"))` | `Row(a=…, b=…)` | source column names |
| `F.struct("a", "b")` | `Row(a=…, b=…)` | strings are *column refs* |
| `F.struct(F.lit(1), F.col("a"))` | `Row(col1=1, a=…)` | unaliased literal -> `col<position>` |
| `F.struct("a", F.lit("k"))` | `Row(a=…, col2='k')` | position, not a running count |
| `F.struct(c.alias("x"))` | `Row(x=…)` | alias wins |
| `.cast(StructType)` | positional rename + retype | arity mismatch raises `AnalysisException` |

**Fix**:
- New `sparkless/core/struct_builder.py` holds the single definition of
  argument unpacking and PySpark's field-naming rules. Both
  `ConditionEvaluator` and `ExpressionEvaluator` delegate to it, so the two
  paths can no longer drift (this also removed ~125 lines of duplicated,
  subtly-wrong logic from `ExpressionEvaluator`).
- `SchemaManager._infer_expression_type` gained `struct`, `cast`, boolean-op
  and `CaseWhen` cases.
- `TypeConverter.cast_to_type` gained a `StructType` branch performing the
  positional rename/retype.

**Impact**:
Any struct assembled in a `select` projection -- the shape used when writing a
struct column to Delta -- was silently NULL. Because the downstream consumer's
test tier runs entirely on sparkless, the projection could not be asserted at
all and the affected production functions carried `# pragma: no cover`.

**Not fixed** (divergence recorded, no behaviour change made):
`F.col("a").cast(StructType(...))` on a scalar returns the scalar unchanged;
PySpark raises `AnalysisException [DATATYPE_MISMATCH.CAST_WITHOUT_SUGGESTION]`.
Sparkless does not currently raise analysis-time errors anywhere, so adding one
here would be an isolated inconsistency.

**Affected Tests**:
- `tests/unit/functions/test_struct_projection_and_cast.py` (new, 13 tests;
  12 of the 13 fail on the pre-fix tree)

---

### BUG-032: An expression in a `when`/`otherwise` value position is discarded
**Status**: Fixed
**Severity**: Critical
**Discovered**: 2026-07-21
**Files**: `sparkless/functions/conditional.py`,
`sparkless/dataframe/schema/schema_manager.py`,
`sparkless/core/type_utils.py`

**Description**:
`CaseWhen._evaluate_column_operation_value` dispatched on five operations --
unary `+`/`-`, binary arithmetic, and `create_map`. Everything else hit:

```python
else:
    # For other operations, try to get the column value
    return ConditionEvaluator._get_column_value(row, operation.column)
```

which returns the value of the operation's *base column*, **discarding the
operation itself**. So:

| Expression in `.otherwise(...)` | sparkless | PySpark 4.0.0 |
|---|---|---|
| `F.datediff(a, b) >= F.lit(90)` | `134` (the day count) | `True` |
| `F.upper(F.col("sku"))` | `'stale'` (unchanged) | `'STALE'` |
| `F.col("sku") == F.lit("stale")` | `'stale'` | `True` |
| `F.col("last_sale_date").isNull()` | `datetime.date(2024, 1, 19)` | `False` |

This is the same shape as BUG-031, but the silent default is *the operand*
rather than `None` -- arguably worse, because the wrong value has a plausible
type for the column and nothing downstream flags it.

The result *type* was wrong too. `CaseWhen.get_result_type` mapped every
non-arithmetic `ColumnOperation` to `StringType`, and neither
`_handle_withcolumn_operation` nor `_infer_expression_type` consulted
`get_result_type` at all -- a `CaseWhen` fell through to `_infer_literal_type`,
which answers `StringType` for anything it does not recognise. So a boolean
CASE WHEN was typed `STRING`.

**Reproduction**:
```python
days_since_sale = F.datediff(F.lit(date(2024, 6, 1)), F.col("last_sale_date"))
aged = F.when(F.col("last_sale_date").isNull(), F.lit(False)).otherwise(
    days_since_sale >= F.lit(90)
)
df.withColumn("aged_stock_flag", aged).collect()
# sparkless:      [Row(..., aged_stock_flag=134), ...]   schema: ... flag:string
# PySpark 4.0.0:  [Row(..., aged_stock_flag=True), ...]  schema: ... flag:boolean
```

**Fix**:
- The `else` branch now delegates to `ConditionEvaluator.evaluate_expression`,
  the shared evaluator that already implements comparisons, the logical
  connectives and the scalar functions. The arithmetic and unary branches are
  kept as-is (`ConditionEvaluator`'s arithmetic path returns `None` for the
  unary form, where `operation.value` is `None`).
- `get_result_type` gained boolean-operation and `cast` cases; the schema
  paths now consult `get_result_type` when the expression exposes it.
- The canonical set of boolean-result operations moved to
  `sparkless.core.type_utils.BOOLEAN_RESULT_OPERATIONS` so the several places
  that infer expression types agree.

**Impact**:
Any derived flag computed in a `when`/`otherwise` was silently wrong -- and
because the wrong value type-checked, the failure only surfaced against a real
cluster.

**Affected Tests**:
- `tests/unit/functions/test_casewhen_expression_values.py` (new, 7 tests;
  6 of the 7 fail on the pre-fix tree -- the seventh guards the arithmetic
  branch that already worked)

---

### BUG-033: A comparison yielding NULL returns FALSE in a `select` projection
**Status**: Fixed
**Severity**: High
**Discovered**: 2026-07-21
**File**: `sparkless/core/condition_evaluator.py`

**Description**:
`ConditionEvaluator._evaluate_comparison_operation` returns `False` when an
operand is NULL, instead of NULL. In SQL a comparison with NULL is NULL, and
`withColumn` already gets this right (it goes through a different path), so the
two projection paths disagree:

```python
df = spark.createDataFrame([(9,), (None,)], "x int")
df.select((F.col("x") >= F.lit(5)).alias("f")).collect()
# sparkless:     [Row(f=True), Row(f=False)]
# PySpark 4.0.0: [Row(f=True), Row(f=None)]
df.withColumn("f", F.col("x") >= F.lit(5)).collect()   # correct: True, None
```

Found while fixing BUG-032 (whose production shape guards the NULL row with a
`when`, so it is unaffected), and deferred there because it touches
`_evaluate_comparison_operation`, which an open PR was editing at the time.

**Second defect, same file (found while fixing this one)**: the *predicate*
path did not share the projection path's kernel at all. It called
`_evaluate_comparison(col_value, op, condition_value)`, whose NULL branch read
`return operation == "!="` -- so `NULL != 9` evaluated to **TRUE** and
`df.filter(F.col("x") != F.lit(9))` kept the NULL row. Three comparison
implementations existed in total (two here, one in `ExpressionEvaluator`), of
which only `ExpressionEvaluator`'s was correct.

**Reference behaviour** (real PySpark 4.0.0, OpenJDK 21):
```
select(x >= 5)   -> [('a', True), ('b', None), ('c', False)]
select(x != 9)   -> [('a', False), ('b', None), ('c', True)]
select(~(x>=5))  -> [('a', False), ('b', None), ('c', True)]
filter(x != 9)   -> ['c']          # NULL row dropped, not kept
filter(~(x>=5))  -> ['c']          # NOT NULL is NULL, not TRUE
```

**Fix**:
- `_evaluate_comparison()` is now the single comparison kernel for this
  evaluator: it returns `None` (not `False`, and not `operation == "!="`) when
  either operand is NULL, accepts both the symbolic and the named operator
  aliases, keeps the existing type coercion, and maps an unreconcilable
  `TypeError` to NULL the way `ExpressionEvaluator` already did.
- `_evaluate_comparison_operation()` now resolves its operands and delegates to
  that kernel instead of carrying its own comparison ladder -- the same
  "predicate path delegates to the shared implementation" move as BUG-034.
- Return types widened to `Optional[bool]`.

`filter()` semantics are unchanged for TRUE/FALSE and now drop NULL-predicate
rows in the cases where the old code coerced NULL to TRUE.

Also filed independently as **BUG-047** while BUG-046 was collapsing the
predicate schism, and deferred there to this sweep.

**Tests**:
- `tests/parity/dataframe/test_null_comparison_semantics.py` (new, 21 tests,
  backend-agnostic `spark` fixture; all 21 also pass under
  `MOCK_SPARK_TEST_BACKEND=pyspark` against PySpark 4.0.0). Reverting the NULL
  branch to `return operation == "!="` fails 17 of the 21. The negation tests
  in that file are parametrised over `select` and `withColumn`, so they also
  guard BUG-046's `~` fix on the projection path.

---

## Medium Priority Issues

### BUG-004: SQL column aliases not properly parsed in SELECT statements
**Status**: Open  
**Severity**: Medium  
**Discovered**: 2025-01-15  
**Files**: `sparkless/session/sql/parser.py`, `sparkless/session/sql/executor.py`

**Description**:
SQL SELECT statements with column aliases (e.g., `SELECT col AS alias`) are not properly parsed. The executor tries to access columns using the full alias expression instead of parsing it correctly.

**Error**:
```
SparkColumnNotFoundError: 'DataFrame' object has no attribute 'name as dept_name'. 
Available columns: dept_id, name
```

**Reproduction**:
```python
result = spark.sql("""
    SELECT e.name, d.name as dept_name
    FROM employees e
    INNER JOIN departments d ON e.dept_id = d.id
""")  # Fails - 'name as dept_name' not parsed correctly
```

**Impact**:
- JOIN queries with aliased columns fail
- Complex SELECT statements with aliases fail
- Breaks PySpark compatibility for SQL queries

**Affected Tests**:
- `test_sql_with_inner_join`
- `test_sql_with_left_join`

---

### BUG-005: SQL CASE WHEN expressions not properly parsed
**Status**: Open  
**Severity**: Medium  
**Discovered**: 2025-01-15  
**Files**: `sparkless/session/sql/parser.py`, `sparkless/session/sql/executor.py`

**Description**:
SQL CASE WHEN expressions are not properly parsed in SELECT statements. The parser treats the entire CASE expression as a column name instead of parsing it as an expression.

**Error**:
```
SparkColumnNotFoundError: 'DataFrame' object has no attribute 'CASE WHEN age < 30 THEN 'Young'...'
```

**Reproduction**:
```python
result = spark.sql("""
    SELECT name, age,
           CASE WHEN age < 30 THEN 'Young'
                WHEN age < 35 THEN 'Middle'
                ELSE 'Senior' END as category
    FROM employees
""")  # Fails - CASE WHEN not parsed
```

**Impact**:
- Conditional SQL expressions fail
- Breaks PySpark compatibility for CASE WHEN statements

**Affected Tests**:
- `test_sql_with_case_when`

---

### BUG-006: SQL HAVING clause not properly supported
**Status**: Open  
**Severity**: Medium  
**Discovered**: 2025-01-15  
**Files**: `sparkless/session/sql/executor.py`

**Description**:
SQL HAVING clause causes aggregation failures because aggregate functions in HAVING are not properly converted to ColumnOperation, hitting the same issue as BUG-002.

**Error**:
```
AssertionError: all exprs should be Column, got AggregateFunction at argument 0.
```

**Reproduction**:
```python
result = spark.sql("""
    SELECT dept, AVG(salary) as avg_salary
    FROM employees
    GROUP BY dept
    HAVING AVG(salary) > 55000
""")  # Fails - HAVING uses AggregateFunction
```

**Impact**:
- All queries with HAVING clause fail
- Cannot filter grouped results by aggregate values

**Related Issues**:
- BUG-002 (root cause)

**Affected Tests**:
- `test_sql_with_having`

---

### BUG-007: SQL UNION operation not properly implemented
**Status**: Open  
**Severity**: Medium  
**Discovered**: 2025-01-15  
**Files**: `sparkless/session/sql/parser.py`, `sparkless/session/sql/executor.py`

**Description**:
SQL UNION operations are not properly parsed or executed. The parser doesn't handle UNION syntax correctly.

**Reproduction**:
```python
result = spark.sql("""
    SELECT name, age FROM table1
    UNION
    SELECT name, age FROM table2
""")  # Fails - UNION not parsed/executed
```

**Impact**:
- UNION queries fail
- Cannot combine results from multiple SELECT statements

**Affected Tests**:
- `test_sql_with_union`

---

### BUG-008: SQL subqueries not properly supported
**Status**: Open  
**Severity**: Medium  
**Discovered**: 2025-01-15  
**Files**: `sparkless/session/sql/parser.py`, `sparkless/session/sql/executor.py`

**Description**:
SQL subqueries (nested SELECT statements) are not properly parsed or executed.

**Reproduction**:
```python
result = spark.sql("""
    SELECT name, salary
    FROM employees
    WHERE salary > (SELECT AVG(salary) FROM employees)
""")  # Fails - subquery not supported
```

**Impact**:
- Subqueries fail
- Cannot use correlated or uncorrelated subqueries

**Affected Tests**:
- `test_sql_with_subquery`

---

### BUG-009: SQL LIKE clause parsing issues
**Status**: Open  
**Severity**: Medium  
**Discovered**: 2025-01-15  
**Files**: `sparkless/session/sql/parser.py`, `sparkless/session/sql/executor.py`

**Description**:
SQL LIKE clause has parsing issues. The pattern matching logic may not be properly implemented.

**Reproduction**:
```python
result = spark.sql("SELECT * FROM employees WHERE name LIKE 'A%'")  # May fail
```

**Impact**:
- Pattern matching queries may fail
- LIKE operations not working correctly

**Affected Tests**:
- `test_sql_with_like`

---

### BUG-010: SQL IN clause parsing issues
**Status**: Open  
**Severity**: Medium  
**Discovered**: 2025-01-15  
**Files**: `sparkless/session/sql/parser.py`, `sparkless/session/sql/executor.py`

**Description**:
SQL IN clause may have parsing or execution issues.

**Reproduction**:
```python
result = spark.sql("SELECT * FROM employees WHERE age IN (25, 35)")  # May fail
```

**Impact**:
- IN clause queries may fail
- Cannot filter by multiple values

**Affected Tests**:
- `test_sql_with_in_clause`

---

### BUG-011: SQL CREATE TABLE AS SELECT not properly implemented
**Status**: Open  
**Severity**: Medium  
**Discovered**: 2025-01-15  
**Files**: `sparkless/session/sql/executor.py`

**Description**:
CREATE TABLE AS SELECT syntax requires column definitions but the parser doesn't extract them properly from the SELECT statement.

**Error**:
```
QueryExecutionException: CREATE TABLE requires column definitions
```

**Reproduction**:
```python
spark.sql("CREATE TABLE IF NOT EXISTS it_employees AS SELECT name, age FROM employees WHERE dept = 'IT'")
# Fails - column definitions not extracted from SELECT
```

**Impact**:
- CREATE TABLE AS SELECT fails
- Cannot create tables from query results

**Affected Tests**:
- `test_create_table_with_select`

---

### BUG-012: SQL INSERT INTO statement execution order incorrect
**Status**: Open  
**Severity**: Medium  
**Discovered**: 2025-01-15  
**Files**: `sparkless/session/sql/executor.py`

**Description**:
INSERT INTO statements appear to insert data but subsequent SELECT queries return data in wrong order or format. The INSERT may not be committing correctly.

**Error**:
```
AssertionError: assert '30' == 'Alice'
```

**Reproduction**:
```python
df = spark.createDataFrame([("Alice", 25)], ["name", "age"])
df.write.mode("overwrite").saveAsTable("insert_test")
spark.sql("INSERT INTO insert_test VALUES ('Bob', 30)")
result = spark.sql("SELECT * FROM insert_test ORDER BY name")
# Result order or values incorrect
```

**Impact**:
- INSERT operations may not work correctly
- Data integrity issues

**Affected Tests**:
- `test_insert_into_table`

---

### BUG-013: SQL UPDATE statement not properly implemented
**Status**: Open  
**Severity**: Medium  
**Discovered**: 2025-01-15  
**Files**: `sparkless/session/sql/executor.py`

**Description**:
SQL UPDATE statements may not be properly executing or the execution may have issues.

**Reproduction**:
```python
spark.sql("UPDATE employees SET age = 26 WHERE name = 'Alice'")
result = spark.sql("SELECT * FROM employees WHERE name = 'Alice'")
# Update may not be reflected
```

**Impact**:
- UPDATE operations may fail or not commit
- Data modification queries don't work

**Affected Tests**:
- `test_update_table`

---

### BUG-014: SQL INSERT INTO ... SELECT not properly implemented
**Status**: Open  
**Severity**: Medium  
**Discovered**: 2025-01-15  
**Files**: `sparkless/session/sql/executor.py`

**Description**:
INSERT INTO ... SELECT statements may not be properly executing.

**Reproduction**:
```python
spark.sql("INSERT INTO target_table SELECT name, age FROM source_table WHERE dept = 'IT'")
# May fail or not insert correctly
```

**Impact**:
- Insert from select queries fail
- Cannot copy data between tables

**Affected Tests**:
- `test_insert_from_select`

---

### BUG-015: SQL SHOW statements return incorrect format
**Status**: Open  
**Severity**: Medium  
**Discovered**: 2025-01-15  
**Files**: `sparkless/session/sql/executor.py`

**Description**:
SHOW DATABASES and SHOW TABLES statements return results in a format that doesn't match PySpark's expected format (e.g., column names differ).

**Error**:
```
AssertionError: assert 'show_test_db' in ['default', 'test']
```

**Reproduction**:
```python
spark.sql("CREATE DATABASE IF NOT EXISTS show_test_db")
result = spark.sql("SHOW DATABASES")
db_names = [row["databaseName"] for row in result.collect()]
# Column name may be wrong or database not in results
```

**Impact**:
- SHOW statements return unexpected format
- Column names don't match PySpark
- Results may be missing expected entries

**Affected Tests**:
- `test_show_databases`
- `test_show_tables`
- `test_show_tables_in_database`

---

### BUG-016: SQL DESCRIBE statements not properly implemented
**Status**: Open  
**Severity**: Medium  
**Discovered**: 2025-01-15  
**Files**: `sparkless/session/sql/executor.py`

**Description**:
DESCRIBE TABLE, DESCRIBE EXTENDED, and DESCRIBE column statements are not properly implemented or return incorrect formats.

**Reproduction**:
```python
result = spark.sql("DESCRIBE employees")
# May fail or return wrong format
```

**Impact**:
- Cannot inspect table schemas via SQL
- DESCRIBE operations fail or return wrong format

**Affected Tests**:
- `test_describe_table`
- `test_describe_extended`
- `test_describe_column`

---

### BUG-017: Array function tests fail due to column name mismatches with expected outputs
**Status**: Fixed  
**Severity**: Medium  
**Discovered**: 2025-01-15  
**Fixed**: 2025-01-15 (PR #39)  
**Files**: `tests/parity/functions/test_array.py`

**Description**:
Array function tests failed because the test code referenced column names that didn't exist in the expected output data. The expected outputs have columns like "arr1", "arr2", "arr3", "id", "value" but tests referenced "tags" or "scores" which don't exist.

**Error**:
```
SparkColumnNotFoundError: 'DataFrame' object has no attribute 'tags'. 
Available columns: arr1, arr2, arr3, id, value
```

**Impact**:
- Tests failed due to data/column name mismatches
- Fixed by aligning test code with expected output schema

**Affected Tests**:
- `test_array_join` - Fixed to use `arr1` with separator `-`
- `test_array_union` - Fixed to use `arr1` and `arr2`
- `test_array_sort` - Fixed to use `arr3`, but skipped due to column name representation mismatch (PySpark uses complex lambda representation, mock uses simple name). Function works correctly.
- `test_array_distinct` - Already using correct column name

**Resolution**:
Fixed in PR #39. Updated all array function tests to use the correct column names that match the expected output schemas. `test_array_sort` was skipped due to a known limitation where PySpark generates a complex lambda function representation in the column name, but our mock generates a simpler name. The function works correctly and data values match.

---

### BUG-018: Null handling function tests fail due to column name mismatches
**Status**: Fixed  
**Severity**: Medium  
**Discovered**: 2025-01-15  
**Fixed**: 2025-01-15 (PR #39)  
**Files**: `tests/parity/functions/test_null_handling.py`

**Description**:
Null handling function tests failed because the test code referenced column names that didn't exist in the expected output data. The expected outputs have columns like "age", "department", "id", "name", "salary" but tests referenced "col1", "col2", "col3", "value" which don't exist.

**Error**:
```
SparkColumnNotFoundError: 'DataFrame' object has no attribute 'col1'. 
Available columns: age, department, id, name, salary
```

**Impact**:
- Tests failed due to data/column name mismatches
- Fixed by aligning test code with expected output schema

**Affected Tests**:
- `test_coalesce` - Fixed to use `salary` and `F.lit(0)` instead of `col1`, `col2`, `col3`
- `test_isnull` - Fixed to use `name` instead of `value`
- `test_isnotnull` - Fixed to use `name` instead of `value`
- `test_when_otherwise` - Fixed to use `salary.isNull()` with literal `0` instead of `age > 30`
- `test_nvl` - Fixed to use `salary` and `F.lit(0)` instead of `value` and `0`
- `test_nullif` - Fixed to use `age` and `F.lit(30)` instead of `col1` and `col2`

**Resolution**:
Fixed in PR #39. Updated all null handling function tests to use the correct column names and expressions that match the expected output schemas.

---

### BUG-019: Datetime function dayofmonth returns incorrect result
**Status**: Fixed  
**Severity**: Medium  
**Discovered**: 2025-01-15  
**Fixed**: 2025-01-15 (PR #38)  
**Files**: `tests/parity/functions/test_datetime.py`

**Description**:
The test for dayofmonth was using the wrong column name (`hire_date` instead of `date`). The dayofmonth function itself was working correctly.

**Reproduction**:
```python
df = spark.createDataFrame([{"date": "2023-01-15"}])
result = df.select(F.dayofmonth(df.date))
# Result matches PySpark correctly
```

**Impact**:
- Test was failing due to column name mismatch
- Function was working correctly all along

**Affected Tests**:
- `test_dayofmonth`

**Resolution**:
Fixed in PR #38. The test was updated to use the correct column name (`df.date` instead of `df.hire_date`) to match the expected output data structure. The dayofmonth function itself was already working correctly and returns the expected values (15, 10, 22 for the test dates).

---

### BUG-020: Catalog.getTable with database parameter argument order incorrect
**Status**: Fixed  
**Severity**: Medium  
**Discovered**: 2025-01-15  
**Fixed**: 2025-01-15 (PR #37)  
**Files**: `sparkless/session/catalog.py`

**Description**:
`Catalog.getTable()` is being called with database as first argument and table as second, but the method signature expects table name first, then optional database name.

**Error**:
```
AnalysisException: Table 'get_table.get_db' does not exist
```

**Reproduction**:
```python
table = spark.catalog.getTable("get_db", "get_table")  # Wrong argument order
# Should be: spark.catalog.getTable("get_table", "get_db")
# Or: spark.catalog.getTable(databaseName="get_db", tableName="get_table")
```

**Impact**:
- getTable with database parameter fails
- API usage confusion

**Affected Tests**:
- `test_get_table_in_database`

**Resolution**:
Fixed in PR #37. The `getTable()` method now supports both argument orders for PySpark compatibility:
- Standard order: `getTable(tableName, dbName)`
- PySpark order: `getTable(dbName, tableName)`

The implementation automatically detects which order to use by trying the standard order first, and if the table isn't found, it tries the PySpark order by swapping the arguments. This maintains backward compatibility while supporting PySpark's expected API.

---

### BUG-021: SQL basic SELECT queries return wrong schema
**Status**: Open  
**Severity**: Medium  
**Discovered**: 2025-01-15  
**Files**: `sparkless/session/sql/executor.py`

**Description**:
Basic SELECT queries return DataFrames with wrong schema compared to PySpark. The schema field count or structure doesn't match.

**Error**:
```
Schema field count mismatch: mock=4, expected=3
```

**Reproduction**:
```python
df = spark.createDataFrame([{"id": 1, "name": "Alice", "age": 25}])
df.write.mode("overwrite").saveAsTable("test_table")
result = spark.sql("SELECT * FROM test_table")
# Schema doesn't match PySpark
```

**Impact**:
- Basic SQL queries return wrong schemas
- Breaks compatibility for simple SELECT statements

**Affected Tests**:
- `test_basic_select`
- `test_filtered_select`
- `test_group_by` (in SQL queries)
- `test_aggregation` (in SQL queries)

---

## Low Priority / Design Issues

### BUG-022: Inconsistent aggregate function return types
**Status**: Open  
**Severity**: Low  
**Discovered**: 2025-01-15  

**Description**:
Aggregate functions have inconsistent return types depending on how they're called:
- Some return `ColumnOperation` (when wrapping is implemented)
- Others return `AggregateFunction` directly
- This creates confusion and breaks strict validation

**Impact**:
- Unpredictable behavior
- Hard to maintain
- Breaks strict validation approach

**Related Issues**:
- BUG-001
- BUG-002

---

---

## Low Priority / Design Issues

### ISSUE-001: Strict validation may be too strict
**Status**: Discussion  
**Severity**: Low  
**Discovered**: 2025-01-15  

**Description**:
The strict validation in `GroupedData.agg()` raises errors for `AggregateFunction` objects, but PySpark actually accepts these in some contexts. The validation may need to be more nuanced.

**Questions**:
- Should Sparkless accept `AggregateFunction` directly in `agg()`?
- Or should all aggregate functions return `ColumnOperation`?
- What is the exact PySpark behavior?

---

## Investigation Notes

### Compatibility Test Behavior
- Compatibility tests in `tests/compatibility/test_aggregations_compatibility.py` use patterns like `F.sum("salary")` and appear to work
- This suggests either:
  1. The bug was introduced after those tests were written
  2. There are different code paths being used
  3. The tests aren't actually running the code that fails

### PySpark Behavior
- PySpark's `groupBy().count()` returns a DataFrame with a "count" column
- PySpark's `agg()` accepts `Column` objects created from aggregate functions
- PySpark aggregate functions return `Column` objects, not `AggregateFunction`

---

## Test Failures Due to Bugs

### Tests Blocked by Aggregation Bugs (BUG-001, BUG-002, BUG-003)

**DataFrame Aggregations** (9 tests):
- `test_sum_aggregation` - BUG-002
- `test_avg_aggregation` - BUG-002
- `test_count_aggregation` - BUG-002
- `test_max_aggregation` - BUG-002
- `test_min_aggregation` - BUG-002
- `test_multiple_aggregations` - BUG-002
- `test_groupby_multiple_columns` - BUG-002
- `test_global_aggregation` - BUG-002
- `test_aggregation_with_nulls` - BUG-002

**Function Aggregations** (5 tests):
- `test_agg_sum` - BUG-002
- `test_agg_avg` - BUG-002
- `test_agg_count` - BUG-002
- `test_agg_max` - BUG-002
- `test_agg_min` - BUG-002

**GroupBy Operations** (2 tests):
- `test_group_by` - BUG-001
- `test_aggregation` - BUG-002

**Window Functions** (3 tests):
- `test_sum_over_window` - BUG-003
- `test_lag` - BUG-003
- `test_lead` - BUG-003

### Tests Blocked by SQL Parsing Bugs (BUG-004 to BUG-011)

**Advanced SQL** (8 tests):
- `test_sql_with_inner_join` - BUG-004 (column aliases)
- `test_sql_with_left_join` - BUG-004 (column aliases)
- `test_sql_with_having` - BUG-006 (HAVING clause)
- `test_sql_with_union` - BUG-007 (UNION)
- `test_sql_with_subquery` - BUG-008 (subqueries)
- `test_sql_with_case_when` - BUG-005 (CASE WHEN)
- `test_sql_with_like` - BUG-009 (LIKE)
- `test_sql_with_in_clause` - BUG-010 (IN clause)

**SQL DDL/DML** (4 tests):
- `test_create_table_with_select` - BUG-011 (CREATE TABLE AS SELECT)
- `test_insert_into_table` - BUG-012 (INSERT execution)
- `test_update_table` - BUG-013 (UPDATE)
- `test_insert_from_select` - BUG-014 (INSERT INTO ... SELECT)

**SQL SHOW/DESCRIBE** (7 tests):
- `test_show_databases` - BUG-015 (SHOW format)
- `test_show_tables` - BUG-015 (SHOW format)
- `test_show_tables_in_database` - BUG-015 (SHOW format)
- `test_describe_table` - BUG-016 (DESCRIBE)
- `test_describe_extended` - BUG-016 (DESCRIBE)
- `test_describe_column` - BUG-016 (DESCRIBE)

**SQL Basic Queries** (4 tests):
- `test_basic_select` - BUG-021 (schema mismatch)
- `test_filtered_select` - BUG-021 (schema mismatch)
- `test_group_by` (SQL) - BUG-021 + BUG-002
- `test_aggregation` (SQL) - BUG-021 + BUG-002

### Tests Blocked by Function Bugs (BUG-017 to BUG-019)

**Array Functions** (4 tests):
- `test_array_join` - BUG-017
- `test_array_union` - BUG-017
- `test_array_sort` - BUG-017
- `test_array_distinct` - BUG-017

**Null Handling Functions** (6 tests):
- `test_coalesce` - BUG-018
- `test_isnull` - BUG-018
- `test_isnotnull` - BUG-018
- `test_when_otherwise` - BUG-018
- `test_nvl` - BUG-018
- `test_nullif` - BUG-018

**Datetime Functions** (1 test):
- `test_dayofmonth` - BUG-019

### Tests Blocked by Catalog Bugs (BUG-020)

**Catalog Operations** (1 test):
- `test_get_table_in_database` - BUG-020 (argument order)

---

## Summary of Test Failures

**Total Failing Tests**: 53  
**Total Passing Tests**: 111  
**Pass Rate**: 68%

**Failures by Category**:
- Aggregation-related: 19 tests (BUG-001, BUG-002, BUG-003)
- SQL parsing/execution: 23 tests (BUG-004 to BUG-016, BUG-021)
- Function implementation: 11 tests (BUG-017 to BUG-019)
- Catalog API: 1 test (BUG-020)

---

## Fix Suggestions

### For BUG-001 and BUG-002:

**Option 1: Make aggregate functions return ColumnOperation**
- Modify all aggregate functions to wrap `AggregateFunction` in `ColumnOperation`
- Ensure `ColumnOperation._aggregate_function` is set for unwrapping in `agg()`
- This matches the pattern used in `corr()` and `covar_samp()`

**Option 2: Allow AggregateFunction in agg()**
- Modify strict validation to accept `AggregateFunction` objects
- Convert them internally to appropriate format
- Less strict but more permissive

**Option 3: Hybrid approach**
- Make convenience methods (like `count()`) return `ColumnOperation`
- Make function calls (like `F.sum()`) return `ColumnOperation`
- Keep strict validation but ensure everything returns `ColumnOperation`

**Recommended**: Option 1 - ensures consistent API and maintains strict validation

---

## Additional Notes

- All issues discovered during parity test migration
- Priority based on impact on test migration and PySpark compatibility
- Some issues may have workarounds that aren't documented yet
- Need to verify against actual PySpark behavior to confirm bugs vs. intentional differences

---

## Testing Recommendations

1. Run compatibility tests to verify current behavior
2. Test aggregate functions with both string and Column arguments
3. Test convenience methods vs. explicit agg() calls
4. Compare behavior with actual PySpark

---

## Bug Summary by Priority

### Critical (3 bugs)
- **BUG-001**: GroupedData.count() returns AggregateFunction
- **BUG-002**: Aggregate functions return AggregateFunction
- **BUG-003**: Window functions with aggregations fail

### Medium (18 bugs)
- **BUG-004**: SQL column aliases not parsed
- **BUG-005**: SQL CASE WHEN not parsed
- **BUG-006**: SQL HAVING clause fails
- **BUG-007**: SQL UNION not implemented
- **BUG-008**: SQL subqueries not supported
- **BUG-009**: SQL LIKE parsing issues
- **BUG-010**: SQL IN clause parsing issues
- **BUG-011**: CREATE TABLE AS SELECT fails
- **BUG-012**: INSERT INTO execution order wrong
- **BUG-013**: UPDATE statement not implemented
- **BUG-014**: INSERT INTO ... SELECT fails
- **BUG-015**: SHOW statements return wrong format
- **BUG-016**: DESCRIBE statements not implemented
- **BUG-017**: Array function test data mismatches
- **BUG-018**: Null handling function test data mismatches
- **BUG-019**: dayofmonth returns wrong result
- **BUG-020**: Catalog.getTable argument order
- **BUG-021**: SQL basic SELECT returns wrong schema

### Low (1 bug)
- **BUG-022**: Inconsistent aggregate function return types

---

## Test Failure Breakdown

**Total Tests**: 164  
**Passing**: 111 (68%)  
**Failing**: 53 (32%)

**Failures by Root Cause**:
- Aggregation bugs (BUG-001, BUG-002, BUG-003): 19 tests (36%)
- SQL parsing/execution bugs (BUG-004 to BUG-016, BUG-021): 23 tests (43%)
- Function/test data bugs (BUG-017, BUG-018, BUG-019): 11 tests (21%)
- Catalog API bug (BUG-020): 1 test (2%)

---

## Quick Reference: Most Impactful Bugs

**Fix these first for maximum impact**:
1. **BUG-002**: Fixes 14 aggregation-related test failures
2. **BUG-001**: Fixes 2 GroupBy test failures  
3. **BUG-003**: Fixes 3 window function test failures
4. **BUG-004**: Fixes 2 SQL JOIN test failures
5. **BUG-021**: Fixes 4 basic SQL query test failures

**Top 5 bugs would fix 25 test failures (47% of all failures)**


---

## 2026-07 PySpark 4.0.0 Compatibility Findings

**Context**: Divergences found while running a large downstream unit-test suite
(~24k tests) on Sparkless. Every bug in this batch is **silent** -- none raises,
each returns a plausible-but-wrong value, so assertions of the form
"not NULL" / "> 0" / "ranks are 1..n" keep passing while the number is wrong.

**Reference engine**: PySpark **4.0.0** on OpenJDK 21 (the Databricks Runtime
17.3 pairing). Every expected value below was produced by executing the
reproduction against real PySpark, not derived from the API docs.

---

### BUG-024: Window.orderBy applies one global sort direction to all keys
**Status**: Fixed
**Severity**: High
**Discovered**: 2026-07-20
**File**: `sparkless/functions/window_execution.py`, `sparkless/dataframe/window_handler.py`

**Description**:
`Window.orderBy` computed a single `reverse` flag as `any(key is desc)` and
performed one `sorted()` call with it. Spark applies the direction **per key**,
so `ORDER BY a DESC, b ASC` must sort `b` ascending within ties on `a`. With the
global flag, any `desc` key reversed *every* key -- the trailing `asc` tie-break
came out backwards.

**Reproduction**:
```python
df = spark.createDataFrame(
    [("g", 1, 5.0, "z"), ("g", 1, 5.0, "a"), ("g", 2, 5.0, "m")],
    ["grp", "prio", "score", "name"],
)
w = Window.partitionBy("grp").orderBy(F.col("score").desc(), F.col("name").asc())
df.withColumn("rn", F.row_number().over(w)).collect()
```

**Expected (PySpark 4.0.0)**: `a, m, z` (all scores tie, so `name` ascending decides)
**Actual (Sparkless)**: `z, m, a` -- exactly `name` *descending*

**Confirmed**: 2026-07-20 against PySpark 4.0.0. Also reproduced with three keys
(`desc, desc, asc` -> expected `m, a, z`, got `m, z, a`) and with an undecorated
trailing key (`orderBy(desc(score), col(name))`), which must default to ascending.

**Impact**:
- Silent. `row_number()` still yields 1..n and `rank()` still looks dense, so the
  shape of the result is right and only the order is wrong.
- Makes any deterministic tie-break unprovable: a downstream project had to mark
  its ordering test `requires_real_spark` because the harness could not reproduce
  the final sort key.
- `lag`/`lead`/`first`/`last` over a mixed-direction window read the wrong
  neighbouring row.
- All-ascending and all-descending windows were unaffected, which is why this
  survived: the existing mixed-direction test (`test_mixed_order_asc_desc`) uses
  `orderBy(asc(grp), desc(val))` while partitioned by `grp`, so the `asc` key is
  constant within each partition and the global reverse happens to be correct.

**Fix**:
Added `sort_indices_multi_key()` to `sparkless/spark_types.py`, which performs
the standard stable multi-key sort (least-significant key first, one pass per
key, each with its own `reverse`). Both order-by implementations now build a
`(is_desc, nulls_last)` pair per key and delegate to it.

Nulls are now ordered with a rank sentinel instead of by substituting
`+/-inf`, which additionally fixes a `TypeError` when ordering a nullable
**string** column (`str` vs `float` comparison).

**Regression tests**: `tests/unit/dataframe/test_window_orderby_per_key_direction.py`
(12 tests, passing under both Sparkless and `MOCK_SPARK_TEST_BACKEND=pyspark`).

---

### BUG-025: Aggregating over an expression collapses to the empty default
**Status**: Fixed
**Severity**: Critical
**Discovered**: 2026-07-20
**File**: `sparkless/dataframe/grouped/base.py`, `sparkless/functions/window_execution.py`

**Description**:
`F.sum(F.when(cond, x))` returned a constant **`0`** -- in both `groupBy().agg()`
and over a window.

An aggregate whose target is an *expression* (rather than a plain column) has to
evaluate that expression for every row. Sparkless gated its per-row branch on
the target exposing an `.operation` attribute. `ColumnOperation` has one;
`CaseWhen` does **not**. So every `F.when(...)` fell through to the plain-column
path, which looked up a column literally named `"CASE WHEN"`, missed on every
row, and returned the aggregate's empty default -- `0` for `sum`, `None` for
`avg`/`max`/`min`.

The window path was worse: it addresses its input purely by *name*
(`get_row_value(row, self.column_name)`), where `column_name` holds only the
rendered expression text. **Any** expression target was therefore broken there,
including plain arithmetic -- `F.sum(F.col("x") * 2).over(w)` returned `0`.

**Reproduction**:
```python
df = spark.createDataFrame(
    [("a", 10.0, True), ("a", 20.0, False), ("b", 40.0, False)],
    ["grp", "x", "flag"],
)
df.groupBy("grp").agg(F.sum(F.when(F.col("flag"), F.col("x"))).alias("s")).collect()
df.withColumn("s", F.sum(F.col("x") * 2).over(Window.partitionBy("grp"))).collect()
```

| expression | PySpark 4.0.0 | Sparkless (before) |
|---|---|---|
| `agg sum(when(flag, x))` | `a=10.0, b=None` | `a=0, b=0` |
| `window sum(when(flag, x))` | `10.0, 10.0, None` | `0.0, 0.0, 0.0` |
| `window sum(x * 2)` | `60.0, 60.0, 80.0` | `0.0, 0.0, 0.0` |
| `window avg(x * 2)` | `30.0, 30.0, 80.0` | `None, None, None` |

**Confirmed**: 2026-07-20 against PySpark 4.0.0.

**Impact**:
- Silent. `0` is a plausible conditional sum, so assertions of "is not null" or
  "sum >= 0" pass on a garbage value. This shape is common in production code
  (`sum(when(cond, 1).otherwise(0))` is the standard conditional-count idiom).
- Independent of BUG-025 (three-valued boolean logic): the `when()` predicate is
  evaluated by `ConditionalEvaluator`, not by the filter path, and
  `sum(when(flag, x))` returns the correct value with or without that fix.

**Fix**:
- Added `is_row_evaluatable_expression()` to `sparkless/core/protocols.py`,
  which recognises both `ColumnOperation` (via `.operation`) and `CaseWhen`
  (via its `conditions`/`default_value` pair, see `CaseWhenLike`). The four
  aggregate branches -- `sum`, `avg`, `max`, `min` -- now gate on it.
- `WindowFunction` now captures its target expression and pre-computes it once
  per row into a synthetic column before dispatch
  (`_with_materialized_target`), so every existing window evaluator keeps
  working unchanged.

**Also corrected -- SUM over nothing is NULL, not 0**:
Spark's `SUM` returns **NULL** when there is no non-NULL value to add up.
Sparkless returned `0`, which is the same silent-zero failure by another route
(`sum(when(cond, x))` over a group where nothing matches is exactly this case).
Both the grouped and the window implementations now return NULL. Verified: zero
new test failures across `tests/unit` + `tests/parity`.

**Known remaining gap (not fixed here)**: `max`, `min`, `collect_set`,
`collect_list` and `stddev` over a window return `None` even for a *plain*
column -- they are missing from `WindowFunction.evaluate()`'s dispatch entirely.
That is a separate, additive fix.

**Regression tests**: `tests/unit/dataframe/test_aggregate_over_expression.py`
(15 tests, passing under both Sparkless and `MOCK_SPARK_TEST_BACKEND=pyspark`).
## Array set/search functions (PySpark 4.0.0 parity)

Numbering follows the assignments made in PR #20, which documented both of
these as open findings. Reference values were produced by executing each
reproduction against real **PySpark 4.0.0** on OpenJDK 21 (the DBR 17.3
pairing), not derived from the API docs.

---

### BUG-028: array_except and array_intersect are not implemented
**Status**: Fixed
**Severity**: Medium
**Discovered**: 2026-07-20
**File**: `sparkless/core/condition_evaluator.py`

**Description**:
Neither function had an evaluator branch anywhere. The constructors
(`sparkless/functions/array.py`) and the re-exports exist, so the call built
fine, matched no `operation_type`, and fell through to the dispatch's default
`return None`.

This is **not** an argument-passing problem: it returned NULL even with a
pure-literal second argument. That is what distinguishes it from BUG-029,
which it superficially resembles.

**Reproduction**:
```python
F.array_except(F.col("doms"), F.array(F.lit("d1")))  # PySpark ['d2'] -> Sparkless None
F.array_except(F.col("doms"), F.array(F.col("domain")))  # PySpark ['d2'] -> Sparkless None
F.array_intersect(F.col("doms"), F.array(F.col("domain")))  # PySpark ['d1'] -> Sparkless None
```

**Reference behaviour (PySpark 4.0.0)**:
Both have set semantics: the result is **deduplicated** and keeps the
first-occurrence order of the left array.
`array_except(['a','b','a','c'], ['a'])` is `['b','c']`;
`array_intersect(['a','a','b'], ['a','b'])` is `['a','b']`.
A NULL array on either side yields NULL; an empty result is `[]`, not NULL.
NULL participates as an ordinary **value**, not as SQL NULL:
`array_except(['a',NULL,'b'], [NULL])` is `['a','b']`.

**Fix**:
One shared branch implementing both (they differ only in the sense of the
membership test), placed next to `array_union` and mirroring its handling of
unhashable elements via a new `_element_key` helper. Both names added to the
function-op whitelist.

---

### BUG-029: array_remove and array_position do not resolve a Column argument
**Status**: Fixed
**Severity**: High
**Discovered**: 2026-07-20
**File**: `sparkless/core/condition_evaluator.py`

**Description**:
Both branches used `operation.value` raw, without resolving it against the
row:

```python
remove_value = operation.value          # still a Column
return [x for x in col_value if x != remove_value]
```

`x != <Column>` invokes `Column.__ne__`, which returns a **truthy
ColumnOperation rather than a bool**, so every element passed the filter and
the array came back unchanged. With a literal second argument both worked
correctly. The neighbouring `array_contains` and `array_union` already
resolved their operand; these two were the outliers.

**`array_position` had the same defect and was worse**: `list.index()` uses
`==`, and `Column.__eq__` is likewise truthy, so it matched index 0
unconditionally — returning a *plausible wrong number* rather than an
obviously-unchanged array.

**Reproduction**:
```python
F.array_remove(F.col("doms"), F.col("domain"))    # PySpark ['d2'] -> Sparkless ['d1','d2']
F.array_position(F.col("doms"), F.col("domain"))  # domain='zz' absent:
                                                  # PySpark 0 -> Sparkless 1
```

**Reference behaviour (PySpark 4.0.0)**:
- `array_remove` removes every occurrence and does **not** deduplicate the
  survivors. A **NULL** value argument makes the whole result NULL (it does
  not mean "remove nothing").
- `array_position` is a 1-based index, `0` when absent. A NULL value argument
  yields NULL. NULL elements occupy a position:
  `array_position(['d1',NULL,'d2'], 'd2')` is `3`.

**Impact**:
- Silent in both cases. The `array_position` variant is the dangerous one: a
  wrong integer index looks like a legitimate answer and passes any
  "not NULL" / "> 0" assertion.
- A downstream `collect_set(domain)` minus-own-value projection — the natural
  Spark formulation of "which *other* values fired" — returned the full set
  **including the row's own value**, silently violating the exclude-self
  invariant the column existed to satisfy.

**Fix**:
Added a shared `_resolve_operand()` helper that resolves a
`Column`/`ColumnOperation`/`Literal` operand against the row while leaving
plain Python scalars (notably bare `str`, which is the PySpark signature for
these functions) as literals. Routed `array_remove`, `array_position` and
both copies of `array_contains` through it, and added the NULL-argument
semantics above.

**Relationship to BUG-028**: distinct root causes. BUG-028 is an unmatched
dispatch falling through to a silent `None` (the failure design shared with
BUG-024/026/027); BUG-029 is a missing operand resolution inside branches
that do exist. Only BUG-029 is addressed by `_resolve_operand`.

**Regression tests**: `tests/unit/functions/test_array_column_argument_resolution.py`
(19 tests, passing under both Sparkless and `MOCK_SPARK_TEST_BACKEND=pyspark`).

---

### BUG-026: F.round ignores its scale argument and rounds half-to-even
**Status**: Fixed
**Severity**: High
**Discovered**: 2026-07-20
**File**: `sparkless/core/condition_evaluator.py`, `sparkless/dataframe/evaluation/expression_evaluator.py`

**Description**:
Two independent defects made `F.round(x, 2)` wrong.

1. **The scale was dropped.** There are three `round` implementations. Two
   called `round(float(value))` with no scale at all. The third,
   `ExpressionEvaluator._func_round`, read the scale from
   `getattr(operation, "precision", 0)` -- but `precision` has never been an
   attribute of `ColumnOperation`; the scale is carried on `operation.value`.
   The `getattr` default therefore won every time and rounded to zero decimal
   places.
2. **The rounding mode was wrong.** All three used Python's built-in `round`,
   which rounds halves to even: `round(2.5) == 2`. Spark's `round` rounds
   halves *away from zero*, giving `3.0`. (Spark's banker's-rounding function
   is `bround`, a different function.)

Spark also rounds the *decimal* representation of the double rather than its
exact binary expansion, so `round(2.675, 2)` is `2.68` -- not the `2.67` the
binary value `2.67499999...` would produce.

**Reproduction**:
```python
df.select(F.round(F.col("x") / 3, 2))
df.select(F.round(F.lit(2.5)), F.round(F.lit(1234.5678), -2))
```

| expression | PySpark 4.0.0 | Sparkless (before) |
|---|---|---|
| `round(10.0/3, 2)` | `3.33` | `3` |
| `round(3.14159, 2)` | `3.14` | `3` |
| `round(3.14159, 3)` | `3.142` | `3` |
| `round(2.5)` | `3.0` | `2` (banker's) |
| `round(-2.5)` | `-3.0` | `-2` |
| `round(0.125, 2)` | `0.13` | `0` |
| `round(2.675, 2)` | `2.68` | `0` |
| `round(1234.5678, -2)` | `1200.0` | `1235` |

**Confirmed**: 2026-07-20 against PySpark 4.0.0.

**Impact**:
- Silent, and the result stays numerically plausible -- a money figure rounded
  to 0 decimals instead of 2 still passes "is not null" and "> 0" assertions.
- The returned value also changed *type*, from float to int.

**Fix**:
Added `spark_round()` to `sparkless/core/math_utils.py`, which quantizes via
`Decimal` with `ROUND_HALF_UP` and derives the decimal from `str(float(v))` --
reproducing Java's `BigDecimal.valueOf(double)`, which is what Spark rounds.
All three implementations now read the scale off the operation and delegate to
it.

**Known related gap (not fixed here)**: `F.bround` has no implementation at all
and returns `None`. Note that Python's built-in `round` -- the function these
three sites were wrongly using -- is precisely the correct semantics for
`bround`.

**Regression tests**: `tests/unit/functions/test_round_scale_argument.py`
(17 tests, passing under both Sparkless and `MOCK_SPARK_TEST_BACKEND=pyspark`).

---

### BUG-035: Most aggregate functions over a window return NULL
**Status**: Fixed
**Severity**: Critical
**Discovered**: 2026-07-21
**File**: `sparkless/functions/window_execution.py`

**Description**:
`WindowFunction.evaluate()` dispatched aggregates through an `elif` chain naming
`sum`, `avg`, `count` and `approx_count_distinct`. Everything else fell into
`else: return [None] * len(data)`. So `max`, `min`, `mean`, `collect_list`,
`collect_set`, `stddev`/`stddev_samp`/`stddev_pop`, `variance`/`var_samp`/
`var_pop`, `skewness`, `kurtosis`, `product`, `median`, `mode`, `any_value` and
`bit_and`/`bit_or`/`bit_xor` over **any** window returned NULL -- for a plain
column, with no expression involved.

**Reproduction**:
```python
df.withColumn("m", F.max("x").over(Window.partitionBy("grp")))  # -> None
```

**Impact**:
An unimplemented function was indistinguishable from a genuine SQL NULL, so the
library reported a plausible wrong answer rather than "not supported".
`max`/`min` are the dangerous cases: NULL reads as "no data".

**Fix**:
New `sparkless/functions/window_frames.py` holds a `{name: reducer}` table
(`REDUCERS`) plus `resolve_frame()`. `WindowFunction._evaluate_frame_aggregate()`
is now the single implementation for every frame-shaped aggregate. Adding an
aggregate is a table entry, not a new branch.

A genuine dispatch miss now emits a `UserWarning` naming the function instead of
returning a silent NULL.

---

### BUG-036: Window frames ignored -- ordered aggregates return the partition total
**Status**: Fixed
**Severity**: Critical
**Discovered**: 2026-07-21
**File**: `sparkless/functions/window_execution.py`

**Description**:
The three aggregates that *were* implemented each approximated the window frame
differently, and none of them correctly. Spark's default frame for an ORDER BY
window is `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` -- a running
aggregate, with tied rows sharing one frame as peers.

* `sum` applied the running frame only when an explicit
  `rowsBetween(unboundedPreceding, currentRow)` was supplied; otherwise it
  returned the whole-partition total.
* `avg` on the ordered path ignored `partitionBy` **entirely**, accumulating a
  running average over the whole DataFrame in physical row order.
* `count` ignored the ordering and returned the partition count.

**Reproduction** (values 10, 20, 30 in one partition, ordered by a distinct key):
```python
df.withColumn("r", F.sum("x").over(Window.partitionBy("grp").orderBy("k")))
# Sparkless: 60, 60, 60      PySpark 4.0.0: 10, 30, 60
```
For `avg`, a two-row partition holding only NULL and 7.0 received 16.0 and 16.25
-- numbers computed from a *different* partition's rows.

**Impact**:
More dangerous than BUG-035, because a wrong *number* survives any assertion
that checks shape or non-nullness. Running totals over an ordered window are
how stock ledgers and trend tables are computed downstream.

Note this is why BUG-035 and BUG-036 are one defect and not two: there was no
shared notion of "the frame", so each function invented its own.

**Fix**:
`resolve_frame()` implements Spark's frame rules once -- default ROWS-whole-
partition without ORDER BY, default RANGE prefix with it, explicit
`rowsBetween` (physical) and `rangeBetween` (value-based, offsets following the
sort direction) -- and every reducer runs over the resolved frame.

This also fixed 15 pre-existing failing tests in the repo, including
`test_issue_392_window_sum_peers`, `test_issue_393_sum_string_column`,
`test_issue_407_stddev_window` and `test_issue_414_row_number_over_descending`.

**Regression tests**: `tests/unit/functions/test_window_aggregate_frames.py`
(47 tests, passing under both Sparkless and `MOCK_SPARK_TEST_BACKEND=pyspark`).

---

### BUG-037: A scalar function wrapping an aggregate returns NULL
**Status**: Fixed
**Severity**: High
**Discovered**: 2026-07-21
**File**: `sparkless/dataframe/grouped/base.py`

**Description**:
(Reported as BUG-026 in the open-findings sweep; renumbered on landing.)
`_evaluate_column_expression` resolved the inner aggregate but gated every
downstream branch on an arithmetic-only operation set (`+ - * / %`), so `sqrt`,
`abs`, `ceil`, `floor`, `coalesce`, `upper` and every other *named* function
matched nothing and hit the literal `return expr_name, None` at the bottom.

```python
df.groupBy("grp").agg(F.sqrt(F.sum("x")).alias("r"))  # PySpark 7.745967 -> None
```

Arithmetic on an aggregate worked, and a scalar function on a plain column
worked; only the combination failed.

**Fix**:
`_resolve_aggregates_to_row()` walks the expression, evaluates each aggregate
node to a scalar into a synthetic one-row dict, and substitutes a plain column
reference for it. The outer expression is then handed to the ordinary
`ExpressionEvaluator`, so scalar functions are supported *by construction*
rather than enumerated -- and the grouped path inherits scalar-evaluator fixes
automatically.

**Regression tests**: `tests/unit/functions/test_grouped_scalar_over_aggregate.py`
(14 tests, passing under both Sparkless and `MOCK_SPARK_TEST_BACKEND=pyspark`).

---

## Findings from the aggregate/window sweep

Reproduced against **PySpark 4.0.0** on OpenJDK 21. Each was found while fixing
BUG-035/036/037 and left out to keep that change to a single concern.
BUG-038/039/040/042 were subsequently fixed together (they share one failure
design -- see BUG-038); BUG-041 remains open.

### BUG-038: `least` and `greatest` return their first argument
**Status**: Fixed
**Severity**: High
**File**: `sparkless/core/condition_evaluator.py`

`ExpressionEvaluator` routes `least`/`greatest` to a stub branch commented
"simplified version" that returns the first operand's value and never looks at
the others:

```python
elif operation_type == "greatest":
    # For greatest, we need multiple values - this is a simplified version
    return value if value is not None else None
```

A correct implementation exists in the same file (~line 1503) and is what
`df.select(F.least(a, b))` reaches, so the plain path is right and only some
callers hit the stub. Verified on a one-row frame `{s0: 30.0, s1: 60.0}`:
`greatest(s0, s1)` returns `30.0`, PySpark returns `60.0`.

This is another *plausible wrong value*, not a NULL, and it is easy to miss:
`greatest(F.sum("x"), F.max("x"))` returns the right answer whenever the sum
happens to be the larger of the two, which is most of the time.

Not fixed here because `condition_evaluator.py` is touched by several in-flight
PRs.


**Root cause (shared with BUG-039/040/042 and with BUG-035)**:
A dispatch whose unmatched case yields a *plausible value* instead of an error.
`ExpressionEvaluator._evaluate_function_call` ends in `return value` -- the
function's first operand -- for any name not in `_function_registry`, and
neither `greatest`/`least` nor `bround` was registered. The same shape produced
BUG-035's `[None] * len(data)` and BUG-039's top-level-only `_AGG_OPS` test.

**Fix**:
`sparkless/core/variadic.py` holds the NULL-skipping semantics once;
`ConditionEvaluator` (the `select` path, already correct) and
`ExpressionEvaluator` (the `withColumn` path, which returned operand 1) both
call it, so they cannot drift apart again. The terminal `return value` now
emits a `UserWarning` naming the function, mirroring what BUG-035's fix did for
the window dispatch -- which is how BUG-046 was found.

`F.greatest(x)` with a single argument now raises, as Spark's
`[WRONG_NUM_COLUMNS]` does; accepting it made `greatest` an identity function.

**Reference behaviour**: `greatest`/`least` **skip** NULL operands rather than
propagating them -- `greatest(NULL, 2, 3)` is `3`, and only an all-NULL
argument list yields NULL. This is the opposite of most functions and was
verified on PySpark 4.0.0 / OpenJDK 21.

**Regression tests**: `tests/unit/functions/test_least_greatest_operand_shapes.py`
(31 tests, passing under both Sparkless and `MOCK_SPARK_TEST_BACKEND=pyspark`).

### BUG-039: A select-level scalar over an aggregate does not collapse the rows
**Status**: Fixed
**Severity**: Medium
**File**: `sparkless/dataframe/lazy.py`

```python
df.select(F.sum("x").alias("r"))        # 1 row, correct
df.select(F.sqrt(F.sum("x")).alias("r"))  # 5 rows of NULL; PySpark: 1 row, 8.944272
```

The wrapped aggregate is not recognised as an aggregate by the select planner,
so the projection stays row-wise. BUG-037's fix covers the `groupBy(...).agg()`
path only. Both the value *and* the row count are wrong.


**Fix**:
The select planner's `_is_agg_expr` tested the **top-level** operation name
against a hardcoded `_AGG_OPS` set, so wrapping an aggregate in anything moved
it out of the tested slot. It now walks the whole expression -- the same move
BUG-037 made for `groupBy().agg()` -- so wrapping works by construction rather
than by enumeration. A window function is deliberately not descended into:
`F.sum(x).over(w)` contains an aggregate but is row-wise, and collapsing it
would be worse than the bug being fixed.

Still divergent (out of scope): `df.select(F.col("x"), F.sum("x"))` returns a
row where PySpark raises `[MISSING_GROUP_BY]`.

**Regression tests**: `tests/unit/functions/test_select_scalar_over_aggregate.py`
(10 tests, passing under both engines).

### BUG-040: `first`/`last` over a window ignore peers and explicit frames
**Status**: Fixed
**Severity**: Medium
**File**: `sparkless/functions/window_execution.py`

`_evaluate_last` returns the current row's value for an ordered window, which is
correct only when the ORDER BY key is unique. With ties, Spark returns the last
value of the *peer group*: for rows `(k=1, x=10)` and `(k=1, x=20)` under
`orderBy("k")`, PySpark gives `20` for both; sparkless gives `10` and `20`.
`_evaluate_first`/`_evaluate_first_value`/`_evaluate_last_value` likewise ignore
an explicit `rowsBetween`/`rangeBetween`.

These are positional rather than aggregate functions, so they were left on their
bespoke branches; routing them through `resolve_frame()` is the natural fix, but
it interacts with `ignoreNulls`, which was not probed.


**Fix**:
`first`/`last`/`first_value`/`last_value` are now `POSITIONAL_REDUCERS` entries
in `window_frames.py` and run through `_evaluate_frame_aggregate`, so they
inherit `resolve_frame()`'s peer groups and explicit `rowsBetween`/
`rangeBetween` instead of re-deriving the frame by hand. Four bespoke methods
and four `elif` branches were deleted.

`ignoreNulls` -- which the note above flagged as unprobed -- turned out to be
parsed onto the `AggregateFunction` and then never read on the window path, so
`F.first(x, True)` silently behaved like `F.first(x)`. `F.last` did not accept
the argument at all. Both now honour it, as do `first_value`/`last_value`.
Confirmed against PySpark 4.0.0: without `ignoreNulls`, a NULL at the frame
edge **is** the answer -- `last` does not fall back to the last non-NULL seen.

**Regression tests**: `tests/unit/functions/test_window_positional_and_distinct.py`
(14 tests, passing under both engines).

### BUG-041: ORDER BY sorts NULLs last; Spark sorts them first on ASC
**Status**: Fixed
**Severity**: Medium
**Files**: `sparkless/spark_types.py`, `sparkless/functions/window_execution.py`,
`sparkless/dataframe/window_handler.py`, `sparkless/dataframe/lazy.py`

Spark's default is `ASC NULLS FIRST` / `DESC NULLS LAST` -- which is why
`asc_nulls_last()` exists as an explicit variant. `_sort_indices_by_columns`
defaults to `nulls_last = True` for a plain ascending sort. Verified: with keys
`[1, 2, 4, NULL]` and a running sum, PySpark places the NULL-key row *first*, so
every subsequent row's running total includes it.

Affects `row_number`, `rank`, `dense_rank`, `lag`/`lead` and the new frame
engine alike -- and it moves rows under every window function at once, so it
needed its own verification pass.

**Not one helper, but three.** The direction-parsing block was duplicated across
`window_execution._sort_indices_by_columns` (rank/row_number/frames),
`window_handler._apply_ordering_to_indices` (lag/lead) and the `orderBy` branch
of `lazy._materialize` (plain `DataFrame.orderBy`/`sort`). All three defaulted
ascending sorts to NULLS LAST, and `lazy`'s copy had drifted further -- it also
mapped `F.asc_nulls_last` and `F.asc` to the same spec, so the explicit variant
was indistinguishable from the default.

**Reference behaviour** (real PySpark 4.0.0, OpenJDK 21), rows
`[(a,1) (b,2) (c,4) (d,NULL) (e,2)]`:
```
orderBy("k")                  -> d a b e c     # ASC  NULLS FIRST
orderBy(col("k").desc())      -> c b e a d     # DESC NULLS LAST
orderBy("k", ascending=False) -> c b e a d
asc_nulls_last                -> a b e c d
desc_nulls_first              -> d c b e a
orderBy("g","k")              -> e c d a b     # per-key NULL placement
row_number over orderBy("k")  -> d=1 a=2 b=3 e=4 c=5
sum("k")   over orderBy("k")  -> d=NULL a=1 b=5 e=5 c=9
lag("id")  over orderBy("k")  -> d=NULL a=d b=a e=b c=e
```

**Fix**:
`spark_types.resolve_order_key(col, default_ascending=True)` resolves one order
key into `(column_name, is_desc, nulls_last)` and is now the only place that
maps a direction to a NULL placement; the three call sites were replaced with a
call to it. Default `nulls_last` mirrors the direction (ASC -> first,
DESC -> last); the explicit `*_nulls_*` variants override it.

**Not changed**: sparkless emits rows in input order after
`df.withColumn(<window expr>)` where Spark happens to emit them in window-sort
order. Spark does not guarantee that order without an explicit `orderBy`, so it
is not pinned by the tests.

**Tests**:
- `tests/parity/dataframe/test_null_ordering.py` (new, 28 tests,
  backend-agnostic `spark` fixture; all 28 also pass under
  `MOCK_SPARK_TEST_BACKEND=pyspark` against PySpark 4.0.0). Covers single key,
  multi-key, mixed ASC/DESC, all four explicit variants, string keys, an
  all-NULL key, NULLs in the partition key, and window vs plain `orderBy`.
  Restoring the NULLS LAST default fails 18 of them; corrupting the explicit
  variants instead fails the other 10.

### BUG-048: `sum()` over an integer column returns a float
**Status**: Open
**Severity**: Low
**Discovered**: 2026-07-21

Noticed while capturing the BUG-041 window reference. `F.sum("k")` over an
`IntegerType` column returns `1.0`/`5.0` in sparkless where PySpark 4.0.0
returns `1`/`5` (SUM of an integral type is `bigint`). Values agree; only the
type differs, so it surfaces as `Row(rs=5.0)` vs `Row(rs=5)` in strict
comparisons.

### BUG-042: DISTINCT aggregates over a window are accepted
**Status**: Fixed
**Severity**: Low
**File**: `sparkless/functions/window_execution.py`

PySpark 4.0.0 rejects `F.count_distinct(...).over(w)` and
`F.sum_distinct(...).over(w)` with
`[DISTINCT_WINDOW_FUNCTION_UNSUPPORTED] ... SQLSTATE: 0A000`. Sparkless computes
a value for the first and NULL for the second. Sparkless being more permissive
than Spark means a query that cannot run in production passes its unit tests.


**Fix**:
`WindowFunction.evaluate()` raises `AnalysisException` with Spark's
`[DISTINCT_WINDOW_FUNCTION_UNSUPPORTED] ... SQLSTATE: 0A000` for the names in
`DISTINCT_WINDOW_FUNCTIONS`. The check is on the evaluation path, not in
`.over()`, because `F.count_distinct(x).over(w)` on its own is a legal Column
in PySpark too -- only *using* it raises. `approx_count_distinct` is not a
DISTINCT aggregate and Spark does permit it over a window, so it is explicitly
excluded; a test pins that the guard does not over-reject.

**Regression tests**: `tests/unit/functions/test_window_positional_and_distinct.py`.

### Note: `F.round` still ignores its scale argument for expression operands

`F.round(F.col("a") / 3, 2)` returns `7` rather than `6.67` on `main`. This is
BUG-025, whose fix (`spark_round()` in `math_utils.py`) is in an unmerged PR.
BUG-037's fix routes the grouped path through the same scalar evaluator, so
`agg(F.round(F.sum("x") / 3, 2))` will start returning `6.67` as soon as that
lands -- no further change needed here.

---

## Findings from the least/greatest sweep

Reproduced against **PySpark 4.0.0** on OpenJDK 21 while fixing
BUG-038/039/040/042.

> Numbering note: the concurrent ordering / NULL-comparison sweep landed on
> `main` as BUG-046/047 while this branch was in flight, so these findings took
> 048/049 on rebase. 044 ended up unused.

### BUG-045: `F.bround` is not implemented
**Status**: Fixed
**Severity**: Medium
**File**: `sparkless/core/math_utils.py`

Flagged as a "known related gap" under BUG-025 and never given its own number.
`bround` was registered in neither evaluator, so it took the same silent
fallthrough as BUG-038: `df.select(F.bround(v, 2))` returned NULL and
`df.withColumn("b", F.bround(v, 2))` returned the **unrounded** value. The
latter is the dangerous one -- a money figure that was never rounded still
passes "is not null" and "> 0".

**Reference behaviour**:
`bround` is HALF_EVEN, but it is **not** Python's built-in `round`, contrary to
the note under BUG-025. Python rounds the exact binary expansion of the double;
Spark rounds its shortest round-tripping decimal string. They disagree wherever
those differ:

| Expression | Python `round` | PySpark 4.0.0 `bround` |
|---|---|---|
| `bround(2.5, 0)` | `2` | `2.0` |
| `bround(3.5, 0)` | `4` | `4.0` |
| `bround(2.675, 2)` | `2.67` | **`2.68`** |
| `bround(1234.5678, -2)` | `1200.0` | `1200.0` |

**Fix**:
`spark_bround()` shares a `_quantize()` helper with `spark_round()`, so the two
differ only in their rounding mode and cannot drift on the decimal-string
detail. Registered in both evaluators.

Known divergence, shared with `spark_round` and therefore left alone: on an
integral column Spark preserves the integer type where sparkless returns a
float.

**Regression tests**: `tests/unit/functions/test_least_greatest_operand_shapes.py`
(`TestBround`, passing under both engines).

### BUG-050: `pow`, `log`, `substring`, `element_at`, `array_distinct` return their first operand
**Status**: Open
**Severity**: High
**File**: `sparkless/dataframe/evaluation/expression_evaluator.py`

Found *by* BUG-038's fix. Making the terminal `return value` warn instead of
answering silently immediately surfaced five more functions taking the same
fallthrough from `withColumn`:

```python
df = spark.createDataFrame([(4.0, 2.0, "hello")], "a double, b double, s string")
df.withColumn("p", F.pow(F.col("a"), F.col("b")))     # 4.0;      PySpark 16.0
df.withColumn("p", F.log(F.col("a")))                 # 4.0;      PySpark 1.3862943611198906
df.withColumn("p", F.substring(F.col("s"), 1, 3))     # 'hello';  PySpark 'hel'

da = spark.createDataFrame([([1, 1, 2],)], "arr array<int>")
da.withColumn("p", F.array_distinct(F.col("arr")))    # [1,1,2];  PySpark [1,2]
da.withColumn("p", F.element_at(F.col("arr"), 1))     # [1,1,2];  PySpark 1
```

Every one is a plausible-looking wrong value rather than a NULL, and
`substring`/`array_distinct` are right whenever the operand is already short
enough or already distinct.

Not fixed here: each needs its own reference pass and its own operand-shape
matrix, and folding five more functions into the least/greatest change would
have made it unreviewable. They are now *loud* rather than silent, which was
the point of the warning.

### BUG-049: `greatest`/`least` accept operands of incompatible types
**Status**: Open
**Severity**: Low
**File**: `sparkless/core/variadic.py`

PySpark rejects `F.greatest(int_col, string_col)` at analysis time with
`[DATATYPE_MISMATCH.DATA_DIFF_TYPES]`. Sparkless returns NULL for every row.

Same family as BUG-042 -- more permissive than the thing being mocked -- but
deferred rather than fixed: the correct fix is an analysis-time type check
against the frame's schema, which sparkless has no phase for. Raising a
`TypeError` from the reducer at collect time would be a guess at the right
error, in the wrong place, and would fire on rows rather than on the query.
NULL is at least a documented answer. Recorded so the gap is not rediscovered
as a mystery.

Related, and also unfixed: sparkless does not perform Spark's implicit numeric
widening, so `greatest(int_col, double_col)` returns `3` where PySpark returns
`3.0`. The value is right, the type is not.
