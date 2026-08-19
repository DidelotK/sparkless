---
"sparkless": minor
---

Rewrite `F.expr`'s SQL parser: honour operator precedence, keep every function argument, and support `uuid()`, day-based `INTERVAL`s, `BETWEEN` and qualified identifiers.

`F.expr` matched operators and keywords with regexes over the raw string, so `quantity * 2 + 1` bound as `quantity * (2 + 1)` (75, where Spark says 51), `concat(sku, dept)` dropped both arguments and evaluated to NULL, `coalesce(name, 'FALLBACK')` lost its fallback, and `size(tags) > 1` was False for a three-element array. None of these raised. It is now a tokenizer, a recursive-descent parser with Spark's precedence and left associativity, and a binder that dispatches every call to the real `F` function -- so `F.expr` can no longer disagree with the programmatic API. Measured on the 119 expressions a downstream data platform actually passes to `F.expr`, agreement with PySpark 4.0.0 goes from 65 to 100.

Newly working: `BETWEEN` / `NOT BETWEEN`, `CAST`/`TRY_CAST`, `count(*)`, `count(DISTINCT x)`, qualified (`a.b`) and backticked identifiers, `NOT IN`, `NOT LIKE`, `ILIKE`, `||`, `<=>`, and chained same-precedence operators (`a - b - c`, which used to raise).

Also fixed, because binding through `F` exposed them:

- **`uuid()` is implemented** (`F.uuid()`). It previously resolved to nothing and evaluated to NULL, so an id column built with `F.expr("uuid()")` was NULL for every row.
- **Nullary functions in `withColumn`.** `uuid()`, `pi()` and `e()` went through an evaluator whose null-propagation guard fired on the *absent* operand and returned NULL for every row.
- **`F.pi()` and `F.e()`** returned NULL through the programmatic API.
- **`date ± INTERVAL n DAYS/WEEKS`** binds to a `timedelta` literal and evaluates correctly; it used to resolve as a column named `INTERVAL_90_DAYS`, so a cutoff filter silently matched nothing.

One behaviour change to be aware of: an expression sparkless cannot evaluate now raises `ParseException` instead of warning and returning a column that evaluates to NULL. This affects month-based `INTERVAL`s (they would need `F.add_months`, which returns NULL for every row), bitwise `~`, and higher-order functions called with a lambda (`filter(xs, x -> ...)`), all of which previously produced NULL or a nonsense column reference.
