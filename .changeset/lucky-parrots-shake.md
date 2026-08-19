---
"sparkless": minor
---

Rewrite `F.expr`'s SQL parser: honour operator precedence, keep every function argument, and raise instead of returning a wrong value.

`F.expr` matched operators and keywords with regexes over the raw string, so `quantity * 2 + 1` bound as `quantity * (2 + 1)` (75, where Spark says 51), `concat(sku, dept)` dropped both arguments and evaluated to NULL, `coalesce(name, 'FALLBACK')` lost its fallback, and `size(tags) > 1` was False for a three-element array. None of these raised. It is now a tokenizer, a recursive-descent parser with Spark's precedence and left associativity, and a binder that dispatches every call to the real `F` function -- so `F.expr` can no longer disagree with the programmatic API. `BETWEEN`, `NOT BETWEEN`, `CAST`/`TRY_CAST`, `count(*)`, `count(DISTINCT x)`, qualified and backticked identifiers, and chained same-precedence operators (`a - b - c`) now work; 53 parity tests pin the results to PySpark 4.0.0.

Two behaviour changes to be aware of:

- An expression sparkless cannot evaluate now raises `ParseException` instead of warning and returning a column that evaluates to NULL. This affects `uuid()` (no sparkless implementation), `INTERVAL` literals, bitwise `~`, and higher-order functions called with a lambda (`filter(xs, x -> ...)`), all of which previously produced NULL or a nonsense column.
- `F.pi()` and `F.e()` returned NULL for every row through the programmatic API; they emit the operation the evaluator implements and now return the constant.
