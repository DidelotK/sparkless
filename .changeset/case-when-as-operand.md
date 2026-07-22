---
"sparkless": patch
---

Evaluate a CASE WHEN used as an operand, sort key, or column name

`F.when(...).otherwise(...)` returns a `CaseWhen`, which is deliberately
neither a `Column` nor a `ColumnOperation`. Three dispatches that resolve
operands did not name it, so each silently mishandled it.

- **Arithmetic operand.** `ConditionEvaluator._get_column_value` fell through
  to `return column` and handed back the *unevaluated* object. `CASE * 2.0`
  then dispatched to `CaseWhen.__mul__`, which builds a new `ColumnOperation`
  instead of multiplying, so an ordinary `F.when(...) * F.lit(2.0)` produced a
  `ColumnOperation` **as the cell value**. Wrapping it hid the object behind a
  plausible number: `F.abs(CASE * lit)` returned `None` and
  `F.greatest(CASE * lit, F.lit(-1.0))` returned `-1.0`.
- **Sort key.** `DataFrame.orderBy` resolved each key to a column *name*. A
  computed key (`orderBy(F.when(...) * F.lit(2))`) matched no stored column,
  so every comparison tied and the frame came back in **input order** —
  silently unsorted. Keys are now resolved to a getter: a row lookup for a
  real column, a per-row evaluation for an expression.
- **Generated column name.** `CaseWhen.otherwise` f-string-interpolated raw
  operands, so a `Literal` rendered via `object.__repr__` and the generated
  name embedded a **memory address** that changed on every run. Only the first
  `WHEN` was rendered. Names now match PySpark's SQL text, including
  multi-branch and nested CASE.

Verified against real PySpark 4.0.0 on OpenJDK 21 (the DBR 17.3 pairing):
`tests/unit/functions/test_case_when_as_operand.py` passes under both engines.
