---
"sparkless": patch
---

Re-release of the fixes intended for 4.2.3. **4.2.3 was never published** — its publish run failed before uploading anything, so no 4.2.3 artifact exists on the feed and none is coming; do not go looking for one. The contents are unchanged from that aborted release:

- **SQL NULL semantics** — boolean columns, negated function results and comparison operators all evaluate under three-valued logic; predicates and logical connectives resolve to booleans instead of to one of their operands; NULLs sort in Spark's order.
- **`CASE WHEN` evaluation** — nested `CASE WHEN`, struct projections and arbitrary expressions inside a `when`/`otherwise` branch, and aggregates computed over a `CASE WHEN`, all return the value Spark returns.
- **Window functions** — `Window.orderBy` applies per-key sort direction, window frames are resolved rather than approximated, and positional window functions are routed through the same frame engine as the other aggregates.
- **Built-in functions** — `F.round` honours its `scale` argument and rounds HALF_UP; `least`/`greatest`/`bround` are implemented; `array_except`/`array_intersect` are added; `Column` arguments are resolved wherever a function accepts one; `last_day`/`trunc` and date predicates evaluate correctly.
- **Schema binding** — binding a schema copies it instead of retaining the caller's object graph, so mutating a schema after use no longer reaches back into an already-built DataFrame.
