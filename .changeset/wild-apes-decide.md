---
"sparkless": patch
---

Stop re-deriving a column's name once per level of nesting. `ColumnOperation.name` walked its operand tree twice (via `_generate_name()` and again via `str(self)`), so naming a node cost `2 ** depth`, and a sub-expression referenced twice was re-walked once per reference on top of that. A 24-level expression over five rows made 18.9 million calls into the name helper and took 12.9 s; it now makes ~200 and takes 0.01 s. Name derivation is memoised for the duration of one outermost walk only, so no name outlives the tree it was derived from.
