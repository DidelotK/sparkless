---
"sparkless": patch
---

`F.countDistinct` and `F.approx_count_distinct` now count when their target is an expression — `F.countDistinct(F.struct(...))`, `F.countDistinct(F.upper(c))`, `F.countDistinct(F.when(...))` — instead of returning 0. Composite target values (a struct is a dict) are keyed rather than hashed directly, so a struct target counts.
