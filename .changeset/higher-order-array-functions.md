---
"sparkless": patch
---

`F.exists`, `F.forall` and `F.filter` now compute their answer with SQL three-valued logic instead of returning NULL for every row, and `F.transform` maps its array instead of raising `LambdaTranslationError` from schema inference. Guards built on them — such as `size(a) != size(array_distinct(a))` over a `transform` — can now fire.
