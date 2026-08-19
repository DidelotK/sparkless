---
"sparkless": patch
---

`F.flatten`, `F.array_min`, `F.array_max` and `F.slice` now compute a value instead of returning NULL for every row on the `select` path, and `F.array_distinct` now deduplicates on the `withColumn`/`agg` path instead of returning the array unchanged. `array_distinct(flatten(collect_list(x)))` therefore deduplicates. `F.slice` rejects `start=0` and a negative `length`, as Spark does, instead of answering NULL. The implementations live in one module both evaluators call, so the two projection paths can no longer disagree.
