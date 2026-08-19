---
"sparkless": patch
---

`F.to_json` now serializes struct, map and array values instead of returning NULL on the `select` path, and `F.to_json(x).alias("j")` names the column `j` instead of re-deriving `to_json(struct(...))`. Spark's three different NULL rules are applied: a NULL struct field is omitted, a NULL array element and a NULL map value are kept.
