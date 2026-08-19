---
"sparkless": patch
---

`F.struct(F.lit(v).alias("name"))` now names the field `name` instead of the positional `col1`. `Literal.alias()` records the alias on `_alias_name`, the attribute every "was this expression named by its author?" check reads, so an aliased literal is no longer indistinguishable from an unaliased one.
