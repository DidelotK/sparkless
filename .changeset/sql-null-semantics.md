---
"sparkless": patch
---

SQL NULL semantics now match Spark: boolean columns, negated function results and comparison operators all evaluate under three-valued logic, predicates and logical connectives resolve to booleans instead of to one of their operands, and NULLs sort in Spark's order.
