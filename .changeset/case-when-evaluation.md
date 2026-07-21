---
"sparkless": patch
---

`when`/`otherwise` chains are evaluated properly: nested `CASE WHEN`, struct projections and arbitrary expressions inside a branch, and aggregates computed over a `CASE WHEN` all return the value Spark returns.
