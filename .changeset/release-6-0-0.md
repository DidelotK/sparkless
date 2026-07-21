---
"sparkless": major
---

**Version jump to 6.0.0. This is not a breaking change.**

Nothing in this release breaks the API. Upgrading from 4.2.2 requires no code changes — the major bump exists purely to escape a version-number collision, and semver is being used here as an escape hatch rather than as a compatibility signal. Read the numbering note below before assuming otherwise.

### Why 6.0.0

Three different things have been publishing under the name `sparkless`, and the numbers had grown incoherent:

| Where | Latest | What it actually is |
| --- | --- | --- |
| public PyPI `sparkless` | 4.13.2 | The upstream Rust/Polars package (`robin-sparkless` engine), unrelated to this fork |
| Azure Artifacts feed `sparkless` | 5.9.0 | A mix: Solya's pure-Python builds **and** upstream releases cached through the feed's PyPI passthrough |
| This repository | 4.2.3 | The Solya fork |

So the Solya fork's 4.2.x line sat *below* a public package of the same name already at 4.13.x, while the feed separately carried a 5.x. Any new 4.2.x or 5.x release would land under something already published.

6.0.0 is above everything on both sides. It is a deliberate discontinuity that gives the fork an unambiguous range of its own.

### Two things to know about the feed

The Azure feed has an upstream passthrough to public PyPI, so the two packages are genuinely mixed in one index. `sparkless 4.2.0` on the feed carries **ten** files: the nine Rust platform wheels from public PyPI plus one `py3-none-any` wheel built here. **Pin the version** — an unpinned `sparkless` can resolve to the upstream package rather than this one.

Also: **4.2.3 was never published.** It was versioned and tagged in the changelog, but its publish run failed before uploading anything, so no 4.2.3 artifact exists on the feed and none is coming. Do not go looking for it.

### What is actually in this release

The fixes that 4.2.3 was meant to carry, unchanged:

- **SQL NULL semantics** — boolean columns, negated function results and comparison operators all evaluate under three-valued logic; predicates and logical connectives resolve to booleans instead of to one of their operands; NULLs sort in Spark's order.
- **`CASE WHEN` evaluation** — nested `CASE WHEN`, struct projections and arbitrary expressions inside a `when`/`otherwise` branch, and aggregates computed over a `CASE WHEN`, all return the value Spark returns.
- **Window functions** — `Window.orderBy` applies per-key sort direction, window frames are resolved rather than approximated, and positional window functions are routed through the same frame engine as the other aggregates.
- **Built-in functions** — `F.round` honours its `scale` argument and rounds HALF_UP; `least`/`greatest`/`bround` are implemented; `array_except`/`array_intersect` are added; `Column` arguments are resolved wherever a function accepts one; `last_day`/`trunc` and date predicates evaluate correctly.
- **Schema binding** — binding a schema copies it instead of retaining the caller's object graph, so mutating a schema after use no longer reaches back into an already-built DataFrame.
