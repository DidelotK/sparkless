---
"sparkless": patch
---

Window functions go through a single frame engine: `Window.orderBy` applies per-key sort direction, window frames are resolved rather than approximated, and positional window functions are routed through the same engine as the other aggregates.
