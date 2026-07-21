---
"sparkless": patch
---

Binding a schema copies it instead of retaining the caller's object graph, so mutating a schema after use no longer reaches back into an already-built DataFrame.
