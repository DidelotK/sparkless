---
"sparkless": patch
---

Correct two claims in the name-memo documentation and repair an unsound assertion in its tests. The runtime is bytecode-identical to 6.0.2; this ships the corrected reasoning, not a behaviour change. The comment claiming an interrupted derivation can only strand the depth counter *low* was wrong -- an asynchronous exception delivered inside the `finally` strands it high, and the clamp cannot recover that -- and the concurrency test compared `id()` of thread-local caches, which a freed-and-reallocated dict can make collide.
