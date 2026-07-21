---
"sparkless": patch
---

Built-in function fixes: `F.round` honours its `scale` argument and rounds HALF_UP, `least`/`greatest`/`bround` are implemented, `array_except`/`array_intersect` are added, `Column` arguments are resolved wherever a function accepts one, and `last_day`/`trunc` plus date predicates evaluate correctly.
