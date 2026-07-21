"""Window frame resolution and aggregate reducers.

Before this module existed, every aggregate over a window was hand-dispatched in
``WindowFunction.evaluate()``: ``sum``, ``avg``, ``count`` and a handful of
positional functions each had a bespoke branch, and every *other* aggregate fell
through an ``elif`` chain into ``return [None] * len(data)``. Two consequences
followed, both silent:

* ``max``, ``min``, ``collect_list``, ``collect_set``, ``stddev``, ``variance``,
  ``product``, ``median``, ... over any window returned NULL. An unimplemented
  function was indistinguishable from a genuine SQL NULL.
* The three branches that *were* implemented each approximated the window frame
  differently and none of them correctly. ``sum`` returned the whole-partition
  total unless an explicit ``rowsBetween(unboundedPreceding, currentRow)`` was
  given -- so a plain ``F.sum(x).over(Window.partitionBy(g).orderBy(k))``
  produced the partition total where Spark produces a running total. ``avg``
  ignored partitioning altogether on the ordered path. ``count`` ignored
  ordering.

Both are the same defect: there was no shared notion of "the frame". This module
supplies one -- ``resolve_frame()`` computes the row set Spark would see, and
``REDUCERS`` maps a function name to the aggregation applied over it -- so that
adding an aggregate is a table entry rather than a new hand-written branch.

Reference behaviour throughout was captured from PySpark 4.0.0 on OpenJDK 21
(the DBR 17.3 pairing); the regression tests in
``tests/unit/functions/test_window_aggregate_frames.py`` run against both
engines so the table cannot drift from it silently.
"""

from __future__ import annotations

import math
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# Window.unboundedPreceding / unboundedFollowing / currentRow sentinels.
UNBOUNDED_PRECEDING = -sys.maxsize - 1
UNBOUNDED_FOLLOWING = sys.maxsize
CURRENT_ROW = 0


# --------------------------------------------------------------------------- #
# Frame resolution
# --------------------------------------------------------------------------- #


def resolve_frame(
    position: int,
    peer_keys: Sequence[Any],
    *,
    has_order_by: bool,
    rows_between: Optional[Tuple[int, int]],
    range_between: Optional[Tuple[int, int]],
    range_keys: Optional[Sequence[Any]] = None,
    descending: bool = False,
) -> Tuple[int, int]:
    """Return the inclusive ``(lo, hi)`` frame bounds for one row.

    Bounds index into the *ordered* partition. ``peer_keys`` holds the full
    ORDER BY key tuple of each row after sorting -- rows with equal tuples are
    peers. ``range_keys`` holds the single scalar key that a numeric RANGE
    offset is measured against (Spark permits only one ORDER BY column with
    such a frame). ``position`` is the offset of the current row.

    Spark's defaults, which this reproduces:

    ==============================  ==========================================
    Window spec                     Frame
    ==============================  ==========================================
    no ORDER BY                     ROWS UNBOUNDED PRECEDING .. UNBOUNDED
                                    FOLLOWING (the whole partition)
    ORDER BY, no explicit frame     RANGE UNBOUNDED PRECEDING .. CURRENT ROW
    explicit ``rowsBetween``        ROWS, physical offsets
    explicit ``rangeBetween``       RANGE, value offsets on the order key
    ==============================  ==========================================

    The RANGE/ROWS distinction is not cosmetic: under RANGE, rows sharing an
    ORDER BY value are *peers* and see an identical frame, which is why
    ``partitionBy(t).orderBy(t)`` yields the partition total rather than a
    running one (issue #392).
    """
    n = len(peer_keys)
    last = n - 1

    if rows_between is not None:
        start, end = rows_between
        lo = 0 if start == UNBOUNDED_PRECEDING else position + start
        hi = last if end == UNBOUNDED_FOLLOWING else position + end
        return max(lo, 0), min(hi, last)

    if not has_order_by:
        return 0, last

    start, end = (
        range_between
        if range_between is not None
        else (UNBOUNDED_PRECEDING, CURRENT_ROW)
    )

    # Peer group of the current row: contiguous run of equal order keys.
    current_peer = peer_keys[position]
    peer_lo = position
    while peer_lo > 0 and _keys_equal(peer_keys[peer_lo - 1], current_peer):
        peer_lo -= 1
    peer_hi = position
    while peer_hi < last and _keys_equal(peer_keys[peer_hi + 1], current_peer):
        peer_hi += 1

    keys = range_keys if range_keys is not None else peer_keys
    lo = _range_bound(start, keys[position], keys, peer_lo, descending, is_start=True)
    hi = _range_bound(end, keys[position], keys, peer_hi, descending, is_start=False)
    return max(lo, 0), min(hi, last)


def _range_bound(
    offset: int,
    current_key: Any,
    ordered_keys: Sequence[Any],
    peer_edge: int,
    descending: bool,
    *,
    is_start: bool,
) -> int:
    """Resolve one side of a RANGE frame to an index into ``ordered_keys``."""
    if offset == UNBOUNDED_PRECEDING:
        return 0
    if offset == UNBOUNDED_FOLLOWING:
        return len(ordered_keys) - 1
    if offset == CURRENT_ROW:
        return peer_edge

    # Numeric offset: a value-based bound on the single order key. Spark
    # measures the offset along the ORDER BY direction, so a DESC ordering
    # negates it (verified on PySpark 4.0.0: under DESC, rangeBetween(0, 2)
    # frames keys in [k - 2, k], not [k, k + 2]).
    if current_key is None:
        # A NULL key compares to nothing: the frame degenerates to its peers.
        return peer_edge
    try:
        bound = current_key - offset if descending else current_key + offset
    except TypeError:
        return peer_edge

    matching = [
        i
        for i, key in enumerate(ordered_keys)
        if key is not None and _within(key, current_key, bound, descending, is_start)
    ]
    if not matching:
        return peer_edge
    return matching[0] if is_start else matching[-1]


def _within(
    key: Any, current: Any, bound: Any, descending: bool, is_start: bool
) -> bool:
    """Whether ``key`` lies on the framed side of ``bound``."""
    try:
        if is_start:
            return bool(key >= bound) if not descending else bool(key <= bound)
        return bool(key <= bound) if not descending else bool(key >= bound)
    except TypeError:
        return False


def _keys_equal(a: Any, b: Any) -> bool:
    """Equality that treats NULL as its own peer group (Spark semantics)."""
    if a is None or b is None:
        return a is None and b is None
    try:
        return bool(a == b)
    except TypeError:  # pragma: no cover - defensive
        return False


# --------------------------------------------------------------------------- #
# Reducers
# --------------------------------------------------------------------------- #


def _numeric(values: List[Any]) -> List[float]:
    """Coerce to float, dropping anything that will not convert."""
    out = []
    for v in values:
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            continue
    return out


def _moment(values: List[float], order: int) -> Optional[float]:
    if not values:
        return None
    mean = sum(values) / len(values)
    return sum((v - mean) ** order for v in values) / len(values)


def _sum(values: List[Any]) -> Any:
    nums = _numeric(values)
    # Spark's SUM over a frame with no non-NULL value is NULL, not 0.
    return sum(nums) if nums else None


def _avg(values: List[Any]) -> Any:
    nums = _numeric(values)
    return sum(nums) / len(nums) if nums else None


def _max(values: List[Any]) -> Any:
    try:
        return max(values) if values else None
    except TypeError:
        return None


def _min(values: List[Any]) -> Any:
    try:
        return min(values) if values else None
    except TypeError:
        return None


def _var(values: List[Any], *, population: bool) -> Any:
    nums = _numeric(values)
    n = len(nums)
    if n == 0 or (not population and n < 2):
        # Sample variance/stddev is undefined for a single observation; Spark
        # returns NULL rather than 0.
        return None
    mean = sum(nums) / n
    ss = sum((v - mean) ** 2 for v in nums)
    return ss / n if population else ss / (n - 1)


def _stddev(values: List[Any], *, population: bool) -> Any:
    var = _var(values, population=population)
    return None if var is None else math.sqrt(var)


def _skewness(values: List[Any]) -> Any:
    nums = _numeric(values)
    if len(nums) < 1:
        return None
    m2 = _moment(nums, 2)
    m3 = _moment(nums, 3)
    if not m2 or m2 <= 0 or m3 is None:
        return None
    return m3 / (m2**1.5)


def _kurtosis(values: List[Any]) -> Any:
    nums = _numeric(values)
    if len(nums) < 1:
        return None
    m2 = _moment(nums, 2)
    m4 = _moment(nums, 4)
    if not m2 or m2 <= 0 or m4 is None:
        return None
    return m4 / (m2**2) - 3.0


def _product(values: List[Any]) -> Any:
    nums = _numeric(values)
    if not nums:
        return None
    result = 1.0
    for v in nums:
        result *= v
    return result


def _median(values: List[Any]) -> Any:
    nums = sorted(_numeric(values))
    if not nums:
        return None
    mid = len(nums) // 2
    if len(nums) % 2:
        return nums[mid]
    return (nums[mid - 1] + nums[mid]) / 2


def _mode(values: List[Any]) -> Any:
    if not values:
        return None
    counts: Dict[Any, int] = {}
    for v in values:
        try:
            counts[v] = counts.get(v, 0) + 1
        except TypeError:  # unhashable
            return None
    best = max(counts.values())
    for v in values:  # first value attaining the max, for determinism
        if counts[v] == best:
            return v
    return None


def _bitwise(values: List[Any], op: str) -> Any:
    ints = []
    for v in values:
        try:
            ints.append(int(v))
        except (TypeError, ValueError):
            continue
    if not ints:
        return None
    acc = ints[0]
    for v in ints[1:]:
        if op == "and":
            acc &= v
        elif op == "or":
            acc |= v
        else:
            acc ^= v
    return acc


def _collect_set(values: List[Any]) -> List[Any]:
    seen: List[Any] = []
    for v in values:
        try:
            if v not in seen:
                seen.append(v)
        except TypeError:  # pragma: no cover - unhashable/uncomparable
            seen.append(v)
    return seen


#: Reducers over the *non-NULL* values in a frame. ``collect_list`` returns an
#: empty array rather than NULL for an empty frame, matching Spark.
REDUCERS: Dict[str, Callable[[List[Any]], Any]] = {
    "sum": _sum,
    "avg": _avg,
    "mean": _avg,
    "max": _max,
    "min": _min,
    "collect_list": lambda vs: list(vs),
    "collect_set": _collect_set,
    "stddev": lambda vs: _stddev(vs, population=False),
    "stddev_samp": lambda vs: _stddev(vs, population=False),
    "stddev_pop": lambda vs: _stddev(vs, population=True),
    "variance": lambda vs: _var(vs, population=False),
    "var_samp": lambda vs: _var(vs, population=False),
    "var_pop": lambda vs: _var(vs, population=True),
    "skewness": _skewness,
    "kurtosis": _kurtosis,
    "product": _product,
    "median": _median,
    "mode": _mode,
    "bit_and": lambda vs: _bitwise(vs, "and"),
    "bit_or": lambda vs: _bitwise(vs, "or"),
    "bit_xor": lambda vs: _bitwise(vs, "xor"),
    "any_value": lambda vs: vs[0] if vs else None,
}

#: Reducers that must see every row in the frame, NULLs included.
NULL_AWARE_REDUCERS: Dict[str, Callable[[List[Any]], Any]] = {
    "count": lambda vs: sum(1 for v in vs if v is not None),
}


def reduce_frame(function_name: str, values: List[Any]) -> Any:
    """Apply the reducer registered for ``function_name``.

    ``values`` are the frame's values with NULLs already removed, except for
    the null-aware reducers which receive them intact.
    """
    reducer = REDUCERS.get(function_name)
    if reducer is None:  # pragma: no cover - guarded by has_reducer()
        raise KeyError(function_name)
    return reducer(values)


def has_reducer(function_name: str) -> bool:
    """Whether a window aggregate reducer is registered for this name."""
    return function_name in REDUCERS or function_name in NULL_AWARE_REDUCERS
