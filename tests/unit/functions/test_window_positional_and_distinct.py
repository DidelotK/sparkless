"""Regression tests for ``first``/``last`` over a window and for DISTINCT
aggregates over a window (BUG-040, BUG-042).

BUG-035/036 fixed every *aggregate* over a window by giving them one shared
notion of "the frame" (``window_frames.resolve_frame``). ``first``, ``last``,
``first_value`` and ``last_value`` were left on their bespoke branches, and
each re-derived the frame by hand from ``partitionBy``/``orderBy``:

* ``last`` over an ordered window returned *the current row's* value, which is
  right only when the ORDER BY key is unique. With ties Spark returns the last
  value of the **peer group**.
* All four ignored an explicit ``rowsBetween``/``rangeBetween`` outright, so
  ``last(x)`` over ``rowsBetween(unboundedPreceding, unboundedFollowing)``
  still returned the current row rather than the partition's last.
* ``ignoreNulls`` was parsed onto the ``AggregateFunction`` and then never read
  on the window path, so ``F.first(x, True)`` behaved like ``F.first(x)`` --
  silently, returning NULL where Spark returns the first non-NULL.

BUG-042 is a different kind of wrong. Spark *rejects*
``F.count_distinct(x).over(w)`` with ``[DISTINCT_WINDOW_FUNCTION_UNSUPPORTED]``;
sparkless computed a number for it and NULL for ``sum_distinct``. Being more
permissive than the thing you are mocking means a query that cannot run in
production passes its unit tests. ``approx_count_distinct`` is *not* a DISTINCT
aggregate and Spark does allow it, so the guard must not over-reject.

Every expectation below was captured from real PySpark 4.0.0 on OpenJDK 21 (the
DBR 17.3 pairing). These tests use the backend-agnostic ``spark`` fixture, so
the same file runs against real PySpark with
``MOCK_SPARK_TEST_BACKEND=pyspark``.
"""

from typing import Any, List, Optional

import pytest

from tests.fixtures.spark_imports import get_spark_imports

#: k=1 is a tie: rows (1, 10) and (1, 20) are peers under ``orderBy("k")``.
#: Without peer handling, `last` returns 10 for the first of them.
TIED_ROWS = [(1, 10), (1, 20), (2, 30)]

#: NULL at both ends, so `ignoreNulls` changes the answer at each edge.
NULL_EDGED_ROWS = [(1, None), (2, 20), (3, None)]


def _frame(spark: Any, rows: List[Any]) -> Any:
    imports = get_spark_imports()
    schema = imports.StructType(
        [
            imports.StructField("k", imports.IntegerType()),
            imports.StructField("x", imports.IntegerType()),
        ]
    )
    return spark.createDataFrame(rows, schema)


def _values(rows: List[Any], key: str) -> List[Optional[int]]:
    """Column ``key``, normalised to int/None across backends."""
    return [None if row[key] is None else int(row[key]) for row in rows]


def _ordered(df: Any, column: Any) -> List[Optional[int]]:
    """Apply ``column``, then read it back in ORDER BY key order.

    Row order out of a window is not guaranteed to match input order on either
    backend, so the rows are sorted by ``k`` before comparison. ``k`` is unique
    per expected value in every fixture except the deliberate tie, whose two
    rows are asserted as a pair.
    """
    rows = df.withColumn("r", column).collect()
    rows = sorted(rows, key=lambda row: (row["k"], row["x"] if row["x"] else 0))
    return _values(rows, "r")


class TestFirstLastPeerGroups:
    """With tied ORDER BY keys, peers share one frame."""

    def test_last_returns_the_peer_group_end(self, spark: Any) -> None:
        """Both tied rows see 20, not 10 and 20."""
        imports = get_spark_imports()
        F, Window = imports.F, imports.Window
        df = _frame(spark, TIED_ROWS)
        assert _ordered(df, F.last("x").over(Window.orderBy("k"))) == [20, 20, 30]

    def test_first_over_a_running_frame(self, spark: Any) -> None:
        """The default frame starts unbounded, so `first` is the partition's first."""
        imports = get_spark_imports()
        F, Window = imports.F, imports.Window
        df = _frame(spark, TIED_ROWS)
        assert _ordered(df, F.first("x").over(Window.orderBy("k"))) == [10, 10, 10]

    def test_last_without_order_by_spans_the_partition(self, spark: Any) -> None:
        """No ORDER BY means the whole partition is the frame."""
        imports = get_spark_imports()
        F, Window = imports.F, imports.Window
        df = _frame(spark, TIED_ROWS)
        assert _ordered(df, F.last("x").over(Window.partitionBy("k"))) == [20, 20, 30]


class TestFirstLastExplicitFrames:
    """An explicit rowsBetween must be honoured, not ignored."""

    def test_last_over_the_whole_partition(self, spark: Any) -> None:
        """``rowsBetween(unbounded, unbounded)`` -- every row sees the last value."""
        imports = get_spark_imports()
        F, Window = imports.F, imports.Window
        df = _frame(spark, TIED_ROWS)
        window = Window.orderBy("k").rowsBetween(
            Window.unboundedPreceding, Window.unboundedFollowing
        )
        assert _ordered(df, F.last("x").over(window)) == [30, 30, 30]

    def test_last_over_a_trailing_two_row_frame(self, spark: Any) -> None:
        """``rowsBetween(-1, 0)`` ends on the current row."""
        imports = get_spark_imports()
        F, Window = imports.F, imports.Window
        df = _frame(spark, TIED_ROWS)
        window = Window.orderBy("k").rowsBetween(-1, 0)
        assert _ordered(df, F.last("x").over(window)) == [10, 20, 30]

    def test_first_over_a_trailing_two_row_frame(self, spark: Any) -> None:
        """The same frame, read from the other end -- the discriminating case.

        ``last`` over ``rowsBetween(-1, 0)`` coincides with the old
        "current row" behaviour, so only ``first`` distinguishes a frame that
        is honoured from one that is ignored.
        """
        imports = get_spark_imports()
        F, Window = imports.F, imports.Window
        df = _frame(spark, TIED_ROWS)
        window = Window.orderBy("k").rowsBetween(-1, 0)
        assert _ordered(df, F.first("x").over(window)) == [10, 10, 20]

    def test_first_value_and_last_value_honour_frames_too(self, spark: Any) -> None:
        """``first_value``/``last_value`` share the implementation."""
        imports = get_spark_imports()
        F, Window = imports.F, imports.Window
        df = _frame(spark, TIED_ROWS)
        window = Window.orderBy("k").rowsBetween(-1, 0)
        assert _ordered(df, F.first_value("x").over(window)) == [10, 10, 20]
        assert _ordered(df, F.last_value("x").over(window)) == [10, 20, 30]


class TestFirstLastIgnoreNulls:
    """``ignoreNulls`` was accepted and then discarded."""

    def _full_frame_window(self) -> Any:
        imports = get_spark_imports()
        Window = imports.Window
        return Window.orderBy("k").rowsBetween(
            Window.unboundedPreceding, Window.unboundedFollowing
        )

    def test_last_without_ignore_nulls_keeps_a_null_edge(self, spark: Any) -> None:
        """The frame's last row is NULL, so ``last`` is NULL.

        Not "the last non-NULL seen" -- getting this backwards would make the
        ``ignoreNulls=True`` test below pass for the wrong reason.
        """
        imports = get_spark_imports()
        F = imports.F
        df = _frame(spark, NULL_EDGED_ROWS)
        assert _ordered(df, F.last("x").over(self._full_frame_window())) == [
            None,
            None,
            None,
        ]

    def test_last_with_ignore_nulls_skips_them(self, spark: Any) -> None:
        """``F.last(x, True)`` used to raise TypeError -- the arg did not exist."""
        imports = get_spark_imports()
        F = imports.F
        df = _frame(spark, NULL_EDGED_ROWS)
        assert _ordered(df, F.last("x", True).over(self._full_frame_window())) == [
            20,
            20,
            20,
        ]

    def test_first_with_ignore_nulls_skips_them(self, spark: Any) -> None:
        """``F.first(x, True)`` took the argument but silently ignored it."""
        imports = get_spark_imports()
        F = imports.F
        df = _frame(spark, NULL_EDGED_ROWS)
        assert _ordered(df, F.first("x", True).over(self._full_frame_window())) == [
            20,
            20,
            20,
        ]


class TestDistinctAggregatesOverWindowAreRejected:
    """Spark raises; being more permissive hides an unrunnable query."""

    @pytest.mark.parametrize(  # type: ignore[misc,untyped-decorator]
        "func", ["count_distinct", "sum_distinct"]
    )
    def test_distinct_aggregate_over_window_raises(self, spark: Any, func: str) -> None:
        """``[DISTINCT_WINDOW_FUNCTION_UNSUPPORTED]`` on both backends."""
        imports = get_spark_imports()
        F, Window = imports.F, imports.Window
        df = _frame(spark, TIED_ROWS)
        column = getattr(F, func)("x").over(Window.partitionBy("k"))
        with pytest.raises(Exception, match="DISTINCT_WINDOW_FUNCTION_UNSUPPORTED"):
            df.withColumn("r", column).collect()

    def test_building_the_column_alone_does_not_raise(self, spark: Any) -> None:
        """Spark rejects it at analysis, not at construction.

        ``F.count_distinct(x).over(w)`` on its own is a perfectly legal Column
        in PySpark; only *using* it fails. Raising in ``.over()`` would be
        stricter than Spark, which is the mirror image of the bug being fixed.
        """
        imports = get_spark_imports()
        F, Window = imports.F, imports.Window
        assert F.count_distinct("x").over(Window.partitionBy("k")) is not None

    def test_approx_count_distinct_over_window_is_allowed(self, spark: Any) -> None:
        """Not a DISTINCT aggregate -- the guard must not catch it."""
        imports = get_spark_imports()
        F, Window = imports.F, imports.Window
        df = _frame(spark, TIED_ROWS)
        column = F.approx_count_distinct("x").over(Window.partitionBy("k"))
        assert _ordered(df, column) == [2, 2, 1]
