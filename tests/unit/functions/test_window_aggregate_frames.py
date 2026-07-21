"""Regression tests for aggregate functions over a window (BUG-035, BUG-036).

Two defects, one cause. ``WindowFunction.evaluate()`` hand-dispatched each
aggregate through an ``elif`` chain:

* Anything the chain did not name -- ``max``, ``min``, ``collect_list``,
  ``collect_set``, ``stddev``, ``variance``, ``product``, ``median``, ... --
  fell into ``return [None] * len(data)``. A window aggregate that was simply
  not implemented was indistinguishable from a genuine SQL NULL (BUG-035).
* The three branches that *were* implemented each approximated the window frame
  differently, and none correctly (BUG-036). ``F.sum(x).over(partitionBy(g)
  .orderBy(k))`` returned the whole-partition total where Spark returns a
  running total; ``avg`` ignored partitioning entirely on the ordered path,
  producing a running average over the whole DataFrame in physical row order;
  ``count`` ignored ordering.

BUG-036 is the more dangerous of the two, because a plausible wrong *number*
survives any assertion that only checks shape or non-nullness. Running totals
over an ordered window are exactly how stock ledgers and trend tables are
computed downstream.

Every expectation below was captured from real PySpark 4.0.0 on OpenJDK 21 (the
DBR 17.3 pairing). These tests use the backend-agnostic ``spark`` fixture, so
the same file runs against real PySpark with
``MOCK_SPARK_TEST_BACKEND=pyspark``.
"""

from typing import Any, Dict, List, Optional

import pytest

from tests.fixtures.spark_imports import get_spark_imports

# grp "a" has a tie at k=1 (peers) and a NULL value at k=3;
# grp "b" has a NULL value first, so its running aggregates start from NULL.
ROWS = [
    ("a", 1, 10.0),
    ("a", 1, 20.0),
    ("a", 2, 30.0),
    ("a", 3, None),
    ("b", 1, None),
    ("b", 2, 7.0),
]


def _frame(spark: Any) -> Any:
    """Standard fixture frame: partition key, order key with ties, nullable value."""
    imports = get_spark_imports()
    schema = imports.StructType(
        [
            imports.StructField("grp", imports.StringType()),
            imports.StructField("k", imports.LongType()),
            imports.StructField("x", imports.DoubleType()),
        ]
    )
    return spark.createDataFrame(ROWS, schema)


def _num(value: Any) -> Any:
    """Normalise numeric types across backends (int/long/float/Decimal)."""
    if value is None or isinstance(value, (str, list)):
        return value
    return round(float(value), 6)


def _collect(df: Any, col: str = "r") -> Dict[Any, Any]:
    """Map each row's ``(grp, k, x)`` identity to its window result."""
    out = {}
    for row in df.collect():
        out[(row["grp"], int(row["k"]), _num(row["x"]))] = row[col]
    return out


def _values(df: Any, col: str = "r") -> Dict[Any, Any]:
    return {k: _num(v) for k, v in _collect(df, col).items()}


A1_LO = ("a", 1, 10.0)
A1_HI = ("a", 1, 20.0)
A2 = ("a", 2, 30.0)
A3 = ("a", 3, None)
B1 = ("b", 1, None)
B2 = ("b", 2, 7.0)


# --------------------------------------------------------------------------
# BUG-036: the default ORDER BY frame is RANGE UNBOUNDED PRECEDING .. CURRENT
# ROW, so an ordered window is a *running* aggregate, and tied rows are peers
# that share one frame.
# --------------------------------------------------------------------------


def test_ordered_sum_is_a_running_total_not_the_partition_total(spark: Any) -> None:
    """The original BUG-036 defect: every row got the partition total, 60.0.

    PySpark 4.0.0: the k=1 peers share a frame (10+20=30), k=2 extends it to 60.
    """
    F, Window = get_spark_imports().F, get_spark_imports().Window
    df = _frame(spark).withColumn(
        "r", F.sum("x").over(Window.partitionBy("grp").orderBy("k"))
    )
    assert _values(df) == {
        A1_LO: 30.0,
        A1_HI: 30.0,
        A2: 60.0,
        A3: 60.0,
        B1: None,
        B2: 7.0,
    }


def test_ordered_avg_respects_the_partition(spark: Any) -> None:
    """``avg`` over an ordered window used to ignore partitionBy entirely.

    It accumulated across the whole DataFrame in physical row order, so group
    "b" received averages computed from group "a" rows (16.0 and 16.25 against
    an input whose group-b values are just NULL and 7.0).
    """
    F, Window = get_spark_imports().F, get_spark_imports().Window
    df = _frame(spark).withColumn(
        "r", F.avg("x").over(Window.partitionBy("grp").orderBy("k"))
    )
    assert _values(df) == {
        A1_LO: 15.0,
        A1_HI: 15.0,
        A2: 20.0,
        A3: 20.0,
        B1: None,
        B2: 7.0,
    }


def test_ordered_count_is_cumulative(spark: Any) -> None:
    """``count`` over an ordered window ignored the ordering and returned the
    partition count for every row."""
    F, Window = get_spark_imports().F, get_spark_imports().Window
    df = _frame(spark).withColumn(
        "r", F.count("x").over(Window.partitionBy("grp").orderBy("k"))
    )
    assert _values(df) == {A1_LO: 2, A1_HI: 2, A2: 3, A3: 3, B1: 0, B2: 1}


def test_order_key_equal_to_partition_key_yields_the_partition_total(
    spark: Any,
) -> None:
    """All rows are peers, so RANGE gives the partition total (issue #392).

    This is the case the old whole-partition ``sum`` got right by accident, and
    it must keep working now that the frame is computed properly.
    """
    F, Window = get_spark_imports().F, get_spark_imports().Window
    df = _frame(spark).withColumn(
        "r", F.sum("x").over(Window.partitionBy("grp").orderBy("grp"))
    )
    assert _values(df) == {
        A1_LO: 60.0,
        A1_HI: 60.0,
        A2: 60.0,
        A3: 60.0,
        B1: 7.0,
        B2: 7.0,
    }


def test_unordered_window_aggregates_the_whole_partition(spark: Any) -> None:
    """Without ORDER BY the frame is the entire partition."""
    F, Window = get_spark_imports().F, get_spark_imports().Window
    df = _frame(spark).withColumn("r", F.sum("x").over(Window.partitionBy("grp")))
    assert _values(df) == {
        A1_LO: 60.0,
        A1_HI: 60.0,
        A2: 60.0,
        A3: 60.0,
        B1: 7.0,
        B2: 7.0,
    }


def test_descending_order_reverses_the_running_total(spark: Any) -> None:
    """Under DESC the frame accumulates from the highest key downward."""
    F, Window = get_spark_imports().F, get_spark_imports().Window
    df = _frame(spark).withColumn(
        "r", F.sum("x").over(Window.partitionBy("grp").orderBy(F.col("k").desc()))
    )
    assert _values(df) == {
        A1_LO: 60.0,
        A1_HI: 60.0,
        A2: 30.0,
        A3: None,
        B1: 7.0,
        B2: 7.0,
    }


# --------------------------------------------------------------------------
# BUG-035: aggregates that had no branch at all returned NULL everywhere.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(  # type: ignore[misc,untyped-decorator]
    "func_name,expected",
    [
        ("max", {"a": 30.0, "b": 7.0}),
        ("min", {"a": 10.0, "b": 7.0}),
        ("sum", {"a": 60.0, "b": 7.0}),
        ("avg", {"a": 20.0, "b": 7.0}),
        ("mean", {"a": 20.0, "b": 7.0}),
        ("stddev", {"a": 10.0, "b": None}),
        ("stddev_samp", {"a": 10.0, "b": None}),
        ("stddev_pop", {"a": 8.164966, "b": 0.0}),
        ("variance", {"a": 100.0, "b": None}),
        ("var_samp", {"a": 100.0, "b": None}),
        ("var_pop", {"a": 66.666667, "b": 0.0}),
        ("skewness", {"a": 0.0, "b": None}),
        ("product", {"a": 6000.0, "b": 7.0}),
        ("median", {"a": 20.0, "b": 7.0}),
    ],
)
def test_unordered_window_aggregate_returns_a_value_not_null(
    spark: Any, func_name: str, expected: Dict[str, Optional[float]]
) -> None:
    """Each of these returned NULL for every row before the fix.

    ``max`` and ``min`` silently returning NULL is the most dangerous of the
    set: they read as "no data" rather than "unimplemented".
    """
    F, Window = get_spark_imports().F, get_spark_imports().Window
    func = getattr(F, func_name)
    df = _frame(spark).withColumn("r", func("x").over(Window.partitionBy("grp")))
    results = _values(df)
    assert results[A1_LO] == expected["a"]
    assert results[A2] == expected["a"]
    assert results[B2] == expected["b"]


def test_kurtosis_over_window(spark: Any) -> None:
    """Kurtosis is excess kurtosis (m4 / m2^2 - 3), matching Spark."""
    F, Window = get_spark_imports().F, get_spark_imports().Window
    df = _frame(spark).withColumn("r", F.kurtosis("x").over(Window.partitionBy("grp")))
    assert _values(df)[A1_LO] == -1.5


def test_collect_list_over_window_returns_the_frame_values(spark: Any) -> None:
    """``collect_list`` returned NULL; Spark returns the frame's non-NULL values."""
    F, Window = get_spark_imports().F, get_spark_imports().Window
    df = _frame(spark).withColumn(
        "r", F.collect_list("x").over(Window.partitionBy("grp"))
    )
    results = _collect(df)
    assert sorted(results[A1_LO]) == [10.0, 20.0, 30.0]
    assert sorted(results[A3]) == [10.0, 20.0, 30.0]
    assert sorted(results[B2]) == [7.0]


def test_collect_list_over_empty_frame_is_an_empty_array_not_null(
    spark: Any,
) -> None:
    """Spark distinguishes "no values" (``[]``) from NULL for collect_list."""
    F, Window = get_spark_imports().F, get_spark_imports().Window
    df = _frame(spark).withColumn(
        "r", F.collect_list("x").over(Window.partitionBy("grp").orderBy("k"))
    )
    # Group b's first row has a NULL value, so its running frame holds nothing.
    assert _collect(df)[B1] == []


def test_collect_set_over_window_deduplicates(spark: Any) -> None:
    """``collect_set`` returned NULL; ordering within the set is unspecified."""
    F, Window = get_spark_imports().F, get_spark_imports().Window
    imports = get_spark_imports()
    schema = imports.StructType(
        [
            imports.StructField("grp", imports.StringType()),
            imports.StructField("k", imports.LongType()),
            imports.StructField("x", imports.DoubleType()),
        ]
    )
    df = spark.createDataFrame(
        [("a", 1, 5.0), ("a", 2, 5.0), ("a", 3, 9.0)], schema
    ).withColumn("r", F.collect_set("x").over(Window.partitionBy("grp")))
    for row in df.collect():
        assert sorted(row["r"]) == [5.0, 9.0]


def test_bitwise_aggregates_over_window(spark: Any) -> None:
    """bit_and / bit_or / bit_xor all returned NULL over a window."""
    F, Window = get_spark_imports().F, get_spark_imports().Window
    imports = get_spark_imports()
    schema = imports.StructType(
        [
            imports.StructField("grp", imports.StringType()),
            imports.StructField("y", imports.LongType()),
        ]
    )
    df = spark.createDataFrame([("a", 2), ("a", 3), ("a", 4)], schema)
    w = Window.partitionBy("grp")
    out = df.withColumn("band", F.bit_and("y").over(w))
    out = out.withColumn("bor", F.bit_or("y").over(w))
    out = out.withColumn("bxor", F.bit_xor("y").over(w))
    first = out.collect()[0]
    assert (int(first["band"]), int(first["bor"]), int(first["bxor"])) == (0, 7, 5)


def test_max_over_a_string_column(spark: Any) -> None:
    """``max`` must work on non-numeric columns too."""
    F, Window = get_spark_imports().F, get_spark_imports().Window
    df = _frame(spark).withColumn("r", F.max("grp").over(Window.partitionBy("grp")))
    assert {row["grp"]: row["r"] for row in df.collect()} == {"a": "a", "b": "b"}


# --------------------------------------------------------------------------
# Operand shape: an aggregate's argument may be an expression, not just a
# bare column. The two axes (which aggregate, what shape of operand) are
# varied independently because a fix to one does not imply the other.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(  # type: ignore[misc,untyped-decorator]
    "func_name,expected_a",
    [("max", 60.0), ("min", 20.0), ("sum", 120.0), ("avg", 40.0)],
)
def test_window_aggregate_over_an_arithmetic_expression(
    spark: Any, func_name: str, expected_a: float
) -> None:
    """``F.max(F.col("x") * 2).over(w)`` -- operand is an expression."""
    F, Window = get_spark_imports().F, get_spark_imports().Window
    df = _frame(spark).withColumn(
        "r", getattr(F, func_name)(F.col("x") * 2).over(Window.partitionBy("grp"))
    )
    assert _values(df)[A1_LO] == expected_a


@pytest.mark.parametrize(  # type: ignore[misc,untyped-decorator]
    "func_name,expected_a",
    [("max", 30.0), ("min", 0.0), ("sum", 50.0)],
)
def test_window_aggregate_over_a_case_when(
    spark: Any, func_name: str, expected_a: float
) -> None:
    """``F.max(F.when(...)).over(w)`` -- operand is a CASE WHEN."""
    F, Window = get_spark_imports().F, get_spark_imports().Window
    operand = F.when(F.col("x") > 10, F.col("x")).otherwise(F.lit(0.0))
    df = _frame(spark).withColumn(
        "r", getattr(F, func_name)(operand).over(Window.partitionBy("grp"))
    )
    assert _values(df)[A1_LO] == expected_a


def test_scalar_function_wrapping_a_window_aggregate(spark: Any) -> None:
    """``F.sqrt(F.sum(x).over(w))`` -- the aggregate is wrapped, not bare."""
    F, Window = get_spark_imports().F, get_spark_imports().Window
    df = _frame(spark).withColumn(
        "r", F.sqrt(F.sum("x").over(Window.partitionBy("grp")))
    )
    assert _values(df)[A1_LO] == 7.745967


# --------------------------------------------------------------------------
# Explicit frames. ROWS counts physical rows and ignores peers; RANGE measures
# along the ORDER BY key and includes them.
# --------------------------------------------------------------------------


def test_rows_between_unbounded_preceding_and_current_row(spark: Any) -> None:
    """A ROWS frame is physical: the k=1 peers get 10.0 and 30.0, not both 30.0."""
    F, Window = get_spark_imports().F, get_spark_imports().Window
    w = (
        Window.partitionBy("grp")
        .orderBy("k")
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    )
    df = _frame(spark).withColumn("r", F.sum("x").over(w))
    results = _values(df)
    assert sorted([results[A1_LO], results[A1_HI]]) == [10.0, 30.0]
    assert results[A2] == 60.0


def test_rows_between_offsets_around_the_current_row(spark: Any) -> None:
    """``rowsBetween(-1, 1)`` is a sliding three-row window."""
    F, Window = get_spark_imports().F, get_spark_imports().Window
    w = Window.partitionBy("grp").orderBy("k").rowsBetween(-1, 1)
    df = _frame(spark).withColumn("r", F.max("x").over(w))
    results = _values(df)
    assert results[A2] == 30.0
    assert results[A3] == 30.0
    assert results[B1] == 7.0


def test_rows_between_entirely_preceding_the_current_row(spark: Any) -> None:
    """``rowsBetween(-2, -1)`` excludes the current row; the first row's frame
    is empty, so ``sum`` is NULL and ``collect_list`` is ``[]``."""
    F, Window = get_spark_imports().F, get_spark_imports().Window
    w = Window.partitionBy("grp").orderBy("k").rowsBetween(-2, -1)
    df = _frame(spark).withColumn("r", F.sum("x").over(w))
    results = _values(df)
    assert results[A2] == 30.0
    assert results[A3] == 50.0
    assert results[B1] is None


def test_rows_between_current_row_and_unbounded_following(spark: Any) -> None:
    """A suffix frame: a reverse running total."""
    F, Window = get_spark_imports().F, get_spark_imports().Window
    w = (
        Window.partitionBy("grp")
        .orderBy("k")
        .rowsBetween(Window.currentRow, Window.unboundedFollowing)
    )
    df = _frame(spark).withColumn("r", F.sum("x").over(w))
    results = _values(df)
    assert results[A2] == 30.0
    assert results[A3] is None
    assert results[B1] == 7.0


def test_range_between_current_row_frames_the_peer_group(spark: Any) -> None:
    """``rangeBetween(0, 0)`` is the peer group, not the single current row."""
    F, Window = get_spark_imports().F, get_spark_imports().Window
    w = Window.partitionBy("grp").orderBy("k").rangeBetween(0, 0)
    df = _frame(spark).withColumn("r", F.sum("x").over(w))
    results = _values(df)
    assert results[A1_LO] == 30.0
    assert results[A1_HI] == 30.0
    assert results[A2] == 30.0


def test_range_between_numeric_offsets_measure_the_order_key(spark: Any) -> None:
    """``rangeBetween(-1, 1)`` frames rows whose key is within 1 of the current."""
    F, Window = get_spark_imports().F, get_spark_imports().Window
    w = Window.partitionBy("grp").orderBy("k").rangeBetween(-1, 1)
    df = _frame(spark).withColumn("r", F.sum("x").over(w))
    results = _values(df)
    # k=1 frames keys 1..2 -> 10+20+30; k=3 frames keys 2..3 -> 30 only.
    assert results[A1_LO] == 60.0
    assert results[A2] == 60.0
    assert results[A3] == 30.0


def test_range_between_offsets_follow_the_sort_direction(spark: Any) -> None:
    """Under DESC, Spark measures the offset the other way: ``rangeBetween(0, 2)``
    frames keys in ``[k - 2, k]``."""
    F, Window = get_spark_imports().F, get_spark_imports().Window
    imports = get_spark_imports()
    schema = imports.StructType(
        [
            imports.StructField("grp", imports.StringType()),
            imports.StructField("k", imports.LongType()),
            imports.StructField("x", imports.DoubleType()),
        ]
    )
    rows = [("a", 1, 1.0), ("a", 2, 2.0), ("a", 4, 4.0)]
    w = Window.partitionBy("grp").orderBy(F.col("k").desc()).rangeBetween(0, 2)
    df = spark.createDataFrame(rows, schema).withColumn("r", F.sum("x").over(w))
    got = {int(row["k"]): _num(row["r"]) for row in df.collect()}
    assert got == {1: 1.0, 2: 3.0, 4: 6.0}


def test_range_between_ascending_offsets(spark: Any) -> None:
    """The ASC counterpart of the test above: ``[k, k + 2]``."""
    F, Window = get_spark_imports().F, get_spark_imports().Window
    imports = get_spark_imports()
    schema = imports.StructType(
        [
            imports.StructField("grp", imports.StringType()),
            imports.StructField("k", imports.LongType()),
            imports.StructField("x", imports.DoubleType()),
        ]
    )
    rows = [("a", 1, 1.0), ("a", 2, 2.0), ("a", 4, 4.0)]
    w = Window.partitionBy("grp").orderBy("k").rangeBetween(0, 2)
    df = spark.createDataFrame(rows, schema).withColumn("r", F.sum("x").over(w))
    got = {int(row["k"]): _num(row["r"]) for row in df.collect()}
    assert got == {1: 3.0, 2: 6.0, 4: 4.0}


# --------------------------------------------------------------------------
# NULL handling, which differs per reducer.
# --------------------------------------------------------------------------


def test_sum_of_an_all_null_frame_is_null_not_zero(spark: Any) -> None:
    """Spark's SUM over no non-NULL value is NULL; returning 0.0 would read as
    a real total."""
    F, Window = get_spark_imports().F, get_spark_imports().Window
    df = _frame(spark).withColumn(
        "r", F.sum("x").over(Window.partitionBy("grp").orderBy("k"))
    )
    assert _values(df)[B1] is None


def test_count_of_an_all_null_frame_is_zero_not_null(spark: Any) -> None:
    """COUNT is the exception: it returns 0, never NULL."""
    F, Window = get_spark_imports().F, get_spark_imports().Window
    df = _frame(spark).withColumn(
        "r", F.count("x").over(Window.partitionBy("grp").orderBy("k"))
    )
    assert _values(df)[B1] == 0


def test_count_star_over_window_counts_null_rows(spark: Any) -> None:
    """``count("*")`` counts rows; ``count(col)`` counts non-NULL values."""
    F, Window = get_spark_imports().F, get_spark_imports().Window
    df = _frame(spark).withColumn("r", F.count("*").over(Window.partitionBy("grp")))
    results = _values(df)
    assert results[A1_LO] == 4
    assert results[B1] == 2


def test_sample_stddev_of_a_single_row_is_null(spark: Any) -> None:
    """Sample stddev is undefined for n=1; Spark returns NULL, not 0.

    Both halves matter. Asserting only the NULL would have passed against the
    original defect, where ``stddev`` over a window returned NULL for every row
    -- so the non-NULL expectations below are what give this test the power to
    fail.
    """
    F, Window = get_spark_imports().F, get_spark_imports().Window
    df = _frame(spark).withColumn(
        "r", F.stddev("x").over(Window.partitionBy("grp").orderBy("k"))
    )
    results = _values(df)
    assert results[B2] is None  # frame holds one value
    assert results[A1_LO] == 7.071068  # frame holds 10.0 and 20.0
    assert results[A2] == 10.0  # frame holds 10.0, 20.0, 30.0


# --------------------------------------------------------------------------
# A dispatch miss must be audible.
# --------------------------------------------------------------------------


def test_unimplemented_window_function_warns_instead_of_silently_nulling() -> None:
    """An unsupported window function must announce itself.

    The whole family above was undetectable precisely because an unhandled
    function and a genuine NULL were the same observation. Sparkless-only: this
    asserts on sparkless internals, so it does not run against real PySpark.
    """
    from sparkless.functions.window_execution import WindowFunction
    from sparkless.window import Window as SparklessWindow

    class _Unknown:
        function_name = "definitely_not_a_window_function"
        column = None

    data: List[Dict[str, Any]] = [{"grp": "a", "x": 1.0}]
    wf = WindowFunction(_Unknown(), SparklessWindow.partitionBy("grp"))

    with pytest.warns(UserWarning, match="not implemented"):
        assert wf.evaluate(data) == [None]
