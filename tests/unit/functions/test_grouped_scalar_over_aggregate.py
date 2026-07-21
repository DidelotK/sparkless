"""Regression tests for a scalar function wrapping an aggregate (BUG-037).

``GroupedData._evaluate_column_expression`` resolved the inner aggregate of an
expression like ``F.sqrt(F.sum("x"))`` but then gated every downstream branch on
an arithmetic-only operation set (``+ - * / %``). ``sqrt``, ``abs``, ``ceil``,
``coalesce``, ``upper`` and every other *named* function matched nothing, fell
past the ``elif`` chain and hit a literal ``return expr_name, None``. The NULL
was the default of an unmatched dispatch, not a computed value -- and
indistinguishable from a group that genuinely aggregated to NULL.

Arithmetic on an aggregate (``F.sum("x") / 3``) worked, and a plain scalar
function on a plain column (``F.sqrt(F.col("x"))``) worked; only the
combination failed, which is why it survived.

The fix resolves the aggregates to scalars and hands the outer expression to the
ordinary ``ExpressionEvaluator``, so scalar functions are supported by
construction rather than enumerated. Expectations captured from real PySpark
4.0.0 on OpenJDK 21; this file also runs against real PySpark via
``MOCK_SPARK_TEST_BACKEND=pyspark``.
"""

from typing import Any, Dict

import pytest

from tests.fixtures.spark_imports import get_spark_imports

ROWS = [
    ("a", 10.0, -1.0, "p"),
    ("a", 20.0, -2.0, "q"),
    ("a", 30.0, -3.0, "p"),
    ("b", 5.0, -4.0, "r"),
    ("b", 15.0, -6.0, "r"),
]


def _frame(spark: Any) -> Any:
    imports = get_spark_imports()
    schema = imports.StructType(
        [
            imports.StructField("grp", imports.StringType()),
            imports.StructField("x", imports.DoubleType()),
            imports.StructField("neg", imports.DoubleType()),
            imports.StructField("s", imports.StringType()),
        ]
    )
    return spark.createDataFrame(ROWS, schema)


def _num(value: Any) -> Any:
    if value is None or isinstance(value, str):
        return value
    return round(float(value), 6)


def _by_group(df: Any, col: str = "r") -> Dict[str, Any]:
    return {row["grp"]: _num(row[col]) for row in df.collect()}


# --------------------------------------------------------------------------
# The defect: a named scalar function wrapping an aggregate.
# --------------------------------------------------------------------------


def test_sqrt_of_sum(spark: Any) -> None:
    """The exact reported repro: returned NULL for every group."""
    F = get_spark_imports().F
    df = _frame(spark).groupBy("grp").agg(F.sqrt(F.sum("x")).alias("r"))
    assert _by_group(df) == {"a": 7.745967, "b": 4.472136}


def test_abs_of_sum(spark: Any) -> None:
    """``abs`` over a negative total -- a wrong sign would be caught, and so
    would the NULL the defect produced."""
    F = get_spark_imports().F
    df = _frame(spark).groupBy("grp").agg(F.abs(F.sum("neg")).alias("r"))
    assert _by_group(df) == {"a": 6.0, "b": 10.0}


@pytest.mark.parametrize(  # type: ignore[misc,untyped-decorator]
    "func_name,expected",
    [
        ("ceil", {"a": 20.0, "b": 10.0}),
        ("floor", {"a": 20.0, "b": 10.0}),
    ],
)
def test_rounding_functions_over_avg(
    spark: Any, func_name: str, expected: Dict[str, float]
) -> None:
    F = get_spark_imports().F
    df = _frame(spark).groupBy("grp").agg(getattr(F, func_name)(F.avg("x")).alias("r"))
    assert _by_group(df) == expected


def test_ceil_and_floor_differ_on_a_fractional_average(spark: Any) -> None:
    """Guards against a fix that merely returns the aggregate unchanged.

    Group b averages 10.0 exactly, so ceil and floor agree there; this frame is
    chosen so they must disagree.
    """
    F = get_spark_imports().F
    imports = get_spark_imports()
    schema = imports.StructType(
        [
            imports.StructField("grp", imports.StringType()),
            imports.StructField("x", imports.DoubleType()),
        ]
    )
    df = spark.createDataFrame([("a", 1.0), ("a", 2.0)], schema)
    ceil_val = _by_group(df.groupBy("grp").agg(F.ceil(F.avg("x")).alias("r")))
    floor_val = _by_group(df.groupBy("grp").agg(F.floor(F.avg("x")).alias("r")))
    assert ceil_val == {"a": 2.0}
    assert floor_val == {"a": 1.0}


def test_upper_of_max_on_a_string_column(spark: Any) -> None:
    """A non-numeric outer function over a non-numeric aggregate."""
    F = get_spark_imports().F
    df = _frame(spark).groupBy("grp").agg(F.upper(F.max("s")).alias("r"))
    assert _by_group(df) == {"a": "Q", "b": "R"}


def test_coalesce_over_an_aggregate(spark: Any) -> None:
    """``coalesce`` takes the aggregate as one of several arguments."""
    F = get_spark_imports().F
    df = _frame(spark).groupBy("grp").agg(F.coalesce(F.sum("x"), F.lit(0.0)).alias("r"))
    assert _by_group(df) == {"a": 60.0, "b": 20.0}


def test_two_aggregates_inside_one_expression(spark: Any) -> None:
    """Both aggregates in a two-operand expression must be resolved.

    A fix that substituted only the first aggregate and left the second as a
    Column would silently read NULL for it.

    (``F.greatest`` / ``F.least`` would be the natural shape here, but they are
    independently broken -- ``ExpressionEvaluator`` returns their first
    argument regardless of value -- so this uses arithmetic instead. See
    BUG-038.)
    """
    F = get_spark_imports().F
    df = _frame(spark).groupBy("grp").agg(F.sqrt(F.sum("x") - F.max("x")).alias("r"))
    assert _by_group(df) == {"a": 5.477226, "b": 2.236068}


def test_scalar_function_over_an_aggregate_of_an_expression(spark: Any) -> None:
    """Both the operand shape *and* the wrapper vary: ``sqrt(sum(x * 2))``."""
    F = get_spark_imports().F
    df = _frame(spark).groupBy("grp").agg(F.sqrt(F.sum(F.col("x") * 2)).alias("r"))
    assert _by_group(df) == {"a": 10.954451, "b": 6.324555}


def test_scalar_function_over_arithmetic_on_an_aggregate(spark: Any) -> None:
    """Nested: a scalar function wrapping arithmetic wrapping an aggregate."""
    F = get_spark_imports().F
    df = _frame(spark).groupBy("grp").agg(F.sqrt(F.sum("x") / 15).alias("r"))
    assert _by_group(df) == {"a": 2.0, "b": 1.154701}


def test_scalar_function_over_an_aggregate_of_a_case_when(spark: Any) -> None:
    """CASE WHEN operand under an aggregate under a scalar function."""
    F = get_spark_imports().F
    operand = F.when(F.col("x") > 10, F.col("x")).otherwise(F.lit(0.0))
    df = _frame(spark).groupBy("grp").agg(F.sqrt(F.sum(operand)).alias("r"))
    assert _by_group(df) == {"a": 7.071068, "b": 3.872983}


# --------------------------------------------------------------------------
# Paths that already worked must keep working.
# --------------------------------------------------------------------------


def test_arithmetic_on_an_aggregate_is_unchanged(spark: Any) -> None:
    """``F.sum("x") / 3`` went through a dedicated branch and was correct."""
    F = get_spark_imports().F
    df = _frame(spark).groupBy("grp").agg((F.sum("x") / 3).alias("r"))
    assert _by_group(df) == {"a": 20.0, "b": 6.666667}


def test_plain_aggregates_are_unchanged(spark: Any) -> None:
    """A bare aggregate must not be routed through the new substitution."""
    F = get_spark_imports().F
    df = (
        _frame(spark)
        .groupBy("grp")
        .agg(F.sum("x").alias("s"), F.max("x").alias("m"), F.count("x").alias("c"))
    )
    got = {
        row["grp"]: (_num(row["s"]), _num(row["m"]), _num(row["c"]))
        for row in df.collect()
    }
    assert got == {"a": (60.0, 30.0, 3.0), "b": (20.0, 15.0, 2.0)}


def test_scalar_function_over_an_all_null_group_stays_null(spark: Any) -> None:
    """A genuine NULL aggregate must still produce NULL -- the fix must not
    manufacture a value where Spark has none."""
    F = get_spark_imports().F
    imports = get_spark_imports()
    schema = imports.StructType(
        [
            imports.StructField("grp", imports.StringType()),
            imports.StructField("x", imports.DoubleType()),
        ]
    )
    df = spark.createDataFrame([("a", None), ("a", None)], schema)
    result = df.groupBy("grp").agg(F.sqrt(F.sum("x")).alias("r"))
    assert _by_group(result) == {"a": None}
