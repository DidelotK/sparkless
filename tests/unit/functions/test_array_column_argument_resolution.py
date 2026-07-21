"""Regression tests for the array set/search functions.

Two independent defects, both silent:

1. **BUG-028** -- ``array_except`` and ``array_intersect`` had no evaluator
   branch at all. The constructors and re-exports exist, so the call built
   fine, matched no ``operation_type``, and fell through to the dispatch's
   default ``return None``. They returned NULL *even with a pure-literal
   second argument*.
2. **BUG-029** -- ``array_remove`` and ``array_position`` used
   ``operation.value`` raw instead of resolving it against the row. With a
   ``Column`` second argument, ``element != <Column>`` invokes
   ``Column.__ne__``, which returns a truthy ``ColumnOperation`` rather than a
   bool. ``array_remove`` therefore returned the array *unchanged*, and
   ``array_position`` -- whose ``list.index`` uses ``==`` -- matched index 0
   unconditionally, returning a plausible **wrong number**.

The two are distinct: the first is a missing implementation, the second a
missing operand resolution. Only the second is fixed by the shared
``_resolve_operand`` helper.

Every expected value below was produced by executing the same expression
against real **PySpark 4.0.0** on OpenJDK 21 (the DBR 17.3 pairing), not
derived from the API docs. These tests are backend-agnostic and also pass
under ``MOCK_SPARK_TEST_BACKEND=pyspark``.
"""

import pytest

from tests.fixtures.spark_imports import get_spark_imports


def _frame(spark):
    """Six rows covering literal/Column, absent, NULL element, empty and NULL array."""
    imports = get_spark_imports()
    schema = imports.StructType(
        [
            imports.StructField("id", imports.StringType()),
            imports.StructField("doms", imports.ArrayType(imports.StringType())),
            imports.StructField("domain", imports.StringType()),
        ]
    )
    rows = [
        ("r1", ["d1", "d2"], "d1"),
        ("r2", ["d1", "d2", "d1"], "d1"),
        ("r3", ["d1", "d2"], "zz"),
        ("r4", ["d1", None, "d2"], None),
        ("r5", [], "d1"),
        ("r6", None, "d1"),
    ]
    return spark.createDataFrame(rows, schema)


def _by_id(spark, column, alias="r"):
    """Evaluate ``column`` over the fixture frame, keyed by row id."""
    df = _frame(spark)
    return {
        row["id"]: row[alias] for row in df.select("id", column.alias(alias)).collect()
    }


class TestArrayPositionColumnArgument:
    """BUG-029, the dangerous half: a wrong *index*, not an unchanged array."""

    def test_position_resolves_column_argument(self, spark) -> None:
        """array_position(arr, col) must compare against the row's value.

        Before the fix every row returned 1 -- including r3, where the value
        is absent and Spark returns 0. A wrong integer index looks like a
        legitimate answer, which is what makes this the priority defect.
        """
        F = get_spark_imports().F
        got = _by_id(spark, F.array_position(F.col("doms"), F.col("domain")))
        assert got["r1"] == 1
        assert got["r2"] == 1
        assert got["r3"] == 0  # "zz" absent -- previously returned 1
        assert got["r4"] is None  # NULL search value -> NULL
        assert got["r5"] == 0  # empty array
        assert got["r6"] is None  # NULL array -> NULL

    def test_position_literal_argument_unchanged(self, spark) -> None:
        """The literal path already worked and must keep working."""
        F = get_spark_imports().F
        got = _by_id(spark, F.array_position(F.col("doms"), "d2"))
        assert got["r1"] == 2
        assert got["r3"] == 2
        assert got["r4"] == 3  # NULL elements occupy a position
        assert got["r5"] == 0
        assert got["r6"] is None

    def test_position_absent_literal_is_zero(self, spark) -> None:
        """An absent literal is 0, not 1."""
        F = get_spark_imports().F
        got = _by_id(spark, F.array_position(F.col("doms"), "zz"))
        assert got["r1"] == 0
        assert got["r5"] == 0

    def test_position_accepts_lit(self, spark) -> None:
        """F.lit() is a Literal, not a Column -- it must resolve to its value."""
        F = get_spark_imports().F
        got = _by_id(spark, F.array_position(F.col("doms"), F.lit("d2")))
        assert got["r1"] == 2
        assert got["r3"] == 2

    def test_position_on_integer_array(self, spark) -> None:
        """Non-string element types resolve identically."""
        imports = get_spark_imports()
        F = imports.F
        schema = imports.StructType(
            [
                imports.StructField("x", imports.ArrayType(imports.IntegerType())),
                imports.StructField("v", imports.IntegerType()),
            ]
        )
        df = spark.createDataFrame([([1, 2, 3], 2)], schema)
        assert (
            df.select(F.array_position(F.col("x"), F.col("v")).alias("r")).collect()[0][
                "r"
            ]
            == 2
        )


class TestArrayRemoveColumnArgument:
    """BUG-029, the visible half: the array came back unchanged."""

    def test_remove_resolves_column_argument(self, spark) -> None:
        """array_remove(arr, col) must drop every occurrence of the row's value."""
        F = get_spark_imports().F
        got = _by_id(spark, F.array_remove(F.col("doms"), F.col("domain")))
        assert got["r1"] == ["d2"]  # previously ["d1", "d2"] -- unchanged
        assert got["r2"] == ["d2"]  # all occurrences removed, no dedupe of others
        assert got["r3"] == ["d1", "d2"]  # absent value removes nothing
        assert got["r4"] is None  # NULL remove value -> NULL result
        assert got["r5"] == []
        assert got["r6"] is None

    def test_remove_literal_argument_unchanged(self, spark) -> None:
        """The literal path already worked and must keep working."""
        F = get_spark_imports().F
        got = _by_id(spark, F.array_remove(F.col("doms"), "d1"))
        assert got["r1"] == ["d2"]
        assert got["r4"] == [None, "d2"]  # NULL elements survive
        assert got["r6"] is None

    def test_remove_does_not_deduplicate(self, spark) -> None:
        """array_remove keeps duplicates of the surviving elements."""
        imports = get_spark_imports()
        F = imports.F
        schema = imports.StructType(
            [imports.StructField("x", imports.ArrayType(imports.StringType()))]
        )
        df = spark.createDataFrame([(["a", "b", "a", "c", "b"],)], schema)
        assert df.select(F.array_remove(F.col("x"), "c").alias("r")).collect()[0][
            "r"
        ] == ["a", "b", "a", "b"]

    def test_remove_accepts_lit(self, spark) -> None:
        """F.lit() resolves to its value rather than being compared as an object."""
        F = get_spark_imports().F
        got = _by_id(spark, F.array_remove(F.col("doms"), F.lit("d1")))
        assert got["r1"] == ["d2"]


class TestArrayExcept:
    """BUG-028: unimplemented -- NULL even with a literal argument."""

    def test_except_with_literal_second_argument(self, spark) -> None:
        """The literal path was broken too, which distinguishes this from BUG-029."""
        F = get_spark_imports().F
        got = _by_id(spark, F.array_except(F.col("doms"), F.array(F.lit("d1"))))
        assert got["r1"] == ["d2"]
        assert got["r3"] == ["d2"]
        assert got["r4"] == [None, "d2"]
        assert got["r5"] == []
        assert got["r6"] is None

    def test_except_with_column_second_argument(self, spark) -> None:
        """The exclude-self projection: which *other* values fired."""
        F = get_spark_imports().F
        got = _by_id(spark, F.array_except(F.col("doms"), F.array(F.col("domain"))))
        assert got["r1"] == ["d2"]
        assert got["r2"] == ["d2"]
        assert got["r3"] == ["d1", "d2"]
        assert got["r4"] == ["d1", "d2"]  # NULL matches the NULL element
        assert got["r5"] == []
        assert got["r6"] is None

    def test_except_deduplicates_and_keeps_order(self, spark) -> None:
        """Spark's array_except has set semantics: the result is deduplicated."""
        imports = get_spark_imports()
        F = imports.F
        schema = imports.StructType(
            [
                imports.StructField("x", imports.ArrayType(imports.StringType())),
                imports.StructField("y", imports.ArrayType(imports.StringType())),
            ]
        )
        df = spark.createDataFrame([(["a", "b", "a", "c"], ["a"])], schema)
        assert df.select(F.array_except("x", "y").alias("r")).collect()[0]["r"] == [
            "b",
            "c",
        ]

    def test_except_excludes_self_value(self, spark) -> None:
        """The downstream invariant this column exists to satisfy.

        ``collect_set(domain)`` minus the row's own ``domain`` must never
        contain that domain. Before the fix the projection returned NULL, and
        the ``array_remove`` formulation returned the full set *including*
        the row's own value.
        """
        F = get_spark_imports().F
        for expr in (
            F.array_except(F.col("doms"), F.array(F.col("domain"))),
            F.array_remove(F.col("doms"), F.col("domain")),
        ):
            got = _by_id(spark, expr)
            for row_id in ("r1", "r2"):
                assert got[row_id] is not None
                assert "d1" not in got[row_id]


class TestArrayIntersect:
    """BUG-028, sibling function -- same missing-branch cause."""

    def test_intersect_with_column_second_argument(self, spark) -> None:
        F = get_spark_imports().F
        got = _by_id(spark, F.array_intersect(F.col("doms"), F.array(F.col("domain"))))
        assert got["r1"] == ["d1"]
        assert got["r3"] == []  # no overlap -- empty array, not NULL
        assert got["r4"] == [None]  # NULL matches the NULL element
        assert got["r5"] == []
        assert got["r6"] is None

    def test_intersect_deduplicates(self, spark) -> None:
        """Set semantics: ['a','a','b'] ∩ ['a','b'] is ['a','b']."""
        imports = get_spark_imports()
        F = imports.F
        schema = imports.StructType(
            [
                imports.StructField("x", imports.ArrayType(imports.StringType())),
                imports.StructField("y", imports.ArrayType(imports.StringType())),
            ]
        )
        df = spark.createDataFrame([(["a", "a", "b"], ["a", "b"])], schema)
        assert df.select(F.array_intersect("x", "y").alias("r")).collect()[0]["r"] == [
            "a",
            "b",
        ]


class TestArrayContainsUnaffected:
    """array_contains already resolved its argument; it must not regress.

    It was refactored onto the shared helper, so it is pinned here.
    """

    @pytest.mark.parametrize(
        "row_id,expected", [("r1", True), ("r3", False), ("r5", False)]
    )
    def test_contains_column_argument(self, spark, row_id, expected) -> None:
        F = get_spark_imports().F
        got = _by_id(spark, F.array_contains(F.col("doms"), F.col("domain")))
        assert got[row_id] is expected

    def test_contains_literal_argument(self, spark) -> None:
        F = get_spark_imports().F
        got = _by_id(spark, F.array_contains(F.col("doms"), "d2"))
        assert got["r1"] is True
        assert got["r5"] is False
