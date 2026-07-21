"""Regression tests for a scalar function wrapping an aggregate in ``select``
(BUG-039).

``df.select(F.sum(x))`` collapsed to one row correctly, but
``df.select(F.sqrt(F.sum(x)))`` returned *one NULL row per input row*. The
select planner decided whether a projection was an aggregation by testing the
**top-level** operation name against a hardcoded ``_AGG_OPS`` set. Wrapping the
aggregate in anything -- ``sqrt``, ``abs``, ``greatest``, or plain arithmetic
-- moved the aggregate out of the top-level slot, the test said "not an
aggregate", and the projection stayed row-wise.

So both the value *and* the row count were wrong, which is the useful thing
about this bug: a NULL is arguably ambiguous, but five rows where Spark gives
one is not.

BUG-037 fixed the same shape on the ``groupBy(...).agg()`` path, by walking the
expression instead of enumerating names. This is that fix applied to ``select``.

The complementary hazard is a window function: ``F.sum(x).over(w)`` *contains*
an aggregate but is row-wise, and collapsing it would be a worse bug than the
one being fixed. ``test_window_aggregate_is_not_collapsed`` pins that.

Every expectation below was captured from real PySpark 4.0.0 on OpenJDK 21 (the
DBR 17.3 pairing). These tests use the backend-agnostic ``spark`` fixture, so
the same file runs against real PySpark with
``MOCK_SPARK_TEST_BACKEND=pyspark``.
"""

from typing import Any, List

import pytest

from tests.fixtures.spark_imports import get_spark_imports

#: sum == 16.0, so sqrt(sum) == 4.0 -- distinct from every input value and
#: from the sum itself, so neither a row-wise result nor an un-wrapped
#: aggregate can be mistaken for the right answer.
VALUES = [1.0, 2.0, 3.0, 4.0, 6.0]


def _frame(spark: Any) -> Any:
    imports = get_spark_imports()
    schema = imports.StructType(
        [
            imports.StructField("x", imports.DoubleType()),
            # Constant partition key. `Window.partitionBy()` with no argument is
            # legal in PySpark but rejected by sparkless, so the window cases
            # below partition on this instead -- an unrelated divergence that
            # would otherwise make this file backend-specific.
            imports.StructField("g", imports.StringType()),
        ]
    )
    return spark.createDataFrame([(value, "g") for value in VALUES], schema)


def _floats(rows: List[Any], key: str) -> List[Any]:
    return [None if row[key] is None else float(row[key]) for row in rows]


class TestSelectScalarOverAggregate:
    """A wrapped aggregate must still collapse the projection to one row."""

    def test_bare_aggregate_is_the_control(self, spark: Any) -> None:
        """The unwrapped case, which was always right."""
        rows = _frame(spark).select(get_spark_imports().F.sum("x").alias("r")).collect()
        assert len(rows) == 1
        assert _floats(rows, "r") == [16.0]

    @pytest.mark.parametrize(  # type: ignore[misc,untyped-decorator]
        "wrapper,expected",
        [
            ("sqrt", 4.0),
            ("abs", 16.0),
        ],
    )
    def test_scalar_function_wrapping_an_aggregate(
        self, spark: Any, wrapper: str, expected: float
    ) -> None:
        """``select(f(sum(x)))`` -- one row, with the value."""
        imports = get_spark_imports()
        F = imports.F
        rows = (
            _frame(spark).select(getattr(F, wrapper)(F.sum("x")).alias("r")).collect()
        )
        assert len(rows) == 1
        assert _floats(rows, "r") == [expected]

    def test_arithmetic_on_an_aggregate(self, spark: Any) -> None:
        """Arithmetic moves the aggregate out of the top-level slot too."""
        imports = get_spark_imports()
        F = imports.F
        rows = _frame(spark).select((F.sum("x") / 2).alias("r")).collect()
        assert len(rows) == 1
        assert _floats(rows, "r") == [8.0]

    def test_variadic_function_over_two_aggregates(self, spark: Any) -> None:
        """``greatest(sum, max)`` -- two aggregates under one wrapper."""
        imports = get_spark_imports()
        F = imports.F
        rows = (
            _frame(spark)
            .select(F.greatest(F.sum("x"), F.max("x")).alias("r"))
            .collect()
        )
        assert len(rows) == 1
        assert _floats(rows, "r") == [16.0]

    def test_row_count_alone(self, spark: Any) -> None:
        """``count()`` on the projection, with no value assertion.

        Kept separate because the row count was wrong independently of the
        values: the bug produced five NULL rows, so a test that only read
        ``rows[0]`` would have seen NULL and a test that only checked
        non-emptiness would have passed.
        """
        imports = get_spark_imports()
        F = imports.F
        assert _frame(spark).select(F.sqrt(F.sum("x")).alias("r")).count() == 1

    def test_alongside_a_bare_aggregate(self, spark: Any) -> None:
        """A sibling bare aggregate used to mask the bug.

        With ``F.max(x)`` also in the projection list the planner already
        flipped into aggregate mode, and the wrapped aggregate came out right.
        Testing only this shape would have shown nothing.
        """
        imports = get_spark_imports()
        F = imports.F
        rows = (
            _frame(spark)
            .select(F.sqrt(F.sum("x")).alias("r"), F.max("x").alias("m"))
            .collect()
        )
        assert len(rows) == 1
        assert _floats(rows, "r") == [4.0]
        assert _floats(rows, "m") == [6.0]

    def test_window_aggregate_is_not_collapsed(self, spark: Any) -> None:
        """``sum(x).over(w)`` contains an aggregate but stays row-wise."""
        imports = get_spark_imports()
        F, Window = imports.F, imports.Window
        rows = (
            _frame(spark)
            .select(F.sum("x").over(Window.partitionBy("g")).alias("r"))
            .collect()
        )
        assert len(rows) == len(VALUES)
        assert _floats(rows, "r") == [16.0] * len(VALUES)

    def test_scalar_over_a_window_aggregate_is_not_collapsed(self, spark: Any) -> None:
        """The wrapped form of the same -- still one row per input row."""
        imports = get_spark_imports()
        F, Window = imports.F, imports.Window
        rows = (
            _frame(spark)
            .select(F.sqrt(F.sum("x").over(Window.partitionBy("g"))).alias("r"))
            .collect()
        )
        assert len(rows) == len(VALUES)
        assert _floats(rows, "r") == [4.0] * len(VALUES)

    def test_plain_projection_is_unaffected(self, spark: Any) -> None:
        """No aggregate anywhere means no collapsing."""
        imports = get_spark_imports()
        F = imports.F
        rows = _frame(spark).select(F.sqrt(F.col("x")).alias("r")).collect()
        assert len(rows) == len(VALUES)
