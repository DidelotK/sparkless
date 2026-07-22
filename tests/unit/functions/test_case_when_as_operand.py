"""A CASE WHEN used as an *operand* rather than as a whole projection.

BUG-054 (arithmetic operand), BUG-055 (sort key), BUG-056 (generated name).

``F.when(...).otherwise(...)`` returns a ``CaseWhen``, which is deliberately
neither a ``Column`` nor a ``ColumnOperation``. Every dispatch that resolves an
operand therefore has to name it explicitly, and three of them did not:

======================================  ==========================================
Site                                    Failure before the fix
======================================  ==========================================
``ConditionEvaluator._get_column_value``  fell through to ``return column`` and
                                        handed back the *unevaluated* object
``LazyDataFrame`` ``orderBy``            resolved the key to a column *name*,
                                        found nothing, tied every row
``CaseWhen.otherwise``                   f-string-interpolated raw operands into
                                        the generated column name
======================================  ==========================================

The first is the dangerous one. ``_get_column_value`` returning the object
means ``CASE * 2.0`` dispatches to ``CaseWhen.__mul__``, which *builds a new
ColumnOperation* instead of multiplying -- so the cell value of a perfectly
ordinary ``F.when(...) * F.lit(2.0)`` was a ColumnOperation object. Worse,
wrapping it hid the object behind a plausible number: ``F.abs(CASE * lit)``
came back ``None`` and ``F.greatest(CASE * lit, F.lit(-1.0))`` came back
``-1.0``. Both are values an assertion can accept without complaint.

Every expectation below was captured from real PySpark 4.0.0 on OpenJDK 21
(the DBR 17.3 pairing). These tests use the backend-agnostic ``spark``
fixture, so the same file runs against real PySpark with
``MOCK_SPARK_TEST_BACKEND=pyspark``.
"""

from typing import Any, List

from tests.fixtures.spark_imports import get_spark_imports


def _frame(spark: Any) -> Any:
    """Two groups, one NULL, and rows fed in a deliberately unsorted order.

    ``v`` is NULL on row ``d`` so the ELSE branch is exercised, and the input
    order (a, b, c, d) is *not* the sorted order -- otherwise a sort that
    silently does nothing would still look correct.
    """
    imports = get_spark_imports()
    schema = imports.StructType(
        [
            imports.StructField("g", imports.StringType(), False),
            imports.StructField("id", imports.StringType(), False),
            imports.StructField("v", imports.DoubleType(), True),
        ]
    )
    return spark.createDataFrame(
        [("g1", "a", 10.0), ("g1", "b", 0.5), ("g2", "c", 4.0), ("g2", "d", None)],
        schema,
    )


def _case(F: Any) -> Any:
    """``CASE WHEN v > 1.0 THEN v ELSE 0.0 END`` -- 10.0 / 0.0 / 4.0 / 0.0."""
    return F.when(F.col("v") > F.lit(1.0), F.col("v")).otherwise(F.lit(0.0))


def _values(rows: List[Any], key: str) -> List[Any]:
    """Column ``key`` from ``rows``, floats normalised across backends.

    Anything that is not a plain scalar is replaced by a description of its
    type. That substitution is load-bearing, not cosmetic: the value this
    module is guarding against is an unevaluated ``ColumnOperation``, and
    ``ColumnOperation.__eq__`` returns *another ColumnOperation* rather than a
    bool. A truthy object makes ``[leaked] == [20.0]`` evaluate to True, so
    the obvious assertion silently passes against the very bug it is meant to
    catch -- these tests did exactly that until the substitution was added.
    """
    out: List[Any] = []
    for row in rows:
        value = row[key]
        if value is None or isinstance(value, bool):
            out.append(value)
        elif isinstance(value, (int, float)):
            out.append(float(value))
        elif isinstance(value, str):
            out.append(value)
        else:
            out.append(f"<non-scalar {type(value).__name__}>")
    return out


class TestCaseWhenAsArithmeticOperand:
    """BUG-054: ``CASE <op> x`` must evaluate, not build a new expression."""

    def test_multiplication_yields_numbers(self, spark: Any) -> None:
        """The reported shape: every cell was a ColumnOperation object."""
        F = get_spark_imports().F
        df = _frame(spark)
        rows = df.select(F.col("id"), (_case(F) * F.lit(2.0)).alias("r")).collect()
        assert _values(rows, "r") == [20.0, 0.0, 8.0, 0.0]

    def test_every_arithmetic_operator(self, spark: Any) -> None:
        """+ - * / all resolve their operands through the same dispatch."""
        F = get_spark_imports().F
        df = _frame(spark)
        case = _case(F)
        assert _values(df.select((case + F.lit(2.0)).alias("r")).collect(), "r") == [
            12.0,
            2.0,
            6.0,
            2.0,
        ]
        assert _values(df.select((case - F.lit(2.0)).alias("r")).collect(), "r") == [
            8.0,
            -2.0,
            2.0,
            -2.0,
        ]
        assert _values(df.select((case / F.lit(2.0)).alias("r")).collect(), "r") == [
            5.0,
            0.0,
            2.0,
            0.0,
        ]

    def test_case_on_the_right_hand_side(self, spark: Any) -> None:
        """``lit * CASE`` takes ``__rmul__``, a different entry point."""
        F = get_spark_imports().F
        rows = _frame(spark).select((F.lit(2.0) * _case(F)).alias("r")).collect()
        assert _values(rows, "r") == [20.0, 0.0, 8.0, 0.0]

    def test_case_on_both_sides(self, spark: Any) -> None:
        """Both operands need resolving, not just the left one."""
        F = get_spark_imports().F
        case = _case(F)
        rows = _frame(spark).select((case * case).alias("r")).collect()
        assert _values(rows, "r") == [100.0, 0.0, 16.0, 0.0]

    def test_wrapping_functions_do_not_hide_the_object(self, spark: Any) -> None:
        """The silent cases: ``abs`` gave None and ``greatest`` gave -1.0.

        These matter more than the bare multiplication. A raw ColumnOperation
        in a cell is at least obviously wrong; ``None`` and ``-1.0`` are the
        kind of values a test asserts against without noticing.
        """
        F = get_spark_imports().F
        df = _frame(spark)
        scaled = _case(F) * F.lit(2.0)
        assert _values(df.select(F.abs(scaled).alias("r")).collect(), "r") == [
            20.0,
            0.0,
            8.0,
            0.0,
        ]
        assert _values(
            df.select(F.greatest(scaled, F.lit(-1.0)).alias("r")).collect(), "r"
        ) == [20.0, 0.0, 8.0, 0.0]
        assert _values(
            df.select(F.coalesce(scaled, F.lit(-1.0)).alias("r")).collect(), "r"
        ) == [20.0, 0.0, 8.0, 0.0]

    def test_select_and_with_column_agree(self, spark: Any) -> None:
        """The two projection paths are evaluated by different code."""
        F = get_spark_imports().F
        df = _frame(spark)
        scaled = _case(F) * F.lit(2.0)
        via_select = _values(df.select(scaled.alias("r")).collect(), "r")
        via_with_column = _values(df.withColumn("r", scaled).collect(), "r")
        assert via_select == via_with_column == [20.0, 0.0, 8.0, 0.0]

    def test_aggregate_over_case_arithmetic(self, spark: Any) -> None:
        """``F.sum`` folded the leaked object into its accumulator."""
        F = get_spark_imports().F
        df = _frame(spark)
        scaled = _case(F) * F.lit(2.0)
        assert _values(df.agg(F.sum(scaled).alias("r")).collect(), "r") == [28.0]
        assert sorted(
            (row["g"], float(row["r"]))
            for row in df.groupBy("g").agg(F.sum(scaled).alias("r")).collect()
        ) == [("g1", 20.0), ("g2", 8.0)]

    def test_nested_case_arithmetic(self, spark: Any) -> None:
        """A CASE whose THEN branch is arithmetic over another CASE."""
        F = get_spark_imports().F
        df = _frame(spark)
        inner = _case(F)
        outer = F.when(inner > F.lit(1.0), inner * F.lit(2.0)).otherwise(F.lit(-1.0))
        assert _values(df.select(outer.alias("r")).collect(), "r") == [
            20.0,
            -1.0,
            8.0,
            -1.0,
        ]

    def test_case_arithmetic_as_a_predicate(self, spark: Any) -> None:
        """The filter path resolves operands through its own dispatch."""
        F = get_spark_imports().F
        df = _frame(spark)
        scaled = _case(F) * F.lit(2.0)
        assert sorted(
            row["id"] for row in df.filter(scaled > F.lit(5.0)).collect()
        ) == ["a", "c"]


class TestCaseWhenAsSortKey:
    """BUG-055: a computed ORDER BY key was resolved by name, so it tied."""

    def test_order_by_case_arithmetic(self, spark: Any) -> None:
        """Returned the input order (a, b, c, d) -- silently unsorted."""
        F = get_spark_imports().F
        df = _frame(spark)
        scaled = _case(F) * F.lit(2.0)
        ordered = df.orderBy(scaled.asc(), F.col("id").asc()).collect()
        assert [row["id"] for row in ordered] == ["b", "d", "c", "a"]

    def test_order_by_case_arithmetic_descending(self, spark: Any) -> None:
        """Direction must still apply once the key actually evaluates."""
        F = get_spark_imports().F
        df = _frame(spark)
        scaled = _case(F) * F.lit(2.0)
        ordered = df.orderBy(scaled.desc(), F.col("id").asc()).collect()
        assert [row["id"] for row in ordered] == ["a", "c", "b", "d"]

    def test_plain_column_sort_keys_still_work(self, spark: Any) -> None:
        """Guards the fix: named keys must keep using the row lookup."""
        F = get_spark_imports().F
        df = _frame(spark)
        assert [row["id"] for row in df.orderBy(F.col("id").desc()).collect()] == [
            "d",
            "c",
            "b",
            "a",
        ]
        # NULLs: Spark sorts ASC NULLS FIRST.
        assert [row["id"] for row in df.orderBy(F.col("v").asc()).collect()][0] == "d"


class TestCaseWhenGeneratedName:
    """BUG-056: the generated column name embedded a memory address."""

    def test_name_matches_spark_sql(self, spark: Any) -> None:
        """PySpark 4.0.0 names it after the SQL text of the expression."""
        F = get_spark_imports().F
        df = _frame(spark)
        assert df.select(_case(F)).columns[0] == (
            "CASE WHEN (v > 1.0) THEN v ELSE 0.0 END"
        )

    def test_name_is_deterministic_across_rebuilds(self, spark: Any) -> None:
        """The address leak made the name differ between two identical builds.

        This is the assertion that fails loudest under the bug: nothing about
        the expression changed, yet the two names differed.
        """
        F = get_spark_imports().F
        df = _frame(spark)
        first = df.select(_case(F)).columns[0]
        second = df.select(_case(F)).columns[0]
        assert first == second
        assert "0x" not in first
        assert "object at" not in first

    def test_multi_branch_name(self, spark: Any) -> None:
        """Only the first WHEN used to be rendered."""
        F = get_spark_imports().F
        df = _frame(spark)
        expr = (
            F.when(F.col("v") > F.lit(1.0), F.col("v"))
            .when(F.col("v") < F.lit(1.0), F.lit(-1.0))
            .otherwise(F.lit(0.0))
        )
        assert df.select(expr).columns[0] == (
            "CASE WHEN (v > 1.0) THEN v WHEN (v < 1.0) THEN -1.0 ELSE 0.0 END"
        )

    def test_name_without_otherwise(self, spark: Any) -> None:
        """No ELSE clause when ``otherwise`` was never called."""
        F = get_spark_imports().F
        df = _frame(spark)
        expr = F.when(F.col("v") > F.lit(1.0), F.col("v"))
        assert df.select(expr).columns[0] == "CASE WHEN (v > 1.0) THEN v END"

    def test_nested_case_name(self, spark: Any) -> None:
        """A nested CASE renders as its own SQL, not as its repr."""
        F = get_spark_imports().F
        df = _frame(spark)
        inner = F.when(F.col("v") > F.lit(5.0), F.lit(9.0)).otherwise(F.lit(8.0))
        expr = F.when(F.col("v") > F.lit(1.0), inner).otherwise(F.lit(0.0))
        assert df.select(expr).columns[0] == (
            "CASE WHEN (v > 1.0) THEN CASE WHEN (v > 5.0) THEN 9.0 ELSE 8.0 END "
            "ELSE 0.0 END"
        )
