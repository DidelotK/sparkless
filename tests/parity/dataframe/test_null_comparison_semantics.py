"""BUG-033 / BUG-046: three-valued logic in boolean projections.

Two defects, both instances of the recurring ``select`` vs ``withColumn`` vs
``filter`` path schism:

BUG-033 -- ``ConditionEvaluator`` collapsed a comparison with a NULL operand to
FALSE. In SQL ``NULL > 5`` is NULL, and the distinction is load-bearing because
``NOT (NULL > 5)`` is NULL, not TRUE. ``withColumn`` (a different evaluator)
already returned NULL, so the two projection paths disagreed. The predicate path
was worse still: its kernel returned ``operation == "!="``, so ``NULL != 9``
evaluated to TRUE.

BUG-046 -- ``ExpressionEvaluator`` had no dispatch branch for ``~`` (emitted as
the operation string ``"!"``). It fell through to the unknown-operation
fallback, which returns the *operand*, so ``withColumn("f", ~(col("x") >= 5))``
silently produced the un-negated value -- a sign inversion on a value-producing
path. Fixed on ``main`` by BUG-046's ``_evaluate_predicate_operation``; these
tests were written against the defect independently and are kept as coverage.

Reference behaviour was captured from real PySpark 4.0.0 on OpenJDK 21. Every
test runs against the backend-agnostic ``spark`` fixture, so
``MOCK_SPARK_TEST_BACKEND=pyspark`` re-checks it against real Spark.
"""

from typing import Any, Dict, List, Optional, Tuple

import pytest

from tests.fixtures.parity_base import ParityTestBase
from tests.fixtures.spark_imports import get_spark_imports


# id, x -- 'b' carries the NULL that every assertion below turns on.
ROWS: List[Tuple[Any, ...]] = [("a", 9), ("b", None), ("c", 1)]


def _schema() -> Any:
    """Build the shared test schema for either backend."""
    imports = get_spark_imports()
    return imports.StructType(
        [
            imports.StructField("id", imports.StringType()),
            imports.StructField("x", imports.IntegerType()),
        ]
    )


def _by_id(df: Any, column: str) -> Dict[str, Optional[bool]]:
    """Collect ``column`` keyed by id, so row order does not matter."""
    return {row["id"]: row[column] for row in df.collect()}


class TestComparisonYieldsNullInProjections(ParityTestBase):
    """A comparison with a NULL operand is NULL, in every projection path."""

    @pytest.mark.parametrize(  # type: ignore[misc,untyped-decorator]
        "op,expected",
        [
            # (a=9, b=NULL, c=1) compared against the literal 5 / 9.
            ("ge", {"a": True, "b": None, "c": False}),
            ("gt", {"a": True, "b": None, "c": False}),
            ("lt", {"a": False, "b": None, "c": True}),
            ("le", {"a": False, "b": None, "c": True}),
            ("eq", {"a": True, "b": None, "c": False}),
            ("ne", {"a": False, "b": None, "c": True}),
        ],
    )
    def test_select_projects_null_not_false(
        self, spark: Any, op: str, expected: Dict[str, Optional[bool]]
    ) -> None:
        """The exact BUG-033 repro, across every comparison operator.

        ``!=`` is the one that used to be actively inverted on the predicate
        path (``NULL != 9`` returned TRUE), so it is not merely a duplicate of
        the others.
        """
        imports = get_spark_imports()
        F = imports.F
        df = spark.createDataFrame(ROWS, _schema())

        col = F.col("x")
        expr = {
            "ge": col >= F.lit(5),
            "gt": col > F.lit(5),
            "lt": col < F.lit(5),
            "le": col <= F.lit(5),
            "eq": col == F.lit(9),
            "ne": col != F.lit(9),
        }[op]

        assert _by_id(df.select("id", expr.alias("f")), "f") == expected

    @pytest.mark.parametrize(  # type: ignore[misc,untyped-decorator]
        "shape",
        ["column_vs_literal", "column_vs_column", "literal_vs_literal"],
    )
    def test_null_propagates_for_every_operand_shape(
        self, spark: Any, shape: str
    ) -> None:
        """Vary the *operand shape*, not just the values.

        A truth table that only ever compares a bare column against a literal
        leaves the column-vs-column and literal-vs-literal shapes unpinned --
        and those route through different resolution branches.
        """
        imports = get_spark_imports()
        F = imports.F
        df = spark.createDataFrame(ROWS, _schema())

        if shape == "column_vs_literal":
            expr, expected = (
                F.col("x") >= F.lit(5),
                {"a": True, "b": None, "c": False},
            )
        elif shape == "column_vs_column":
            # x > x is FALSE where x is known and NULL where it is not.
            expr, expected = (
                F.col("x") > F.col("x"),
                {"a": False, "b": None, "c": False},
            )
        else:
            # A NULL literal makes the comparison NULL for every row.
            expr, expected = (
                F.lit(None) > F.lit(5),
                {"a": None, "b": None, "c": None},
            )

        assert _by_id(df.select("id", expr.alias("f")), "f") == expected

    def test_select_and_with_column_agree(self, spark: Any) -> None:
        """The two projection paths must produce the same column.

        This is the invariant BUG-033 was reported against: ``withColumn`` was
        right and ``select`` was wrong. Asserting agreement (rather than each
        one separately) is what keeps them from drifting apart again.
        """
        imports = get_spark_imports()
        F = imports.F
        df = spark.createDataFrame(ROWS, _schema())

        expr = F.col("x") >= F.lit(5)
        via_select = _by_id(df.select("id", expr.alias("f")), "f")
        via_with_column = _by_id(df.withColumn("f", expr).select("id", "f"), "f")

        assert via_select == via_with_column == {"a": True, "b": None, "c": False}

    def test_null_comparison_survives_and_or(self, spark: Any) -> None:
        """Kleene AND/OR must see NULL, not a coerced FALSE."""
        imports = get_spark_imports()
        F = imports.F
        df = spark.createDataFrame(ROWS, _schema())

        conj = (F.col("x") >= F.lit(5)) & (F.col("x") < F.lit(20))
        disj = (F.col("x") >= F.lit(5)) | F.lit(False)
        assert _by_id(df.select("id", conj.alias("f")), "f") == {
            "a": True,
            "b": None,
            "c": False,
        }
        assert _by_id(df.select("id", disj.alias("f")), "f") == {
            "a": True,
            "b": None,
            "c": False,
        }

    def test_true_dominates_or_even_with_a_null_operand(self, spark: Any) -> None:
        """``NULL OR TRUE`` is TRUE -- NULL must not swallow the whole term."""
        imports = get_spark_imports()
        F = imports.F
        df = spark.createDataFrame(ROWS, _schema())

        expr = (~(F.col("x") >= F.lit(5))) | F.lit(True)
        assert _by_id(df.select("id", expr.alias("f")), "f") == {
            "a": True,
            "b": True,
            "c": True,
        }


class TestNegatedComparisonProjections(ParityTestBase):
    """BUG-046: ``~`` must actually negate, and must keep NULL as NULL."""

    @pytest.mark.parametrize(  # type: ignore[misc,untyped-decorator]
        "projection", ["select", "withColumn"]
    )
    def test_negated_comparison(self, spark: Any, projection: str) -> None:
        """``~(x >= 5)`` inverts TRUE/FALSE and leaves NULL alone.

        Parametrised over both projection paths because they used to fail in
        *opposite* directions: ``select`` returned TRUE for the NULL row,
        ``withColumn`` returned the un-negated value for the others.
        """
        imports = get_spark_imports()
        F = imports.F
        df = spark.createDataFrame(ROWS, _schema())

        expr = ~(F.col("x") >= F.lit(5))
        if projection == "select":
            result = _by_id(df.select("id", expr.alias("f")), "f")
        else:
            result = _by_id(df.withColumn("f", expr).select("id", "f"), "f")

        assert result == {"a": False, "b": None, "c": True}

    @pytest.mark.parametrize(  # type: ignore[misc,untyped-decorator]
        "projection", ["select", "withColumn"]
    )
    def test_negated_conjunction(self, spark: Any, projection: str) -> None:
        """``~(A AND B)`` negates the whole conjunction, not just its head."""
        imports = get_spark_imports()
        F = imports.F
        df = spark.createDataFrame(ROWS, _schema())

        expr = ~((F.col("x") >= F.lit(5)) & (F.col("x") < F.lit(20)))
        if projection == "select":
            result = _by_id(df.select("id", expr.alias("f")), "f")
        else:
            result = _by_id(df.withColumn("f", expr).select("id", "f"), "f")

        assert result == {"a": False, "b": None, "c": True}

    @pytest.mark.parametrize(  # type: ignore[misc,untyped-decorator]
        "projection", ["select", "withColumn"]
    )
    def test_negated_is_null_is_never_null(self, spark: Any, projection: str) -> None:
        """``~x.isNull()`` is total: ``isNull`` never yields NULL itself."""
        imports = get_spark_imports()
        F = imports.F
        df = spark.createDataFrame(ROWS, _schema())

        expr = ~F.col("x").isNull()
        if projection == "select":
            result = _by_id(df.select("id", expr.alias("f")), "f")
        else:
            result = _by_id(df.withColumn("f", expr).select("id", "f"), "f")

        assert result == {"a": True, "b": False, "c": True}


class TestNullComparisonInFilter(ParityTestBase):
    """A NULL predicate drops the row, and stays NULL under negation."""

    def test_filter_drops_the_null_row(self, spark: Any) -> None:
        """``WHERE x >= 5`` keeps only rows where the predicate is TRUE."""
        imports = get_spark_imports()
        F = imports.F
        df = spark.createDataFrame(ROWS, _schema())

        result = [row["id"] for row in df.filter(F.col("x") >= F.lit(5)).collect()]
        assert result == ["a"]

    def test_filter_on_a_negated_comparison_also_drops_the_null_row(
        self, spark: Any
    ) -> None:
        """``WHERE NOT (x >= 5)`` must not resurrect the NULL row.

        With the comparison collapsed to FALSE, ``NOT FALSE`` was TRUE and the
        NULL row survived. Under three-valued logic ``NOT NULL`` is NULL, which
        is not TRUE, so the row is filtered out.
        """
        imports = get_spark_imports()
        F = imports.F
        df = spark.createDataFrame(ROWS, _schema())

        result = [row["id"] for row in df.filter(~(F.col("x") >= F.lit(5))).collect()]
        assert result == ["c"]

    def test_filter_on_inequality_drops_the_null_row(self, spark: Any) -> None:
        """``WHERE x != 9`` drops NULL: ``NULL != 9`` is NULL, not TRUE.

        The predicate kernel used to special-case ``!=`` and return TRUE for a
        NULL operand, which let the NULL row through.
        """
        imports = get_spark_imports()
        F = imports.F
        df = spark.createDataFrame(ROWS, _schema())

        result = [row["id"] for row in df.filter(F.col("x") != F.lit(9)).collect()]
        assert result == ["c"]
