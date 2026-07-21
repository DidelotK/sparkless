"""PySpark parity tests for boolean predicates (BUG-046).

``Column.isNotNull()`` used to evaluate to the *column's own value* rather than
to a boolean whenever it was projected with ``withColumn`` or used inside
``when`` / ``&`` / ``|``::

    df.withColumn("c", F.col("price").isNotNull())
    # sparkless -> [100.0, 200.0, 150.0]   PySpark -> [True, True, True]

The value is truthy, so an enclosing ``when`` matched every row. The same
defect hit ``isin``, ``eqNullSafe``, ``~``, ``&`` and ``|``: none of them had a
handler in ``ExpressionEvaluator``, so all of them fell through to
``_evaluate_function_call``, whose terminal ``return value`` hands back the
operand.

Every expectation in this module is the value captured from **PySpark 4.0.0 on
OpenJDK 21**, so the file is written against the backend-agnostic ``spark``
fixture and passes under ``MOCK_SPARK_TEST_BACKEND=pyspark`` too.
"""

from typing import Any, List, Optional

import pytest

from tests.fixtures.parity_base import ParityTestBase
from tests.fixtures.spark_imports import get_spark_imports


def _values(rows: List[Any], key: str) -> List[Any]:
    """Extract one column from collected rows, in row order."""
    return [row[key] for row in rows]


class TestPredicateProjectionParity(ParityTestBase):
    """A projected predicate must be a boolean, never its own operand."""

    def test_is_not_null_projects_true_not_the_value(self, spark: Any) -> None:
        """``withColumn`` on ``isNotNull()`` returns booleans, not the prices."""
        F = get_spark_imports().F
        df = spark.createDataFrame(
            [{"price": 100.0}, {"price": 200.0}, {"price": 150.0}]
        )

        result = df.withColumn("c", F.col("price").isNotNull()).collect()

        # PySpark 4.0.0: [True, True, True]. The bug returned the prices.
        assert _values(result, "c") == [True, True, True]

    def test_is_not_null_projection_agrees_with_select(self, spark: Any) -> None:
        """``withColumn`` and ``select`` must agree -- they used to not."""
        F = get_spark_imports().F
        df = spark.createDataFrame([{"price": 100.0}, {"price": 200.0}])

        via_with_column = _values(
            df.withColumn("c", F.col("price").isNotNull()).collect(), "c"
        )
        via_select = _values(
            df.select(F.col("price").isNotNull().alias("c")).collect(), "c"
        )

        assert via_with_column == via_select == [True, True]

    @pytest.mark.parametrize(  # type: ignore[misc,untyped-decorator]
        "predicate_name,expected",
        [
            ("isNull", False),
            ("isNotNull", True),
        ],
    )
    def test_null_predicates_on_a_non_null_row(
        self, spark: Any, predicate_name: str, expected: bool
    ) -> None:
        """Null predicates over a non-null value, in all three paths."""
        F = get_spark_imports().F
        df = spark.createDataFrame([{"price": 100.0}])
        predicate = getattr(F.col("price"), predicate_name)()

        assert df.withColumn("c", predicate).collect()[0]["c"] is expected
        assert df.select(predicate.alias("c")).collect()[0]["c"] is expected
        assert df.filter(predicate).count() == (1 if expected else 0)

    @pytest.mark.parametrize(  # type: ignore[misc,untyped-decorator]
        "predicate_name,expected",
        [
            ("isNull", True),
            ("isNotNull", False),
        ],
    )
    def test_null_predicates_on_a_null_row(
        self, spark: Any, predicate_name: str, expected: bool
    ) -> None:
        """Null predicates over a NULL value, in all three paths."""
        imports = get_spark_imports()
        F = imports.F
        schema = imports.StructType(
            [imports.StructField("price", imports.DoubleType(), True)]
        )
        df = spark.createDataFrame([(None,)], schema=schema)
        predicate = getattr(F.col("price"), predicate_name)()

        assert df.withColumn("c", predicate).collect()[0]["c"] is expected
        assert df.select(predicate.alias("c")).collect()[0]["c"] is expected
        assert df.filter(predicate).count() == (1 if expected else 0)

    def test_isin_projects_a_boolean(self, spark: Any) -> None:
        """``isin`` projected returns a boolean, not the operand nor NULL."""
        F = get_spark_imports().F
        df = spark.createDataFrame([{"n": 5}, {"n": 7}])

        predicate = F.col("n").isin([1, 5, 9])

        assert _values(df.withColumn("c", predicate).collect(), "c") == [True, False]
        assert _values(df.select(predicate.alias("c")).collect(), "c") == [True, False]
        assert df.filter(predicate).count() == 1

    def test_between_projects_a_boolean(self, spark: Any) -> None:
        """``between`` projected returns a boolean; ``select`` used to give NULL."""
        F = get_spark_imports().F
        df = spark.createDataFrame([{"n": 5}, {"n": 50}])

        predicate = F.col("n").between(1, 10)

        assert _values(df.withColumn("c", predicate).collect(), "c") == [True, False]
        assert _values(df.select(predicate.alias("c")).collect(), "c") == [True, False]
        assert df.filter(predicate).count() == 1

    def test_eq_null_safe_projects_a_boolean(self, spark: Any) -> None:
        """``eqNullSafe`` projected returns a boolean; it used to give NULL."""
        F = get_spark_imports().F
        df = spark.createDataFrame([{"n": 5}, {"n": 7}])

        predicate = F.col("n").eqNullSafe(5)

        assert _values(df.withColumn("c", predicate).collect(), "c") == [True, False]
        assert _values(df.select(predicate.alias("c")).collect(), "c") == [True, False]
        assert df.filter(predicate).count() == 1

    @pytest.mark.parametrize(  # type: ignore[misc,untyped-decorator]
        "predicate_name,argument,expected",
        [
            ("like", "a%", True),
            ("rlike", "^a", True),
            ("contains", "b", True),
            ("startswith", "a", True),
            ("endswith", "c", True),
            ("like", "z%", False),
            ("contains", "z", False),
            ("startswith", "z", False),
            ("endswith", "z", False),
        ],
    )
    def test_string_predicates_project_booleans(
        self, spark: Any, predicate_name: str, argument: str, expected: bool
    ) -> None:
        """String predicates project booleans in every path."""
        F = get_spark_imports().F
        df = spark.createDataFrame([{"s": "abc"}])
        predicate = getattr(F.col("s"), predicate_name)(argument)

        assert df.withColumn("c", predicate).collect()[0]["c"] is expected
        assert df.select(predicate.alias("c")).collect()[0]["c"] is expected
        assert df.filter(predicate).count() == (1 if expected else 0)


class TestLogicalConnectiveParity(ParityTestBase):
    """``&`` / ``|`` / ``~`` must combine operands, not return the left one."""

    @pytest.mark.parametrize(  # type: ignore[misc,untyped-decorator]
        "left_threshold,right_threshold,expected",
        [
            (1, 100, False),  # True AND False
            (100, 1, False),  # False AND True
            (1, 2, True),  # True AND True
            (100, 200, False),  # False AND False
        ],
    )
    def test_and_combines_both_operands(
        self, spark: Any, left_threshold: int, right_threshold: int, expected: bool
    ) -> None:
        """``&`` used to return its LEFT operand, so `True & False` gave True."""
        F = get_spark_imports().F
        df = spark.createDataFrame([{"n": 5}])

        predicate = (F.col("n") > left_threshold) & (F.col("n") > right_threshold)

        assert df.withColumn("c", predicate).collect()[0]["c"] is expected
        assert df.select(predicate.alias("c")).collect()[0]["c"] is expected

    @pytest.mark.parametrize(  # type: ignore[misc,untyped-decorator]
        "left_threshold,right_threshold,expected",
        [
            (100, 1, True),  # False OR True
            (1, 100, True),  # True OR False
            (100, 200, False),  # False OR False
            (1, 2, True),  # True OR True
        ],
    )
    def test_or_combines_both_operands(
        self, spark: Any, left_threshold: int, right_threshold: int, expected: bool
    ) -> None:
        """``|`` used to return its LEFT operand, so `False | True` gave False."""
        F = get_spark_imports().F
        df = spark.createDataFrame([{"n": 5}])

        predicate = (F.col("n") > left_threshold) | (F.col("n") > right_threshold)

        assert df.withColumn("c", predicate).collect()[0]["c"] is expected
        assert df.select(predicate.alias("c")).collect()[0]["c"] is expected

    def test_not_inverts_its_operand(self, spark: Any) -> None:
        """``~`` used to return the operand unchanged in a projection."""
        F = get_spark_imports().F
        df = spark.createDataFrame([{"n": 5}])

        predicate = ~(F.col("n") > 1)

        assert df.withColumn("c", predicate).collect()[0]["c"] is False
        assert df.select(predicate.alias("c")).collect()[0]["c"] is False
        assert df.filter(predicate).count() == 0

    def test_is_not_null_anded_with_a_false_comparison(self, spark: Any) -> None:
        """The compound shape from the downstream report."""
        F = get_spark_imports().F
        df = spark.createDataFrame(
            [{"price": 100.0}, {"price": 200.0}, {"price": 150.0}]
        )

        predicate = F.col("price").isNotNull() & (F.col("price") < 0)

        # PySpark 4.0.0: [False, False, False]. The bug returned the prices.
        assert _values(df.withColumn("c", predicate).collect(), "c") == [
            False,
            False,
            False,
        ]
        assert df.filter(predicate).count() == 0


class TestPredicateAggregationParity(ParityTestBase):
    """The decisive downstream check: a validation-rule invalid-row count."""

    def test_sum_of_when_over_a_null_guarded_condition_is_zero(
        self, spark: Any
    ) -> None:
        """``sum(when(isNotNull() & (price < 0), 1).otherwise(0))`` must be 0.

        This is the exact shape a downstream validation framework builds its
        rules from. With ``isNotNull()`` leaking the price, the condition was
        truthy for every row and the invalid-row count came out as 3 of 3.
        """
        F = get_spark_imports().F
        df = spark.createDataFrame(
            [{"price": 100.0}, {"price": 200.0}, {"price": 150.0}]
        )

        condition = F.col("price").isNotNull() & (F.col("price") < 0)
        invalid = df.agg(F.sum(F.when(condition, 1).otherwise(0)).alias("v")).collect()[
            0
        ]["v"]

        assert invalid == 0

    def test_sum_of_when_counts_the_genuinely_invalid_rows(self, spark: Any) -> None:
        """A guard that must be able to fail: two rows really are negative."""
        F = get_spark_imports().F
        df = spark.createDataFrame(
            [
                {"price": 100.0},
                {"price": -5.0},
                {"price": 150.0},
                {"price": -1.0},
            ]
        )

        condition = F.col("price").isNotNull() & (F.col("price") < 0)
        invalid = df.agg(F.sum(F.when(condition, 1).otherwise(0)).alias("v")).collect()[
            0
        ]["v"]

        assert invalid == 2

    def test_when_over_a_predicate_selects_the_right_branch(self, spark: Any) -> None:
        """``when`` on a bare predicate, row by row."""
        F = get_spark_imports().F
        df = spark.createDataFrame([{"n": 5}, {"n": 50}])

        result = df.withColumn(
            "c", F.when(F.col("n").isin([5]), "hit").otherwise("miss")
        ).collect()

        assert _values(result, "c") == ["hit", "miss"]


class TestPredicateNullSemanticsParity(ParityTestBase):
    """Three-valued logic, captured from PySpark 4.0.0."""

    @staticmethod
    def _null_row(spark: Any) -> Any:
        """One row whose string and double columns are both NULL."""
        imports = get_spark_imports()
        schema = imports.StructType(
            [
                imports.StructField("s", imports.StringType(), True),
                imports.StructField("d", imports.DoubleType(), True),
            ]
        )
        return spark.createDataFrame([(None, None)], schema=schema)

    @pytest.mark.parametrize(  # type: ignore[misc,untyped-decorator]
        "predicate_name,argument",
        [
            ("like", "a%"),
            ("rlike", "^a"),
            ("contains", "b"),
            ("startswith", "a"),
            ("endswith", "c"),
        ],
    )
    def test_string_predicates_are_null_over_null(
        self, spark: Any, predicate_name: str, argument: str
    ) -> None:
        """PySpark 4.0.0 returns NULL, not FALSE, for a NULL operand."""
        F = get_spark_imports().F
        df = self._null_row(spark)
        predicate = getattr(F.col("s"), predicate_name)(argument)

        assert df.select(predicate.alias("c")).collect()[0]["c"] is None
        assert df.filter(predicate).count() == 0

    @pytest.mark.parametrize(  # type: ignore[misc,untyped-decorator]
        "build_predicate_name,argument",
        [
            ("isin", [1.0, 2.0]),
            ("between", None),
        ],
    )
    def test_membership_predicates_are_null_over_null(
        self, spark: Any, build_predicate_name: str, argument: Optional[List[Any]]
    ) -> None:
        """``isin`` / ``between`` over NULL are NULL in PySpark 4.0.0."""
        F = get_spark_imports().F
        df = self._null_row(spark)
        column = F.col("d")
        predicate = (
            column.isin(argument)
            if build_predicate_name == "isin"
            else column.between(1, 10)
        )

        assert df.select(predicate.alias("c")).collect()[0]["c"] is None
        assert df.filter(predicate).count() == 0

    def test_isin_is_null_when_unmatched_and_the_list_holds_null(
        self, spark: Any
    ) -> None:
        """SQL IN: no match plus a NULL in the list is NULL, not FALSE."""
        F = get_spark_imports().F
        df = spark.createDataFrame([{"d": 3.0}])

        unmatched = F.col("d").isin([1.0, None])
        matched = F.col("d").isin([3.0, None])

        assert df.select(unmatched.alias("c")).collect()[0]["c"] is None
        # A match wins over the NULL in the list.
        assert df.select(matched.alias("c")).collect()[0]["c"] is True

    @pytest.mark.parametrize(  # type: ignore[misc,untyped-decorator]
        "predicate_name,expected",
        [
            ("isNull", True),
            ("isNotNull", False),
        ],
    )
    def test_null_predicates_never_return_null(
        self, spark: Any, predicate_name: str, expected: bool
    ) -> None:
        """``isNull`` / ``isNotNull`` are total: they never yield NULL."""
        F = get_spark_imports().F
        df = self._null_row(spark)
        predicate = getattr(F.col("d"), predicate_name)()

        assert df.select(predicate.alias("c")).collect()[0]["c"] is expected

    def test_eq_null_safe_never_returns_null(self, spark: Any) -> None:
        """``<=>``: NULL <=> NULL is TRUE, NULL <=> value is FALSE."""
        F = get_spark_imports().F
        null_df = self._null_row(spark)
        value_df = spark.createDataFrame([{"d": 1.0}])

        both_null = F.col("d").eqNullSafe(F.lit(None).cast("double"))
        null_vs_value = F.col("d").eqNullSafe(F.lit(1.0))

        assert null_df.select(both_null.alias("c")).collect()[0]["c"] is True
        assert null_df.select(null_vs_value.alias("c")).collect()[0]["c"] is False
        assert value_df.select(both_null.alias("c")).collect()[0]["c"] is False
        assert value_df.select(null_vs_value.alias("c")).collect()[0]["c"] is True

    def test_not_of_a_null_predicate_stays_null(self, spark: Any) -> None:
        """``NOT NULL`` is NULL, so the row is still filtered out."""
        F = get_spark_imports().F
        df = self._null_row(spark)

        predicate = ~F.col("d").isNull()

        # isNull() is TRUE here, so NOT of it is FALSE -- not NULL.
        assert df.withColumn("c", predicate).collect()[0]["c"] is False
        assert df.filter(predicate).count() == 0

    def test_and_short_circuits_to_false_over_null(self, spark: Any) -> None:
        """Kleene AND: ``NULL AND FALSE`` is FALSE, ``NULL AND TRUE`` is NULL."""
        F = get_spark_imports().F
        df = self._null_row(spark)

        null_and_false = (F.col("d") > 1) & F.col("d").isNotNull()
        null_and_true = (F.col("d") > 1) & F.col("d").isNull()

        assert df.withColumn("c", null_and_false).collect()[0]["c"] is False
        assert df.withColumn("c", null_and_true).collect()[0]["c"] is None

    def test_or_short_circuits_to_true_over_null(self, spark: Any) -> None:
        """Kleene OR: ``NULL OR TRUE`` is TRUE, ``NULL OR FALSE`` is NULL."""
        F = get_spark_imports().F
        df = self._null_row(spark)

        null_or_true = (F.col("d") > 1) | F.col("d").isNull()
        null_or_false = (F.col("d") > 1) | F.col("d").isNotNull()

        assert df.withColumn("c", null_or_true).collect()[0]["c"] is True
        assert df.withColumn("c", null_or_false).collect()[0]["c"] is None


class TestIsNaNParity(ParityTestBase):
    """``Column.isNaN`` -- the PySpark method sparkless was missing."""

    def test_is_nan_distinguishes_nan_from_null_and_from_a_number(
        self, spark: Any
    ) -> None:
        """NaN is TRUE; a number and NULL are both FALSE -- never NULL."""
        imports = get_spark_imports()
        F = imports.F
        schema = imports.StructType(
            [imports.StructField("d", imports.DoubleType(), True)]
        )
        df = spark.createDataFrame([(float("nan"),), (1.0,), (None,)], schema=schema)

        predicate = F.col("d").isNaN()

        assert _values(df.withColumn("c", predicate).collect(), "c") == [
            True,
            False,
            False,
        ]
        assert _values(df.select(predicate.alias("c")).collect(), "c") == [
            True,
            False,
            False,
        ]
        assert df.filter(predicate).count() == 1
