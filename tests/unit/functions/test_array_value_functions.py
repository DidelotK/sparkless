"""Regression tests for the array value functions on the ``select`` path.

``F.flatten``, ``F.array_min``, ``F.array_max`` and ``F.slice`` all returned
NULL for every row under ``df.select(...)``. The visible consequence reported
on Solya-app/solya-data-platform#2420 is that
``F.array_distinct(F.flatten(F.collect_list(TAGS)))`` -- the tag rollup for
product dimensions -- came back **with its duplicates intact**, the exact
opposite of what the expression exists to guarantee. NULL is not a value a
caller checks for here, so the wrong answer travelled.

Root cause: :class:`sparkless.core.condition_evaluator.ConditionEvaluator`,
the evaluator the lazy ``select`` path uses, dispatches on a whitelist of
operation names and returns ``None`` for anything absent from it. These four
were absent. The ``withColumn`` path
(:class:`sparkless.dataframe.evaluation.expression_evaluator.ExpressionEvaluator`)
*did* implement ``flatten``, so the two paths answered differently for the
same expression on the same data -- which is why ``test_select_and_withcolumn_agree``
below is not a redundant assertion but the one that would have caught the drift.

Every expectation was measured against PySpark 4.0.0 (``local[1]``); these
tests are backend-agnostic and pass under ``MOCK_SPARK_TEST_BACKEND=pyspark``.
"""

import pytest

from tests.fixtures.spark_imports import get_spark_imports


@pytest.fixture
def arrays_df(spark):
    """Rows covering the empty, NULL, and NULL-element array shapes."""
    imports = get_spark_imports()
    StructType, StructField = imports.StructType, imports.StructField
    ArrayType, IntegerType = imports.ArrayType, imports.IntegerType

    schema = StructType(
        [
            StructField("nested", ArrayType(ArrayType(IntegerType()))),
            StructField("nums", ArrayType(IntegerType())),
        ]
    )
    return spark.createDataFrame(
        [
            ([[1, 2], [3]], [3, 1, 2]),
            ([[1, 2], None], []),
            (None, None),
            ([[]], [5, None, 2]),
        ],
        schema,
    )


def _column(df, expression):
    """Collect a single projected expression as a plain list."""
    return [row[0] for row in df.select(expression).collect()]


class TestFlatten:
    """``flatten`` concatenates one level; it is not NULL."""

    def test_flatten_concatenates_one_level(self, arrays_df) -> None:
        """The reported failure: every row came back NULL."""
        F = get_spark_imports().F

        assert _column(arrays_df, F.flatten(F.col("nested"))) == [
            [1, 2, 3],
            None,
            None,
            [],
        ]

    def test_flatten_of_null_inner_array_is_null(self, arrays_df) -> None:
        """A NULL inner array poisons the whole result rather than vanishing.

        Dropping it would be the plausible-looking wrong answer: the row would
        still carry an array, just one element short.
        """
        F = get_spark_imports().F

        assert _column(arrays_df, F.flatten(F.col("nested")))[1] is None

    def test_flatten_select_and_withcolumn_agree(self, arrays_df) -> None:
        """The two projection paths must not answer differently.

        They did: ``withColumn`` implemented ``flatten`` and ``select`` did not.
        """
        F = get_spark_imports().F
        expression = F.flatten(F.col("nested"))

        selected = _column(arrays_df, expression)
        with_column = [
            row["f"] for row in arrays_df.withColumn("f", expression).collect()
        ]

        assert selected == with_column

    def test_array_distinct_of_flatten_removes_duplicates(self, spark) -> None:
        """The data-platform tag rollup: duplicates must not survive.

        ``array_distinct`` was never the defect -- it is correct on its own --
        but it cannot deduplicate a NULL, so the whole expression silently kept
        every duplicate.
        """
        F = get_spark_imports().F
        df = spark.createDataFrame([(["a", "b", "a"],), (["z"],)], ["tags"])

        rolled_up = df.agg(
            F.array_distinct(F.flatten(F.collect_list("tags"))).alias("tags")
        ).collect()[0]["tags"]

        assert sorted(rolled_up) == ["a", "b", "z"]


class TestArrayMinMax:
    """``array_min`` / ``array_max`` skip NULLs and are NULL when empty."""

    def test_array_min_skips_nulls(self, arrays_df) -> None:
        F = get_spark_imports().F

        assert _column(arrays_df, F.array_min(F.col("nums"))) == [1, None, None, 2]

    def test_array_max_skips_nulls(self, arrays_df) -> None:
        F = get_spark_imports().F

        assert _column(arrays_df, F.array_max(F.col("nums"))) == [3, None, None, 5]

    def test_empty_array_is_null_not_an_error(self, arrays_df) -> None:
        """An empty array yields NULL; it must not raise or return 0."""
        F = get_spark_imports().F

        assert _column(arrays_df, F.array_min(F.col("nums")))[1] is None


class TestSlice:
    """``slice`` is 1-based, truncates past the end, and rejects start=0."""

    def test_slice_is_one_based(self, arrays_df) -> None:
        F = get_spark_imports().F

        assert _column(arrays_df, F.slice(F.col("nums"), 1, 2)) == [
            [3, 1],
            [],
            None,
            [5, None],
        ]

    def test_negative_start_counts_from_the_end(self, arrays_df) -> None:
        F = get_spark_imports().F

        assert _column(arrays_df, F.slice(F.col("nums"), -2, 2)) == [
            [1, 2],
            [],
            None,
            [None, 2],
        ]

    def test_length_past_the_end_truncates(self, arrays_df) -> None:
        F = get_spark_imports().F

        assert _column(arrays_df, F.slice(F.col("nums"), 2, 10)) == [
            [1, 2],
            [],
            None,
            [None, 2],
        ]

    def test_start_past_the_end_is_empty_not_null(self, spark) -> None:
        """Past the end gives ``[]``. NULL here would be indistinguishable
        from the unimplemented-function answer this whole file is about."""
        F = get_spark_imports().F
        df = spark.createDataFrame([([1, 2, 3],)], ["nums"])

        assert _column(df, F.slice(F.col("nums"), 10, 2)) == [[]]

    def test_zero_start_is_rejected(self, spark) -> None:
        """Spark refuses ``start=0``; answering NULL would hide the mistake."""
        F = get_spark_imports().F
        df = spark.createDataFrame([([1, 2, 3],)], ["nums"])

        with pytest.raises(Exception) as excinfo:
            df.select(F.slice(F.col("nums"), 0, 2)).collect()

        assert "start" in str(excinfo.value)

    def test_negative_length_is_rejected(self, spark) -> None:
        """Spark refuses a negative ``length`` for the same reason."""
        F = get_spark_imports().F
        df = spark.createDataFrame([([1, 2, 3],)], ["nums"])

        with pytest.raises(Exception) as excinfo:
            df.select(F.slice(F.col("nums"), 1, -1)).collect()

        assert "length" in str(excinfo.value)
