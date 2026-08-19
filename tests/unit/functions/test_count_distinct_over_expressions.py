"""Regression tests for ``countDistinct`` / ``approx_count_distinct`` targets.

Both returned **0** whenever their argument was an expression rather than a
plain column -- ``F.countDistinct(F.struct(...))``, ``F.countDistinct(F.upper(c))``,
``F.countDistinct(F.when(...))``. 0 is a legitimate answer for these functions,
so nothing distinguished "no distinct values" from "not computed".

Solya-app/solya-data-platform#2417 reports the struct form, and the site it
reports it from is ``pipelines/core/validation/batch_engine.py:388`` -- a
**validation counter**. A control that reads 0 because the engine measured
nothing, while reporting that nothing is wrong, is the worst available shape
of this bug, which is why the tests below assert a non-zero count for data
that has one rather than merely asserting "not NULL".

Expectations measured against PySpark 4.0.0 (``local[1]``); the file is
backend-agnostic and passes under ``MOCK_SPARK_TEST_BACKEND=pyspark``.
"""

import pytest

from tests.fixtures.spark_imports import get_spark_imports


@pytest.fixture
def skus_df(spark):
    """Four rows, one NULL department, two departments repeated."""
    return spark.createDataFrame(
        [("A", "eng"), ("B", "ops"), ("C", "eng"), ("D", None)],
        ["sku", "dept"],
    )


class TestCountDistinctOverExpressions:
    """The target may be any expression, not only a column reference."""

    def test_plain_column_still_counts(self, skus_df) -> None:
        """Baseline: the column form was never broken and must stay correct."""
        F = get_spark_imports().F

        assert (
            skus_df.agg(F.countDistinct(F.col("dept")).alias("n")).collect()[0]["n"]
            == 2
        )

    def test_function_call_target_counts(self, skus_df) -> None:
        """``countDistinct(upper(c))`` answered 0, not 2."""
        F = get_spark_imports().F

        assert (
            skus_df.agg(F.countDistinct(F.upper(F.col("dept"))).alias("n")).collect()[
                0
            ]["n"]
            == 2
        )

    def test_struct_target_counts(self, skus_df) -> None:
        """The reported form. Four rows, four distinct (sku, dept) pairs.

        A struct is never NULL, so no row is skipped -- including the one whose
        ``dept`` is NULL.
        """
        F = get_spark_imports().F

        assert (
            skus_df.agg(
                F.countDistinct(F.struct(F.col("sku"), F.col("dept"))).alias("n")
            ).collect()[0]["n"]
            == 4
        )

    def test_struct_with_a_null_field_is_still_a_distinct_value(self, skus_df) -> None:
        """``struct(dept)`` over eng/ops/eng/NULL is 3, not 2.

        The NULL is inside the struct, not in place of it, so it is a value
        like any other. Skipping it would be the plausible wrong answer.
        """
        F = get_spark_imports().F

        assert (
            skus_df.agg(F.countDistinct(F.struct(F.col("dept"))).alias("n")).collect()[
                0
            ]["n"]
            == 3
        )

    def test_case_when_target_counts(self, skus_df) -> None:
        """A CASE WHEN target is an expression too."""
        F = get_spark_imports().F

        assert (
            skus_df.agg(
                F.countDistinct(F.when(F.col("dept") == "eng", F.col("sku"))).alias("n")
            ).collect()[0]["n"]
            == 2
        )

    def test_approx_count_distinct_over_an_expression(self, skus_df) -> None:
        """The approximate variant had the same defect."""
        F = get_spark_imports().F

        assert (
            skus_df.agg(
                F.approx_count_distinct(F.upper(F.col("dept"))).alias("n")
            ).collect()[0]["n"]
            == 2
        )

    def test_grouped_count_distinct_over_an_expression(self, skus_df) -> None:
        """Per group, not only over the whole frame."""
        F = get_spark_imports().F

        counts = {
            row["dept"]: row["n"]
            for row in skus_df.groupBy("dept")
            .agg(F.countDistinct(F.upper(F.col("sku"))).alias("n"))
            .collect()
        }

        assert counts == {"eng": 2, "ops": 1, None: 1}
