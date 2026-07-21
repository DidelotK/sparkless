"""Regression tests for aggregating over an *expression* rather than a column.

``F.sum(F.col("x"))`` targets a plain column and can be read straight out of
each row by name. ``F.sum(F.col("x") * 2)`` and ``F.sum(F.when(cond, x))``
target an expression that must be evaluated per row first -- there is no column
of that name to look up.

Sparkless gated its "evaluate per row" branch on the target having an
``.operation`` attribute. ``CaseWhen`` does not have one, so every
``F.sum(F.when(...))`` fell through to the plain-column path, looked up a column
literally named ``"CASE WHEN"``, missed on every row, and collapsed to the empty
default -- a constant ``0``. The window path had the same problem for *any*
expression target, including plain arithmetic.

The failure is silent in the worst way: ``0`` is a plausible conditional sum, so
an assertion of "is not null" or "sum >= 0" passes on a garbage value.

Also covered: Spark's ``SUM`` returns **NULL**, not ``0``, when a group has no
non-NULL value to add up. Returning ``0`` there is the same silent-zero failure
by a different route.

Verified against PySpark 4.0.0 (the DBR 17.3 runtime). These tests are
backend-agnostic and also pass under ``MOCK_SPARK_TEST_BACKEND=pyspark``.
"""

from tests.fixtures.spark_imports import get_spark_imports


def _frame(spark):
    """Group 'a' has one flagged row; group 'b' has none."""
    return spark.createDataFrame(
        [
            ("a", 10.0, True),
            ("a", 20.0, False),
            ("b", 40.0, False),
        ],
        ["grp", "x", "flag"],
    )


def _by_group(rows, key):
    return {r["grp"]: r[key] for r in rows}


class TestAggregateOverCaseWhen:
    """groupBy().agg() over a CASE WHEN target."""

    def test_sum_of_when_without_otherwise(self, spark) -> None:
        """sum(when(flag, x)) adds only the flagged rows."""
        imports = get_spark_imports()
        F = imports.F
        rows = (
            _frame(spark)
            .groupBy("grp")
            .agg(F.sum(F.when(F.col("flag"), F.col("x"))).alias("s"))
            .collect()
        )

        result = _by_group(rows, "s")
        assert result["a"] == 10.0
        # No flagged row in group 'b': Spark yields NULL, not 0.
        assert result["b"] is None

    def test_sum_of_when_with_otherwise(self, spark) -> None:
        """An explicit otherwise(0) makes the empty group a real 0."""
        imports = get_spark_imports()
        F = imports.F
        rows = (
            _frame(spark)
            .groupBy("grp")
            .agg(
                F.sum(F.when(F.col("flag"), F.col("x")).otherwise(F.lit(0.0))).alias(
                    "s"
                )
            )
            .collect()
        )

        result = _by_group(rows, "s")
        assert result["a"] == 10.0
        assert result["b"] == 0.0

    def test_conditional_counting_idiom(self, spark) -> None:
        """sum(when(cond, 1).otherwise(0)) -- the count-matching-rows idiom."""
        imports = get_spark_imports()
        F = imports.F
        rows = (
            _frame(spark)
            .groupBy("grp")
            .agg(
                F.sum(F.when(F.col("x") > 15, F.lit(1)).otherwise(F.lit(0))).alias("n")
            )
            .collect()
        )

        result = _by_group(rows, "n")
        assert result["a"] == 1
        assert result["b"] == 1

    def test_avg_of_when(self, spark) -> None:
        """avg() over a CASE WHEN ignores the NULL (non-matching) rows."""
        imports = get_spark_imports()
        F = imports.F
        rows = (
            _frame(spark)
            .groupBy("grp")
            .agg(F.avg(F.when(F.col("flag"), F.col("x"))).alias("a"))
            .collect()
        )

        result = _by_group(rows, "a")
        assert result["a"] == 10.0
        assert result["b"] is None

    def test_max_and_min_of_when(self, spark) -> None:
        """max()/min() over a CASE WHEN see only the matching rows."""
        imports = get_spark_imports()
        F = imports.F
        rows = (
            _frame(spark)
            .groupBy("grp")
            .agg(
                F.max(F.when(F.col("flag"), F.col("x"))).alias("mx"),
                F.min(F.when(F.col("flag"), F.col("x"))).alias("mn"),
            )
            .collect()
        )

        assert _by_group(rows, "mx")["a"] == 10.0
        assert _by_group(rows, "mn")["a"] == 10.0

    def test_arithmetic_target_still_works(self, spark) -> None:
        """The pre-existing arithmetic-expression path must not regress."""
        imports = get_spark_imports()
        F = imports.F
        rows = (
            _frame(spark).groupBy("grp").agg(F.sum(F.col("x") * 2).alias("s")).collect()
        )

        result = _by_group(rows, "s")
        assert result["a"] == 60.0
        assert result["b"] == 80.0

    def test_plain_column_still_works(self, spark) -> None:
        """The plain-column path must not regress."""
        imports = get_spark_imports()
        F = imports.F
        rows = _frame(spark).groupBy("grp").agg(F.sum("x").alias("s")).collect()

        result = _by_group(rows, "s")
        assert result["a"] == 30.0
        assert result["b"] == 40.0


class TestSumReturnsNullWhenNothingToAdd:
    """Spark's SUM is NULL -- not 0 -- when no non-NULL value is aggregated."""

    def test_sum_of_never_matching_condition_is_null(self, spark) -> None:
        """Every row NULL under the CASE WHEN -> NULL for every group."""
        imports = get_spark_imports()
        F = imports.F
        rows = (
            _frame(spark)
            .groupBy("grp")
            .agg(F.sum(F.when(F.lit(False), F.col("x"))).alias("s"))
            .collect()
        )

        result = _by_group(rows, "s")
        assert result["a"] is None
        assert result["b"] is None

    def test_sum_of_all_null_column_is_null(self, spark) -> None:
        """A column that is NULL throughout sums to NULL."""
        imports = get_spark_imports()
        F = imports.F
        # An explicit schema is required: neither engine can infer the type of
        # a column that is NULL in every row.
        schema = imports.StructType(
            [
                imports.StructField("grp", imports.StringType(), True),
                imports.StructField("x", imports.DoubleType(), True),
            ]
        )
        df = spark.createDataFrame([("a", None), ("a", None)], schema)
        rows = df.groupBy("grp").agg(F.sum("x").alias("s")).collect()

        assert rows[0]["s"] is None


class TestWindowAggregateOverExpression:
    """The same expression targets, evaluated over a window."""

    def test_window_sum_of_when(self, spark) -> None:
        """sum(when(flag, x)) over a partition adds only the flagged rows."""
        imports = get_spark_imports()
        F, Window = imports.F, imports.Window
        window = Window.partitionBy("grp")

        rows = (
            _frame(spark)
            .withColumn("s", F.sum(F.when(F.col("flag"), F.col("x"))).over(window))
            .collect()
        )

        by_group = {r["grp"]: r["s"] for r in rows}
        assert by_group["a"] == 10.0
        # Partition 'b' has no flagged row -> NULL, not 0.
        assert by_group["b"] is None

    def test_window_sum_of_arithmetic_expression(self, spark) -> None:
        """sum(x * 2) over a partition -- the window path took no expressions."""
        imports = get_spark_imports()
        F, Window = imports.F, imports.Window
        window = Window.partitionBy("grp")

        rows = (
            _frame(spark).withColumn("s", F.sum(F.col("x") * 2).over(window)).collect()
        )

        by_group = {r["grp"]: r["s"] for r in rows}
        assert by_group["a"] == 60.0
        assert by_group["b"] == 80.0

    def test_window_avg_of_arithmetic_expression(self, spark) -> None:
        """avg(x * 2) over a partition."""
        imports = get_spark_imports()
        F, Window = imports.F, imports.Window
        window = Window.partitionBy("grp")

        rows = (
            _frame(spark).withColumn("a", F.avg(F.col("x") * 2).over(window)).collect()
        )

        by_group = {r["grp"]: r["a"] for r in rows}
        assert by_group["a"] == 30.0
        assert by_group["b"] == 80.0

    def test_window_count_of_when(self, spark) -> None:
        """count(when(flag, 1)) counts only non-NULL results."""
        imports = get_spark_imports()
        F, Window = imports.F, imports.Window
        window = Window.partitionBy("grp")

        rows = (
            _frame(spark)
            .withColumn("n", F.count(F.when(F.col("flag"), F.lit(1))).over(window))
            .collect()
        )

        by_group = {r["grp"]: r["n"] for r in rows}
        assert by_group["a"] == 1
        assert by_group["b"] == 0

    def test_window_sum_of_plain_column_still_works(self, spark) -> None:
        """The plain-column window path must not regress."""
        imports = get_spark_imports()
        F, Window = imports.F, imports.Window
        window = Window.partitionBy("grp")

        rows = _frame(spark).withColumn("s", F.sum("x").over(window)).collect()

        by_group = {r["grp"]: r["s"] for r in rows}
        assert by_group["a"] == 30.0
        assert by_group["b"] == 40.0

    def test_window_running_sum_still_works(self, spark) -> None:
        """An ordered running total must keep accumulating correctly."""
        imports = get_spark_imports()
        F, Window = imports.F, imports.Window
        df = spark.createDataFrame(
            [("a", 1, 10.0), ("a", 2, 20.0), ("a", 3, 30.0)],
            ["grp", "k", "x"],
        )
        window = (
            Window.partitionBy("grp")
            .orderBy("k")
            .rowsBetween(Window.unboundedPreceding, Window.currentRow)
        )

        rows = df.withColumn("running", F.sum("x").over(window)).collect()

        by_k = {r["k"]: r["running"] for r in rows}
        assert by_k[1] == 10.0
        assert by_k[2] == 30.0
        assert by_k[3] == 60.0
