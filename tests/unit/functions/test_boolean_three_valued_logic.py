"""Regression tests for SQL three-valued (Kleene) boolean logic.

Spark evaluates boolean expressions under SQL three-valued logic, where NULL
means "unknown":

    NOT TRUE  = FALSE     TRUE  AND NULL = NULL     TRUE  OR NULL = TRUE
    NOT FALSE = TRUE      FALSE AND NULL = FALSE    FALSE OR NULL = NULL
    NOT NULL  = NULL      NULL  AND NULL = NULL     NULL  OR NULL = NULL

and a predicate that evaluates to NULL filters the row **out** (only TRUE
passes the filter).

Sparkless previously evaluated a bare boolean column as a *presence* check
(``value is not None``) rather than reading the stored boolean. That made
``~F.col(flag)`` mean "flag IS NULL", so ``filter(~col)`` returned the NULL
rows and dropped the explicitly-false ones — a precise inversion — while
``col == F.lit(False)`` stayed correct. The logical connectives additionally
used Python's ``not``/``and``/``or``, which disagree with Kleene logic for
``NOT NULL``, ``NULL AND FALSE`` and ``NULL OR FALSE``.

Every expectation below was captured from real PySpark 4.0.0 (the DBR 17.3
runtime) on Java 21 and re-confirmed on PySpark 3.5.3 / Java 17; these tests
also pass when run against real PySpark via ``SPARKLESS_TEST_BACKEND=pyspark``.
"""

from tests.fixtures.spark_imports import get_spark_imports


def _bool_frame(spark):
    """Frame with one true / one false / one NULL boolean row."""
    imports = get_spark_imports()
    schema = imports.StructType(
        [
            imports.StructField("vid", imports.StringType()),
            imports.StructField("flag", imports.BooleanType()),
        ]
    )
    return spark.createDataFrame(
        [("v-active", True), ("v-inactive", False), ("v-unknown", None)],
        schema,
    )


def _pair_frame(spark):
    """Frame covering all nine (a, b) combinations of TRUE/FALSE/NULL."""
    imports = get_spark_imports()
    schema = imports.StructType(
        [
            imports.StructField("a", imports.BooleanType()),
            imports.StructField("b", imports.BooleanType()),
        ]
    )
    values = [True, False, None]
    rows = [(a, b) for a in values for b in values]
    return spark.createDataFrame(rows, schema)


# --------------------------------------------------------------------------
# Filtering: only TRUE passes; NULL predicates exclude the row.
# --------------------------------------------------------------------------


def test_negated_boolean_column_filter_matches_false_not_null(spark) -> None:
    """``~F.col(flag)`` must select the FALSE row, never the NULL row.

    This is the original defect: sparkless returned ``['v-unknown']``.
    """
    F = get_spark_imports().F
    df = _bool_frame(spark)

    result = sorted(row["vid"] for row in df.filter(~F.col("flag")).collect())

    assert result == ["v-inactive"]


def test_bare_boolean_column_filter_selects_only_true(spark) -> None:
    """``filter(F.col(flag))`` keeps TRUE only, not "every non-null row"."""
    F = get_spark_imports().F
    df = _bool_frame(spark)

    result = sorted(row["vid"] for row in df.filter(F.col("flag")).collect())

    assert result == ["v-active"]


def test_negation_agrees_with_equality_against_literal_false(spark) -> None:
    """``~col``, ``col == False`` and ``col == F.lit(False)`` must agree."""
    F = get_spark_imports().F
    df = _bool_frame(spark)

    def vids(predicate):
        return sorted(row["vid"] for row in df.filter(predicate).collect())

    assert (
        vids(~F.col("flag"))
        == vids(F.col("flag") == False)  # noqa: E712 - exercising the API
        == vids(F.col("flag") == F.lit(False))
        == ["v-inactive"]
    )


def test_double_negation_round_trips(spark) -> None:
    """``~~col`` is ``col``: TRUE only, NULL stays NULL through both NOTs."""
    F = get_spark_imports().F
    df = _bool_frame(spark)

    result = sorted(row["vid"] for row in df.filter(~~F.col("flag")).collect())

    assert result == ["v-active"]


def test_negated_is_null_is_unaffected(spark) -> None:
    """``~col.isNull()`` stays a two-valued predicate over a total function."""
    F = get_spark_imports().F
    df = _bool_frame(spark)

    result = sorted(row["vid"] for row in df.filter(~F.col("flag").isNull()).collect())

    assert result == ["v-active", "v-inactive"]


def test_bare_boolean_column_combines_with_comparison(spark) -> None:
    """A bare boolean operand mixed with a comparison operand."""
    F = get_spark_imports().F
    imports = get_spark_imports()
    schema = imports.StructType(
        [
            imports.StructField("n", imports.IntegerType()),
            imports.StructField("flag", imports.BooleanType()),
        ]
    )
    df = spark.createDataFrame([(1, True), (2, False), (3, None)], schema)

    assert [r["n"] for r in df.filter(F.col("flag") & (F.col("n") > 0)).collect()] == [
        1
    ]
    assert [
        r["n"] for r in df.filter((~F.col("flag")) & (F.col("n") > 0)).collect()
    ] == [2]


# --------------------------------------------------------------------------
# Projection: the full Kleene truth table.
# --------------------------------------------------------------------------


# value -> NOT value, captured from real PySpark 4.0.0
NOT_TRUTH_TABLE = [
    (True, False),
    (False, True),
    (None, None),
]

# (a, b) -> (a AND b, a OR b), captured from real PySpark 4.0.0
AND_OR_TRUTH_TABLE = [
    (True, True, True, True),
    (True, False, False, True),
    (True, None, None, True),
    (False, True, False, True),
    (False, False, False, False),
    (False, None, False, None),
    (None, True, None, True),
    (None, False, False, None),
    (None, None, None, None),
]


def test_not_truth_table(spark) -> None:
    """NOT TRUE = FALSE, NOT FALSE = TRUE, NOT NULL = NULL."""
    F = get_spark_imports().F
    df = _bool_frame(spark)

    rows = {
        row["flag"]: row["not_flag"]
        for row in df.select("flag", (~F.col("flag")).alias("not_flag")).collect()
    }

    for value, expected in NOT_TRUTH_TABLE:
        assert rows[value] is expected, f"NOT {value} should be {expected}"


def test_and_or_truth_table(spark) -> None:
    """Full Kleene AND/OR table, including the FALSE/TRUE dominance cases.

    ``NULL AND FALSE`` is FALSE and ``NULL OR TRUE`` is TRUE — the result is
    determined regardless of what the unknown stands for. Python's ``and`` /
    ``or`` get these two cells wrong.
    """
    F = get_spark_imports().F
    df = _pair_frame(spark)

    projected = df.select(
        "a",
        "b",
        (F.col("a") & F.col("b")).alias("conj"),
        (F.col("a") | F.col("b")).alias("disj"),
    ).collect()
    table = {(row["a"], row["b"]): (row["conj"], row["disj"]) for row in projected}

    for a, b, expected_and, expected_or in AND_OR_TRUTH_TABLE:
        assert table[(a, b)] == (expected_and, expected_or), (
            f"a={a} b={b}: expected AND={expected_and} OR={expected_or}, "
            f"got {table[(a, b)]}"
        )


def test_and_filter_keeps_only_both_true(spark) -> None:
    """``filter(a & b)`` passes only the TRUE/TRUE row."""
    F = get_spark_imports().F
    df = _pair_frame(spark)

    result = [(r["a"], r["b"]) for r in df.filter(F.col("a") & F.col("b")).collect()]

    assert sorted(result, key=str) == [(True, True)]


def test_or_filter_keeps_rows_with_a_true_operand(spark) -> None:
    """``filter(a | b)`` passes exactly the rows where either side is TRUE."""
    F = get_spark_imports().F
    df = _pair_frame(spark)

    result = [(r["a"], r["b"]) for r in df.filter(F.col("a") | F.col("b")).collect()]

    assert sorted(result, key=str) == sorted(
        [(True, True), (True, False), (True, None), (False, True), (None, True)],
        key=str,
    )
