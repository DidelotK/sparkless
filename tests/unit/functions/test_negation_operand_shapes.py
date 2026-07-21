"""Three-valued-logic regression tests for negation, indexed by operand *shape*.

`tests/unit/functions/test_boolean_three_valued_logic.py` covers the truth
values (TRUE/FALSE/NULL) but only for one operand shape: a **bare boolean
column**. That left the other shapes a boolean expression can take completely
uncovered, and two of them were wrong:

* ``~F.coalesce(col, F.lit(False))`` — the predicate path evaluated function
  operations with a scalar helper that only saw the *first* operand and
  returned NULL whenever it was NULL. `coalesce(NULL, False)` therefore
  evaluated to NULL instead of FALSE, so the row was dropped.
* ``~F.when(...).otherwise(...)`` — ``CaseWhen.__invert__`` emits the operation
  string ``"~"``, which no evaluator branch recognised, so the negation fell
  through to ``False`` for every row and the filter returned nothing at all.

These are *shape* bugs, not truth-value bugs: the same three input values that
the truth-table file already exercised produced the correct answer through a
bare column and the wrong one through a function result. So this file varies
the shape and keeps the data fixed.

Every expectation was captured from real PySpark 4.0.0 on Java 21 (the DBR 17.3
runtime). The file uses the repo's backend-agnostic ``spark`` fixture, so it can
be re-verified against real PySpark with ``SPARKLESS_TEST_BACKEND=pyspark``.
"""

from tests.fixtures.spark_imports import get_spark_imports


def _forecast_frame(spark):
    """The shape of the real production data that surfaced this bug.

    ``is_forecast`` is a nullable boolean: legacy rows predate the flag and
    carry NULL, and the production filter treats them as observed via
    ``~F.coalesce(F.col("is_forecast"), F.lit(False))``.
    """
    imports = get_spark_imports()
    schema = imports.StructType(
        [
            imports.StructField("vid", imports.StringType()),
            imports.StructField("is_forecast", imports.BooleanType()),
        ]
    )
    return spark.createDataFrame(
        [("obs", False), ("fc", True), ("legacy", None)],
        schema,
    )


def _vids(df, predicate):
    return sorted(row["vid"] for row in df.filter(predicate).collect())


# --------------------------------------------------------------------------
# Shape: negation of a function result.
# --------------------------------------------------------------------------


def test_negated_coalesce_keeps_null_derived_rows(spark) -> None:
    """``~coalesce(col, lit(False))`` must keep NULL-derived rows.

    ``coalesce(NULL, False)`` is FALSE, and ``NOT FALSE`` is TRUE, so the legacy
    row is kept. Regression: this returned ``['obs']``, silently dropping real
    production rows from a historical computation.
    """
    F = get_spark_imports().F
    df = _forecast_frame(spark)

    assert _vids(df, ~F.coalesce(F.col("is_forecast"), F.lit(False))) == [
        "legacy",
        "obs",
    ]


def test_coalesce_itself_is_unchanged(spark) -> None:
    """The un-negated function result was always correct; pin it so the fix
    cannot be "achieved" by breaking coalesce instead."""
    F = get_spark_imports().F
    df = _forecast_frame(spark)

    assert _vids(df, F.coalesce(F.col("is_forecast"), F.lit(False))) == ["fc"]


def test_negated_coalesce_projects_three_valued_result(spark) -> None:
    """Projection agrees with the filter: NOT coalesce(NULL, False) is TRUE."""
    F = get_spark_imports().F
    df = _forecast_frame(spark)

    values = {
        row["vid"]: row["neg"]
        for row in df.select(
            "vid", (~F.coalesce(F.col("is_forecast"), F.lit(False))).alias("neg")
        ).collect()
    }

    assert values == {"obs": True, "fc": False, "legacy": True}


def test_double_negated_coalesce_round_trips(spark) -> None:
    """``~~coalesce(...)`` collapses back to the un-negated result."""
    F = get_spark_imports().F
    df = _forecast_frame(spark)

    assert _vids(df, ~~F.coalesce(F.col("is_forecast"), F.lit(False))) == ["fc"]


# --------------------------------------------------------------------------
# Shape: negation of a CASE WHEN.
# --------------------------------------------------------------------------


def test_negated_case_when_with_otherwise(spark) -> None:
    """``~when(cond, True).otherwise(False)`` mirrors the coalesce case.

    Regression: ``CaseWhen.__invert__`` emits ``"~"``, which no evaluator branch
    matched, so this filter returned an empty result set.
    """
    F = get_spark_imports().F
    df = _forecast_frame(spark)

    negated = ~F.when(F.col("is_forecast"), F.lit(True)).otherwise(F.lit(False))

    assert _vids(df, negated) == ["legacy", "obs"]


def test_negated_case_when_without_otherwise_is_null(spark) -> None:
    """Without ``.otherwise``, unmatched rows are NULL, so NOT is NULL too.

    Only the matched row yields a non-NULL value (``NOT TRUE`` = FALSE), so the
    filter keeps nothing. Verified against real PySpark 4.0.0 — this asserts
    NULL propagation, not an empty result by accident.
    """
    F = get_spark_imports().F
    df = _forecast_frame(spark)

    assert _vids(df, ~F.when(F.col("is_forecast"), F.lit(True))) == []


def test_negated_case_when_projects_three_valued_result(spark) -> None:
    """CASE WHEN negation projects TRUE/FALSE/TRUE, and NULL where unmatched."""
    F = get_spark_imports().F
    df = _forecast_frame(spark)

    with_otherwise = ~F.when(F.col("is_forecast"), F.lit(True)).otherwise(F.lit(False))
    without_otherwise = ~F.when(F.col("is_forecast"), F.lit(True))

    rows = {
        row["vid"]: (row["with_else"], row["no_else"])
        for row in df.select(
            "vid",
            with_otherwise.alias("with_else"),
            without_otherwise.alias("no_else"),
        ).collect()
    }

    assert rows == {
        "obs": (True, None),
        "fc": (False, False),
        "legacy": (True, None),
    }


# --------------------------------------------------------------------------
# Shape: negation of a compound / already-boolean expression.
# --------------------------------------------------------------------------


def test_negated_conjunction_of_function_result(spark) -> None:
    """A function result nested inside AND, then negated."""
    F = get_spark_imports().F
    df = _forecast_frame(spark)

    negated = ~(F.coalesce(F.col("is_forecast"), F.lit(False)) & F.lit(True))

    assert _vids(df, negated) == ["legacy", "obs"]


def test_negated_null_check_functions(spark) -> None:
    """``~isnull`` / ``~isNotNull`` are total, so they stay two-valued."""
    F = get_spark_imports().F
    df = _forecast_frame(spark)

    assert _vids(df, ~F.isnull(F.col("is_forecast"))) == ["fc", "obs"]
    assert _vids(df, ~F.col("is_forecast").isNotNull()) == ["legacy"]


def test_negated_bare_column_still_correct(spark) -> None:
    """The bare-column shape (BUG-023) keeps working — no re-regression."""
    F = get_spark_imports().F
    df = _forecast_frame(spark)

    assert _vids(df, ~F.col("is_forecast")) == ["obs"]
    assert _vids(df, ~(F.col("is_forecast") & F.lit(True))) == ["obs"]
    assert _vids(df, ~(F.col("is_forecast") | F.lit(False))) == ["obs"]
