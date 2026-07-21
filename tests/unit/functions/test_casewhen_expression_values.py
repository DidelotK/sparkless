"""Regression tests for expressions in ``when`` / ``otherwise`` value positions.

``CaseWhen._evaluate_column_operation_value`` dispatched on a hardcoded list
of five operations -- unary ``+``/``-``, binary arithmetic, and
``create_map``. Everything else hit an ``else`` that returned
``operation.column``'s value, **discarding the operation itself**:

- ``F.when(c, x).otherwise(F.datediff(a, b) >= 30)`` returned the day count
  (``134``) instead of ``True``;
- ``.otherwise(F.upper(col))`` returned the string unchanged;
- ``.otherwise(col == lit("a"))`` returned the string being compared.

The returned value was of a plausible type for the underlying column, so
nothing downstream flagged it. The branch now delegates to the shared
``ConditionEvaluator``.

The result *type* was wrong too: ``get_result_type`` mapped every
non-arithmetic ``ColumnOperation`` to ``StringType``, and the ``withColumn``
schema path did not consult ``get_result_type`` at all, falling through to a
literal-inference helper that answers ``StringType`` for anything it does not
recognise.

All expectations verified against PySpark 4.0.0 (the DBR 17.3 runtime).
These tests are backend-agnostic and also pass under
``MOCK_SPARK_TEST_BACKEND=pyspark``.
"""

import datetime

import pytest

from tests.fixtures.spark_imports import get_spark_imports


def _field_type(schema, name):
    """Look up a top-level field's dataType by name (backend-agnostic)."""
    for field in schema.fields:
        if field.name == name:
            return field.dataType
    raise KeyError(name)


SNAPSHOT_DATE = datetime.date(2024, 6, 1)
AGED_THRESHOLD_DAYS = 90


@pytest.fixture
def stock_df(spark):
    """Positions with a stale, a fresh, and a never-sold last-sale date."""
    imports = get_spark_imports()
    StructType, StructField, StringType, DateType = (
        imports.StructType,
        imports.StructField,
        imports.StringType,
        imports.DateType,
    )
    schema = StructType(
        [
            StructField("sku", StringType()),
            StructField("last_sale_date", DateType()),
        ]
    )
    return spark.createDataFrame(
        [
            ("stale", datetime.date(2024, 1, 19)),  # 134 days -> aged
            ("fresh", datetime.date(2024, 5, 30)),  # 2 days   -> not aged
            ("never", None),  # NULL     -> not aged
        ],
        schema,
    )


def _aged_stock_flag(F):
    """The production shape: a datediff comparison in the ``otherwise`` branch."""
    days_since_sale = F.datediff(F.lit(SNAPSHOT_DATE), F.col("last_sale_date"))
    return F.when(F.col("last_sale_date").isNull(), F.lit(False)).otherwise(
        days_since_sale >= F.lit(AGED_THRESHOLD_DAYS)
    )


class TestComparisonInOtherwise:
    """A comparison in the else branch must evaluate to a boolean."""

    def test_datediff_comparison_yields_booleans(self, stock_df) -> None:
        """The reported failure: the day count leaked out instead of the flag."""
        F = get_spark_imports().F

        rows = stock_df.withColumn("aged", _aged_stock_flag(F)).collect()
        flags = {row["sku"]: row["aged"] for row in rows}

        assert flags == {"stale": True, "fresh": False, "never": False}

    def test_datediff_comparison_is_not_the_raw_day_count(self, stock_df) -> None:
        """Guard the specific regression: 134 must never be returned as a flag."""
        F = get_spark_imports().F

        aged = stock_df.withColumn("aged", _aged_stock_flag(F)).collect()[0]["aged"]

        assert isinstance(aged, bool)
        assert aged is not 134  # noqa: F632 - identity is the point, 134 is the bug

    def test_result_column_is_typed_boolean(self, stock_df) -> None:
        """A CASE WHEN over booleans is BOOLEAN, not STRING."""
        imports = get_spark_imports()
        F, BooleanType = imports.F, imports.BooleanType

        schema = stock_df.withColumn("aged", _aged_stock_flag(F)).schema

        assert isinstance(_field_type(schema, "aged"), BooleanType)

    def test_equality_comparison_in_otherwise(self, stock_df) -> None:
        """A plain ``==`` in the else branch is evaluated, not short-circuited."""
        F = get_spark_imports().F

        rows = stock_df.withColumn(
            "is_stale",
            F.when(F.lit(False), F.lit(False)).otherwise(
                F.col("sku") == F.lit("stale")
            ),
        ).collect()

        assert {row["sku"]: row["is_stale"] for row in rows} == {
            "stale": True,
            "fresh": False,
            "never": False,
        }

    def test_comparison_in_then_branch(self, stock_df) -> None:
        """The ``then`` value position has the same evaluator, so test it too."""
        F = get_spark_imports().F

        rows = stock_df.withColumn(
            "is_stale",
            F.when(F.lit(True), F.col("sku") == F.lit("stale")).otherwise(F.lit(False)),
        ).collect()

        assert {row["sku"]: row["is_stale"] for row in rows} == {
            "stale": True,
            "fresh": False,
            "never": False,
        }


class TestScalarFunctionsInOtherwise:
    """Scalar functions in a value position were dropped just as silently."""

    def test_upper_is_applied(self, stock_df) -> None:
        """``F.upper`` returned the unmodified string."""
        F = get_spark_imports().F

        rows = stock_df.withColumn(
            "name",
            F.when(F.lit(False), F.lit("never")).otherwise(F.upper(F.col("sku"))),
        ).collect()

        assert sorted(row["name"] for row in rows) == ["FRESH", "NEVER", "STALE"]

    def test_isnull_predicate_is_applied(self, stock_df) -> None:
        """``isNull()`` returned the date rather than a boolean."""
        F = get_spark_imports().F

        rows = stock_df.withColumn(
            "missing",
            F.when(F.lit(False), F.lit(False)).otherwise(
                F.col("last_sale_date").isNull()
            ),
        ).collect()

        assert {row["sku"]: row["missing"] for row in rows} == {
            "stale": False,
            "fresh": False,
            "never": True,
        }

    def test_arithmetic_still_works(self, spark) -> None:
        """The branches that already worked must keep working."""
        F = get_spark_imports().F
        df = spark.createDataFrame([(10,), (20,)], ["n"])

        rows = df.withColumn(
            "doubled",
            F.when(F.col("n") > F.lit(15), F.col("n") * F.lit(2)).otherwise(
                F.col("n") + F.lit(1)
            ),
        ).collect()

        assert sorted(row["doubled"] for row in rows) == [11, 40]
