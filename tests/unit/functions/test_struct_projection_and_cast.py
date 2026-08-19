"""Regression tests for ``F.struct`` in ``select`` and ``.cast(StructType)``.

``df.select(F.struct(...))`` returned ``NULL`` for the whole struct, and the
column was typed ``STRING``. Three independent gaps produced that:

1. :class:`sparkless.core.condition_evaluator.ConditionEvaluator` -- the
   evaluator the lazy ``select`` path uses -- had no ``struct`` case in its
   operation whitelist, and its fall-through returns ``None``. An
   unimplemented operation was therefore indistinguishable from a genuine
   SQL NULL.
2. ``SchemaManager._infer_expression_type`` (the ``select`` type-inference
   table) had no ``struct`` and no ``cast`` case, defaulting both to
   ``StringType``. The parallel ``withColumn`` table did handle ``cast``,
   so the two paths disagreed.
3. ``TypeConverter.cast_to_type`` handled ``ArrayType`` and ``MapType`` but
   not ``StructType``, so a struct-to-struct cast fell through its ``else``
   and returned the value unchanged -- a silent no-op instead of the
   positional rename/retype Spark performs.

Separately, ``F.struct("a", "b")`` treated the *second* string as a literal
(``{"a": ..., "col2": "b"}``) instead of a column reference.

All expectations verified against PySpark 4.0.0 (the DBR 17.3 runtime).
These tests are backend-agnostic and also pass under
``MOCK_SPARK_TEST_BACKEND=pyspark``.
"""

import pytest

from tests.fixtures.spark_imports import get_spark_imports


def _fields(struct_value):
    """Read a struct value as a plain dict.

    PySpark hands back a ``Row``; sparkless a ``dict``. Both index by field
    name and both preserve field order.
    """
    if hasattr(struct_value, "asDict"):
        return struct_value.asDict()
    return dict(struct_value)


def _field_names(struct_value):
    """Ordered field names of a struct value."""
    if hasattr(struct_value, "__fields__"):
        return list(struct_value.__fields__)
    return list(struct_value.keys())


def _field_type(schema, name):
    """Look up a top-level field's dataType by name.

    ``StructType.__getitem__`` accepts a field name in PySpark but only an
    index in sparkless, so go through ``.fields`` for both.
    """
    for field in schema.fields:
        if field.name == name:
            return field.dataType
    raise KeyError(name)


@pytest.fixture
def scores_df(spark):
    """One row of numeric scores, the shape a decision-vector projection has."""
    return spark.createDataFrame([(1.5, 2.5, "k1")], ["urgency", "risk", "key"])


class TestStructInSelect:
    """``select`` must build the struct, not collapse it to NULL."""

    def test_select_struct_of_columns_returns_values(self, scores_df) -> None:
        """The whole point: select(struct(...)) is not NULL."""
        F = get_spark_imports().F

        row = scores_df.select(
            F.struct(F.col("urgency"), F.col("risk")).alias("scores")
        ).collect()[0]

        assert row["scores"] is not None
        assert _fields(row["scores"]) == {"urgency": 1.5, "risk": 2.5}

    def test_select_struct_is_typed_as_struct_not_string(self, scores_df) -> None:
        """The projected column's declared type must be a StructType."""
        imports = get_spark_imports()
        F, StructType = imports.F, imports.StructType

        schema = scores_df.select(
            F.struct(F.col("urgency"), F.col("risk")).alias("scores")
        ).schema

        assert isinstance(_field_type(schema, "scores"), StructType)

    def test_select_and_withcolumn_agree(self, scores_df) -> None:
        """The two projection paths must not disagree about the same expression."""
        F = get_spark_imports().F
        expr = F.struct(F.col("urgency"), F.col("risk"))

        selected = scores_df.select(expr.alias("scores")).collect()[0]["scores"]
        with_col = scores_df.withColumn("scores", expr).collect()[0]["scores"]

        assert _fields(selected) == _fields(with_col)

    def test_string_arguments_are_column_references(self, scores_df) -> None:
        """F.struct("a", "b") names columns; the strings are not literals."""
        F = get_spark_imports().F

        row = scores_df.select(F.struct("urgency", "risk").alias("scores")).collect()[0]

        assert _fields(row["scores"]) == {"urgency": 1.5, "risk": 2.5}

    def test_aliased_arguments_name_their_fields(self, scores_df) -> None:
        """An alias on an argument becomes the struct field name."""
        F = get_spark_imports().F

        row = scores_df.select(
            F.struct(
                F.col("urgency").alias("restock_urgency"),
                F.col("risk").alias("stockout_risk"),
            ).alias("scores")
        ).collect()[0]

        assert _fields(row["scores"]) == {
            "restock_urgency": 1.5,
            "stockout_risk": 2.5,
        }

    def test_unaliased_literal_is_named_by_position(self, scores_df) -> None:
        """An unaliased literal takes the name col<N> for its 1-based position."""
        F = get_spark_imports().F

        row = scores_df.select(
            F.struct(F.col("urgency"), F.lit(0.0)).alias("scores")
        ).collect()[0]

        assert _field_names(row["scores"]) == ["urgency", "col2"]
        assert _fields(row["scores"])["col2"] == 0.0

    def test_aliased_literal_keeps_its_alias(self, scores_df) -> None:
        """An alias on a *literal* argument names the field, exactly as on a column.

        ``Literal.alias`` used to rewrite only the literal's display name and
        leave ``_alias_name`` unset, so ``field_name_for`` saw an unaliased
        literal and fell through to the positional ``col<N>``. The asymmetry
        was the tell: ``F.col("x").alias("n")`` kept its name here while
        ``F.lit(v).alias("n")`` silently did not (#2417).
        """
        F = get_spark_imports().F

        row = scores_df.select(
            F.struct(
                F.lit("axis").alias("axis_type"),
                F.col("key").alias("value_id"),
            ).alias("s")
        ).collect()[0]

        assert _field_names(row["s"]) == ["axis_type", "value_id"]
        assert _fields(row["s"]) == {"axis_type": "axis", "value_id": "k1"}

    def test_aliased_literal_is_named_in_the_schema_too(self, scores_df) -> None:
        """The declared schema must carry the alias, not just the collected value.

        A consumer that reads the field by name off ``df.schema`` is the one
        that breaks in production; asserting only the row value would let a
        schema that still says ``col1`` pass.
        """
        F = get_spark_imports().F

        schema = scores_df.select(
            F.struct(
                F.lit("axis").alias("axis_type"),
                F.col("key").alias("value_id"),
            ).alias("s")
        ).schema

        assert [f.name for f in _field_type(schema, "s").fields] == [
            "axis_type",
            "value_id",
        ]

    def test_aliased_and_unaliased_literals_mix(self, scores_df) -> None:
        """Aliasing one literal must not renumber the positional name of another."""
        F = get_spark_imports().F

        row = scores_df.select(
            F.struct(
                F.lit("axis").alias("axis_type"),
                F.lit(0.0),
                F.col("key").alias("value_id"),
            ).alias("s")
        ).collect()[0]

        assert _field_names(row["s"]) == ["axis_type", "col2", "value_id"]

    def test_expression_argument_is_evaluated(self, scores_df) -> None:
        """A computed argument contributes its computed value."""
        F = get_spark_imports().F

        row = scores_df.select(
            F.struct((F.col("urgency") + F.col("risk")).alias("total")).alias("scores")
        ).collect()[0]

        assert _fields(row["scores"]) == {"total": 4.0}


class TestStructCast:
    """``.cast(StructType)`` renames and retypes positionally."""

    def test_cast_to_matching_struct_preserves_values(self, scores_df) -> None:
        """The reported failure: casting an assembled struct yielded NULL."""
        imports = get_spark_imports()
        F, StructType, StructField, DoubleType = (
            imports.F,
            imports.StructType,
            imports.StructField,
            imports.DoubleType,
        )
        target = StructType(
            [
                StructField("urgency", DoubleType()),
                StructField("risk", DoubleType()),
            ]
        )

        row = scores_df.select(
            F.struct(F.col("urgency"), F.col("risk")).cast(target).alias("scores")
        ).collect()[0]

        assert row["scores"] is not None
        assert _fields(row["scores"]) == {"urgency": 1.5, "risk": 2.5}

    def test_cast_renames_fields_positionally(self, scores_df) -> None:
        """Field N of the source becomes field N of the target, under its name."""
        imports = get_spark_imports()
        F, StructType, StructField, DoubleType = (
            imports.F,
            imports.StructType,
            imports.StructField,
            imports.DoubleType,
        )
        target = StructType(
            [
                StructField("restock_urgency", DoubleType()),
                StructField("stockout_risk", DoubleType()),
            ]
        )

        row = scores_df.select(
            F.struct(F.col("urgency"), F.col("risk")).cast(target).alias("scores")
        ).collect()[0]

        assert _fields(row["scores"]) == {
            "restock_urgency": 1.5,
            "stockout_risk": 2.5,
        }

    def test_cast_retypes_fields(self, scores_df) -> None:
        """A struct cast converts each field to the declared target type."""
        imports = get_spark_imports()
        F, StructType, StructField, StringType = (
            imports.F,
            imports.StructType,
            imports.StructField,
            imports.StringType,
        )
        target = StructType(
            [
                StructField("urgency", StringType()),
                StructField("risk", StringType()),
            ]
        )

        row = scores_df.select(
            F.struct(F.col("urgency"), F.col("risk")).cast(target).alias("scores")
        ).collect()[0]

        assert _fields(row["scores"]) == {"urgency": "1.5", "risk": "2.5"}

    def test_cast_result_carries_the_target_schema(self, scores_df) -> None:
        """The declared cast type wins over the StringType default."""
        imports = get_spark_imports()
        F, StructType, StructField, DoubleType = (
            imports.F,
            imports.StructType,
            imports.StructField,
            imports.DoubleType,
        )
        target = StructType(
            [
                StructField("urgency", DoubleType()),
                StructField("risk", DoubleType()),
            ]
        )

        schema = scores_df.select(
            F.struct(F.col("urgency"), F.col("risk")).cast(target).alias("scores")
        ).schema

        assert isinstance(_field_type(schema, "scores"), StructType)
        assert [f.name for f in _field_type(schema, "scores").fields] == [
            "urgency",
            "risk",
        ]

    def test_scalar_cast_in_select_is_typed_by_its_target(self, scores_df) -> None:
        """A plain cast in select() was also typed STRING; the target type wins."""
        imports = get_spark_imports()
        F, IntegerType = imports.F, imports.IntegerType

        projected = scores_df.select(F.col("urgency").cast("int").alias("n"))

        assert isinstance(_field_type(projected.schema, "n"), IntegerType)
        assert projected.collect()[0]["n"] == 1
