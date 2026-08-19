"""Regression tests for ``F.to_json``.

``df.select(F.to_json(F.struct(...)))`` returned NULL for every row.
Solya-app/solya-data-platform#2417 reports it from
``pipelines/shared/data_quality.py:165``, where the serialised struct is what
carries the quality report -- so the report shipped empty rather than absent.

The issue attributes it to sparkless typing a ``F.struct(...)`` column as
``StringType``. That is **not** the cause: ``df.schema`` already reports
``StructType`` (see ``test_struct_projection_and_cast.py``). The cause is that
``ConditionEvaluator``, the evaluator the lazy ``select`` path uses, has no
``to_json`` handler and its fall-through returns ``None``.

Rendering measured against PySpark 4.0.0 (``local[1]``):

=============================================  =============================
expression                                     result
=============================================  =============================
``to_json(struct(sku, dept))``                 ``{"sku":"A","dept":"eng"}``
                                               -- no spaces, field order kept
``to_json(struct(NULL as z, 1 as a))``         ``{"a":1}`` -- a NULL **struct
                                               field** is omitted entirely
``to_json(array_col)`` with a NULL element     ``["x",null]`` -- a NULL
                                               **array element** is kept
``to_json(map)`` with a NULL value             ``{"k":null}`` -- a NULL **map
                                               value** is kept
``to_json(NULL array_col)``                    NULL
=============================================  =============================

The three NULL rules differ from each other, so each has its own test.
"""

import datetime

import pytest

from tests.fixtures.spark_imports import get_spark_imports


@pytest.fixture
def report_df(spark):
    """One row carrying every value shape ``to_json`` renders differently."""
    imports = get_spark_imports()
    StructType, StructField = imports.StructType, imports.StructField
    StringType, ArrayType = imports.StringType, imports.ArrayType
    BooleanType, DateType = imports.BooleanType, imports.DateType

    schema = StructType(
        [
            StructField("sku", StringType()),
            StructField("dept", StringType()),
            StructField("quoted", StringType()),
            StructField("missing", StringType()),
            StructField("accented", StringType()),
            StructField("flag", BooleanType()),
            StructField("day", DateType()),
            StructField("tags", ArrayType(StringType())),
        ]
    )
    return spark.createDataFrame(
        [("A", "eng", 'a"q', None, "é", True, datetime.date(2026, 1, 2), ["x", None])],
        schema,
    )


def _first(df, expression):
    """Collect a single projected expression from the first row."""
    return df.select(expression).collect()[0][0]


class TestToJsonStruct:
    """``to_json`` of a struct produces the JSON object, not NULL."""

    def test_struct_is_serialised(self, report_df) -> None:
        """The reported failure: every row came back NULL."""
        F = get_spark_imports().F

        assert (
            _first(report_df, F.to_json(F.struct("sku", "dept")))
            == '{"sku":"A","dept":"eng"}'
        )

    def test_field_order_is_the_struct_order(self, report_df) -> None:
        """Not alphabetical, and not the source DataFrame's column order."""
        F = get_spark_imports().F

        assert (
            _first(report_df, F.to_json(F.struct("dept", "sku")))
            == '{"dept":"eng","sku":"A"}'
        )

    def test_null_struct_field_is_omitted(self, report_df) -> None:
        """Spark drops a NULL struct field rather than writing ``null``.

        This is the rule that differs from arrays and maps below, so getting
        it from the array rule would produce valid-looking, wrong JSON.
        """
        F = get_spark_imports().F

        assert _first(report_df, F.to_json(F.struct("missing", "sku"))) == '{"sku":"A"}'

    def test_nested_struct_is_serialised(self, report_df) -> None:
        F = get_spark_imports().F

        assert (
            _first(
                report_df,
                F.to_json(F.struct(F.struct(F.col("sku").alias("inner")).alias("o"))),
            )
            == '{"o":{"inner":"A"}}'
        )

    def test_scalars_render_as_json_types(self, report_df) -> None:
        """A boolean is ``true``, a date is a quoted ISO day, and ``"`` escapes."""
        F = get_spark_imports().F

        assert (
            _first(report_df, F.to_json(F.struct("flag", "day", "quoted")))
            == '{"flag":true,"day":"2026-01-02","quoted":"a\\"q"}'
        )

    def test_unicode_is_not_escaped(self, report_df) -> None:
        """Spark emits the character, not ``\\u00e9``."""
        F = get_spark_imports().F

        assert _first(report_df, F.to_json(F.struct("accented"))) == '{"accented":"é"}'

    def test_alias_names_the_column(self, report_df) -> None:
        """``F.to_json(x).alias("j")`` produces a column called ``j``.

        Schema inference re-derived the *default* name for to_json and to_csv,
        discarding whatever the caller had asked for -- in ``select`` and in
        ``withColumn`` alike.
        """
        F = get_spark_imports().F

        assert report_df.select(F.to_json(F.struct("sku")).alias("j")).columns == ["j"]

    def test_select_and_withcolumn_agree(self, report_df) -> None:
        """The two projection paths must not disagree about the same expression."""
        F = get_spark_imports().F
        expression = F.to_json(F.struct("sku", "dept"))

        selected = _first(report_df, expression)
        with_column = report_df.withColumn("j", expression).collect()[0]["j"]

        assert selected == with_column


class TestToJsonCollections:
    """Arrays and maps keep their NULLs, unlike struct fields."""

    def test_null_array_element_is_kept(self, report_df) -> None:
        """``["x", null]`` -- dropping it would silently shorten the array."""
        F = get_spark_imports().F

        assert _first(report_df, F.to_json(F.col("tags"))) == '["x",null]'

    def test_array_inside_a_struct_keeps_its_nulls(self, report_df) -> None:
        """The struct rule must not reach into the array it contains."""
        F = get_spark_imports().F

        assert _first(report_df, F.to_json(F.struct("tags"))) == '{"tags":["x",null]}'

    def test_map_is_serialised(self, report_df) -> None:
        F = get_spark_imports().F

        assert (
            _first(report_df, F.to_json(F.create_map(F.lit("k"), F.lit(1))))
            == '{"k":1}'
        )


class TestToJsonNull:
    """A NULL input column is NULL, not the string ``"null"``."""

    def test_null_input_is_null(self, spark) -> None:
        imports = get_spark_imports()
        F, StructType, StructField = imports.F, imports.StructType, imports.StructField
        ArrayType, StringType = imports.ArrayType, imports.StringType

        df = spark.createDataFrame(
            [(None,)], StructType([StructField("tags", ArrayType(StringType()))])
        )

        assert _first(df, F.to_json(F.col("tags"))) is None
