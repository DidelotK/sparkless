"""BUG-035: binding a schema must not alias the caller's schema object graph.

``spark.createDataFrame(data, schema=...)`` used to store the caller's
``StructType`` -- and therefore the caller's ``StructField`` and ``DataType``
objects -- directly on the DataFrame.  Any in-place mutation of a bound schema
then leaked back into the caller.

The leak is unbounded because of a very common idiom:

    _SOURCE = StructType([...])
    _DERIVED = StructType([StructField("k", StringType()), *_SOURCE.fields])

Unpacking shares the ``StructField`` *objects* between the two schemas, so a
mutation reached through ``_DERIVED`` also corrupts ``_SOURCE`` -- for every
later user in the same process.  Under ``pytest -n auto`` that surfaces as an
unrelated test failing on a worker that happens to run after the polluter,
which is the worst possible signature.

Reference behaviour captured from real PySpark 4.0.0 on OpenJDK 21::

    REF A1 df.schema is caller schema                : False
    REF A2 df.schema.fields is caller list           : False
    REF A3 df.schema.fields[0] is caller field       : False
    REF A4 df.schema == caller schema                : True
    REF A5 repeated df.schema is stable object       : True
    REF B1 caller field names after mutating df.schema: ['a', 'b']
    REF B2 df.schema field names after mutating it   : ['a', 'b', 'leaked']
    REF D1 source schema after binding a derived one : unchanged

PySpark round-trips the schema through the JVM, so the DataFrame always owns a
fresh object graph.  Sparkless now copies at the same boundary.
"""

from typing import Any, Iterator, List, Tuple

import pytest

from sparkless import SparkSession
from sparkless.spark_types import (
    ArrayType,
    DoubleType,
    IntegerType,
    MapType,
    StringType,
    StructField,
    StructType,
    copy_schema,
)


@pytest.fixture  # type: ignore[misc,untyped-decorator]
def sparkless_session() -> Iterator[SparkSession]:
    """A private sparkless session.

    These tests assert on sparkless-specific APIs (``add_field``,
    ``ArrayType.element_type``) and on object identity, so they deliberately
    do not go through the backend-switching ``spark`` fixture.  The PySpark
    reference for every assertion is recorded in the module docstring instead.
    """
    session = SparkSession("bug035_schema_binding")
    try:
        yield session
    finally:
        session.stop()


def _source_schema() -> StructType:
    """A schema whose field objects a caller may share with other schemas."""
    return StructType(
        [
            StructField("a", DoubleType(), nullable=True),
            StructField("b", IntegerType(), nullable=True),
            StructField("flag", StringType(), nullable=True),
        ]
    )


def _names(schema: StructType) -> List[str]:
    return [field.name for field in schema.fields]


def _shape(schema: StructType) -> List[Tuple[str, str, bool]]:
    return [(f.name, repr(f.dataType), f.nullable) for f in schema.fields]


class TestBoundSchemaIsNotTheCallersObject:
    """The DataFrame must own its schema, exactly as PySpark does."""

    def test_bound_schema_is_a_distinct_object(
        self, sparkless_session: SparkSession
    ) -> None:
        """``df.schema is caller_schema`` is False (PySpark reference A1)."""
        schema = _source_schema()
        df = sparkless_session.createDataFrame([(1.0, 2, "x")], schema=schema)

        assert df.schema is not schema

    def test_bound_schema_does_not_share_the_fields_list(
        self, sparkless_session: SparkSession
    ) -> None:
        """The DataFrame gets its own ``fields`` list (reference A2)."""
        schema = _source_schema()
        df = sparkless_session.createDataFrame([(1.0, 2, "x")], schema=schema)

        assert df.schema.fields is not schema.fields

    def test_bound_schema_does_not_share_field_objects(
        self, sparkless_session: SparkSession
    ) -> None:
        """No ``StructField`` object is shared with the caller (reference A3).

        This is the object that the ``[*OTHER.fields, ...]`` idiom aliases, so
        it is the one that must not be reachable from the DataFrame.
        """
        schema = _source_schema()
        df = sparkless_session.createDataFrame([(1.0, 2, "x")], schema=schema)

        for bound, original in zip(df.schema.fields, schema.fields):
            assert bound is not original

    def test_bound_schema_does_not_share_data_type_objects(
        self, sparkless_session: SparkSession
    ) -> None:
        """``DataType`` objects are copied too -- they carry ``nullable``."""
        schema = _source_schema()
        df = sparkless_session.createDataFrame([(1.0, 2, "x")], schema=schema)

        for bound, original in zip(df.schema.fields, schema.fields):
            assert bound.dataType is not original.dataType

    def test_bound_schema_still_compares_equal(
        self, sparkless_session: SparkSession
    ) -> None:
        """Copying must not change what the schema *means* (reference A4)."""
        schema = _source_schema()
        df = sparkless_session.createDataFrame([(1.0, 2, "x")], schema=schema)

        assert df.schema == schema
        assert _shape(df.schema) == _shape(schema)

    def test_repeated_schema_access_returns_the_same_object(
        self, sparkless_session: SparkSession
    ) -> None:
        """``df.schema is df.schema`` stays True (reference A5).

        Guards against "fixing" the aliasing by copying on every read, which
        would silently break code that mutates ``df.schema``.
        """
        schema = _source_schema()
        df = sparkless_session.createDataFrame([(1.0, 2, "x")], schema=schema)

        assert df.schema is df.schema


class TestMutatingTheBoundSchemaDoesNotReachTheCaller:
    """The actual corruption channel, and the one the fix closes."""

    def test_add_field_on_bound_schema_does_not_leak(
        self, sparkless_session: SparkSession
    ) -> None:
        """Reference B1/B2: the caller keeps its own field list."""
        schema = _source_schema()
        df = sparkless_session.createDataFrame([(1.0, 2, "x")], schema=schema)

        df.schema.add_field(StructField("leaked", StringType()))

        assert _names(schema) == ["a", "b", "flag"]
        assert "leaked" in _names(df.schema)

    def test_field_mutation_does_not_reach_a_schema_sharing_that_field(
        self, sparkless_session: SparkSession
    ) -> None:
        """The downstream failure, reduced.

        ``derived`` shares ``StructField`` objects with ``source``.  Before the
        fix the DataFrame bound those very objects, so anything mutating the
        bound schema's fields corrupted ``source`` -- which other tests in the
        same process still use.
        """
        source = _source_schema()
        before = _shape(source)
        derived = StructType(
            [StructField("k", StringType(), nullable=False), *source.fields]
        )
        assert derived.fields[1] is source.fields[0], (
            "precondition: unpacking shares the field objects"
        )

        df = sparkless_session.createDataFrame([("k1", 1.0, 2, "x")], schema=derived)

        # Mutate the bound schema the way sparkless internals may.
        df.schema.fields[1].nullable = False
        df.schema.fields[1].dataType.nullable = False

        assert _shape(source) == before

    def test_nested_types_are_not_shared(self, sparkless_session: SparkSession) -> None:
        """Array/map element types are copied recursively, not aliased."""
        element = StringType()
        schema = StructType(
            [
                StructField("arr", ArrayType(element), nullable=True),
                StructField("m", MapType(StringType(), IntegerType()), nullable=True),
            ]
        )
        df = sparkless_session.createDataFrame([(["x"], {"k": 1})], schema=schema)

        bound_array = df.schema.fields[0].dataType
        bound_map = df.schema.fields[1].dataType
        assert bound_array.element_type is not element
        assert bound_map.key_type is not schema.fields[1].dataType.key_type
        assert bound_map.value_type is not schema.fields[1].dataType.value_type

    def test_same_schema_object_bound_twice_yields_independent_schemas(
        self, sparkless_session: SparkSession
    ) -> None:
        """A module-level schema reused across tests stays reusable."""
        schema = _source_schema()
        df1 = sparkless_session.createDataFrame([(1.0, 2, "x")], schema=schema)
        df2 = sparkless_session.createDataFrame([(3.0, 4, "y")], schema=schema)

        df1.schema.add_field(StructField("only_on_df1", StringType()))

        assert _names(df2.schema) == ["a", "b", "flag"]
        assert _names(schema) == ["a", "b", "flag"]


class TestBindingStillWorks:
    """Copying must be transparent to every observable behaviour."""

    def test_values_round_trip(self, sparkless_session: SparkSession) -> None:
        """Data is unaffected by the schema copy."""
        schema = _source_schema()
        rows = sparkless_session.createDataFrame(
            [(1.0, 2, "x"), (3.0, 4, "y")], schema=schema
        )

        collected = [(r["a"], r["b"], r["flag"]) for r in rows.collect()]
        assert collected == [(1.0, 2, "x"), (3.0, 4, "y")]

    def test_empty_dataframe_preserves_the_schema(
        self, sparkless_session: SparkSession
    ) -> None:
        """An explicit schema on empty data survives the copy."""
        schema = _source_schema()
        df = sparkless_session.createDataFrame([], schema=schema)

        assert _shape(df.schema) == _shape(schema)
        assert df.collect() == []

    @pytest.mark.parametrize(  # type: ignore[misc,untyped-decorator]
        "data_type",
        [
            StringType(),
            IntegerType(),
            DoubleType(),
            ArrayType(StringType()),
            MapType(StringType(), IntegerType()),
            StructType([StructField("inner", StringType(), nullable=True)]),
        ],
    )
    def test_copy_schema_is_equal_but_disjoint(self, data_type: Any) -> None:
        """``copy_schema`` round-trips every supported type shape."""
        schema = StructType([StructField("c", data_type, nullable=True)])
        copied = copy_schema(schema)

        assert copied == schema
        assert copied is not schema
        assert copied.fields[0] is not schema.fields[0]
        assert copied.fields[0].dataType is not schema.fields[0].dataType
        assert repr(copied.fields[0].dataType) == repr(schema.fields[0].dataType)

    def test_copy_schema_preserves_metadata_without_sharing_it(self) -> None:
        """Field metadata is copied, not aliased."""
        metadata = {"comment": "hello"}
        schema = StructType(
            [StructField("c", StringType(), nullable=True, metadata=metadata)]
        )
        copied = copy_schema(schema)

        assert copied.fields[0].metadata == metadata
        assert copied.fields[0].metadata is not schema.fields[0].metadata

    def test_copy_schema_handles_the_empty_schema(self) -> None:
        """An empty schema copies to an empty schema, not to None."""
        copied = copy_schema(StructType([]))

        assert isinstance(copied, StructType)
        assert copied.fields == []


def test_session_level_repro_of_the_reported_corruption() -> None:
    """End-to-end repro on a private session, no fixtures involved.

    Mirrors the downstream shape exactly: a module-level schema, a second
    schema derived from it by unpacking, and a bind of the derived one.
    """
    spark = SparkSession("bug035")
    try:
        module_level = _source_schema()
        snapshot = _shape(module_level)
        derived = StructType(
            [StructField("shop_id", StringType(), nullable=False), *module_level.fields]
        )

        df = spark.createDataFrame([("s1", 1.0, 2, "x")], schema=derived)
        df.collect()
        df.schema.add_field(StructField("injected", StringType()))
        df.schema.fields[1].nullable = False

        assert _shape(module_level) == snapshot
        assert len(module_level.fields) == 3
    finally:
        spark.stop()
