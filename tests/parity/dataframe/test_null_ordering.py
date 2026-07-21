"""BUG-041: NULL placement in ORDER BY must match Spark's defaults.

Spark orders ``ASC NULLS FIRST`` and ``DESC NULLS LAST`` -- which is precisely
why ``asc_nulls_last()`` and ``desc_nulls_first()`` exist as explicit variants.
Sparkless defaulted every ascending sort to NULLS LAST, which moved rows under
every window function (``row_number``, ``rank``, ``lag``/``lead``, running
aggregates, ``first``/``last``), silently changing computed values rather than
just presentation.

Reference behaviour in this file was captured from real PySpark 4.0.0 on
OpenJDK 21 (the DBR 17.3 pairing). Every test runs against the backend-agnostic
``spark`` fixture, so ``MOCK_SPARK_TEST_BACKEND=pyspark`` re-checks the same
assertions against real Spark.

Note on row order after a window function: Spark does not guarantee the output
row order of ``df.withColumn(w)``, and sparkless preserves input order where
Spark happens to emit sort order. These tests therefore assert the *computed
value per row*, keyed by id -- the thing BUG-041 corrupted -- and never the
emitted row sequence.
"""

from typing import Any, Dict, List, Tuple

import pytest

from tests.fixtures.parity_base import ParityTestBase
from tests.fixtures.spark_imports import get_spark_imports


def _schema() -> Any:
    """Build the shared test schema for either backend."""
    imports = get_spark_imports()
    return imports.StructType(
        [
            imports.StructField("id", imports.StringType()),
            imports.StructField("k", imports.IntegerType()),
            imports.StructField("g", imports.StringType()),
        ]
    )


# id, k (ordering key, one NULL), g (partition key, two NULLs)
ROWS: List[Tuple[Any, ...]] = [
    ("a", 1, "x"),
    ("b", 2, "x"),
    ("c", 4, None),
    ("d", None, "x"),
    ("e", 2, None),
]


def _ids(df: Any) -> List[str]:
    """Collect the id column in emitted row order."""
    return [row["id"] for row in df.collect()]


def _by_id(df: Any, column: str) -> Dict[str, Any]:
    """Collect ``column`` keyed by id, so row order does not matter."""
    return {row["id"]: row[column] for row in df.collect()}


class TestNullOrderingDefaults(ParityTestBase):
    """Plain ``orderBy``/``sort``: NULLs first ascending, last descending."""

    @pytest.mark.parametrize(  # type: ignore[misc,untyped-decorator]
        "spec,expected",
        [
            # Bare string key -> ASC NULLS FIRST.
            ("plain_string", ["d", "a", "b", "e", "c"]),
            # Column.asc() -> ASC NULLS FIRST.
            ("col_asc", ["d", "a", "b", "e", "c"]),
            # Column.desc() -> DESC NULLS LAST.
            ("col_desc", ["c", "b", "e", "a", "d"]),
            # ascending=False on a string key -> DESC NULLS LAST.
            ("string_descending_kwarg", ["c", "b", "e", "a", "d"]),
            # F.asc()/F.desc() free functions take the same defaults.
            ("f_asc", ["d", "a", "b", "e", "c"]),
            ("f_desc", ["c", "b", "e", "a", "d"]),
            # sort() is an alias of orderBy() and must agree.
            ("sort_alias", ["d", "a", "b", "e", "c"]),
        ],
    )
    def test_default_null_placement(
        self, spark: Any, spec: str, expected: List[str]
    ) -> None:
        """Default NULL placement follows the sort direction, per Spark."""
        imports = get_spark_imports()
        F = imports.F
        df = spark.createDataFrame(ROWS, _schema())

        ordered = {
            "plain_string": lambda: df.orderBy("k"),
            "col_asc": lambda: df.orderBy(F.col("k").asc()),
            "col_desc": lambda: df.orderBy(F.col("k").desc()),
            "string_descending_kwarg": lambda: df.orderBy("k", ascending=False),
            "f_asc": lambda: df.orderBy(F.asc("k")),
            "f_desc": lambda: df.orderBy(F.desc("k")),
            "sort_alias": lambda: df.sort("k"),
        }[spec]()

        assert _ids(ordered) == expected

    @pytest.mark.parametrize(  # type: ignore[misc,untyped-decorator]
        "variant,expected",
        [
            ("asc_nulls_first", ["d", "a", "b", "e", "c"]),
            ("asc_nulls_last", ["a", "b", "e", "c", "d"]),
            ("desc_nulls_first", ["d", "c", "b", "e", "a"]),
            ("desc_nulls_last", ["c", "b", "e", "a", "d"]),
        ],
    )
    def test_explicit_nulls_variants_override_the_default(
        self, spark: Any, variant: str, expected: List[str]
    ) -> None:
        """Explicit ``*_nulls_*`` variants win over the direction default.

        ``asc_nulls_last`` and ``desc_nulls_first`` are the two that differ from
        the default, so this pins that the override is real and not a no-op
        that happens to agree with the new default.
        """
        imports = get_spark_imports()
        F = imports.F
        df = spark.createDataFrame(ROWS, _schema())

        ordered = df.orderBy(getattr(F.col("k"), variant)())
        assert _ids(ordered) == expected

    def test_string_column_nulls_sort_first_ascending(self, spark: Any) -> None:
        """NULL placement is not numeric-only: a string key behaves the same."""
        imports = get_spark_imports()
        schema = imports.StructType(
            [
                imports.StructField("id", imports.StringType()),
                imports.StructField("s", imports.StringType()),
            ]
        )
        df = spark.createDataFrame(
            [("a", "p"), ("b", None), ("c", "q"), ("d", "r"), ("e", None)],
            schema,
        )
        assert _ids(df.orderBy("s")) == ["b", "e", "a", "c", "d"]

    def test_all_null_column_keeps_input_order(self, spark: Any) -> None:
        """A key that is NULL for every row leaves the rows in input order."""
        imports = get_spark_imports()
        F = imports.F
        schema = imports.StructType(
            [
                imports.StructField("id", imports.StringType()),
                imports.StructField("k", imports.IntegerType()),
            ]
        )
        df = spark.createDataFrame([("a", None), ("b", None), ("c", None)], schema)

        assert _ids(df.orderBy("k")) == ["a", "b", "c"]
        assert _ids(df.orderBy(F.col("k").desc())) == ["a", "b", "c"]


class TestMultiKeyNullOrdering(ParityTestBase):
    """Each key carries its own direction *and* its own NULL placement."""

    def test_two_ascending_keys(self, spark: Any) -> None:
        """Both keys ASC: NULLs first on the partition key and the sort key."""
        df = spark.createDataFrame(ROWS, _schema())
        # g ASC NULLS FIRST -> (c, e) then (a, b, d); within each, k ASC NULLS
        # FIRST -> e(2) before c(4); d(NULL) before a(1) before b(2).
        assert _ids(df.orderBy("g", "k")) == ["e", "c", "d", "a", "b"]

    def test_desc_then_asc_keeps_per_key_null_placement(self, spark: Any) -> None:
        """``ORDER BY g DESC, k ASC``: NULLs last on g, first on k."""
        imports = get_spark_imports()
        F = imports.F
        df = spark.createDataFrame(ROWS, _schema())
        assert _ids(df.orderBy(F.col("g").desc(), F.col("k").asc())) == [
            "d",
            "a",
            "b",
            "e",
            "c",
        ]

    def test_asc_then_desc_keeps_per_key_null_placement(self, spark: Any) -> None:
        """``ORDER BY g ASC, k DESC``: NULLs first on g, last on k."""
        imports = get_spark_imports()
        F = imports.F
        df = spark.createDataFrame(ROWS, _schema())
        assert _ids(df.orderBy(F.col("g").asc(), F.col("k").desc())) == [
            "c",
            "e",
            "b",
            "a",
            "d",
        ]

    def test_explicit_variant_on_one_key_only(self, spark: Any) -> None:
        """An explicit variant on one key does not leak onto the other."""
        imports = get_spark_imports()
        F = imports.F
        df = spark.createDataFrame(ROWS, _schema())
        # g ASC NULLS LAST -> (a, b, d) then (c, e); k keeps ASC NULLS FIRST.
        assert _ids(df.orderBy(F.col("g").asc_nulls_last(), F.col("k").asc())) == [
            "d",
            "a",
            "b",
            "e",
            "c",
        ]

    def test_descending_kwarg_applies_to_every_string_key(self, spark: Any) -> None:
        """``ascending=False`` makes both keys DESC NULLS LAST."""
        df = spark.createDataFrame(ROWS, _schema())
        assert _ids(df.orderBy("g", "k", ascending=False)) == [
            "b",
            "a",
            "d",
            "c",
            "e",
        ]


class TestWindowOrderingMovesRows(ParityTestBase):
    """The NULL-key row moves to the front of every ordered window.

    These are the assertions BUG-041 actually broke: the *values* the window
    functions compute, not the order rows are printed in.
    """

    def test_row_number_puts_the_null_key_row_first(self, spark: Any) -> None:
        """``row_number`` over ``ASC`` numbers the NULL-key row 1."""
        imports = get_spark_imports()
        F, Window = imports.F, imports.Window
        df = spark.createDataFrame(ROWS, _schema())

        result = _by_id(
            df.withColumn("rn", F.row_number().over(Window.orderBy("k"))), "rn"
        )
        assert result == {"d": 1, "a": 2, "b": 3, "e": 4, "c": 5}

    def test_rank_and_dense_rank_over_ascending_order(self, spark: Any) -> None:
        """``rank``/``dense_rank`` see the NULL row as the first group."""
        imports = get_spark_imports()
        F, Window = imports.F, imports.Window
        df = spark.createDataFrame(ROWS, _schema())
        w = Window.orderBy("k")

        assert _by_id(df.withColumn("r", F.rank().over(w)), "r") == {
            "d": 1,
            "a": 2,
            "b": 3,
            "e": 3,
            "c": 5,
        }
        assert _by_id(df.withColumn("r", F.dense_rank().over(w)), "r") == {
            "d": 1,
            "a": 2,
            "b": 3,
            "e": 3,
            "c": 4,
        }

    def test_running_sum_includes_the_null_key_row_first(self, spark: Any) -> None:
        """A running total over ``ASC`` starts at the NULL-key row.

        This is the shape from the BUG-041 report: with keys ``[1, 2, 4, NULL]``
        every subsequent row's running total depends on where the NULL lands.
        The NULL-key row's own total is NULL (a SUM over only NULL), and the
        tied k=2 peers share the same RANGE frame total.
        """
        imports = get_spark_imports()
        F, Window = imports.F, imports.Window
        df = spark.createDataFrame(ROWS, _schema())

        result = _by_id(df.withColumn("rs", F.sum("k").over(Window.orderBy("k"))), "rs")
        assert result["d"] is None
        assert result["a"] == 1
        assert result["b"] == 5
        assert result["e"] == 5
        assert result["c"] == 9

    def test_lag_and_lead_shift_around_the_null_key_row(self, spark: Any) -> None:
        """``lag``/``lead`` read neighbours in Spark's NULLS FIRST order."""
        imports = get_spark_imports()
        F, Window = imports.F, imports.Window
        df = spark.createDataFrame(ROWS, _schema())
        w = Window.orderBy("k")

        assert _by_id(df.withColumn("v", F.lag("id").over(w)), "v") == {
            "d": None,
            "a": "d",
            "b": "a",
            "e": "b",
            "c": "e",
        }
        assert _by_id(df.withColumn("v", F.lead("id").over(w)), "v") == {
            "d": "a",
            "a": "b",
            "b": "e",
            "e": "c",
            "c": None,
        }

    def test_first_over_window_is_the_null_key_row(self, spark: Any) -> None:
        """``first`` over an ascending window returns the NULL-key row's id."""
        imports = get_spark_imports()
        F, Window = imports.F, imports.Window
        df = spark.createDataFrame(ROWS, _schema())

        result = _by_id(
            df.withColumn("v", F.first("id").over(Window.orderBy("k"))), "v"
        )
        assert set(result.values()) == {"d"}

    def test_descending_window_puts_the_null_key_row_last(self, spark: Any) -> None:
        """``DESC`` keeps NULLs last, so the NULL-key row gets the last number."""
        imports = get_spark_imports()
        F, Window = imports.F, imports.Window
        df = spark.createDataFrame(ROWS, _schema())

        result = _by_id(
            df.withColumn("rn", F.row_number().over(Window.orderBy(F.col("k").desc()))),
            "rn",
        )
        assert result == {"c": 1, "b": 2, "e": 3, "a": 4, "d": 5}

    def test_explicit_nulls_last_window_overrides_the_default(self, spark: Any) -> None:
        """``asc_nulls_last()`` in a window still puts the NULL row last."""
        imports = get_spark_imports()
        F, Window = imports.F, imports.Window
        df = spark.createDataFrame(ROWS, _schema())

        result = _by_id(
            df.withColumn(
                "rn", F.row_number().over(Window.orderBy(F.col("k").asc_nulls_last()))
            ),
            "rn",
        )
        assert result == {"a": 1, "b": 2, "e": 3, "c": 4, "d": 5}

    def test_multi_key_window_applies_each_direction(self, spark: Any) -> None:
        """A window ordered by ``g DESC, k ASC`` keeps per-key NULL placement."""
        imports = get_spark_imports()
        F, Window = imports.F, imports.Window
        df = spark.createDataFrame(ROWS, _schema())

        w = Window.orderBy(F.col("g").desc(), F.col("k").asc())
        result = _by_id(df.withColumn("rn", F.row_number().over(w)), "rn")
        assert result == {"d": 1, "a": 2, "b": 3, "e": 4, "c": 5}


class TestNullsInThePartitionKey(ParityTestBase):
    """A NULL partition key forms its own partition, ordered independently."""

    def test_row_number_within_a_null_partition(self, spark: Any) -> None:
        """Rows with a NULL partition key are numbered among themselves."""
        imports = get_spark_imports()
        F, Window = imports.F, imports.Window
        df = spark.createDataFrame(ROWS, _schema())

        w = Window.partitionBy("g").orderBy("k")
        result = _by_id(df.withColumn("rn", F.row_number().over(w)), "rn")
        # g=NULL partition: e(k=2), c(k=4). g='x' partition: d(NULL), a(1), b(2).
        assert result == {"e": 1, "c": 2, "d": 1, "a": 2, "b": 3}

    def test_running_sum_within_a_null_partition(self, spark: Any) -> None:
        """Each partition's running total starts from its own first row."""
        imports = get_spark_imports()
        F, Window = imports.F, imports.Window
        df = spark.createDataFrame(ROWS, _schema())

        w = Window.partitionBy("g").orderBy("k")
        result = _by_id(df.withColumn("rs", F.sum("k").over(w)), "rs")
        assert result["e"] == 2
        assert result["c"] == 6
        # d sorts first inside g='x' now, so its running total is SUM(NULL).
        assert result["d"] is None
        assert result["a"] == 1
        assert result["b"] == 3
