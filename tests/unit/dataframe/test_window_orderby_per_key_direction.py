"""Regression tests for per-key sort direction in ``Window.orderBy``.

Spark applies the sort direction **per key**: ``ORDER BY a DESC, b ASC`` sorts
``a`` descending and, within ties on ``a``, ``b`` *ascending*.

Sparkless previously computed a single global ``reverse`` flag --
``reverse=any(key is desc)`` -- and sorted once. Any ``desc`` key therefore
reversed *every* key, so the trailing ``asc`` tie-break came out backwards.
The failure is silent: the window still produces plausible ranks, they are just
ordered wrongly, so a test asserting "row_number() is 1..n" still passes.

Verified against PySpark 4.0.0 (the DBR 17.3 runtime). These tests are
backend-agnostic and also pass under ``MOCK_SPARK_TEST_BACKEND=pyspark``.
"""

from tests.fixtures.spark_imports import get_spark_imports


def _ranked_names(df, window, F):
    """Return names ordered by the window's row_number()."""
    rows = df.withColumn("rn", F.row_number().over(window)).collect()
    return [r["name"] for r in sorted(rows, key=lambda r: r["rn"])]


def _tie_frame(spark):
    """Rows that tie on ``score`` so the later sort keys decide the order."""
    return spark.createDataFrame(
        [
            ("g", 1, 5.0, "z"),
            ("g", 1, 5.0, "a"),
            ("g", 2, 5.0, "m"),
        ],
        ["grp", "prio", "score", "name"],
    )


class TestWindowOrderByPerKeyDirection:
    """Each order key must keep its own direction."""

    def test_desc_then_asc_tiebreak_sorts_tiebreak_ascending(self, spark) -> None:
        """desc(score), asc(name): the asc tie-break must not be reversed."""
        imports = get_spark_imports()
        F, Window = imports.F, imports.Window
        df = _tie_frame(spark)

        window = Window.partitionBy("grp").orderBy(
            F.col("score").desc(), F.col("name").asc()
        )

        # All scores tie, so `name` alone decides -- ascending.
        assert _ranked_names(df, window, F) == ["a", "m", "z"]

    def test_three_keys_mixed_directions(self, spark) -> None:
        """desc(score), desc(prio), asc(name) -- the trailing asc key survives."""
        imports = get_spark_imports()
        F, Window = imports.F, imports.Window
        df = _tie_frame(spark)

        window = Window.partitionBy("grp").orderBy(
            F.col("score").desc(),
            F.col("prio").desc(),
            F.col("name").asc(),
        )

        # score ties -> prio desc puts prio=2 ("m") first -> then name asc: a, z.
        assert _ranked_names(df, window, F) == ["m", "a", "z"]

    def test_bare_column_after_desc_defaults_to_ascending(self, spark) -> None:
        """An undecorated trailing key defaults to ascending, not to desc."""
        imports = get_spark_imports()
        F, Window = imports.F, imports.Window
        df = _tie_frame(spark)

        window = Window.partitionBy("grp").orderBy(F.col("score").desc(), F.col("name"))

        assert _ranked_names(df, window, F) == ["a", "m", "z"]

    def test_asc_then_desc_tiebreak_sorts_tiebreak_descending(self, spark) -> None:
        """The mirror case: a desc tie-break after an asc key stays descending."""
        imports = get_spark_imports()
        F, Window = imports.F, imports.Window
        df = _tie_frame(spark)

        window = Window.partitionBy("grp").orderBy(
            F.col("score").asc(), F.col("name").desc()
        )

        assert _ranked_names(df, window, F) == ["z", "m", "a"]

    def test_all_ascending_unchanged(self, spark) -> None:
        """All-ascending was already correct and must stay correct."""
        imports = get_spark_imports()
        F, Window = imports.F, imports.Window
        df = _tie_frame(spark)

        window = Window.partitionBy("grp").orderBy(
            F.col("score").asc(), F.col("name").asc()
        )

        assert _ranked_names(df, window, F) == ["a", "m", "z"]

    def test_all_descending_unchanged(self, spark) -> None:
        """All-descending was already correct and must stay correct."""
        imports = get_spark_imports()
        F, Window = imports.F, imports.Window
        df = _tie_frame(spark)

        window = Window.partitionBy("grp").orderBy(
            F.col("score").desc(), F.col("name").desc()
        )

        assert _ranked_names(df, window, F) == ["z", "m", "a"]

    def test_significant_key_still_dominates(self, spark) -> None:
        """The first key must remain the most significant one."""
        imports = get_spark_imports()
        F, Window = imports.F, imports.Window
        df = spark.createDataFrame(
            [
                ("g", 1.0, "a"),
                ("g", 3.0, "b"),
                ("g", 2.0, "c"),
            ],
            ["grp", "score", "name"],
        )

        window = Window.partitionBy("grp").orderBy(
            F.col("score").desc(), F.col("name").asc()
        )

        # No ties: score desc decides everything.
        assert _ranked_names(df, window, F) == ["b", "c", "a"]

    def test_rank_uses_the_same_ordering(self, spark) -> None:
        """rank() goes through the same ordering path as row_number()."""
        imports = get_spark_imports()
        F, Window = imports.F, imports.Window
        df = _tie_frame(spark)

        window = Window.partitionBy("grp").orderBy(
            F.col("score").desc(), F.col("name").asc()
        )
        rows = df.withColumn("rk", F.rank().over(window)).collect()

        by_name = {r["name"]: r["rk"] for r in rows}
        # score+name is unique across the three rows, so rank is a dense 1,2,3.
        assert by_name["a"] == 1
        assert by_name["m"] == 2
        assert by_name["z"] == 3

    def test_lag_follows_mixed_direction_ordering(self, spark) -> None:
        """lag() must read the previous row of the *correctly* ordered window."""
        imports = get_spark_imports()
        F, Window = imports.F, imports.Window
        df = _tie_frame(spark)

        window = Window.partitionBy("grp").orderBy(
            F.col("score").desc(), F.col("name").asc()
        )
        rows = df.withColumn("prev", F.lag("name", 1).over(window)).collect()

        by_name = {r["name"]: r["prev"] for r in rows}
        # Order is a, m, z.
        assert by_name["a"] is None
        assert by_name["m"] == "a"
        assert by_name["z"] == "m"


class TestWindowOrderByNullPlacement:
    """Null placement must stay correct now that direction is per key."""

    def _null_frame(self, spark):
        return spark.createDataFrame(
            [
                ("g", 1, "b"),
                ("g", 1, None),
                ("g", 1, "a"),
            ],
            ["grp", "score", "name"],
        )

    def test_nulls_last_on_ascending_tiebreak(self, spark) -> None:
        """asc_nulls_last after a desc key keeps nulls at the end."""
        imports = get_spark_imports()
        F, Window = imports.F, imports.Window
        df = self._null_frame(spark)

        window = Window.partitionBy("grp").orderBy(
            F.col("score").desc(), F.col("name").asc_nulls_last()
        )

        assert _ranked_names(df, window, F) == ["a", "b", None]

    def test_nulls_first_on_ascending_tiebreak(self, spark) -> None:
        """asc_nulls_first after a desc key keeps nulls at the front."""
        imports = get_spark_imports()
        F, Window = imports.F, imports.Window
        df = self._null_frame(spark)

        window = Window.partitionBy("grp").orderBy(
            F.col("score").desc(), F.col("name").asc_nulls_first()
        )

        assert _ranked_names(df, window, F) == [None, "a", "b"]

    def test_nulls_in_a_string_column_do_not_raise(self, spark) -> None:
        """Ordering a nullable string column must not compare str against float.

        The previous implementation substituted ``float("inf")`` for nulls,
        which raises TypeError when compared against a string value.
        """
        imports = get_spark_imports()
        F, Window = imports.F, imports.Window
        df = self._null_frame(spark)

        window = Window.partitionBy("grp").orderBy(F.col("name").desc_nulls_last())

        assert _ranked_names(df, window, F) == ["b", "a", None]
