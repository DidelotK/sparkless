"""Regression tests for ``F.exists`` / ``F.forall`` / ``F.filter`` / ``F.transform``.

All three predicates returned **NULL for every row**, and ``F.transform``
raised ``LambdaTranslationError``. NULL is falsy in every downstream branch, so
a guard written on top of them answers "clean" for every input — which is what
Solya-app/solya-data-platform#2419 reports for the duplicate-axis guard in
``pipelines/shared/sku_id.py``: sparkless said "no duplicate" on input real
Spark flags as duplicated, and never raised, because wrapping ``F.transform``
in ``F.size`` never materialises the lambda.

Three-valued logic is the substance of these functions, so it is what the tests
assert. Every expectation below was measured against PySpark 4.0.0
(``local[1]``); the file is backend-agnostic and passes under
``MOCK_SPARK_TEST_BACKEND=pyspark``.

===============================  ==============================================
call                             answer
===============================  ==============================================
``exists(NULL, f)``              NULL
``exists([], f)``                FALSE
``exists`` with no TRUE and at   NULL — not FALSE. An unknown element could
least one NULL predicate         have been the one that matched.
``forall(NULL, f)``              NULL
``forall([], f)``                TRUE
``forall`` with no FALSE and at  NULL
least one NULL predicate
``filter``                       keeps only TRUE; a NULL predicate drops the
                                 element, and a NULL array stays NULL
``transform``                    maps element-wise; a NULL element maps through
                                 the expression, so ``x * 2`` gives NULL
===============================  ==============================================
"""

import pytest

from tests.fixtures.spark_imports import get_spark_imports


@pytest.fixture
def arrays_df(spark):
    """Rows covering the populated, empty, NULL and NULL-element array shapes."""
    imports = get_spark_imports()
    StructType, StructField = imports.StructType, imports.StructField
    ArrayType, IntegerType, StringType = (
        imports.ArrayType,
        imports.IntegerType,
        imports.StringType,
    )

    schema = StructType(
        [
            StructField("nums", ArrayType(IntegerType())),
            StructField("tags", ArrayType(StringType())),
        ]
    )
    return spark.createDataFrame(
        [
            ([1, 2, 3], ["a", "b", "a"]),
            ([], []),
            (None, None),
            ([5, None, 2], ["x", None]),
        ],
        schema,
    )


def _column(df, expression):
    """Collect a single projected expression as a plain list."""
    return [row[0] for row in df.select(expression).collect()]


class TestExists:
    """``exists`` is TRUE on a match, NULL when a match cannot be ruled out."""

    def test_exists_finds_a_matching_element(self, arrays_df) -> None:
        """The reported failure: every row came back NULL."""
        F = get_spark_imports().F

        assert _column(arrays_df, F.exists("nums", lambda x: x > 2)) == [
            True,
            False,
            None,
            True,
        ]

    def test_no_match_is_false_not_null(self, arrays_df) -> None:
        """A row that genuinely matches nothing answers FALSE.

        This is the assertion the NULL bug hid: FALSE and NULL are the same
        thing to an ``if``, so a caller could not tell "no match" from
        "not implemented".
        """
        F = get_spark_imports().F

        assert _column(arrays_df, F.exists("tags", lambda x: x.isNull()))[0] is False

    def test_unknown_element_makes_the_answer_null(self, arrays_df) -> None:
        """No TRUE but a NULL predicate gives NULL, not FALSE.

        ``[5, NULL, 2]`` against ``x > 99``: the NULL element could have been
        the match, so Spark refuses to answer FALSE.
        """
        F = get_spark_imports().F

        assert _column(arrays_df, F.exists("nums", lambda x: x > 99)) == [
            False,
            False,
            None,
            None,
        ]

    def test_empty_array_is_false_and_null_array_is_null(self, arrays_df) -> None:
        """The two empty-ish shapes answer differently and must not be conflated."""
        F = get_spark_imports().F

        answers = _column(arrays_df, F.exists("nums", lambda x: x > 2))
        assert answers[1] is False
        assert answers[2] is None


class TestForall:
    """``forall`` is FALSE on a counter-example, NULL when one cannot be ruled out."""

    def test_forall_over_all_matching_elements(self, arrays_df) -> None:
        F = get_spark_imports().F

        assert _column(arrays_df, F.forall("nums", lambda x: x > 0)) == [
            True,
            True,
            None,
            None,
        ]

    def test_a_counter_example_wins_over_an_unknown(self, arrays_df) -> None:
        """FALSE dominates NULL: ``[5, NULL, 2]`` against ``x > 2`` is FALSE.

        Returning NULL here would be the plausible-looking wrong answer -- the
        array does contain an unknown -- but ``2 > 2`` already settles it.
        """
        F = get_spark_imports().F

        assert _column(arrays_df, F.forall("nums", lambda x: x > 2)) == [
            False,
            True,
            None,
            False,
        ]

    def test_empty_array_is_true(self, arrays_df) -> None:
        """Vacuous truth, as in Spark -- and the opposite of ``exists([])``."""
        F = get_spark_imports().F

        assert _column(arrays_df, F.forall("nums", lambda x: x > 2))[1] is True


class TestFilter:
    """``filter`` keeps the TRUE elements and nothing else."""

    def test_filter_keeps_matching_elements(self, arrays_df) -> None:
        F = get_spark_imports().F

        assert _column(arrays_df, F.filter("nums", lambda x: x > 2)) == [
            [3],
            [],
            None,
            [5],
        ]

    def test_null_predicate_drops_the_element(self, arrays_df) -> None:
        """``[5, NULL, 2]`` filtered on ``x > 2`` keeps only ``5``."""
        F = get_spark_imports().F

        assert _column(arrays_df, F.filter("nums", lambda x: x > 2))[3] == [5]

    def test_null_array_stays_null_not_empty(self, arrays_df) -> None:
        """NULL in, NULL out. ``[]`` would claim the row had no matches."""
        F = get_spark_imports().F

        assert _column(arrays_df, F.filter("nums", lambda x: x > 2))[2] is None


class TestTransform:
    """``transform`` maps element-wise instead of raising."""

    def test_transform_maps_each_element(self, arrays_df) -> None:
        """Previously raised ``LambdaTranslationError`` from schema inference."""
        F = get_spark_imports().F

        assert _column(arrays_df, F.transform("nums", lambda x: x * 2)) == [
            [2, 4, 6],
            [],
            None,
            [10, None, 4],
        ]

    def test_transform_can_call_other_functions(self, arrays_df) -> None:
        """The lambda body is a Spark expression, not just an operator."""
        F = get_spark_imports().F

        assert _column(
            arrays_df,
            F.transform(F.array_sort("tags"), lambda s: F.concat(s, F.lit("="))),
        )[0] == ["a=", "a=", "b="]

    def test_null_element_maps_to_null(self, arrays_df) -> None:
        """A NULL element is transformed, not skipped: the array keeps its length."""
        F = get_spark_imports().F

        assert _column(arrays_df, F.transform("nums", lambda x: x * 2))[3] == [
            10,
            None,
            4,
        ]


class TestDuplicateAxisGuard:
    """The composite from ``pipelines/shared/sku_id.py`` that motivated the issue."""

    def test_duplicate_axis_is_detected(self, spark) -> None:
        """``size(a) != size(array_distinct(a))`` must flag a duplicated axis.

        Under the NULL behaviour both sides collapsed and the guard answered
        FALSE -- "clean" -- for every input, including the duplicated one. A
        data-integrity check that can never fire is worse than an absent one.
        """
        F = get_spark_imports().F
        df = spark.createDataFrame([(["a", "b", "a"],), (["z"],)], ["tags"])

        rendered = F.transform(F.col("tags"), lambda slot: slot)
        guard = F.size(rendered) != F.size(F.array_distinct(rendered))

        assert _column(df, guard) == [True, False]

    def test_null_slot_is_detected(self, spark) -> None:
        """``exists(rendered, slot -> slot.isNull())``, the other half of the guard."""
        F = get_spark_imports().F
        df = spark.createDataFrame([(["a", "b"],), (["a", None],)], ["tags"])

        assert _column(df, F.exists("tags", lambda slot: slot.isNull())) == [
            False,
            True,
        ]
