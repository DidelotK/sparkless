"""PySpark parity tests for ``F.expr``'s SQL parser.

``F.expr`` had no real parser. It looked for operators and keywords with
:func:`re.search` over the raw expression string, split at the first match, and
required the split to yield exactly two pieces. Three consequences, all of them
*silent* -- every one of these returned a value and none of them warned:

* **Operator precedence was inverted.** ``q * 2 + 1`` split at ``*`` into
  ``q`` and ``2 + 1``, binding ``q * (2 + 1)``: 75 where Spark says 51.
  ``salary / 2 - 100`` gave ``-510.2`` instead of ``24900.0``.
* **Function arguments were dropped.** A call bound to
  ``ColumnOperation(None, name, args)``, a shape no evaluator implements, so
  ``concat(sku, dept)`` rendered as ``concat()`` and evaluated to NULL --
  while ``F.concat(F.col("sku"), F.col("dept"))`` was correct all along.
  ``coalesce(name, 'FALLBACK')`` lost its fallback the same way.
* **Predicates returned the wrong boolean.** ``sku RLIKE '^[A-Z]$'`` was NULL
  and ``size(tags) > 1`` was False for a three-element array, because
  ``size(...)`` evaluated to -1.

Expressions with two operators of the same precedence (``q - 5 - 3``) split
into three pieces, fell through every branch and were reported as an invalid
identifier -- so the loud half of the defect hid the silent half.

Every expectation below is the value collected from **PySpark 4.0.0 on
OpenJDK 17**, on the frame built by :func:`_frame`. The file uses the repo's
backend-agnostic ``spark`` fixture, so it can be re-verified against real
PySpark with ``SPARKLESS_TEST_BACKEND=pyspark``.

Tracked as Solya-app/solya-data-platform#2418.
"""

from datetime import date, timedelta
from typing import Any, List

import pytest

from tests.fixtures.parity_base import ParityTestBase
from tests.fixtures.spark_imports import get_spark_imports

_ROWS = [
    {
        "q": 25,
        "age": 25,
        "salary": 50000.0,
        "revenue": 100.0,
        "cost": 60.0,
        "sku": "A",
        "dept": "eng",
        "name": "Alice",
        "tag": "abc",
        "currency": None,
        "tags": ["a", "b", "c"],
        "nullname": None,
    },
    {
        "q": 10,
        "age": 40,
        "salary": 60000.0,
        "revenue": 200.0,
        "cost": 50.0,
        "sku": "B",
        "dept": "ops",
        "name": "Bob",
        "tag": "xyz",
        "currency": "EUR",
        "tags": ["z"],
        "nullname": "set",
    },
]


def _frame(spark: Any) -> Any:
    """The two-row frame every expectation in this module was measured on."""
    return spark.createDataFrame(_ROWS)


def _evaluate(spark: Any, expression: str) -> List[Any]:
    """Collect ``F.expr(expression)`` over :func:`_frame`, in row order."""
    F = get_spark_imports().F
    rows = _frame(spark).select(F.expr(expression).alias("r")).collect()
    return [row["r"] for row in rows]


class TestExprArithmeticPrecedenceParity(ParityTestBase):
    """Arithmetic must bind at Spark's precedence, left-associatively."""

    @pytest.mark.parametrize(  # type: ignore[misc,untyped-decorator]
        "expression,expected",
        [
            # Multiplication before addition, not the other way round.
            ("q * 2 + 1", [51, 21]),
            ("age + 2 * 3", [31, 46]),
            ("salary / 2 - 100", [24900.0, 29900.0]),
            # Same-precedence operators chain left to right. Both of these
            # used to raise ParseException.
            ("q - 5 - 3", [17, 2]),
            ("salary / 2 / 5", [5000.0, 6000.0]),
            # Parentheses were the one arithmetic shape that already worked;
            # they must keep working.
            ("(revenue - cost) / revenue * 100", [40.0, 75.0]),
            ("q * (2 + 1)", [75, 30]),
            ("q % 4", [1, 2]),
            ("q * 2", [50, 20]),
        ],
    )
    def test_arithmetic_binds_like_spark(
        self, spark: Any, expression: str, expected: List[Any]
    ) -> None:
        """PySpark 4.0.0's value for each arithmetic expression."""
        assert _evaluate(spark, expression) == expected

    def test_comparison_is_not_swallowed_by_arithmetic(self, spark: Any) -> None:
        """``q * 2 + 1 > 50`` is a boolean, not the arithmetic operand.

        The comparison used to be dropped outright: the expression returned
        ``[50, 20]``, the value of ``q * 2``.
        """
        assert _evaluate(spark, "q * 2 + 1 > 50") == [True, False]

    def test_unary_minus_applies_to_the_column(self, spark: Any) -> None:
        """``-q + 100`` negates the column; it used to raise."""
        assert _evaluate(spark, "-q + 100") == [75, 90]

    def test_subtracting_a_negative_literal(self, spark: Any) -> None:
        """``q - -5`` is addition, not a parse error."""
        assert _evaluate(spark, "q - -5") == [30, 15]


class TestExprFunctionArgumentsParity(ParityTestBase):
    """Every argument of a call must reach the function."""

    @pytest.mark.parametrize(  # type: ignore[misc,untyped-decorator]
        "expression,expected",
        [
            ("concat(sku, dept)", ["Aeng", "Bops"]),
            ("concat(tag, 'X')", ["abcX", "xyzX"]),
            ("concat_ws('|', sku, dept)", ["A|eng", "B|ops"]),
            ("coalesce(nullname, 'FALLBACK')", ["FALLBACK", "set"]),
            ("coalesce(nullname, name)", ["Alice", "set"]),
            ("substring(name, 1, 3)", ["Ali", "Bob"]),
            ("substr(name, 2, 2)", ["li", "ob"]),
            ("cast(age as string)", ["25", "40"]),
            ("lower(regexp_extract(name, '(A)', 1))", ["a", ""]),
            ("round(salary / 3, 2)", [16666.67, 20000.0]),
            # Single-argument calls were the shape that already worked.
            ("upper(name)", ["ALICE", "BOB"]),
            ("length(name)", [5, 3]),
        ],
    )
    def test_call_arguments_reach_the_function(
        self, spark: Any, expression: str, expected: List[Any]
    ) -> None:
        """PySpark 4.0.0's value for each call."""
        assert _evaluate(spark, expression) == expected

    def test_deeply_nested_calls(self, spark: Any) -> None:
        """The production id expression from ``core/utils/uuid_utils.py``.

        Four levels of nesting, mixing literals and columns. It evaluated to
        NULL because the outermost ``concat`` never saw its arguments.
        """
        expression = (
            "concat(substr(md5(concat_ws('|','org',cast(sku as string))),1,8),'-',sku)"
        )
        assert _evaluate(spark, expression) == ["c6903eff-A", "2fd87b67-B"]

    def test_a_literal_argument_is_a_literal_not_a_column_name(
        self, spark: Any
    ) -> None:
        """``'dept'`` quoted is the string, not the ``dept`` column.

        This is the distinction that made ``coalesce(name, 'FALLBACK')``
        return NULL: passed through as a bare Python string, the fallback was
        resolved as a column name instead of a value.
        """
        assert _evaluate(spark, "concat(sku, 'dept')") == ["Adept", "Bdept"]


class TestExprPredicateParity(ParityTestBase):
    """Predicates must return Spark's boolean."""

    @pytest.mark.parametrize(  # type: ignore[misc,untyped-decorator]
        "expression,expected",
        [
            ("sku RLIKE '^[A-Z]$'", [True, True]),
            ("currency IS NULL OR currency RLIKE '^[A-Z]{3}$'", [True, True]),
            ("size(tags) > 1", [True, False]),
            ("size(tags)", [3, 1]),
            ("revenue BETWEEN 0 AND 150", [True, False]),
            ("revenue NOT BETWEEN 0 AND 150", [False, True]),
            ("dept NOT IN ('eng', 'hr')", [False, True]),
            ("name NOT LIKE 'A%'", [False, True]),
            # The predicate class that already worked, kept under test so the
            # fix cannot regress it.
            ("age > 30", [False, True]),
            ("nullname IS NULL", [True, False]),
            ("nullname IS NOT NULL", [False, True]),
            ("dept IN ('eng', 'hr')", [True, False]),
            ("name LIKE 'A%'", [True, False]),
            ("age > 20 AND dept = 'eng'", [True, False]),
            ("age < 20 OR dept = 'ops'", [False, True]),
            ("NOT (age > 30)", [True, False]),
        ],
    )
    def test_predicate_matches_spark(
        self, spark: Any, expression: str, expected: List[Any]
    ) -> None:
        """PySpark 4.0.0's value for each predicate."""
        assert _evaluate(spark, expression) == expected

    def test_and_binds_tighter_than_or(self, spark: Any) -> None:
        """``a OR b AND c`` is ``a OR (b AND c)``.

        Row 1 (age 25, dept eng): False OR (True AND True) -> True.
        Row 2 (age 40, dept ops): True OR (False AND False) -> True.
        Read as ``(a OR b) AND c`` row 2 would be False.
        """
        expression = "age > 30 OR age = 25 AND dept = 'eng'"
        assert _evaluate(spark, expression) == [True, True]

    def test_case_when_still_works(self, spark: Any) -> None:
        """CASE WHEN was already correct and must stay correct."""
        expression = "CASE WHEN age > 30 THEN 'old' ELSE 'young' END"
        assert _evaluate(spark, expression) == ["young", "old"]

    def test_case_with_an_operand(self, spark: Any) -> None:
        """The simple ``CASE operand WHEN value`` form."""
        expression = "CASE dept WHEN 'eng' THEN 1 ELSE 0 END"
        assert _evaluate(spark, expression) == [1, 0]


class TestExprUuidParity(ParityTestBase):
    """``uuid()`` must produce identifiers, not NULL."""

    def _ids(self, spark: Any) -> List[Any]:
        """Project ``uuid()`` over a three-row frame."""
        F = get_spark_imports().F
        df = spark.createDataFrame([{"n": 1}, {"n": 2}, {"n": 3}])
        return [row["id"] for row in df.select(F.expr("uuid()").alias("id")).collect()]

    def test_uuid_returns_a_uuid_string(self, spark: Any) -> None:
        """Every row gets a 36-character UUID."""
        ids = self._ids(spark)
        assert all(isinstance(i, str) and len(i) == 36 for i in ids)

    def test_uuid_differs_per_row(self, spark: Any) -> None:
        """Spark's uuid() is per-row, not a constant."""
        assert len(set(self._ids(spark))) == 3

    def test_uuid_in_with_column(self, spark: Any) -> None:
        """``withColumn`` is the shape most pipeline call sites use.

        It went through a different evaluator, whose null-propagation guard
        returned NULL for every nullary function -- so an ``id`` column built
        this way was NULL for every row.
        """
        F = get_spark_imports().F
        df = spark.createDataFrame([{"n": 1}, {"n": 2}, {"n": 3}])

        ids = [row["id"] for row in df.withColumn("id", F.expr("uuid()")).collect()]

        assert len(set(ids)) == 3
        assert all(isinstance(i, str) and len(i) == 36 for i in ids)


class TestExprCastParity(ParityTestBase):
    """``CAST`` and ``TRY_CAST`` are syntax, not function calls."""

    def test_try_cast_of_a_backticked_column(self, spark: Any) -> None:
        """``try_cast(`revenue` AS DOUBLE)`` used to raise ParseException."""
        assert _evaluate(spark, "try_cast(`revenue` AS DOUBLE)") == [100.0, 200.0]

    def test_try_cast_of_an_unconvertible_value_is_null(self, spark: Any) -> None:
        """``try_cast('A' AS DOUBLE)`` is NULL, not an error."""
        assert _evaluate(spark, "try_cast(sku AS DOUBLE)") == [None, None]

    def test_cast_of_a_backticked_column(self, spark: Any) -> None:
        """Backticked identifiers resolve inside CAST too."""
        assert _evaluate(spark, "cast(`age` AS STRING)") == ["25", "40"]


class TestExprIntervalParity(ParityTestBase):
    """Day-based ``INTERVAL`` arithmetic, the shape the pipelines use."""

    def _dates(self, spark: Any, expression: str) -> List[Any]:
        """Collect a date expression over a two-row frame of known dates."""
        F = get_spark_imports().F
        df = spark.createDataFrame([{"d": date(2026, 8, 19)}, {"d": date(2026, 1, 1)}])
        return [
            str(row["r"]) for row in df.select(F.expr(expression).alias("r")).collect()
        ]

    @pytest.mark.parametrize(  # type: ignore[misc,untyped-decorator]
        "expression,expected",
        [
            ("d - INTERVAL 90 DAYS", ["2026-05-21", "2025-10-03"]),
            ("d + INTERVAL 30 DAYS", ["2026-09-18", "2026-01-31"]),
            ("d - INTERVAL 1 WEEK", ["2026-08-12", "2025-12-25"]),
            ("d - INTERVAL 1 DAY", ["2026-08-18", "2025-12-31"]),
        ],
    )
    def test_day_interval_arithmetic(
        self, spark: Any, expression: str, expected: List[Any]
    ) -> None:
        """PySpark 4.0.0's date for each interval expression."""
        assert self._dates(spark, expression) == expected

    def test_interval_as_its_own_expression(self, spark: Any) -> None:
        """``current_date() - F.expr("INTERVAL n DAYS")``.

        This is how every pipeline call site writes it: the interval is its
        own ``F.expr`` and the subtraction happens in Python. It used to bind
        to a *column reference* named ``INTERVAL_90_DAYS``, so the comparison
        below silently matched nothing.
        """
        F = get_spark_imports().F
        df = spark.createDataFrame(
            [
                {"d": date.today() - timedelta(days=5)},
                {"d": date.today() - timedelta(days=200)},
            ]
        )
        cutoff = F.current_date() - F.expr("INTERVAL 90 DAYS")

        kept = df.filter(F.col("d") >= cutoff).collect()

        assert len(kept) == 1


class TestExprAggregateParity(ParityTestBase):
    """``F.expr`` inside ``agg`` must aggregate, not return NULL."""

    def _aggregate(self, spark: Any, expression: str) -> Any:
        """Collect ``agg(F.expr(expression))`` over :func:`_frame`."""
        F = get_spark_imports().F
        rows = _frame(spark).agg(F.expr(expression).alias("r")).collect()
        return rows[0]["r"]

    def test_sum(self, spark: Any) -> None:
        """``sum(salary)`` was NULL."""
        assert self._aggregate(spark, "sum(salary)") == 110000.0

    def test_avg(self, spark: Any) -> None:
        """``avg(age)`` was 0."""
        assert self._aggregate(spark, "avg(age)") == 32.5

    def test_count_of_a_literal(self, spark: Any) -> None:
        """``count(1)`` was 0."""
        assert self._aggregate(spark, "count(1)") == 2

    def test_count_star(self, spark: Any) -> None:
        """``count(*)`` used to raise ParseException."""
        assert self._aggregate(spark, "count(*)") == 2

    def test_count_distinct(self, spark: Any) -> None:
        """``count(DISTINCT dept)`` counts the two distinct departments."""
        assert self._aggregate(spark, "count(DISTINCT dept)") == 2
