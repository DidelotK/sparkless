"""Deriving a column's name must not re-walk its operand tree exponentially.

The defect: :attr:`ColumnOperation.name` walked its subtree twice (once via
``_generate_name()``, once via ``str(self)``), so the cost of naming a node
doubled with every level of nesting, and a sub-expression referenced twice was
re-walked once per reference on top of that.

Measured on the expression these tests rebuild -- ``gold.decision_context``'s
projected stock-out date, 24 levels with two shared operands -- a five-row
``collect()`` made 17 469 222 calls into the name helper and took 12.9 s. The
same expression one level lower took 0.083 s.

These tests pin the *work*, not the wall clock: a timing assertion would be
flaky on a loaded CI box, whereas the call count is deterministic and is the
thing that was exponential.
"""

import datetime
import threading
from typing import Any

import pytest

from sparkless import functions as F
from sparkless.functions.core.column import Column, ColumnOperation
from sparkless.session import SparkSession
from sparkless.spark_types import DoubleType, StructField, StructType

_SCHEMA = StructType(
    [
        StructField("current_stock", DoubleType(), True),
        StructField("forecast_30d", DoubleType(), True),
    ]
)

_ROWS = [
    (10.0, 30.0),
    (10.0, 90.0),
    (10.0, 0.0),
    (25.0, 2.53e-4),
    (0.0, 30.0),
]

_SNAPSHOT = datetime.date(2026, 8, 14)

#: The unfixed evaluator made 17.5 million calls for this expression. One walk
#: per node would be a few hundred; the bound is deliberately loose so ordinary
#: refactoring of the name helper does not trip it, and still four orders of
#: magnitude below the defect.
_CALL_BUDGET = 20_000


def _days_of_cover() -> Any:
    """``max(current_stock / (forecast_30d / 30), 0)``, NULL below the floor.

    Deliberately built the way the caller builds it: ``days`` is bound once and
    referenced twice, so the expression is a DAG rather than a tree.
    """
    forecast = F.col("forecast_30d").cast("double")
    current_stock = F.col("current_stock").cast("double")
    significant = forecast * (forecast >= F.lit(0.5)).cast("double")
    divisor = F.nullif(significant, F.lit(0.0))
    days = current_stock * F.lit(30.0) / divisor
    return (F.abs(days) + days) / F.lit(2.0)


def _projected_stock_out_date() -> Any:
    """``snapshot_date + floor(days_of_cover)``, as a DATE."""
    offset_days = F.floor(_days_of_cover()).cast("int")
    epoch_day = (_SNAPSHOT - datetime.date(1970, 1, 1)).days
    return F.date_from_unix_date(F.lit(epoch_day) + offset_days)


@pytest.fixture
def spark():
    session = SparkSession.builder.appName("nested-expression-naming").getOrCreate()
    yield session
    session.stop()


class TestNestedExpressionIsWalkedOnce:
    """The operand tree is visited a bounded number of times, not 2**depth."""

    def test_collect_over_a_deeply_nested_expression_stays_within_budget(
        self, spark, monkeypatch
    ) -> None:
        """FALSIFIED BY (measured): reverting the memo -> 17 469 222 calls."""
        calls = [0]
        original = ColumnOperation._generate_name_early_helper

        def counting(column, operation, value):
            calls[0] += 1
            return original(column, operation, value)

        monkeypatch.setattr(
            ColumnOperation, "_generate_name_early_helper", staticmethod(counting)
        )

        df = spark.createDataFrame(_ROWS, schema=_SCHEMA)
        rows = df.withColumn("projected", _projected_stock_out_date()).collect()

        assert len(rows) == 5
        assert calls[0] < _CALL_BUDGET, (
            f"name derivation made {calls[0]} calls into the name helper; "
            "the operand tree is being re-walked per level again"
        )

    def test_the_nested_expression_still_computes_the_right_dates(self, spark) -> None:
        """The speed-up must not move a single value.

        Row 1 covers 10 whole days, row 2 covers 3.33 (floored to 3), rows 3 and
        4 have no usable velocity (zero, and below the 0.5 floor), and row 5
        holds no stock so it runs out on the snapshot date itself.
        """
        df = spark.createDataFrame(_ROWS, schema=_SCHEMA)
        out = df.withColumn("projected", _projected_stock_out_date()).collect()

        assert [row["projected"] for row in out] == [
            datetime.date(2026, 8, 24),
            datetime.date(2026, 8, 17),
            None,
            None,
            datetime.date(2026, 8, 14),
        ]

    def test_each_added_layer_costs_about_the_same(self, spark, monkeypatch) -> None:
        """Cost must grow with the tree, not double per level.

        This is the shape of the defect rather than its size: wrapping the
        expression in one more layer used to multiply the work, so the ratio
        between adjacent layers is the thing to hold down.

        FALSIFIED BY (measured): reverting the memo ->
        ``[133929, 310218, 1311054, 4586528, 18962252]``, ~3x per layer.
        Memoised, the same five layers cost ``[190, 122, 211, 285, 202]``.
        """
        calls = []
        original = ColumnOperation._generate_name_early_helper

        df = spark.createDataFrame(_ROWS, schema=_SCHEMA)
        layers = [
            _days_of_cover,
            lambda: F.floor(_days_of_cover()),
            lambda: F.floor(_days_of_cover()).cast("int"),
            lambda: F.lit(1) + F.floor(_days_of_cover()).cast("int"),
            _projected_stock_out_date,
        ]

        for build in layers:
            counter = [0]

            def counting(column, operation, value, _c=counter):
                _c[0] += 1
                return original(column, operation, value)

            monkeypatch.setattr(
                ColumnOperation, "_generate_name_early_helper", staticmethod(counting)
            )
            try:
                df.withColumn("x", build()).collect()
            finally:
                monkeypatch.undo()
            calls.append(counter[0])

        assert all(count > 0 for count in calls)
        assert calls[-1] < 8 * calls[0], f"cost per layer is compounding again: {calls}"


class TestTheMemoDoesNotOutliveItsPass:
    """A cached name must never be served to a later, unrelated derivation."""

    def test_a_rebound_operand_renames_the_node(self) -> None:
        """Mutating an operand between two reads changes the second answer."""
        node = F.col("a") + F.lit(1)
        before = node.name

        node.column = Column("b")

        assert node._generate_name() != before
        assert "b" in node._generate_name()

    def test_a_failed_derivation_leaves_no_entries_behind(self) -> None:
        """An exception mid-walk must still drain the memo.

        Otherwise the next derivation reads names cached against a tree that has
        since moved on -- the exact staleness the pass scope exists to prevent.
        """
        from sparkless.functions.core import name_memo

        class Exploding:
            @property
            def name(self):
                raise RuntimeError("boom")

            def __str__(self):
                raise RuntimeError("boom")

        node = ColumnOperation(F.col("a"), "+", F.lit(1))
        node.column = Exploding()

        with pytest.raises(RuntimeError):
            node._generate_name()

        assert name_memo._STATE.cache == {}
        assert name_memo._STATE.depth == 0

    def test_each_thread_gets_its_own_cache_and_depth_counter(self) -> None:
        """Concurrent derivations must not share the memo's bookkeeping.

        Comparing results alone would not test this -- distinct nodes give
        distinct answers out of a shared cache too. What a shared cache would
        actually break is the *depth* counter: one thread's increments would
        hold ``depth`` above zero while another thread's outermost derivation
        returns, so that thread's entries would never be dropped and would go on
        answering for a tree that has since been mutated. So this asserts on the
        bookkeeping itself -- one cache object per thread, and every thread's
        state drained afterwards.
        """
        from sparkless.functions.core import name_memo

        cache_ids = {}
        depths = {}
        names = {}
        barrier = threading.Barrier(4)

        def derive(tag):
            node = F.col(tag).cast("double") * F.lit(2.0)
            barrier.wait()
            for _ in range(200):
                names[tag] = node.name
            cache_ids[tag] = id(name_memo._STATE.cache)
            depths[tag] = name_memo._STATE.depth
            assert not name_memo._STATE.cache

        tags = ("north", "south", "east", "west")
        threads = [threading.Thread(target=derive, args=(t,)) for t in tags]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(cache_ids) == len(tags), "a worker thread died"
        assert len(set(cache_ids.values())) == len(tags), (
            f"threads shared a cache object: {cache_ids}"
        )
        assert set(depths.values()) == {0}, f"depth counter leaked: {depths}"
        assert all(tag in names[tag] for tag in tags)
        assert len(set(names.values())) == len(tags)
