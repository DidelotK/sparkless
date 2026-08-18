"""Deriving a column's name must not re-walk its operand tree exponentially.

The defect: :attr:`ColumnOperation.name` walked its subtree twice (once via
``_generate_name()``, once via ``str(self)``), so the cost of naming a node
doubled with every level of nesting, and a sub-expression referenced twice was
re-walked once per reference on top of that.

Measured on the expression these tests rebuild -- ``gold.decision_context``'s
projected stock-out date, 24 levels with two shared operands -- a five-row
``collect()`` made 18 962 252 calls into the name helper and took 12.9 s. The
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
        4 yield no date, and row 5 holds no stock so it runs out on the snapshot
        date itself.

        Rows 3 and 4 reach NULL by a different route here than on a real
        cluster, and that is worth being explicit about rather than papering
        over: sparkless does not implement ``nullif`` (it warns and returns its
        first operand), so the zero and below-floor velocities divide by a
        literal zero and reach NULL through sparkless's ``x / 0 -> NULL``
        instead of through the ``nullif`` gate. Same values, and the same values
        the pipeline gets from this engine -- which is what this test pins. The
        `nullif` gap is pre-existing and out of scope here.
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


class TestTheComparisonStringPathIsWalkedOnce:
    """``__str__`` has its own recursion, and it needs its own memo.

    For the comparison operators ``__str__`` recurses through ``str()`` on
    *both* operands rather than through ``_generate_name()``, so neither the
    ``name`` memo nor the ``_generate_name`` memo covers it. Without a test
    here the ``@memoise_within_pass("str")`` decoration could be deleted and
    the rest of this file would stay green -- measured.
    """

    def test_a_shared_comparison_operand_is_stringified_once_per_node(self) -> None:
        """A node reachable from both sides must not be re-stringified per path.

        FALSIFIED BY (measured): dropping the ``str`` memo takes this from 37 to
        262 143 entries into ``__str__``.
        """
        entries = [0]
        original = ColumnOperation.__str__

        def counting(self):
            entries[0] += 1
            return original(self)

        node = F.col("a") == F.col("b")
        for _ in range(17):
            node = node == node  # one object on both sides: a DAG, not a tree

        ColumnOperation.__str__ = counting  # type: ignore[method-assign]
        try:
            rendered = str(node)
        finally:
            ColumnOperation.__str__ = original  # type: ignore[method-assign]

        assert rendered.startswith("(")
        assert entries[0] < 1000, (
            f"__str__ was entered {entries[0]} times for 18 distinct nodes; "
            "the comparison operands are being re-walked per reference"
        )


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

        The exploding operand is placed *above* several levels of ordinary
        expression, and the walk is checked to have populated the cache before
        it blows up. A shallower arrangement -- raising on the first operand
        touched -- would leave the cache empty for want of anything to cache, so
        the assertion below would hold no matter what the ``finally`` did.
        """
        from sparkless.functions.core import name_memo

        depth_seen = []

        class Exploding:
            @property
            def name(self):
                depth_seen.append((name_memo._STATE.depth, len(name_memo._STATE.cache)))
                raise RuntimeError("boom")

            def __str__(self):
                raise RuntimeError("boom")

        deep = F.col("a").cast("double") * F.lit(2.0)
        deep = (F.abs(deep) + deep) / F.lit(3.0)
        # The deep tree goes in the *value* slot and the poison in the column
        # slot, because the name helper stringifies `value` before it reads
        # `column.name`: that ordering is what gets entries into the cache
        # before the walk blows up. The poison is swapped in after construction
        # because building a ColumnOperation over it would raise in __init__.
        node = ColumnOperation(F.col("z"), "+", deep)
        assert node.name
        node.column = Exploding()

        with pytest.raises(RuntimeError):
            node._generate_name()

        assert depth_seen, "the exploding operand was never reached"
        mid_depth, mid_entries = depth_seen[-1]
        assert mid_depth > 0, "the walk was not in flight when it raised"
        assert mid_entries > 0, (
            "nothing was cached before the failure, so draining is untested"
        )

        assert name_memo._STATE.cache == {}
        assert name_memo._STATE.depth == 0

    def test_a_drifted_depth_counter_re_synchronises(self) -> None:
        """A counter that drifted below zero must not disable invalidation.

        The counter is the only thing that decides when the memo drains, and a
        counter that never returns to its clearing value turns the memo into a
        permanent process-wide cache that never invalidates and pins every node
        it ever saw -- silent and unbounded.

        The wrapper increments *inside* its ``try`` precisely so that the only
        drift an asynchronous exception (SIGINT, ``interrupt_main``,
        ``PyThreadState_SetAsyncExc``) can cause is an unmatched *decrement*,
        never an unmatched increment: a stranded-high counter would be
        unrecoverable, whereas a low one is absorbed here. That ordering is not
        directly testable -- the window is a few bytecodes wide -- but its
        consequence is, so this pins the recovery.

        FALSIFIED BY (measured): restoring ``if state.depth == 0`` in place of
        the ``<= 0`` clamp leaves the memo poisoned and this fails.
        """
        from sparkless.functions.core import name_memo

        node = F.col("a") + F.lit(1)
        assert "a" in node.name

        # Simulate unmatched decrements, the drift an interrupted walk leaves.
        name_memo._STATE.depth = -3
        try:
            node.name
            drained_depth = name_memo._STATE.depth
            drained_entries = len(name_memo._STATE.cache)
        finally:
            name_memo._STATE.depth = 0
            name_memo._STATE.cache.clear()

        assert drained_depth == 0, (
            f"counter stayed drifted at {drained_depth}; the memo would never "
            "invalidate again"
        )
        assert drained_entries == 0, (
            f"{drained_entries} entries survived the outermost derivation"
        )

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

        Every observation is carried back to the main thread and asserted there.
        An ``assert`` inside the worker would be reported as a
        ``PytestUnhandledThreadExceptionWarning`` and the test would still pass.
        """
        from sparkless.functions.core import name_memo

        cache_ids = {}
        depths = {}
        residues = {}
        names = {}
        barrier = threading.Barrier(4)

        def derive(tag):
            node = F.col(tag).cast("double") * F.lit(2.0)
            barrier.wait()
            for _ in range(200):
                names[tag] = node.name
            cache_ids[tag] = id(name_memo._STATE.cache)
            depths[tag] = name_memo._STATE.depth
            residues[tag] = len(name_memo._STATE.cache)

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
        assert set(residues.values()) == {0}, (
            f"a thread's cache was not drained: {residues}"
        )
        assert all(tag in names[tag] for tag in tags)
        assert len(set(names.values())) == len(tags)
