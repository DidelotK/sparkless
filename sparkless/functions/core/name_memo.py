"""Pass-scoped memoisation for column-name derivation.

A :class:`~sparkless.functions.core.column.ColumnOperation` derives its display
name by walking its operand tree: :attr:`~ColumnOperation.name` asks
``_generate_name()``, whose helper reads ``column.name`` and ``str(value)`` on
the operands, and those recurse the same way. Two independent multipliers turn
that walk into an exponential one:

* :attr:`~ColumnOperation.name` walks the subtree **twice** -- once through
  ``_generate_name()`` and once through ``str(self)`` -- to decide whether an
  explicitly-set ``_name`` differs from the derived one. So the cost doubles
  with every level of nesting: ``2 ** depth``.
* an expression is a DAG, not a tree. A sub-expression bound to a Python local
  and referenced twice -- ``(abs(days) + days) / 2`` -- is re-walked once per
  reference, multiplying again at every shared node.

Measured on the ``gold.decision_context`` projected-stock-out-date expression
(24 levels, two shared operands): a five-row ``collect()`` made **18 962 252**
calls into the name helper and took **12.9 s**. The same expression one level
lower took **0.083 s**.

This rests on an **assumption**, so it is worth stating as one: name derivation
is *pure*. Every branch reachable from here reads ``column`` / ``operation`` /
``value`` / ``_name`` / ``_alias_name`` and returns a string, mutating nothing.
If that holds, a node reached twice in one walk must produce the same answer
both times, and memoising the answer for the duration of a single outermost
derivation collapses the walk to one visit per node with no change to any result.

The strongest evidence for it is not about types but about *reach*: a pass is a
closed world. Profiling the whole test suite with ``sys.setprofile`` toggled at
every outermost entry and exit, exactly three sparkless modules ever execute
inside a pass --

===========================  ==========
module                       calls
===========================  ==========
``functions/core/column.py``    112 048
``functions/core/literals.py``      978
``spark_types.py``                   92
===========================  ==========

-- and every one of them is read-only with respect to the tree. Nothing from
``session/``, nothing from ``dataframe/``. In particular the two places that
*do* rebind operands after construction, ``LazyDataFrame``'s column
normalisation and ``GroupedData``'s aggregate substitution, are not among them,
so they cannot run while a name is being derived. An audit that recomputed
every derivation the suite performs against an unmemoised reference agrees:
0 mismatches, 0 mutations observed.

What is *not* enforced is the operand slot's type. ``ColumnOperation.__init__``
accepts ``value: Any`` and the helper calls ``str(value)`` on it, so a
caller-supplied object whose ``__str__`` rewrote the tree would widen that
closed world and could observe a stale name within the walk it corrupted. No
such type exists in this library; a foreign one would be violating this
contract.

The memo is deliberately **not** kept between derivations. Expression nodes do
get mutated after construction -- ``LazyDataFrame`` normalises column references
onto a copied node, ``GroupedData`` substitutes evaluated aggregates into one --
so a name cached across calls could go stale and serve a name for an operand the
node no longer has. A memo that lives only inside one walk cannot: nothing
mutates a node while that node's name is being derived.
"""

from __future__ import annotations

import functools
import threading
from typing import Any, Callable, Dict, Tuple, TypeVar

_Derivation = TypeVar("_Derivation", bound=Callable[[Any], str])


class _MemoState(threading.local):
    """Per-thread state for the derivation currently in flight.

    Thread-local because two threads may derive names concurrently and a shared
    dict would let one thread's ``depth`` bookkeeping clear the other's entries.
    """

    def __init__(self) -> None:
        #: Nesting depth of the derivation in flight. ``0`` means no walk is
        #: running, so the cache must be empty.
        self.depth: int = 0
        #: ``(id(node), kind)`` -> ``(node, derived string)``. The node is kept
        #: in the value to pin it for the lifetime of the walk: were it
        #: collected, a later object could reuse its ``id()`` and read the dead
        #: node's cached name.
        self.cache: Dict[Tuple[int, str], Tuple[Any, str]] = {}


_STATE = _MemoState()


def memoise_within_pass(kind: str) -> Callable[[_Derivation], _Derivation]:
    """Memoise a pure, tree-recursive name derivation for one outermost walk.

    Args:
        kind: Discriminator for the derivation being wrapped. ``name``,
            ``str`` and ``_generate_name`` return *different* strings for the
            same node, so each needs its own cache slot.

    Returns:
        A decorator for a no-argument method that derives a string from ``self``.
    """

    def decorate(func: _Derivation) -> _Derivation:
        @functools.wraps(func)
        def wrapper(self: Any) -> str:
            state = _STATE
            key = (id(self), kind)
            hit = state.cache.get(key)
            if hit is not None:
                return hit[1]

            try:
                # Inside the `try`, not before it. A stranded-HIGH counter is
                # the harmful direction: it silently turns this into a
                # permanent global cache that never invalidates and pins every
                # node it ever saw, and the clamp below cannot recover it.
                # Putting the increment before the `try` leaves a window where
                # it has already run but the handler is not yet installed --
                # reachable by a `RecursionError` raised from the attribute
                # access itself, which is realistic in a library that walks
                # deep trees. Inside, that same failure strands the counter
                # LOW, which the clamp absorbs.
                #
                # This narrows the exposure; it does not close it. An
                # asynchronous exception delivered at the first line of the
                # `finally`, before the decrement runs, still strands the
                # counter high -- measured with `sys.settrace`: depth 1
                # afterwards, the next derivation serves a stale name. That
                # residual window is irreducible in pure Python, because the
                # `finally` cannot itself be made atomic.
                state.depth += 1
                result = func(self)
            finally:
                state.depth -= 1
                # Back at the top: the walk is over, so every entry becomes
                # potentially stale and is dropped. This also runs when `func`
                # raised, which is why it lives in `finally`. `<= 0` rather than
                # `== 0` so a counter that slipped negative re-synchronises
                # instead of never clearing again. It cannot rescue a counter
                # stranded high -- see the note above.
                if state.depth <= 0:
                    state.depth = 0
                    state.cache.clear()

            # Only worth remembering while an enclosing walk can still ask for
            # this node again; at depth 0 the cache was just cleared.
            if state.depth:
                state.cache[key] = (self, result)
            return result

        return wrapper  # type: ignore[return-value]

    return decorate


__all__ = ["memoise_within_pass"]
