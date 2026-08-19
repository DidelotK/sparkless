"""Canonical JSON rendering for ``F.to_json``.

Spark's JSON writer applies three *different* NULL rules, and they are the
reason this cannot be ``json.dumps``:

=====================================  ======================================
NULL in a...                           renders as
=====================================  ======================================
struct field                           **omitted** -- ``struct(NULL as z,
                                       1 as a)`` gives ``{"a":1}``
array element                          ``null`` -- kept, so the array keeps
                                       its length
map value                              ``null`` -- kept
whole input column                     SQL NULL, not the string ``"null"``
=====================================  ======================================

A ``Decimal`` also keeps its scale unquoted (``1.50``, not ``"1.50"`` and not
``1.5``), which ``json.dumps`` cannot express, so the renderer here is
explicit rather than delegated.

Struct and map both evaluate to a Python ``dict`` in sparkless, so the value
alone cannot say which NULL rule applies. The *expression* can: callers pass
the operand of ``to_json`` and the renderer reads ``create_map`` and friends
as maps, ``struct``/``named_struct`` as structs -- recursively, using each
struct field's own argument expression.

A dict reached without an expression to describe it -- a struct column
referenced by name, ``F.to_json(F.col("s"))`` -- is rendered with **struct**
rules, that being what ``to_json`` is overwhelmingly applied to. The residual
divergence is narrow and worth stating plainly: a *map column referenced by
name* whose values contain NULL drops those entries, where Spark keeps them.
Maps built inline through ``F.create_map`` are detected and unaffected.

Rendering verified against PySpark 4.0.0 (``local[1]``). Timestamps are
rendered from the stored value without applying a session time zone, which
Spark does; only the date form is asserted in the tests.
"""

import base64
import datetime
import decimal
import json
import math
from typing import Any, List, Optional

__all__ = ["to_json_value"]

#: Operations that build a map. Everything else that evaluates to a ``dict``
#: is treated as a struct -- see the module docstring.
_MAP_OPERATIONS = frozenset(
    {
        "create_map",
        "map_from_arrays",
        "map_from_entries",
        "map_concat",
        "map_filter",
        "transform_keys",
        "transform_values",
    }
)

_STRUCT_OPERATIONS = frozenset({"struct", "named_struct"})


def _is_map_expression(expression: Any) -> bool:
    """Whether ``expression`` builds a map rather than a struct."""
    return getattr(expression, "operation", None) in _MAP_OPERATIONS


def _struct_field_expressions(expression: Any) -> Optional[List[Any]]:
    """Ordered per-field expressions of a struct, or ``None`` if unknown.

    Positional: ``build_struct_value`` fills the dict in this same order, so
    zipping the two lines each field up with the expression that produced it.
    """
    operation = getattr(expression, "operation", None)
    if operation not in _STRUCT_OPERATIONS:
        return None

    from .struct_builder import struct_argument_columns, struct_argument_source

    if operation == "named_struct":
        args = getattr(expression, "value", None) or ()
        if not isinstance(args, (list, tuple)):
            args = (args,)
        return [
            struct_argument_source(args[index + 1])
            for index in range(0, len(args) - 1, 2)
        ]
    return [
        struct_argument_source(item) for item in struct_argument_columns(expression)
    ]


def _render_string(value: str) -> str:
    """A JSON string literal, with the character itself rather than ``\\uXXXX``."""
    return json.dumps(value, ensure_ascii=False)


def _render_scalar(value: Any) -> str:
    """Render a non-collection value."""
    if value is None:
        return "null"
    # bool before int: bool is an int subclass and would render as 1/0.
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, decimal.Decimal):
        # str() keeps the declared scale, which Spark also keeps and which a
        # float round-trip would lose (1.50 -> 1.5).
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return '"NaN"'
        if math.isinf(value):
            return '"Infinity"' if value > 0 else '"-Infinity"'
        return repr(value)
    if isinstance(value, datetime.datetime):
        millis = value.microsecond // 1000
        return _render_string(value.strftime("%Y-%m-%dT%H:%M:%S.") + f"{millis:03d}Z")
    if isinstance(value, datetime.date):
        return _render_string(value.isoformat())
    if isinstance(value, (bytes, bytearray)):
        return _render_string(base64.b64encode(bytes(value)).decode("ascii"))
    return _render_string(str(value))


def _render(value: Any, expression: Any) -> str:
    """Render one value, using ``expression`` to tell a struct from a map."""
    if isinstance(value, dict):
        if _is_map_expression(expression):
            # A map keeps its NULL values.
            map_entries = [
                f"{_render_string(str(key))}:{_render(item, None)}"
                for key, item in value.items()
            ]
            return "{" + ",".join(map_entries) + "}"

        field_expressions = _struct_field_expressions(expression)
        entries = []
        for position, (key, item) in enumerate(value.items()):
            if item is None:
                # A NULL struct field is omitted entirely.
                continue
            field_expression = (
                field_expressions[position]
                if field_expressions is not None and position < len(field_expressions)
                else None
            )
            entries.append(
                f"{_render_string(str(key))}:{_render(item, field_expression)}"
            )
        return "{" + ",".join(entries) + "}"

    if isinstance(value, (list, tuple)):
        # An array keeps its NULL elements; its elements have no expression of
        # their own to describe them.
        return "[" + ",".join(_render(item, None) for item in value) + "]"

    return _render_scalar(value)


def to_json_value(value: Any, expression: Any = None) -> Optional[str]:
    """Serialize a struct, map or array value to a JSON string.

    Args:
        value: The already-evaluated operand value.
        expression: The operand expression, used only to tell a struct from a
            map. Optional; see the module docstring for the default.

    Returns:
        The JSON text, or ``None`` for a NULL input -- SQL NULL, never the
        string ``"null"``.
    """
    if value is None:
        return None
    return _render(value, expression)
