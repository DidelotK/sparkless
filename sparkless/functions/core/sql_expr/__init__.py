"""SQL expression support for :func:`F.expr`.

Three stages, each testable on its own:

``tokenizer``
    Source text to tokens. One pass, positions kept for error messages.
``parser``
    Tokens to an AST, with Spark's operator precedence and left associativity.
``binder``
    AST to sparkless columns, dispatching every function call to the real
    ``F`` namespace.
"""

from typing import Any

from . import nodes
from .binder import SQLExpressionBinder, bind
from .parser import parse


def parse_and_bind(source: str) -> Any:
    """Parse a SQL expression and bind it to a sparkless column.

    Args:
        source: The SQL expression text, e.g. ``"concat(sku, dept)"``.

    Returns:
        A ``Column``, ``ColumnOperation`` or ``CaseWhen``.

    Raises:
        ParseException: If the expression is not valid SQL, or uses a
            construct sparkless cannot evaluate. It never returns a value it
            could not compute.
    """
    return bind(parse(source), source)


__all__ = ["SQLExpressionBinder", "bind", "nodes", "parse", "parse_and_bind"]
