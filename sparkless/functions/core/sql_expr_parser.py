"""SQL expression parsing for :func:`F.expr` compatibility.

``F.expr`` accepts SQL syntax (``"id IS NOT NULL"``) rather than Python
expressions (``"col('id').isNotNull()"``). The work is done by
:mod:`sparkless.functions.core.sql_expr`: a tokenizer, a recursive-descent
parser with Spark's operator precedence, and a binder that dispatches to the
real ``F`` functions.

This module is the stable entry point those three stages sit behind.
"""

from typing import Any, Union

from .column import Column, ColumnOperation
from .sql_expr import SQLExpressionBinder, parse

__all__ = ["SQLExprParser"]


class SQLExprParser:
    """Parses SQL expressions into sparkless column expressions.

    Supported:

    * column references, qualified (``a.b``) and quoted (`` `a b` ``)
    * literals: numbers, strings, ``TRUE``/``FALSE``/``NULL``
    * arithmetic (``+ - * / %``) and ``||``, at Spark's precedence
    * comparisons, ``AND``/``OR``/``NOT``, ``IS [NOT] NULL``
    * ``BETWEEN``, ``IN``, ``LIKE``, ``ILIKE``, ``RLIKE``, ``REGEXP``
    * ``CASE WHEN``, ``CAST``/``TRY_CAST``
    * any function in the ``F`` namespace, with all of its arguments

    Anything else raises ``ParseException``. The parser never returns a value
    it could not compute.
    """

    @staticmethod
    def parse(expr: str) -> Union[Column, ColumnOperation, Any]:
        """Parse a SQL expression string into a column expression.

        Args:
            expr: SQL expression string, e.g. ``"quantity * 2 + 1"``.

        Returns:
            The ``Column``, ``ColumnOperation`` or ``CaseWhen`` the expression
            denotes. An expression that is nothing but a literal (``"TRUE"``,
            ``"'x'"``, ``"123"``) returns that literal as a plain Python
            value; :func:`F.expr` wraps it in a column at its own boundary.

        Raises:
            ParseException: If the SQL is invalid or has no sparkless
                equivalent.
        """
        return SQLExpressionBinder(expr).bind(parse(expr))
