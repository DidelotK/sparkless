"""Recursive-descent parser for the SQL expression grammar of ``F.expr``.

The grammar below is Spark's operator precedence, lowest binding first. The
implementation that this replaced had no precedence at all: it tried operators
in the fixed order ``* / % + -``, split the string at the *first* occurrence of
the first one it found, and required the split to yield exactly two pieces. So
``q * 2 + 1`` split at ``*`` into ``q`` and ``2 + 1`` and bound as
``q * (2 + 1)`` -- 75 where Spark says 51 -- while ``q - 5 - 3`` split into
three pieces, fell through every branch, and was reported as an invalid
identifier.

::

    expression      := or_expression
    or_expression   := and_expression (OR and_expression)*
    and_expression  := not_expression (AND not_expression)*
    not_expression  := NOT not_expression | predicate
    predicate       := additive (comparison_suffix)*
    comparison_suffix
                    := (= | == | <> | != | < | <= | > | >= | <=>) additive
                     | IS [NOT] (NULL | TRUE | FALSE)
                     | [NOT] BETWEEN additive AND additive
                     | [NOT] IN "(" expression ("," expression)* ")"
                     | [NOT] (LIKE | ILIKE | RLIKE | REGEXP) additive
    additive        := multiplicative ((+ | - | "||") multiplicative)*
    multiplicative  := unary ((* | / | % | DIV) unary)*
    unary           := (- | + | ~) unary | primary
    primary         := literal | column | function_call | case | cast
                     | interval | lambda | "(" expression ")"

All binary operators are parsed left-associative, which is what makes
``q - 5 - 3`` mean ``(q - 5) - 3`` and ``salary / 2 / 5`` mean
``(salary / 2) / 5``.
"""

from typing import List, NoReturn, Optional, Sequence, Tuple

from ....core.exceptions.analysis import ParseException
from . import nodes
from .tokenizer import Token, TokenType, tokenize

_COMPARISON_OPERATORS = frozenset({"=", "==", "<>", "!=", "<", "<=", ">", ">=", "<=>"})
_ADDITIVE_OPERATORS = frozenset({"+", "-", "||"})
_MULTIPLICATIVE_OPERATORS = frozenset({"*", "/", "%"})
_PATTERN_KEYWORDS = frozenset({"LIKE", "ILIKE", "RLIKE", "REGEXP"})

# Words that may never be read as a column name. Anything outside this set is
# available as an identifier, which is what lets a column called ``filter`` or
# ``value`` keep working.
_RESERVED = frozenset(
    {
        "AND",
        "OR",
        "NOT",
        "IS",
        "NULL",
        "TRUE",
        "FALSE",
        "BETWEEN",
        "IN",
        "LIKE",
        "ILIKE",
        "RLIKE",
        "REGEXP",
        "CASE",
        "WHEN",
        "THEN",
        "ELSE",
        "END",
        "CAST",
        "TRY_CAST",
        "AS",
        "INTERVAL",
        "DISTINCT",
        "DIV",
    }
)

# Recognised by the parser purely so that the binder can reject them with an
# explanation instead of the tokenizer failing on an unfamiliar shape.
_INTERVAL_UNITS = frozenset(
    {
        "YEAR",
        "YEARS",
        "MONTH",
        "MONTHS",
        "WEEK",
        "WEEKS",
        "DAY",
        "DAYS",
        "HOUR",
        "HOURS",
        "MINUTE",
        "MINUTES",
        "SECOND",
        "SECONDS",
        "MILLISECOND",
        "MILLISECONDS",
        "MICROSECOND",
        "MICROSECONDS",
    }
)


class SQLExpressionParser:
    """Parses one SQL expression into an AST.

    The parser never imports a sparkless function and never evaluates anything;
    :mod:`sparkless.functions.core.sql_expr.binder` does that from the AST.
    """

    def __init__(self, source: str) -> None:
        """Initialise the parser over ``source``.

        Args:
            source: The SQL expression text.
        """
        self.source = source
        self.tokens: List[Token] = tokenize(source)
        self.index = 0

    # -- token helpers ----------------------------------------------------

    @property
    def current(self) -> Token:
        """The token about to be consumed."""
        return self.tokens[self.index]

    def _advance(self) -> Token:
        """Consume and return the current token."""
        token = self.tokens[self.index]
        if token.type is not TokenType.END:
            self.index += 1
        return token

    def _at_keyword(self, *keywords: str) -> bool:
        """Whether the current token is one of ``keywords``."""
        token = self.current
        return token.type is TokenType.IDENTIFIER and token.upper in keywords

    def _accept_keyword(self, *keywords: str) -> Optional[Token]:
        """Consume the current token if it is one of ``keywords``."""
        if self._at_keyword(*keywords):
            return self._advance()
        return None

    def _expect_keyword(self, keyword: str) -> Token:
        """Consume ``keyword`` or fail."""
        token = self._accept_keyword(keyword)
        if token is None:
            self._fail(f"expected {keyword}")
        return token

    def _at_operator(self, *symbols: str) -> bool:
        """Whether the current token is one of the given operator symbols."""
        token = self.current
        return token.type is TokenType.OPERATOR and token.text in symbols

    def _accept_operator(self, *symbols: str) -> Optional[Token]:
        """Consume the current token if it is one of ``symbols``."""
        if self._at_operator(*symbols):
            return self._advance()
        return None

    def _at_punctuation(self, symbol: str) -> bool:
        """Whether the current token is the given punctuation."""
        token = self.current
        return token.type is TokenType.PUNCTUATION and token.text == symbol

    def _accept_punctuation(self, symbol: str) -> Optional[Token]:
        """Consume the current token if it is the given punctuation."""
        if self._at_punctuation(symbol):
            return self._advance()
        return None

    def _expect_punctuation(self, symbol: str) -> Token:
        """Consume the given punctuation or fail."""
        token = self._accept_punctuation(symbol)
        if token is None:
            self._fail(f"expected {symbol!r}")
        return token

    def _fail(self, what: str) -> NoReturn:
        """Raise a ParseException pointing at the current token."""
        token = self.current
        found = "end of expression" if token.type is TokenType.END else repr(token.text)
        raise ParseException(
            f"Invalid SQL expression {self.source!r}: {what} at position "
            f"{token.position}, found {found}"
        )

    # -- entry point ------------------------------------------------------

    def parse(self) -> nodes.Node:
        """Parse the whole source and require it to be fully consumed.

        Returns:
            The root AST node.

        Raises:
            ParseException: On any syntax error, including trailing input.
        """
        node = self._parse_expression()
        if self.current.type is not TokenType.END:
            self._fail("unexpected trailing input")
        return node

    # -- grammar ----------------------------------------------------------

    def _parse_expression(self) -> nodes.Node:
        """Parse a full expression (lowest precedence)."""
        return self._parse_or()

    def _parse_or(self) -> nodes.Node:
        """Parse ``a OR b OR c``, left-associative."""
        node = self._parse_and()
        while self._accept_keyword("OR") is not None:
            node = nodes.BinaryOperation("OR", node, self._parse_and())
        return node

    def _parse_and(self) -> nodes.Node:
        """Parse ``a AND b AND c``, left-associative."""
        node = self._parse_not()
        while self._accept_keyword("AND") is not None:
            node = nodes.BinaryOperation("AND", node, self._parse_not())
        return node

    def _parse_not(self) -> nodes.Node:
        """Parse ``NOT x``."""
        if self._accept_keyword("NOT") is not None:
            return nodes.UnaryOperation("NOT", self._parse_not())
        return self._parse_predicate()

    def _parse_predicate(self) -> nodes.Node:
        """Parse a comparison and its SQL predicate suffixes."""
        node = self._parse_additive()

        while True:
            if self.current.type is TokenType.OPERATOR and (
                self.current.text in _COMPARISON_OPERATORS
            ):
                operator = self._advance().text
                node = nodes.BinaryOperation(operator, node, self._parse_additive())
                continue

            if self._accept_keyword("IS") is not None:
                node = self._parse_is_suffix(node)
                continue

            negated = self._accept_keyword("NOT") is not None
            suffix = self._parse_predicate_suffix(node, negated)
            if suffix is None:
                if negated:
                    self._fail("expected BETWEEN, IN, LIKE, ILIKE, RLIKE or REGEXP")
                return node
            node = suffix

    def _parse_is_suffix(self, operand: nodes.Node) -> nodes.Node:
        """Parse what follows ``IS``: ``[NOT] NULL|TRUE|FALSE``."""
        negated = self._accept_keyword("NOT") is not None
        if self._accept_keyword("NULL") is not None:
            return nodes.IsNull(operand, negated=negated)
        if self._accept_keyword("TRUE") is not None:
            return nodes.IsBoolean(operand, expected=True, negated=negated)
        if self._accept_keyword("FALSE") is not None:
            return nodes.IsBoolean(operand, expected=False, negated=negated)
        self._fail("expected NULL, TRUE or FALSE after IS")

    def _parse_predicate_suffix(
        self, operand: nodes.Node, negated: bool
    ) -> Optional[nodes.Node]:
        """Parse ``BETWEEN``/``IN``/``LIKE``-family suffixes, if present."""
        if self._accept_keyword("BETWEEN") is not None:
            lower = self._parse_additive()
            self._expect_keyword("AND")
            upper = self._parse_additive()
            return nodes.Between(operand, lower, upper, negated=negated)

        if self._accept_keyword("IN") is not None:
            self._expect_punctuation("(")
            items: List[nodes.Node] = []
            if not self._at_punctuation(")"):
                items.append(self._parse_expression())
                while self._accept_punctuation(",") is not None:
                    items.append(self._parse_expression())
            self._expect_punctuation(")")
            return nodes.InList(operand, items, negated=negated)

        token = self.current
        if token.type is TokenType.IDENTIFIER and token.upper in _PATTERN_KEYWORDS:
            kind = self._advance().upper
            pattern = self._parse_additive()
            return nodes.PatternMatch(kind, operand, pattern, negated=negated)

        return None

    def _parse_additive(self) -> nodes.Node:
        """Parse ``+``, ``-`` and ``||``, left-associative."""
        node = self._parse_multiplicative()
        while self.current.type is TokenType.OPERATOR and (
            self.current.text in _ADDITIVE_OPERATORS
        ):
            operator = self._advance().text
            node = nodes.BinaryOperation(operator, node, self._parse_multiplicative())
        return node

    def _parse_multiplicative(self) -> nodes.Node:
        """Parse ``*``, ``/``, ``%`` and ``DIV``, left-associative."""
        node = self._parse_unary()
        while True:
            if self.current.type is TokenType.OPERATOR and (
                self.current.text in _MULTIPLICATIVE_OPERATORS
            ):
                operator = self._advance().text
                node = nodes.BinaryOperation(operator, node, self._parse_unary())
                continue
            if self._accept_keyword("DIV") is not None:
                node = nodes.BinaryOperation("DIV", node, self._parse_unary())
                continue
            return node

    def _parse_unary(self) -> nodes.Node:
        """Parse prefix ``-``, ``+`` and ``~``."""
        token = self._accept_operator("-", "+", "~")
        if token is not None:
            operand = self._parse_unary()
            if token.text == "+":
                return operand
            if token.text == "-" and isinstance(operand, nodes.Literal):
                value = operand.value
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    return nodes.Literal(-value)
            return nodes.UnaryOperation(token.text, operand)
        return self._parse_primary()

    def _parse_primary(self) -> nodes.Node:
        """Parse a literal, column, call, or parenthesised expression."""
        token = self.current

        if token.type is TokenType.NUMBER or token.type is TokenType.STRING:
            self._advance()
            return nodes.Literal(token.value)

        if token.type is TokenType.OPERATOR and token.text == "*":
            self._advance()
            return nodes.Star()

        if self._at_punctuation("("):
            return self._parse_parenthesised()

        if token.type is TokenType.QUOTED_IDENTIFIER:
            self._advance()
            return self._parse_reference_suffix([str(token.value)])

        if token.type is TokenType.IDENTIFIER:
            return self._parse_identifier_primary(token)

        self._fail("expected an expression")

    def _parse_parenthesised(self) -> nodes.Node:
        """Parse ``( ... )``, which may also be a lambda parameter list."""
        start = self.index
        self._expect_punctuation("(")

        parameters = self._try_parse_lambda_parameters()
        if parameters is not None:
            body = self._parse_expression()
            return nodes.Lambda(parameters, body)

        self.index = start
        self._expect_punctuation("(")
        node = self._parse_expression()
        self._expect_punctuation(")")
        return node

    def _try_parse_lambda_parameters(self) -> Optional[Sequence[str]]:
        """Parse ``(x, y) ->`` if that is what follows, else return None."""
        parameters: List[str] = []
        while self.current.type is TokenType.IDENTIFIER:
            parameters.append(self._advance().text)
            if self._accept_punctuation(",") is not None:
                continue
            break
        if not parameters:
            return None
        if self._accept_punctuation(")") is None:
            return None
        if self._accept_operator("->") is None:
            return None
        return parameters

    def _parse_identifier_primary(self, token: Token) -> nodes.Node:
        """Parse an identifier: keyword literal, call, lambda or column."""
        upper = token.upper

        if upper == "CASE":
            return self._parse_case()
        if upper in ("CAST", "TRY_CAST"):
            return self._parse_cast(try_cast=upper == "TRY_CAST")
        if upper == "INTERVAL":
            return self._parse_interval()
        if upper == "NULL":
            self._advance()
            return nodes.Literal(None)
        if upper == "TRUE":
            self._advance()
            return nodes.Literal(True)
        if upper == "FALSE":
            self._advance()
            return nodes.Literal(False)

        if upper in _RESERVED:
            self._fail(f"{token.text} is not valid here")

        self._advance()

        if self._at_operator("->"):
            self._advance()
            return nodes.Lambda([token.text], self._parse_expression())

        if self._at_punctuation("("):
            return self._parse_function_call(token.text)

        return self._parse_reference_suffix([token.text])

    def _parse_reference_suffix(self, parts: List[str]) -> nodes.Node:
        """Parse the ``.b.c`` tail of a qualified column reference."""
        while self._at_punctuation("."):
            self._advance()
            token = self.current
            if token.type not in (TokenType.IDENTIFIER, TokenType.QUOTED_IDENTIFIER):
                self._fail("expected a field name after '.'")
            self._advance()
            parts.append(str(token.value))
        return nodes.ColumnReference(parts)

    def _parse_function_call(self, name: str) -> nodes.Node:
        """Parse ``name(...)`` including ``DISTINCT`` and ``*``."""
        self._expect_punctuation("(")
        distinct = self._accept_keyword("DISTINCT") is not None
        arguments: List[nodes.Node] = []

        if not self._at_punctuation(")"):
            arguments.append(self._parse_expression())
            while self._accept_punctuation(",") is not None:
                arguments.append(self._parse_expression())

        self._expect_punctuation(")")
        return nodes.FunctionCall(name, arguments, distinct=distinct)

    def _parse_case(self) -> nodes.Node:
        """Parse ``CASE [operand] WHEN ... THEN ... [ELSE ...] END``."""
        self._expect_keyword("CASE")

        operand: Optional[nodes.Node] = None
        if not self._at_keyword("WHEN"):
            operand = self._parse_expression()

        branches: List[Tuple[nodes.Node, nodes.Node]] = []
        while self._accept_keyword("WHEN") is not None:
            condition = self._parse_expression()
            self._expect_keyword("THEN")
            branches.append((condition, self._parse_expression()))

        if not branches:
            self._fail("expected at least one WHEN in CASE")

        else_value: Optional[nodes.Node] = None
        if self._accept_keyword("ELSE") is not None:
            else_value = self._parse_expression()

        self._expect_keyword("END")
        return nodes.CaseWhen(list(branches), else_value, operand)

    def _parse_cast(self, try_cast: bool) -> nodes.Node:
        """Parse ``CAST(x AS type)`` / ``TRY_CAST(x AS type)``."""
        self._advance()
        self._expect_punctuation("(")
        operand = self._parse_expression()
        self._expect_keyword("AS")
        type_name = self._parse_type_name()
        self._expect_punctuation(")")
        return nodes.Cast(operand, type_name, try_cast=try_cast)

    def _parse_type_name(self) -> str:
        """Parse a type name such as ``STRING``, ``DECIMAL(10,2)``."""
        token = self.current
        if token.type is not TokenType.IDENTIFIER:
            self._fail("expected a type name")
        start_offset = token.position
        self._advance()

        end_offset = self.current.position
        depth = 0
        while True:
            token = self.current
            if token.type is TokenType.END:
                break
            if token.type is TokenType.PUNCTUATION and token.text == "(":
                depth += 1
            elif token.type is TokenType.PUNCTUATION and token.text == ")":
                if depth == 0:
                    break
                depth -= 1
            elif token.type is TokenType.OPERATOR and token.text == "<":
                depth += 1
            elif token.type is TokenType.OPERATOR and token.text == ">":
                depth -= 1
            self._advance()
            end_offset = token.position + len(token.text)

        return self.source[start_offset:end_offset].strip()

    def _parse_interval(self) -> nodes.Node:
        """Parse an ``INTERVAL`` literal well enough to reject it clearly."""
        start_offset = self.current.position
        self._advance()
        end_offset = self.current.position

        while True:
            token = self.current
            if token.type is TokenType.NUMBER or token.type is TokenType.STRING:
                self._advance()
                end_offset = token.position + len(token.text)
                continue
            if token.type is TokenType.IDENTIFIER and token.upper in _INTERVAL_UNITS:
                self._advance()
                end_offset = token.position + len(token.text)
                continue
            break

        return nodes.Interval(self.source[start_offset:end_offset].strip())


def parse(source: str) -> nodes.Node:
    """Parse a SQL expression into an AST.

    Args:
        source: The SQL expression text.

    Returns:
        The root AST node.

    Raises:
        ParseException: If ``source`` is empty or not a valid SQL expression.
    """
    if not source.strip():
        raise ParseException("Empty SQL expression")
    return SQLExpressionParser(source).parse()
