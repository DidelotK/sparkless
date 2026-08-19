"""Unit tests for the SQL expression parser behind ``F.expr``.

`tests/parity/functions/test_expr_sql_parity.py` pins the *values* against
PySpark 4.0.0. This file pins the two things a value test cannot see:

1. **The shape the parser produced.** ``q * 2 + 1`` and ``q * (2 + 1)`` are
   both numbers; only the tree says which one was built. Asserting on the AST
   catches a precedence regression even if some evaluator happens to paper
   over it.
2. **That an expression sparkless cannot evaluate raises.** The defect this
   parser replaced was silent -- a dropped argument yielded NULL and nothing
   warned. A parser that cannot express something must say so, so every
   unsupported construct here is asserted to raise ``ParseException`` rather
   than return a column.
"""

from typing import Any

import pytest

from sparkless.core.exceptions.analysis import ParseException
from sparkless.functions.core.sql_expr import nodes, parse
from sparkless.functions.core.sql_expr.tokenizer import TokenType, tokenize


class TestTokenizer:
    """The token layer the previous implementation did not have."""

    def test_operators_inside_string_literals_are_not_operators(self) -> None:
        """``' AND '`` is one string token, not a logical operator."""
        tokens = tokenize("dept = ' AND '")
        assert [token.type for token in tokens[:3]] == [
            TokenType.IDENTIFIER,
            TokenType.OPERATOR,
            TokenType.STRING,
        ]
        assert tokens[2].value == " AND "

    def test_keywords_inside_identifiers_are_not_keywords(self) -> None:
        """``brand`` contains ``AND`` but is a single identifier."""
        tokens = tokenize("brand")
        assert tokens[0].type is TokenType.IDENTIFIER
        assert tokens[0].value == "brand"

    def test_doubled_quote_escapes_inside_a_string(self) -> None:
        """``'it''s'`` decodes to ``it's``."""
        assert tokenize("'it''s'")[0].value == "it's"

    def test_backticked_identifier_keeps_its_spaces(self) -> None:
        """A quoted identifier is not normalised."""
        token = tokenize("`total revenue`")[0]
        assert token.type is TokenType.QUOTED_IDENTIFIER
        assert token.value == "total revenue"

    def test_tokens_carry_their_position(self) -> None:
        """Positions are what lets an error point at the right character."""
        tokens = tokenize("a + b")
        assert [token.position for token in tokens[:3]] == [0, 2, 4]

    def test_unterminated_string_raises(self) -> None:
        """An unclosed literal is a parse error, not a silent truncation."""
        with pytest.raises(ParseException, match="Unterminated string"):
            tokenize("dept = 'eng")


class TestArithmeticPrecedence:
    """Precedence and associativity, asserted on the tree."""

    def test_multiplication_binds_tighter_than_addition(self) -> None:
        """``q * 2 + 1`` is ``(q * 2) + 1``, not ``q * (2 + 1)``."""
        tree = parse("q * 2 + 1")
        assert isinstance(tree, nodes.BinaryOperation)
        assert tree.operator == "+"
        assert isinstance(tree.left, nodes.BinaryOperation)
        assert tree.left.operator == "*"
        assert tree.right == nodes.Literal(1)

    def test_addition_after_multiplication(self) -> None:
        """``age + 2 * 3`` is ``age + (2 * 3)``."""
        tree = parse("age + 2 * 3")
        assert isinstance(tree, nodes.BinaryOperation)
        assert tree.operator == "+"
        assert isinstance(tree.right, nodes.BinaryOperation)
        assert tree.right.operator == "*"

    def test_subtraction_is_left_associative(self) -> None:
        """``q - 5 - 3`` is ``(q - 5) - 3``."""
        tree = parse("q - 5 - 3")
        assert isinstance(tree, nodes.BinaryOperation)
        assert tree.operator == "-"
        assert tree.right == nodes.Literal(3)
        assert isinstance(tree.left, nodes.BinaryOperation)
        assert tree.left.operator == "-"

    def test_parentheses_override_precedence(self) -> None:
        """``q * (2 + 1)`` keeps the addition inside the multiplication."""
        tree = parse("q * (2 + 1)")
        assert isinstance(tree, nodes.BinaryOperation)
        assert tree.operator == "*"
        assert isinstance(tree.right, nodes.BinaryOperation)
        assert tree.right.operator == "+"

    def test_comparison_binds_looser_than_arithmetic(self) -> None:
        """``q * 2 + 1 > 50`` compares the whole arithmetic expression."""
        tree = parse("q * 2 + 1 > 50")
        assert isinstance(tree, nodes.BinaryOperation)
        assert tree.operator == ">"
        assert isinstance(tree.left, nodes.BinaryOperation)
        assert tree.left.operator == "+"

    def test_and_binds_tighter_than_or(self) -> None:
        """``a OR b AND c`` is ``a OR (b AND c)``."""
        tree = parse("a = 1 OR b = 2 AND c = 3")
        assert isinstance(tree, nodes.BinaryOperation)
        assert tree.operator == "OR"
        assert isinstance(tree.right, nodes.BinaryOperation)
        assert tree.right.operator == "AND"

    def test_not_binds_looser_than_comparison(self) -> None:
        """``NOT a = 1`` negates the comparison."""
        tree = parse("NOT a = 1")
        assert isinstance(tree, nodes.UnaryOperation)
        assert tree.operator == "NOT"
        assert isinstance(tree.operand, nodes.BinaryOperation)


class TestCallArguments:
    """Arguments must survive parsing -- all of them, in order."""

    def test_every_argument_is_kept(self) -> None:
        """``concat(sku, dept)`` has two arguments, not zero."""
        tree = parse("concat(sku, dept)")
        assert isinstance(tree, nodes.FunctionCall)
        assert tree.name == "concat"
        assert tree.arguments == [
            nodes.ColumnReference(["sku"]),
            nodes.ColumnReference(["dept"]),
        ]

    def test_an_operator_inside_a_call_does_not_split_the_call(self) -> None:
        """``round(salary / 3, 2)`` is one call with two arguments.

        The previous implementation split on ``/`` without tracking
        parentheses, producing the fragments ``round(salary`` and ``3, 2)``.
        """
        tree = parse("round(salary / 3, 2)")
        assert isinstance(tree, nodes.FunctionCall)
        assert len(tree.arguments) == 2
        assert isinstance(tree.arguments[0], nodes.BinaryOperation)
        assert tree.arguments[1] == nodes.Literal(2)

    def test_a_comma_inside_a_string_does_not_split_arguments(self) -> None:
        """``concat(a, ', ', b)`` has three arguments."""
        tree = parse("concat(a, ', ', b)")
        assert isinstance(tree, nodes.FunctionCall)
        assert len(tree.arguments) == 3

    def test_nested_calls_keep_their_own_arguments(self) -> None:
        """Nesting does not flatten or drop inner arguments."""
        tree = parse("concat(substr(md5(sku), 1, 8), '-', sku)")
        assert isinstance(tree, nodes.FunctionCall)
        assert len(tree.arguments) == 3
        inner = tree.arguments[0]
        assert isinstance(inner, nodes.FunctionCall)
        assert inner.name == "substr"
        assert len(inner.arguments) == 3

    def test_no_argument_call(self) -> None:
        """``current_date()`` parses with an empty argument list."""
        tree = parse("current_date()")
        assert isinstance(tree, nodes.FunctionCall)
        assert tree.arguments == []

    def test_count_star(self) -> None:
        """``count(*)`` keeps the star as an argument node."""
        tree = parse("count(*)")
        assert isinstance(tree, nodes.FunctionCall)
        assert isinstance(tree.arguments[0], nodes.Star)

    def test_count_distinct(self) -> None:
        """``count(DISTINCT dept)`` records the DISTINCT."""
        tree = parse("count(DISTINCT dept)")
        assert isinstance(tree, nodes.FunctionCall)
        assert tree.distinct is True


class TestPredicateForms:
    """SQL predicate syntax the previous parser could not see."""

    def test_between(self) -> None:
        """``BETWEEN`` does not lose its upper bound to ``AND``."""
        tree = parse("revenue BETWEEN 0 AND 150")
        assert isinstance(tree, nodes.Between)
        assert tree.lower == nodes.Literal(0)
        assert tree.upper == nodes.Literal(150)

    def test_between_inside_a_conjunction(self) -> None:
        """The ``AND`` after a BETWEEN's bounds is the outer conjunction."""
        tree = parse("revenue BETWEEN 0 AND 150 AND dept = 'eng'")
        assert isinstance(tree, nodes.BinaryOperation)
        assert tree.operator == "AND"
        assert isinstance(tree.left, nodes.Between)

    def test_not_between(self) -> None:
        """``NOT BETWEEN`` is one predicate, negated."""
        tree = parse("revenue NOT BETWEEN 0 AND 150")
        assert isinstance(tree, nodes.Between)
        assert tree.negated is True

    def test_is_not_null(self) -> None:
        """``IS NOT NULL`` is negated; ``IS NULL`` is not."""
        assert parse("a IS NOT NULL") == nodes.IsNull(
            nodes.ColumnReference(["a"]), negated=True
        )
        assert parse("a IS NULL") == nodes.IsNull(
            nodes.ColumnReference(["a"]), negated=False
        )

    def test_is_null_on_a_call(self) -> None:
        """``IS NULL`` applies to the whole call, not to its last argument."""
        tree = parse("coalesce(a, b) IS NULL")
        assert isinstance(tree, nodes.IsNull)
        assert isinstance(tree.operand, nodes.FunctionCall)

    def test_rlike(self) -> None:
        """``RLIKE`` keeps the pattern with its regex metacharacters."""
        tree = parse("sku RLIKE '^[A-Z]{3}$'")
        assert isinstance(tree, nodes.PatternMatch)
        assert tree.kind == "RLIKE"
        assert tree.pattern == nodes.Literal("^[A-Z]{3}$")

    def test_qualified_column_reference(self) -> None:
        """``scores.risk`` is one qualified reference."""
        tree = parse("scores.risk")
        assert isinstance(tree, nodes.ColumnReference)
        assert tree.name == "scores.risk"

    def test_backticked_column_in_a_cast(self) -> None:
        """``try_cast(`revenue` AS DOUBLE)`` parses as a cast."""
        tree = parse("try_cast(`revenue` AS DOUBLE)")
        assert isinstance(tree, nodes.Cast)
        assert tree.try_cast is True
        assert tree.type_name == "DOUBLE"

    def test_a_column_may_be_named_like_a_function(self) -> None:
        """``value`` and ``filter`` are usable as column names."""
        assert parse("value > 1").left == nodes.ColumnReference(["value"])  # type: ignore[attr-defined]


class TestUnsupportedExpressionsRaise:
    """What sparkless cannot evaluate must raise, never return a value."""

    @pytest.mark.parametrize(  # type: ignore[misc,untyped-decorator]
        "expression",
        [
            "",
            "   ",
            "q +",
            "concat(a,",
            "dept = 'eng",
            "CASE WHEN a THEN 1",
            "a IS",
            "a NOT 5",
            "* 3",
        ],
    )
    def test_malformed_expressions_raise(self, expression: str) -> None:
        """A syntax error is a ParseException, not a column."""
        with pytest.raises(ParseException):
            parse_and_bind_expression(expression)

    def test_an_unknown_function_raises(self) -> None:
        """A function sparkless does not implement must say so.

        The old parser bound *every* unknown name to a hand-built operation
        that evaluated to NULL for every row -- a silent wrong answer.
        """
        with pytest.raises(ParseException, match="Undefined function 'no_such_fn'"):
            parse_and_bind_expression("no_such_fn(a)")

    def test_a_month_based_interval_raises(self) -> None:
        """Months cannot be approximated in days, so they are refused.

        Day-based intervals bind (see the parity tests); month-based ones
        would need ``F.add_months``, which returns NULL for every row.
        """
        with pytest.raises(ParseException, match="day-based INTERVAL"):
            parse_and_bind_expression("day_col - INTERVAL 5 YEARS")

    def test_a_day_interval_binds_to_a_timedelta(self) -> None:
        """``INTERVAL 90 DAYS`` on its own is a 90-day timedelta literal."""
        import datetime

        bound = parse_and_bind_expression("INTERVAL 90 DAYS")

        assert bound.value == datetime.timedelta(days=90)

    def test_a_higher_order_lambda_raises(self) -> None:
        """A lambda would bind to a function that returns NULL for every row.

        sparkless's ``filter``/``exists``/``forall`` are defective
        (solya-data-platform#2419), so binding the lambda would replace
        today's loud failure with a silent one.
        """
        with pytest.raises(ParseException, match="2419"):
            parse_and_bind_expression("filter(tags, x -> x IS NOT NULL)")

    def test_bitwise_not_raises(self) -> None:
        """Every sparkless spelling of bitwise NOT evaluates to NULL."""
        with pytest.raises(ParseException, match="bitwise NOT"):
            parse_and_bind_expression("~flags")

    def test_a_column_where_a_literal_is_required_raises(self) -> None:
        """``substring``'s length is a literal in sparkless, not a column."""
        with pytest.raises(ParseException, match="takes a literal"):
            parse_and_bind_expression("substring(name, 1, other_col)")

    def test_a_star_outside_count_raises(self) -> None:
        """``*`` is only meaningful as ``count(*)``."""
        with pytest.raises(ParseException, match=r"count\(\*\)"):
            parse_and_bind_expression("sum(*)")


def parse_and_bind_expression(expression: str) -> Any:
    """Parse and bind ``expression`` the way ``F.expr`` does.

    Binding is where an unresolvable function or an unsupported construct is
    rejected, so the "must raise" tests have to go through it rather than
    stopping at :func:`parse`.
    """
    from sparkless.functions.core.sql_expr import parse_and_bind

    return parse_and_bind(expression)
