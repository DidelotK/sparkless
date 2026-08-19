"""Tokenizer for the SQL expression grammar accepted by :func:`F.expr`.

The previous implementation had no token layer at all: it looked for operators
and keywords with :func:`re.search` over the raw expression text, so ``AND``
inside an identifier, ``-`` inside a function call and ``LIKE`` inside a string
literal were all indistinguishable from the real thing. Everything downstream
(precedence, associativity, argument splitting) inherited that ambiguity.

This module turns the source text into a flat list of tokens exactly once.
Every token carries its offset in the source so the parser can point at the
offending character when it rejects an expression.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional, Sequence, Tuple

from ....core.exceptions.analysis import ParseException


class TokenType(Enum):
    """The lexical classes of the SQL expression grammar."""

    NUMBER = "NUMBER"
    STRING = "STRING"
    IDENTIFIER = "IDENTIFIER"
    QUOTED_IDENTIFIER = "QUOTED_IDENTIFIER"
    OPERATOR = "OPERATOR"
    PUNCTUATION = "PUNCTUATION"
    END = "END"


@dataclass(frozen=True)
class Token:
    """One lexical token and where it came from.

    Attributes:
        type: Lexical class of the token.
        text: The exact source text the token was cut from.
        value: Decoded value for ``NUMBER``/``STRING`` tokens, the identifier
            name for identifiers, and the symbol itself otherwise.
        position: Offset of the token's first character in the source.
    """

    type: TokenType
    text: str
    value: Any
    position: int

    @property
    def upper(self) -> str:
        """The token text upper-cased, for case-insensitive keyword tests."""
        return self.text.upper()


# Longest first: the scanner takes the first symbol that matches, so "<=" must
# be tried before "<" and "<=>" before "<=".
_OPERATORS = (
    "<=>",
    "||",
    "->",
    ">=",
    "<=",
    "<>",
    "!=",
    "==",
    "=",
    "<",
    ">",
    "+",
    "-",
    "*",
    "/",
    "%",
    "&",
    "|",
    "^",
    "~",
)

_PUNCTUATION = ("(", ")", ",", ".", "[", "]")

_QUOTE_CHARACTERS = ("'", '"')


def tokenize(source: str) -> List[Token]:
    """Split a SQL expression into tokens.

    Args:
        source: The SQL expression text.

    Returns:
        The tokens, terminated by a single ``END`` token.

    Raises:
        ParseException: On an unterminated string or quoted identifier, or on a
            character that starts no token.
    """
    tokens: List[Token] = []
    index = 0
    length = len(source)

    while index < length:
        character = source[index]

        if character.isspace():
            index += 1
            continue

        if character in _QUOTE_CHARACTERS:
            token, index = _scan_string(source, index)
            tokens.append(token)
            continue

        if character == "`":
            token, index = _scan_quoted_identifier(source, index)
            tokens.append(token)
            continue

        if character.isdigit() or (
            character == "."
            and index + 1 < length
            and source[index + 1].isdigit()
            # A dot only starts a number when it is not a field access: "a.5"
            # is not valid anyway, but "x.1" must not swallow the dot.
            and not _previous_is_value(tokens)
        ):
            token, index = _scan_number(source, index)
            tokens.append(token)
            continue

        if character.isalpha() or character == "_":
            token, index = _scan_identifier(source, index)
            tokens.append(token)
            continue

        operator = _match_symbol(source, index, _OPERATORS)
        if operator is not None:
            tokens.append(
                Token(TokenType.OPERATOR, operator, operator, index),
            )
            index += len(operator)
            continue

        if character in _PUNCTUATION:
            tokens.append(Token(TokenType.PUNCTUATION, character, character, index))
            index += 1
            continue

        raise ParseException(
            f"Unexpected character {character!r} at position {index} in SQL "
            f"expression: {source}"
        )

    tokens.append(Token(TokenType.END, "", None, length))
    return tokens


def _previous_is_value(tokens: List[Token]) -> bool:
    """Whether the last token can be the left side of a field access."""
    if not tokens:
        return False
    last = tokens[-1]
    return last.type in (
        TokenType.IDENTIFIER,
        TokenType.QUOTED_IDENTIFIER,
    ) or (last.type == TokenType.PUNCTUATION and last.text in (")", "]"))


def _match_symbol(source: str, index: int, symbols: Sequence[str]) -> Optional[str]:
    """Return the first symbol in ``symbols`` that starts at ``index``."""
    for symbol in symbols:
        if source.startswith(symbol, index):
            return symbol
    return None


def _scan_string(source: str, index: int) -> Tuple[Token, int]:
    """Scan a string literal, honouring ``''`` and backslash escapes."""
    quote = source[index]
    start = index
    index += 1
    characters: List[str] = []

    while index < len(source):
        character = source[index]
        if character == "\\" and index + 1 < len(source):
            characters.append(_unescape(source[index + 1]))
            index += 2
            continue
        if character == quote:
            # A doubled quote is an escaped quote, not the end of the literal.
            if index + 1 < len(source) and source[index + 1] == quote:
                characters.append(quote)
                index += 2
                continue
            return (
                Token(
                    TokenType.STRING,
                    source[start : index + 1],
                    "".join(characters),
                    start,
                ),
                index + 1,
            )
        characters.append(character)
        index += 1

    raise ParseException(
        f"Unterminated string literal starting at position {start} in SQL "
        f"expression: {source}"
    )


_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "0": "\0"}


def _unescape(character: str) -> str:
    """Decode the character following a backslash."""
    return _ESCAPES.get(character, character)


def _scan_quoted_identifier(source: str, index: int) -> Tuple[Token, int]:
    """Scan a backtick-quoted identifier, honouring ```` `` ```` escapes."""
    start = index
    index += 1
    characters: List[str] = []

    while index < len(source):
        character = source[index]
        if character == "`":
            if index + 1 < len(source) and source[index + 1] == "`":
                characters.append("`")
                index += 2
                continue
            return (
                Token(
                    TokenType.QUOTED_IDENTIFIER,
                    "".join(characters),
                    "".join(characters),
                    start,
                ),
                index + 1,
            )
        characters.append(character)
        index += 1

    raise ParseException(
        f"Unterminated quoted identifier starting at position {start} in SQL "
        f"expression: {source}"
    )


def _scan_number(source: str, index: int) -> Tuple[Token, int]:
    """Scan an integer or floating point literal, including exponents."""
    start = index
    length = len(source)
    seen_dot = False
    seen_exponent = False

    while index < length:
        character = source[index]
        if character.isdigit():
            index += 1
            continue
        if character == "." and not seen_dot and not seen_exponent:
            seen_dot = True
            index += 1
            continue
        if character in ("e", "E") and not seen_exponent:
            following = index + 1
            if following < length and source[following] in ("+", "-"):
                following += 1
            if following < length and source[following].isdigit():
                seen_exponent = True
                index = following
                continue
        break

    text = source[start:index]
    value: Any = float(text) if (seen_dot or seen_exponent) else int(text)
    return Token(TokenType.NUMBER, text, value, start), index


def _scan_identifier(source: str, index: int) -> Tuple[Token, int]:
    """Scan an unquoted identifier or keyword."""
    start = index
    length = len(source)
    while index < length and (source[index].isalnum() or source[index] == "_"):
        index += 1
    text = source[start:index]
    return Token(TokenType.IDENTIFIER, text, text, start), index
