"""
Condition evaluation utilities for Sparkless.

This module provides shared condition evaluation logic to avoid duplication
between DataFrame and conditional function modules.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union, cast
from ..functions.base import Column, ColumnOperation
from ..spark_types import get_row_value
from .datetime_utils import coerce_temporal_pair, spark_last_day, spark_trunc
from .math_utils import spark_bround, spark_round
from .variadic import variadic_reducer


def _round_scale(operation: Any) -> int:
    """Extract the scale argument of a round() operation.

    ``F.round(col, 2)`` stores the 2 on the operation's ``value``. Anything
    else (a missing scale, or a non-integer) means zero decimal places.
    """
    scale = getattr(operation, "value", 0)
    if isinstance(scale, bool) or not isinstance(scale, int):
        return 0
    return scale


#: Named comparison aliases accepted alongside their symbolic form.
_COMPARISON_ALIASES: Dict[str, str] = {
    "eq": "==",
    "ne": "!=",
    "lt": "<",
    "le": "<=",
    "gt": ">",
    "ge": ">=",
}

logger = logging.getLogger(__name__)


#: Operations whose *value* is exactly the boolean that the predicate path
#: (:meth:`ConditionEvaluator._evaluate_column_operation`) computes for them.
#: Projecting one of these must give the same answer as filtering on it, so the
#: value path delegates rather than keeping a second, drifting implementation.
#:
#: The logical connectives are deliberately absent: ``_evaluate_logical_operation``
#: already owns them on the value path.
_VALUE_IS_PREDICATE_OPERATIONS: frozenset = frozenset(
    {
        "isNull",
        "isnull",
        "isNotNull",
        "isnotnull",
        "isnan",
        "isNaN",
        "eqNullSafe",
        "isin",
        "between",
        "like",
        "rlike",
        "contains",
        "startswith",
        "startsWith",
        "endswith",
        "endsWith",
    }
)


class ConditionEvaluator:
    """Shared condition evaluation logic."""

    @staticmethod
    def _to_sql_boolean(value: Any) -> Optional[bool]:
        """Interpret an already-evaluated expression result as a SQL boolean.

        NULL stays NULL (rather than collapsing to False) so that an enclosing
        NOT/AND/OR sees SQL NULL and applies three-valued logic to it. A row
        whose predicate is NULL is filtered out, but ``NOT NULL`` must remain
        NULL instead of becoming True.
        """
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        return bool(value)

    @staticmethod
    def _kleene_not(value: Optional[bool]) -> Optional[bool]:
        """SQL three-valued NOT.

        NOT TRUE = FALSE, NOT FALSE = TRUE, NOT NULL = NULL.

        Python's ``not`` operator is *not* a substitute here: ``not None``
        evaluates to ``True``, which inverts the result for NULL inputs
        (a NULL predicate must filter the row out, not in).
        """
        if value is None:
            return None
        return not value

    @staticmethod
    def _kleene_and(left: Optional[bool], right: Optional[bool]) -> Optional[bool]:
        """SQL three-valued AND.

        FALSE dominates: ``NULL AND FALSE`` is FALSE (not NULL), because the
        result is FALSE regardless of what the NULL stands for. Python's
        ``and`` short-circuits and returns ``None`` for ``None and False``,
        which is why it cannot be used directly.
        """
        if left is False or right is False:
            return False
        if left is None or right is None:
            return None
        return True

    @staticmethod
    def _kleene_or(left: Optional[bool], right: Optional[bool]) -> Optional[bool]:
        """SQL three-valued OR.

        TRUE dominates: ``NULL OR TRUE`` is TRUE (not NULL). Python's ``or``
        returns ``False`` for ``None or False``, which must be NULL instead.
        """
        if left is True or right is True:
            return True
        if left is None or right is None:
            return None
        return False

    @staticmethod
    def evaluate_expression(row: Dict[str, Any], expression: Any) -> Any:
        """Evaluate an expression (arithmetic, function, etc.) for a given row.

        Args:
            row: The data row to evaluate against.
            expression: The expression to evaluate.

        Returns:
            The evaluated result.
        """
        if isinstance(expression, ColumnOperation):
            return ConditionEvaluator._evaluate_column_operation_value(row, expression)
        elif hasattr(expression, "evaluate"):
            return expression.evaluate(row)
        elif hasattr(expression, "value"):
            return expression.value
        else:
            return expression

    @staticmethod
    def evaluate_condition(row: Dict[str, Any], condition: Any) -> Optional[bool]:
        """Evaluate a condition for a given row.

        Args:
            row: The data row to evaluate against.
            condition: The condition to evaluate.

        Returns:
            True if condition is met, False otherwise.
        """
        # Check ColumnOperation BEFORE Column since ColumnOperation is a subclass of Column
        # This ensures comparison operations (==, !=, etc.) are properly evaluated
        if isinstance(condition, ColumnOperation):
            return ConditionEvaluator._evaluate_column_operation(row, condition)

        # CaseWhen (F.when(...).otherwise(...)) is not a Column subclass. Without
        # this branch it fell through to the truthiness fallback below, where the
        # object itself is always truthy — so `F.when(...)` was True for every row
        # and `~F.when(...)` was False for every row (an empty result set).
        if hasattr(condition, "evaluate") and hasattr(condition, "conditions"):
            return ConditionEvaluator._to_sql_boolean(condition.evaluate(row))

        if isinstance(condition, Column):
            # A bare column used as a predicate must yield its *value*, not a
            # presence check. For a boolean column PySpark evaluates the stored
            # boolean under three-valued logic; NULL stays NULL so that both the
            # row filter and any enclosing NOT/AND/OR see SQL NULL rather than
            # a coerced False.
            column_value = get_row_value(row, condition.name)
            if column_value is None:
                return None
            if isinstance(column_value, bool):
                return column_value
            # Non-boolean, non-null columns keep the historical truthy
            # behaviour (PySpark itself rejects these with an AnalysisException).
            return True

        # Unwrap Literal objects
        if hasattr(condition, "value") and not isinstance(
            condition, (Column, ColumnOperation)
        ):
            return bool(condition.value) if condition.value is not None else False

        # For simple values, check if truthy
        return bool(condition) if condition is not None else False

    @staticmethod
    def _evaluate_column_operation_value(
        row: Dict[str, Any], operation: ColumnOperation
    ) -> Optional[Any]:
        """Evaluate a column operation and return the value (not boolean).

        Args:
            row: The data row to evaluate against.
            operation: The column operation to evaluate.

        Returns:
            The evaluated result value.
        """
        operation_type = operation.operation

        # Predicates whose *value* is the boolean the predicate path already
        # computes. Dispatched first so that projecting a predicate
        # (``df.select(col.isin(...))``) agrees with filtering on it
        # (``df.filter(col.isin(...))``) -- the two used to be evaluated by
        # different tables, and `isin` / `between` / `eqNullSafe` were missing
        # from this one, so projecting them yielded NULL (BUG-046).
        if operation_type in _VALUE_IS_PREDICATE_OPERATIONS:
            return ConditionEvaluator._evaluate_column_operation(row, operation)

        # Arithmetic operations
        if operation_type in ["+", "-", "*", "/", "%"]:
            left_value = ConditionEvaluator._get_column_value(row, operation.column)
            right_value = ConditionEvaluator._get_column_value(row, operation.value)

            if left_value is None or right_value is None:
                return None

            # PySpark automatically casts string values to numeric (Double) for arithmetic
            def _coerce_to_numeric(value: Any) -> Any:
                if isinstance(value, str):
                    try:
                        return float(value.strip())
                    except (ValueError, TypeError):
                        return None
                return value

            # Coerce strings to numeric if at least one operand is numeric or both are strings
            if isinstance(left_value, str) or isinstance(right_value, str):
                left_value = _coerce_to_numeric(left_value)
                right_value = _coerce_to_numeric(right_value)
                if left_value is None or right_value is None:
                    return None

            try:
                if operation_type == "+":
                    # PySpark compatibility: String concatenation with + operator returns None
                    # when DataFrame is cached. Check if both operands are strings.
                    is_string_concatenation = isinstance(
                        left_value, str
                    ) and isinstance(right_value, str)
                    if is_string_concatenation and get_row_value(
                        row, "__dataframe_is_cached__", False
                    ):
                        # Check if we're in a cached DataFrame context
                        return None
                    add_result: Any = left_value + right_value
                    return cast("Optional[bool]", add_result)
                elif operation_type == "-":
                    return cast("Optional[bool]", left_value - right_value)
                elif operation_type == "*":
                    return cast("bool", left_value * right_value)
                elif operation_type == "/":
                    if right_value == 0:
                        return None
                    return cast("bool", left_value / right_value)
                elif operation_type == "%":
                    if right_value == 0:
                        return None
                    return cast("bool", left_value % right_value)
            except (TypeError, ValueError):
                return None

        # Struct construction. Collected here rather than in the function
        # whitelist below because struct returns a composite value (a dict of
        # field name -> value) instead of a scalar.
        elif operation_type in ("struct", "named_struct"):
            return ConditionEvaluator._evaluate_struct_operation(row, operation)

        # Higher-order array functions. Collected here rather than in the
        # function whitelist below because each one needs the *row* to
        # evaluate the expression its lambda builds, not just the already
        # resolved value of its first operand. Absent from both tables, all
        # four fell to the NULL fall-through -- and NULL is falsy, so a guard
        # written on top of them answered "clean" for every input (#2419).
        elif operation_type in ("exists", "forall", "filter", "transform"):
            return ConditionEvaluator._evaluate_higher_order_operation(row, operation)

        # Cast operations
        elif operation_type == "cast":
            value = ConditionEvaluator._get_column_value(row, operation.column)
            target_type = operation.value

            if value is None:
                return None

            try:
                # Handle DataType objects (Issue #5 fix)
                from sparkless.dataframe.casting.type_converter import TypeConverter
                from sparkless.spark_types import DataType, DateType, TimestampType

                if isinstance(target_type, DataType):
                    # Use TypeConverter for proper DataType handling
                    return TypeConverter.cast_to_type(value, target_type)

                # Handle string type names (legacy support)
                if target_type == "long" or target_type == "bigint":
                    # Convert to Unix timestamp if it's a datetime string
                    if isinstance(value, str) and ("-" in value or ":" in value):
                        from datetime import datetime

                        dt = datetime.fromisoformat(value.replace(" ", "T"))
                        return int(dt.timestamp())
                    return int(float(value))
                elif target_type == "int":
                    return int(float(value))
                elif target_type == "double" or target_type == "float":
                    return float(value)
                elif target_type == "string":
                    return str(value)
                elif target_type == "boolean":
                    return bool(value)
                elif target_type in ("date", "timestamp"):
                    # Delegate to the same converter the DataType branch above
                    # uses. Without this, `F.lit("2026-01-01").cast("date")`
                    # fell to `return value` and stayed a *string* on this
                    # path, while the projection path produced a real date.
                    # A predicate then compared date >= str, which raises and
                    # was swallowed into NULL -- silently dropping every row
                    # of `df.filter(F.last_day(d) >= lit(s).cast("date"))`
                    # (BUG-053).
                    return TypeConverter.cast_to_type(
                        value,
                        DateType() if target_type == "date" else TimestampType(),
                    )
                else:
                    return value
            except (TypeError, ValueError):
                return None

        # Function operations
        elif operation_type in [
            "md5",
            "sha1",
            "crc32",
            "upper",
            "lower",
            "length",
            "trim",
            "abs",
            "round",
            "bround",
            "log10",
            "log",
            "log2",
            "concat",
            "split",
            "regexp_replace",
            "coalesce",
            "ceil",
            "floor",
            "sqrt",
            "exp",
            "sin",
            "cos",
            "tan",
            "asin",
            "acos",
            "atan",
            "sinh",
            "cosh",
            "tanh",
            "degrees",
            "radians",
            "sign",
            "greatest",
            "least",
            "when",
            "otherwise",
            "isnull",
            "isnotnull",
            "isnan",
            "nvl",
            "nvl2",
            "current_date",
            "current_timestamp",
            "to_date",
            "to_timestamp",
            "try_to_timestamp",
            "hour",
            "day",
            "dayofmonth",
            "month",
            "year",
            "dayofweek",
            "dayofyear",
            "weekofyear",
            "quarter",
            "minute",
            "second",
            "date_add",
            "date_sub",
            "datediff",
            "last_day",
            "trunc",
            "months_between",
            "unix_timestamp",
            "from_unixtime",
            "array_distinct",
            "array_sort",
            "sort_array",
            "initcap",
            "concat_ws",
            "pi",
            "e",
            "substr",
            "substring",
            "translate",
            "substring_index",
            "levenshtein",
            "soundex",
            "regexp_extract",
            "regexp_extract_all",
            "create_map",
            "udf",
            "date_trunc",
            "date_format",
            "lpad",
            "rpad",
            "replace",
            "reverse",
            "repeat",
            "like",
            "rlike",
            "contains",
            "startswith",
            "endswith",
            "getItem",
            "getField",
            "xxhash64",
            "get_json_object",
            "json_tuple",
            "size",
            "array_contains",
            "explode",
            "array",
            "nullif",
            "nanvl",
            "pow",
            "power",
            "ltrim",
            "rtrim",
            "ascii",
            "hex",
            "base64",
            "array_position",
            "element_at",
            "array_join",
            "array_remove",
            "array_union",
            "array_except",
            "array_intersect",
            # Array value functions. Absent from this whitelist, each of them
            # fell to the NULL fall-through below while the withColumn path
            # answered (flatten) or returned its own operand (array_min,
            # array_max, slice) -- two evaluators, two different wrong answers
            # for one expression (#2420).
            "flatten",
            "array_min",
            "array_max",
            "slice",
            # Absent from this whitelist, to_json fell to the NULL fall-through
            # below: df.select(F.to_json(F.struct(...))) answered NULL for
            # every row with no warning (#2417).
            "to_json",
        ]:
            return ConditionEvaluator._evaluate_function_operation_value(row, operation)

        # Comparison operations (return boolean)
        elif operation_type in [
            "==",
            "!=",
            "<",
            ">",
            "<=",
            ">=",
            "eq",
            "ne",
            "lt",
            "le",
            "gt",
            "ge",
        ]:
            return ConditionEvaluator._evaluate_comparison_operation(row, operation)

        # Logical operations (return boolean)
        elif operation_type in ["and", "&", "or", "|", "not", "!", "~"]:
            return ConditionEvaluator._evaluate_logical_operation(row, operation)

        # Default fallback. Reaching here means `operation_type` matched none
        # of the branches above, i.e. the operation is unimplemented rather
        # than genuinely NULL -- the two are indistinguishable downstream, so
        # log it to keep the gap diagnosable.
        logger.debug(
            "ConditionEvaluator has no value handler for operation %r; "
            "returning NULL. This is an unimplemented operation, not a SQL NULL.",
            operation_type,
        )
        return None

    @staticmethod
    def _evaluate_higher_order_operation(
        row: Dict[str, Any], operation: ColumnOperation
    ) -> Any:
        """Evaluate ``exists`` / ``forall`` / ``filter`` / ``transform``.

        The lambda is applied to each array element as a literal, and whatever
        expression it builds is evaluated by this same evaluator -- so a
        lambda body may use any function, exactly as in Spark.

        Args:
            row: The data row to evaluate against.
            operation: The higher-order ``ColumnOperation``.

        Returns:
            The function's result, or NULL for a NULL array.
        """
        from .higher_order import (
            element_applier,
            exists_value,
            filter_value,
            forall_value,
            transform_value,
        )

        col_value = ConditionEvaluator._get_column_value(row, operation.column)
        if col_value is None:
            return None

        def evaluate(expression: Any) -> Any:
            return ConditionEvaluator.evaluate_expression(row, expression)

        apply = element_applier(operation.value, evaluate)
        if operation.operation == "exists":
            return exists_value(col_value, apply)
        if operation.operation == "forall":
            return forall_value(col_value, apply)
        if operation.operation == "filter":
            return filter_value(col_value, apply)
        return transform_value(col_value, apply)

    @staticmethod
    def _evaluate_struct_operation(
        row: Dict[str, Any], operation: ColumnOperation
    ) -> Dict[str, Any]:
        """Evaluate a ``struct`` / ``named_struct`` operation to a dict.

        Args:
            row: The data row to evaluate against.
            operation: The struct operation to evaluate.

        Returns:
            Mapping of struct field name to evaluated value.
        """
        from .struct_builder import build_struct_value

        def resolve(item: Any) -> Any:
            return ConditionEvaluator._get_column_value(row, item)

        return build_struct_value(operation, resolve)

    @staticmethod
    def _evaluate_function_operation_value(
        row: Dict[str, Any], operation: ColumnOperation
    ) -> Any:
        """Evaluate a function operation and return the value.

        Args:
            row: The data row to evaluate against.
            operation: The function operation to evaluate.

        Returns:
            The evaluated result value.
        """
        operation_type = operation.operation

        # Handle constant functions that don't need column values
        if operation_type == "pi":
            import math

            return math.pi
        elif operation_type == "e":
            import math

            return math.e

        col_value = ConditionEvaluator._get_column_value(row, operation.column)

        # Handle function operations that return values
        if operation_type == "upper":
            return str(col_value).upper() if col_value is not None else None
        elif operation_type == "lower":
            return str(col_value).lower() if col_value is not None else None
        elif operation_type == "length":
            return len(str(col_value)) if col_value is not None else None
        elif operation_type == "trim":
            # PySpark trim only removes ASCII space characters (0x20), not tabs/newlines
            return str(col_value).strip(" ") if col_value is not None else None
        elif operation_type == "ltrim":
            # PySpark ltrim only removes ASCII space characters (0x20), not tabs/newlines
            return str(col_value).lstrip(" ") if col_value is not None else None
        elif operation_type == "rtrim":
            # PySpark rtrim only removes ASCII space characters (0x20), not tabs/newlines
            return str(col_value).rstrip(" ") if col_value is not None else None
        elif operation_type == "ascii":
            if col_value is None:
                return None
            s = str(col_value)
            if not s:
                return None
            return ord(s[0])
        elif operation_type == "hex":
            if col_value is None:
                return None
            # PySpark hex: for strings, encode each byte as hex uppercase
            if isinstance(col_value, (int, float)):
                return hex(int(col_value))[2:].upper()
            return str(col_value).encode("utf-8").hex().upper()
        elif operation_type == "base64":
            if col_value is None:
                return None
            import base64 as base64_mod

            return base64_mod.b64encode(str(col_value).encode("utf-8")).decode("utf-8")
        elif operation_type == "initcap":
            # Capitalize first letter of each word
            if col_value is None:
                return None
            return " ".join(word.capitalize() for word in str(col_value).split())
        elif operation_type == "concat":
            # Concatenate multiple columns: operation.column is first, operation.value is remaining
            if col_value is None:
                return None
            parts = [str(col_value)]
            if hasattr(operation, "value") and operation.value is not None:
                remaining = (
                    operation.value
                    if isinstance(operation.value, (list, tuple))
                    else [operation.value]
                )
                for col_ref in remaining:
                    val = ConditionEvaluator._get_column_value(row, col_ref)
                    if val is None:
                        return None  # PySpark returns null if any arg is null
                    parts.append(str(val))
            return "".join(parts)
        elif operation_type == "concat_ws":
            # Concatenate with separator - operation.value is (sep, [columns])
            # For concat_ws, we need to get values from multiple columns
            # The operation.value should be a tuple: (separator, [column1, column2, ...])
            if not hasattr(operation, "value") or not isinstance(
                operation.value, tuple
            ):
                return None
            sep, columns = operation.value
            # Get values for all columns (including the first column from operation.column)
            values = []
            # First column is in operation.column
            first_val = ConditionEvaluator._get_column_value(row, operation.column)
            if first_val is not None:
                values.append(str(first_val))
            # Additional columns are in the tuple
            for col in columns:
                col_val = ConditionEvaluator._get_column_value(row, col)
                if col_val is not None:
                    values.append(str(col_val))
            # Join with separator
            return sep.join(values) if values else None
        elif operation_type == "regexp_replace":
            # Regex replace - operation.value is (pattern, replacement)
            if col_value is None:
                return None
            import re

            if not hasattr(operation, "value") or not isinstance(
                operation.value, tuple
            ):
                return str(col_value)
            pattern, replacement = operation.value
            try:
                return re.sub(pattern, replacement, str(col_value))
            except Exception:
                return str(col_value)
        elif operation_type == "abs":
            return abs(float(col_value)) if col_value is not None else None
        elif operation_type == "round":
            # The scale lives on the operation; dropping it rounded every
            # round(x, n) to zero decimal places.
            return spark_round(col_value, _round_scale(operation))
        elif operation_type == "ceil":
            import math

            return math.ceil(float(col_value)) if col_value is not None else None
        elif operation_type == "floor":
            import math

            return math.floor(float(col_value)) if col_value is not None else None
        elif operation_type == "sqrt":
            import math

            return (
                math.sqrt(float(col_value))
                if col_value is not None and float(col_value) >= 0
                else None
            )
        elif operation_type == "exp":
            import math

            return math.exp(float(col_value)) if col_value is not None else None
        elif operation_type == "sin":
            import math

            return math.sin(float(col_value)) if col_value is not None else None
        elif operation_type == "cos":
            import math

            return math.cos(float(col_value)) if col_value is not None else None
        elif operation_type == "tan":
            import math

            return math.tan(float(col_value)) if col_value is not None else None
        elif operation_type == "asin":
            import math

            return math.asin(float(col_value)) if col_value is not None else None
        elif operation_type == "acos":
            import math

            return math.acos(float(col_value)) if col_value is not None else None
        elif operation_type == "atan":
            import math

            return math.atan(float(col_value)) if col_value is not None else None
        elif operation_type == "sinh":
            import math

            return math.sinh(float(col_value)) if col_value is not None else None
        elif operation_type == "cosh":
            import math

            return math.cosh(float(col_value)) if col_value is not None else None
        elif operation_type == "tanh":
            import math

            return math.tanh(float(col_value)) if col_value is not None else None
        elif operation_type == "degrees":
            import math

            return math.degrees(float(col_value)) if col_value is not None else None
        elif operation_type == "radians":
            import math

            return math.radians(float(col_value)) if col_value is not None else None
        elif operation_type == "sign":
            return (
                1
                if col_value > 0
                else (-1 if col_value < 0 else 0)
                if col_value is not None
                else None
            )
        elif operation_type == "current_date":
            from datetime import date

            return date.today()
        elif operation_type == "current_timestamp":
            from datetime import datetime

            return datetime.now()
        elif operation_type == "unix_timestamp":
            if isinstance(col_value, str) and ("-" in col_value or ":" in col_value):
                from datetime import datetime

                dt = datetime.fromisoformat(col_value.replace(" ", "T"))
                return int(dt.timestamp())
            return None
        elif operation_type == "datediff":
            # For datediff, we need two dates - get both values
            end_date = ConditionEvaluator._get_column_value(row, operation.column)
            start_date = ConditionEvaluator._get_column_value(row, operation.value)

            if end_date is None or start_date is None:
                return None

            try:
                from datetime import datetime

                # Parse end date
                if isinstance(end_date, str):
                    if " " in end_date:  # Has time component
                        end_dt = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S")
                    else:
                        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                else:
                    end_dt = end_date

                # Parse start date
                if isinstance(start_date, str):
                    if " " in start_date:  # Has time component
                        start_dt = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
                    else:
                        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                else:
                    start_dt = start_date

                # Calculate difference in days
                return (end_dt - start_dt).days
            except (ValueError, AttributeError):
                return None
        elif operation_type == "last_day":
            # DATE-valued even for a TIMESTAMP operand; see core.datetime_utils.
            return spark_last_day(col_value)
        elif operation_type == "trunc":
            return spark_trunc(col_value, operation.value)
        elif operation_type == "months_between":
            # For months_between, we need two dates - get both values
            end_date = ConditionEvaluator._get_column_value(row, operation.column)
            start_date = ConditionEvaluator._get_column_value(row, operation.value)

            if end_date is None or start_date is None:
                return None

            try:
                from datetime import datetime

                # Parse end date
                if isinstance(end_date, str):
                    if " " in end_date:  # Has time component
                        end_dt = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S")
                    else:
                        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                else:
                    end_dt = end_date

                # Parse start date
                if isinstance(start_date, str):
                    if " " in start_date:  # Has time component
                        start_dt = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
                    else:
                        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                else:
                    start_dt = start_date

                # Calculate difference in months using PySpark formula:
                # (year1 - year2) * 12 + (month1 - month2) + (day1 - day2) / 31.0
                year_diff = end_dt.year - start_dt.year
                month_diff = end_dt.month - start_dt.month
                day_diff = end_dt.day - start_dt.day
                return year_diff * 12 + month_diff + day_diff / 31.0
            except (ValueError, AttributeError):
                return None
        elif operation_type == "array_distinct":
            from .array_values import array_distinct_value

            return array_distinct_value(col_value)
        elif operation_type == "flatten":
            from .array_values import flatten_value

            return flatten_value(col_value)
        elif operation_type == "array_min":
            from .array_values import array_min_value

            return array_min_value(col_value)
        elif operation_type == "array_max":
            from .array_values import array_max_value

            return array_max_value(col_value)
        elif operation_type == "slice":
            from .array_values import slice_value

            params = operation.value
            if not isinstance(params, tuple) or len(params) < 2:  # noqa: PLR2004
                return None
            return slice_value(col_value, int(params[0]), int(params[1]))
        elif operation_type == "to_json":
            from .json_values import to_json_value

            # The operand expression goes with the value: struct and map both
            # evaluate to a dict, and Spark's NULL rule differs between them.
            return to_json_value(col_value, operation.column)
        elif operation_type == "array_sort" or operation_type == "sort_array":
            # Sort array elements - operation.value contains asc boolean
            if not isinstance(col_value, list):
                return None
            asc = True
            if hasattr(operation, "value") and operation.value is not None:
                asc = operation.value
            # Sort while preserving type
            try:
                return sorted(col_value, reverse=not asc)
            except TypeError:
                # If items are not directly comparable, convert to strings
                return sorted(col_value, key=str, reverse=not asc)
        elif operation_type in ("substr", "substring"):
            if col_value is None:
                return None
            s = str(col_value)
            # operation.value is (start, length) tuple
            params = operation.value
            if isinstance(params, tuple) and len(params) >= 2:
                start, length = int(params[0]), int(params[1])
            elif isinstance(params, (int, float)):
                start, length = int(params), len(s)
            else:
                return None
            # PySpark substr is 1-based
            if start > 0:
                idx = start - 1
            elif start < 0:
                idx = max(0, len(s) + start)
            else:
                idx = 0
            return s[idx : idx + length]

        elif operation_type == "translate":
            if col_value is None:
                return None
            # operation.value is (matching, replace) tuple
            params = operation.value
            if isinstance(params, tuple) and len(params) >= 2:
                matching, replace = str(params[0]), str(params[1])
                table = str.maketrans(
                    matching, replace[: len(matching)].ljust(len(matching), "\x00")
                )
                return str(col_value).translate(table).replace("\x00", "")
            return str(col_value)

        elif operation_type == "substring_index":
            if col_value is None:
                return None
            params = operation.value
            if isinstance(params, tuple) and len(params) >= 2:
                delim, count = str(params[0]), int(params[1])
                if count == 0 or delim == "":
                    return ""
                parts = str(col_value).split(delim)
                if count > 0:
                    return delim.join(parts[:count])
                elif count < 0:
                    return delim.join(parts[count:])
            return str(col_value)

        elif operation_type == "levenshtein":
            val1 = str(col_value) if col_value is not None else ""
            val2_raw = ConditionEvaluator._get_column_value(row, operation.value)
            val2 = str(val2_raw) if val2_raw is not None else ""
            if col_value is None or val2_raw is None:
                return None
            # DP edit distance
            m, n = len(val1), len(val2)
            dp = list(range(n + 1))
            for i in range(1, m + 1):
                prev, dp[0] = dp[0], i
                for j in range(1, n + 1):
                    temp = dp[j]
                    dp[j] = (
                        prev
                        if val1[i - 1] == val2[j - 1]
                        else 1 + min(dp[j], dp[j - 1], prev)
                    )
                    prev = temp
            return dp[n]

        elif operation_type == "soundex":
            if col_value is None:
                return None
            s = str(col_value).upper()
            if not s:
                return ""
            codes = {
                "B": "1",
                "F": "1",
                "P": "1",
                "V": "1",
                "C": "2",
                "G": "2",
                "J": "2",
                "K": "2",
                "Q": "2",
                "S": "2",
                "X": "2",
                "Z": "2",
                "D": "3",
                "T": "3",
                "L": "4",
                "M": "5",
                "N": "5",
                "R": "6",
            }
            soundex_result: str = s[0]
            prev_code: str = codes.get(s[0], "0")
            for c in s[1:]:
                cur_code: str = codes.get(c, "0")
                if cur_code != "0" and cur_code != prev_code:
                    soundex_result += cur_code
                # Always update prev_code: vowels/non-coded chars reset it
                # so same-coded letters separated by a vowel are coded twice
                prev_code = cur_code
            return (soundex_result + "000")[:4]

        elif operation_type == "regexp_extract":
            if col_value is None:
                return None
            import re as re_mod

            params = operation.value
            if isinstance(params, tuple) and len(params) >= 2:
                pattern, idx = str(params[0]), int(params[1])
                match = re_mod.search(pattern, str(col_value))
                if match:
                    try:
                        return match.group(idx)
                    except IndexError:
                        return ""
                return ""
            return ""

        elif operation_type == "regexp_extract_all":
            if col_value is None:
                return None
            import re as re_mod

            params = operation.value
            pattern = str(params) if not isinstance(params, tuple) else str(params[0])
            return re_mod.findall(pattern, str(col_value))

        elif operation_type == "create_map":
            # operation.value is a tuple of (key_col, val_col, key_col, val_col, ...)
            args = operation.value if operation.value else ()
            if not args:
                return {}
            result_map = {}
            items = list(args)
            for j in range(0, len(items) - 1, 2):
                k = ConditionEvaluator._get_column_value(row, items[j])
                v = ConditionEvaluator._get_column_value(row, items[j + 1])
                result_map[k] = v
            return result_map

        elif operation_type == "size":
            if col_value is None:
                return -1  # PySpark returns -1 for null arrays
            if isinstance(col_value, (list, tuple)):
                return len(col_value)
            if isinstance(col_value, dict):
                return len(col_value)
            return -1

        elif operation_type == "array_contains":
            if col_value is None:
                return None
            search_value = ConditionEvaluator._resolve_operand(row, operation.value)
            if isinstance(col_value, (list, tuple)):
                return search_value in col_value
            return False

        elif operation_type == "array_position":
            if col_value is None:
                return None
            search_value = ConditionEvaluator._resolve_operand(row, operation.value)
            # PySpark: a NULL search value yields NULL, not 0.
            if search_value is None:
                return None
            if isinstance(col_value, (list, tuple)):
                for idx, item in enumerate(col_value):
                    if item == search_value:
                        return idx + 1  # 1-based
                return 0
            return 0

        elif operation_type == "element_at":
            if col_value is None:
                return None
            idx = operation.value
            if isinstance(idx, float):
                idx = int(idx)
            if isinstance(col_value, dict):
                return col_value.get(idx)
            if isinstance(col_value, (list, tuple)):
                if idx > 0:
                    # 1-based indexing
                    if idx <= len(col_value):
                        return col_value[idx - 1]
                    return None
                elif idx < 0:
                    # Negative indexing from end
                    if abs(idx) <= len(col_value):
                        return col_value[idx]
                    return None
            return None

        elif operation_type == "array_join":
            if col_value is None:
                return None
            if not isinstance(col_value, (list, tuple)):
                return None
            # operation.value is (delimiter, null_replacement)
            delimiter = ","
            null_replacement = None
            if hasattr(operation, "value") and operation.value is not None:
                if isinstance(operation.value, tuple):
                    delimiter = (
                        str(operation.value[0])
                        if operation.value[0] is not None
                        else ","
                    )
                    null_replacement = (
                        operation.value[1] if len(operation.value) > 1 else None
                    )
                elif isinstance(operation.value, str):
                    delimiter = operation.value
            parts = []
            for item in col_value:
                if item is None:
                    if null_replacement is not None:
                        parts.append(str(null_replacement))
                    # else skip None items (PySpark behavior)
                else:
                    parts.append(str(item))
            return delimiter.join(parts)

        elif operation_type == "array_remove":
            if col_value is None:
                return None
            if not isinstance(col_value, (list, tuple)):
                return None
            remove_value = ConditionEvaluator._resolve_operand(row, operation.value)
            # PySpark: removing a NULL value yields NULL for the whole array.
            if remove_value is None:
                return None
            return [x for x in col_value if x != remove_value]

        elif operation_type in ("array_except", "array_intersect"):
            if col_value is None:
                return None
            arr2 = ConditionEvaluator._get_column_value(row, operation.value)
            if arr2 is None:
                return None
            if not isinstance(col_value, (list, tuple)) or not isinstance(
                arr2, (list, tuple)
            ):
                return None
            # PySpark set semantics: the result is deduplicated and keeps the
            # first-occurrence order of the left array. NULL participates as a
            # value (``array_except([a, NULL], [NULL])`` -> ``[a]``).
            other = {ConditionEvaluator._element_key(item) for item in arr2}
            want_present = operation_type == "array_intersect"
            seen_keys: set = set()  # type: ignore[no-redef]
            result: list = []  # type: ignore[no-redef]
            for item in col_value:
                key = ConditionEvaluator._element_key(item)
                if (key in other) is not want_present:
                    continue
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                result.append(item)
            return result

        elif operation_type == "array_union":
            if col_value is None:
                return None
            arr2 = ConditionEvaluator._get_column_value(row, operation.value)
            if arr2 is None:
                return None
            if not isinstance(col_value, (list, tuple)) or not isinstance(
                arr2, (list, tuple)
            ):
                return None
            # Union preserving order, removing duplicates
            seen: set = set()  # type: ignore[no-redef]
            result: list = []  # type: ignore[no-redef]
            for item in list(col_value) + list(arr2):
                try:
                    key = (
                        item
                        if isinstance(item, (int, float, str, bool, type(None)))
                        else repr(item)
                    )
                except TypeError:
                    key = repr(item)
                if key not in seen:
                    seen.add(key)
                    result.append(item)
            return result

        elif operation_type == "explode":
            # explode is handled at the select level for row expansion
            # Here we just return the array value itself
            return col_value

        elif operation_type == "array":
            # Collect values from multiple columns into an array
            array_result = []
            if col_value is not None or (
                hasattr(operation, "column")
                and hasattr(operation.column, "name")
                and operation.column.name != "__array_empty_base__"
            ):
                array_result.append(col_value)
            # Check for empty array case
            if (
                hasattr(operation, "column")
                and hasattr(operation.column, "name")
                and operation.column.name == "__array_empty_base__"
            ):
                return []
            # Add remaining values from operation.value
            if hasattr(operation, "value") and operation.value is not None:
                if isinstance(operation.value, (list, tuple)):
                    for item in operation.value:
                        val = ConditionEvaluator._get_column_value(row, item)
                        array_result.append(val)
                else:
                    val = ConditionEvaluator._get_column_value(row, operation.value)
                    array_result.append(val)
            return array_result

        elif operation_type == "udf":
            udf_func = getattr(operation, "_udf_func", None)
            udf_cols = getattr(operation, "_udf_cols", None)
            if udf_func is None:
                return None
            args = []
            if udf_cols:
                for col_ref in udf_cols:
                    val = ConditionEvaluator._get_column_value(row, col_ref)
                    args.append(val)
            else:
                args.append(col_value)
            try:
                return udf_func(*args)
            except Exception:
                return None

        elif operation_type == "date_trunc":
            if col_value is None:
                return None
            from datetime import datetime, date

            unit = str(operation.value).lower() if operation.value else "day"
            dt = col_value
            if isinstance(dt, str):
                try:
                    dt = datetime.fromisoformat(dt.replace(" ", "T"))
                except ValueError:
                    return None
            if isinstance(dt, date) and not isinstance(dt, datetime):  # type: ignore[unreachable]
                dt = datetime(dt.year, dt.month, dt.day)  # type: ignore[unreachable]
            if not isinstance(dt, datetime):  # type: ignore[unreachable]
                return None  # type: ignore[unreachable]
            if unit in ("year", "yyyy", "yy"):
                return dt.replace(
                    month=1, day=1, hour=0, minute=0, second=0, microsecond=0
                )
            elif unit in ("month", "mon", "mm"):
                return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            elif unit in ("day", "dd"):
                return dt.replace(hour=0, minute=0, second=0, microsecond=0)
            elif unit in ("hour",):
                return dt.replace(minute=0, second=0, microsecond=0)
            elif unit in ("minute",):
                return dt.replace(second=0, microsecond=0)
            elif unit in ("second",):
                return dt.replace(microsecond=0)
            return dt

        elif operation_type == "date_format":
            if col_value is None:
                return None
            from datetime import datetime, date

            fmt = str(operation.value) if operation.value else "yyyy-MM-dd"
            dt = col_value
            if isinstance(dt, str):
                try:
                    dt = datetime.fromisoformat(dt.replace(" ", "T"))
                except ValueError:
                    return None
            if isinstance(dt, date) and not isinstance(dt, datetime):  # type: ignore[unreachable]
                dt = datetime(dt.year, dt.month, dt.day)  # type: ignore[unreachable]
            if not isinstance(dt, datetime):  # type: ignore[unreachable]
                return None  # type: ignore[unreachable]
            # Convert Java-style format to Python strftime
            py_fmt = (
                fmt.replace("yyyy", "%Y")
                .replace("yy", "%y")
                .replace("MM", "%m")
                .replace("dd", "%d")
                .replace("HH", "%H")
                .replace("mm", "%M")
                .replace("ss", "%S")
            )
            return dt.strftime(py_fmt)

        elif operation_type in ("lpad", "rpad"):
            if col_value is None:
                return None
            params = operation.value
            if isinstance(params, tuple) and len(params) >= 2:
                length, pad = int(params[0]), str(params[1])
                s = str(col_value)
                if operation_type == "lpad":
                    return s.rjust(length, pad[0]) if pad else s
                else:
                    return s.ljust(length, pad[0]) if pad else s
            return str(col_value)

        elif operation_type == "replace":
            if col_value is None:
                return None
            params = operation.value
            if isinstance(params, tuple) and len(params) >= 2:
                return str(col_value).replace(str(params[0]), str(params[1]))
            return str(col_value)

        elif operation_type == "reverse":
            if col_value is None:
                return None
            if isinstance(col_value, list):
                return list(reversed(col_value))
            return str(col_value)[::-1]

        elif operation_type == "repeat":
            if col_value is None:
                return None
            n = int(operation.value) if operation.value else 1
            return str(col_value) * n

        elif operation_type in ("like", "rlike"):
            if col_value is None:
                return None
            import re as re_mod

            pattern = str(operation.value)
            if operation_type == "like":
                # Convert SQL LIKE to regex
                regex = "^" + pattern.replace("%", ".*").replace("_", ".") + "$"
                return bool(re_mod.match(regex, str(col_value)))
            else:
                return bool(re_mod.search(pattern, str(col_value)))

        elif operation_type == "contains":
            if col_value is None:
                return None
            return str(operation.value) in str(col_value)

        elif operation_type in ("startswith", "startsWith"):
            if col_value is None:
                return None
            return str(col_value).startswith(str(operation.value))

        elif operation_type in ("endswith", "endsWith"):
            if col_value is None:
                return None
            return str(col_value).endswith(str(operation.value))

        elif operation_type == "xxhash64":
            # PySpark uses xxhash64 with seed=42
            SEED = 42
            vals = [col_value]
            if hasattr(operation, "value") and operation.value is not None:
                extra = (
                    operation.value
                    if isinstance(operation.value, (list, tuple))
                    else [operation.value]
                )
                for v in extra:
                    vals.append(ConditionEvaluator._get_column_value(row, v))
            # If all values are None, return the seed (PySpark behavior)
            if all(v is None for v in vals):
                return SEED
            try:
                import xxhash

                h = xxhash.xxh64(seed=SEED)
                for v in vals:
                    if v is None:
                        h.update(b"\x00")
                    else:
                        h.update(str(v).encode("utf-8"))
                return h.intdigest()
            except ImportError:
                # Fallback if xxhash not installed
                import hashlib

                hash_input = "|".join(str(v) if v is not None else "null" for v in vals)
                h_bytes = hashlib.sha256(hash_input.encode()).digest()
                return int.from_bytes(h_bytes[:8], byteorder="big", signed=True)

        elif operation_type == "get_json_object":
            if col_value is None:
                return None
            import json as json_lib

            path = str(operation.value) if operation.value else "$"
            try:
                obj = json_lib.loads(str(col_value))
            except (json_lib.JSONDecodeError, TypeError):
                return None
            # Navigate JSON path (simple $.key.key support)
            parts = path.lstrip("$").split(".")
            current_obj = obj
            for part in parts:
                if not part:
                    continue
                # Handle array index like [0]
                import re as re_mod

                arr_match = re_mod.match(r"(\w*)\[(\d+)\]", part)
                if arr_match:
                    key, idx = arr_match.group(1), int(arr_match.group(2))
                    if key and isinstance(current_obj, dict):
                        current_obj = current_obj.get(key)
                    if isinstance(current_obj, list) and idx < len(current_obj):
                        current_obj = current_obj[idx]
                    else:
                        return None
                elif isinstance(current_obj, dict):
                    current_obj = current_obj.get(part)
                    if current_obj is None:
                        return None
                else:
                    return None
            if isinstance(current_obj, (dict, list)):
                return json_lib.dumps(current_obj, separators=(",", ":"))
            return str(current_obj) if current_obj is not None else None

        elif operation_type == "json_tuple":
            if col_value is None:
                return None
            import json as json_lib

            try:
                obj = json_lib.loads(str(col_value))
            except (json_lib.JSONDecodeError, TypeError):
                return None
            # operation.value is a tuple of field names
            fields = (
                operation.value
                if isinstance(operation.value, (list, tuple))
                else [operation.value]
            )
            if isinstance(obj, dict):
                return tuple(
                    str(obj.get(f)) if obj.get(f) is not None else None for f in fields
                )
            return None

        elif operation_type in ("getItem", "getField"):
            if col_value is None:
                return None
            key = operation.value
            if isinstance(key, Column):
                key = ConditionEvaluator._get_column_value(row, key)
            if isinstance(col_value, dict):
                return col_value.get(key)
            elif isinstance(col_value, (list, tuple)):
                try:
                    return col_value[int(key)]
                except (IndexError, TypeError, ValueError):
                    return None
            return None

        elif operation_type == "split":
            if col_value is None:
                return None
            import re as re_split_mod

            # operation.value is (pattern, limit)
            if isinstance(operation.value, tuple) and len(operation.value) >= 1:
                pattern = operation.value[0]
                limit = operation.value[1] if len(operation.value) > 1 else None
                if limit is not None and limit > 0:
                    # PySpark limit=N means N parts (split at most N-1 times)
                    if limit == 1:
                        # limit=1 means no split at all, return original as single-element list
                        return [str(col_value)]
                    return re_split_mod.split(
                        pattern, str(col_value), maxsplit=limit - 1
                    )
                else:
                    # limit is None, 0, or negative: split all
                    return re_split_mod.split(pattern, str(col_value))
            return str(col_value).split()

        elif operation_type == "coalesce":
            # Return the first non-null value from the column list
            # operation.column is the first column, operation.value is a list of remaining columns
            if col_value is not None:
                return col_value
            # Check remaining columns
            if hasattr(operation, "value") and operation.value is not None:
                remaining = (
                    operation.value
                    if isinstance(operation.value, (list, tuple))
                    else [operation.value]
                )
                for col_ref in remaining:
                    val = ConditionEvaluator._get_column_value(row, col_ref)
                    if val is not None:
                        return val
            return None

        elif operation_type == "to_date":
            if col_value is None:
                return None
            try:
                from datetime import datetime

                fmt: Any = getattr(operation, "value", None)  # type: ignore[no-redef]
                if fmt and isinstance(fmt, str):
                    # Convert Java/Spark date format to Python strftime format
                    python_fmt = (
                        fmt.replace("yyyy", "%Y")
                        .replace("yy", "%y")
                        .replace("MM", "%m")
                        .replace("dd", "%d")
                        .replace("HH", "%H")
                        .replace("mm", "%M")
                        .replace("ss", "%S")
                    )
                    return datetime.strptime(str(col_value), python_fmt).date()
                else:
                    return datetime.strptime(str(col_value), "%Y-%m-%d").date()
            except ValueError:
                return None

        elif operation_type == "try_to_timestamp":
            if col_value is None:
                return None
            try:
                from datetime import datetime

                fmt_val: Any = getattr(operation, "value", None)
                if hasattr(fmt_val, "value"):
                    fmt_val = fmt_val.value  # Unwrap Literal
                if fmt_val and isinstance(fmt_val, str):
                    python_fmt = (
                        fmt_val.replace("yyyy", "%Y")
                        .replace("yy", "%y")
                        .replace("MM", "%m")
                        .replace("dd", "%d")
                        .replace("HH", "%H")
                        .replace("mm", "%M")
                        .replace("ss", "%S")
                    )
                    return datetime.strptime(str(col_value), python_fmt)
                try:
                    return datetime.fromisoformat(str(col_value).replace(" ", "T"))
                except ValueError:
                    return datetime.strptime(str(col_value), "%Y-%m-%d %H:%M:%S")
            except (ValueError, TypeError):
                return None

        elif operation_type == "log":
            import math

            if col_value is None:
                return None
            try:
                val = float(col_value)
                if val <= 0:
                    return None
                if operation.value is not None:
                    base_raw = (
                        ConditionEvaluator._get_column_value(row, operation.value)
                        if isinstance(operation.value, (Column, ColumnOperation))
                        else operation.value
                    )
                    base = float(base_raw)
                    if base <= 0 or base == 1:
                        return None
                    return math.log(val, base)
                return math.log(val)
            except (ValueError, TypeError):
                return None
        elif operation_type in ("pow", "power"):
            if col_value is None:
                return None
            try:
                base_val = float(col_value)
                exp_val = ConditionEvaluator._get_column_value(row, operation.value)
                if exp_val is None:
                    return None
                return float(base_val) ** float(exp_val)
            except (ValueError, TypeError):
                return None
        elif operation_type == "nullif":
            val2 = ConditionEvaluator._get_column_value(row, operation.value)
            if col_value is not None and col_value == val2:
                return None
            return col_value
        elif operation_type == "nanvl":
            import math as _math

            val2 = ConditionEvaluator._get_column_value(row, operation.value)
            if col_value is None:
                return None
            try:
                if isinstance(col_value, float) and _math.isnan(col_value):
                    return val2
            except (TypeError, ValueError):
                pass
            return col_value
        elif operation_type == "isnull":
            return col_value is None
        elif operation_type == "isnotnull":
            return col_value is not None
        elif operation_type == "date_add":
            if col_value is None:
                return None
            try:
                from datetime import datetime, timedelta, date as date_type

                if isinstance(col_value, str):
                    try:
                        dt = datetime.strptime(col_value, "%Y-%m-%d").date()  # type: ignore[assignment]
                    except ValueError:
                        dt = datetime.fromisoformat(col_value.replace(" ", "T")).date()  # type: ignore[assignment]
                elif isinstance(col_value, datetime):
                    dt = col_value.date()  # type: ignore[assignment]
                elif isinstance(col_value, date_type):
                    dt = col_value  # type: ignore[assignment]
                else:
                    dt = col_value
                days = int(operation.value) if operation.value is not None else 1
                return dt + timedelta(days=days)
            except (ValueError, AttributeError, TypeError):
                return None
        elif operation_type == "date_sub":
            if col_value is None:
                return None
            try:
                from datetime import datetime, timedelta, date as date_type

                if isinstance(col_value, str):
                    try:
                        dt = datetime.strptime(col_value, "%Y-%m-%d").date()  # type: ignore[assignment]
                    except ValueError:
                        dt = datetime.fromisoformat(col_value.replace(" ", "T")).date()  # type: ignore[assignment]
                elif isinstance(col_value, datetime):
                    dt = col_value.date()  # type: ignore[assignment]
                elif isinstance(col_value, date_type):
                    dt = col_value  # type: ignore[assignment]
                else:
                    dt = col_value
                days = int(operation.value) if operation.value is not None else 1
                return dt - timedelta(days=days)
            except (ValueError, AttributeError, TypeError):
                return None
        elif (reducer := variadic_reducer(cast("str", operation_type))) is not None:
            # greatest/least. The NULL-skipping semantics live in core.variadic
            # so this path and ExpressionEvaluator (which reaches the same
            # functions from `withColumn`) cannot drift apart -- they did
            # before, and only this one was right (BUG-038).
            values = [col_value]
            if hasattr(operation, "value") and operation.value is not None:
                remaining = (
                    operation.value
                    if isinstance(operation.value, (list, tuple))
                    else [operation.value]
                )
                values.extend(
                    ConditionEvaluator._get_column_value(row, col_ref)
                    for col_ref in remaining
                )
            return reducer(values)
        elif operation_type == "bround":
            # Same HALF_EVEN semantics as the withColumn path; see
            # ExpressionEvaluator._func_bround.
            if not isinstance(col_value, (int, float)) or isinstance(col_value, bool):
                return col_value
            return spark_bround(col_value, _round_scale(operation))
        else:
            # For other functions, delegate to the existing function evaluation
            # operation_type is guaranteed to be a string in ColumnOperation
            op_str: str = cast("str", operation_type)
            return ConditionEvaluator._evaluate_function_operation(
                col_value, op_str, _round_scale(operation)
            )

    @staticmethod
    def _evaluate_comparison_operation(
        row: Dict[str, Any], operation: ColumnOperation
    ) -> Optional[bool]:
        """Evaluate a comparison operation against a row.

        Resolves both operands, then defers to the shared
        :meth:`_evaluate_comparison` kernel so that a projection
        (``df.select(col("x") >= 5)``) and a predicate
        (``df.filter(col("x") >= 5)``) agree on NULL handling.

        Args:
            row: The data row to evaluate against.
            operation: The comparison operation to evaluate.

        Returns:
            The comparison result, or ``None`` when either operand is NULL.
        """
        left_value = ConditionEvaluator._get_column_value(row, operation.column)
        right_value = ConditionEvaluator._get_column_value(row, operation.value)

        # Resolve CaseWhen objects that _get_column_value doesn't handle
        if hasattr(left_value, "evaluate") and hasattr(left_value, "conditions"):
            left_value = left_value.evaluate(row)
        if hasattr(right_value, "evaluate") and hasattr(right_value, "conditions"):
            right_value = right_value.evaluate(row)
        # Unwrap Literal objects
        if hasattr(left_value, "value") and not isinstance(
            left_value, (Column, ColumnOperation)
        ):
            left_value = left_value.value
        if hasattr(right_value, "value") and not isinstance(
            right_value, (Column, ColumnOperation)
        ):
            right_value = right_value.value

        return ConditionEvaluator._evaluate_comparison(
            left_value, cast("str", operation.operation), right_value
        )

    @staticmethod
    def _evaluate_logical_operation(
        row: Dict[str, Any], operation: ColumnOperation
    ) -> Optional[bool]:
        """Evaluate a logical operation.

        Args:
            row: The data row to evaluate against.
            operation: The logical operation to evaluate.

        Returns:
            The logical result.
        """
        operation_type = operation.operation

        if operation_type in ["and", "&"]:
            left_result = ConditionEvaluator.evaluate_condition(row, operation.column)
            right_result = ConditionEvaluator.evaluate_condition(row, operation.value)
            return ConditionEvaluator._kleene_and(left_result, right_result)
        elif operation_type in ["or", "|"]:
            left_result = ConditionEvaluator.evaluate_condition(row, operation.column)
            right_result = ConditionEvaluator.evaluate_condition(row, operation.value)
            return ConditionEvaluator._kleene_or(left_result, right_result)
        elif operation_type in ["not", "!", "~"]:
            return ConditionEvaluator._kleene_not(
                ConditionEvaluator.evaluate_condition(row, operation.column)
            )
        else:
            return False

    @staticmethod
    def _evaluate_column_operation(
        row: Dict[str, Any], operation: ColumnOperation
    ) -> Optional[bool]:
        """Evaluate a column operation.

        Args:
            row: The data row to evaluate against.
            operation: The column operation to evaluate.

        Returns:
            True if operation evaluates to true, False otherwise.
        """
        operation_type = operation.operation
        col_value = ConditionEvaluator._get_column_value(row, operation.column)

        # Null checks
        if operation_type in ["isNotNull", "isnotnull"]:
            return col_value is not None
        elif operation_type in ["isNull", "isnull"]:
            return col_value is None
        elif operation_type in ["isnan", "isNaN"]:
            # isnan(NULL) is FALSE in Spark, never NULL.
            import math

            return isinstance(col_value, float) and math.isnan(col_value)

        # Null-safe equality (`<=>`). Never NULL: NULL <=> NULL is TRUE and
        # NULL <=> <non-null> is FALSE. `_evaluate_comparison` cannot express
        # that -- it collapses any NULL operand to `operation == "!="`.
        if operation_type == "eqNullSafe":
            other = operation.value
            if isinstance(other, (ColumnOperation, Column)):
                other = ConditionEvaluator._get_column_value(row, other)
            elif hasattr(other, "value") and not isinstance(
                other, (Column, ColumnOperation)
            ):
                other = other.value  # Unwrap Literal
            if col_value is None or other is None:
                return col_value is None and other is None
            return ConditionEvaluator._evaluate_comparison(col_value, "==", other)

        # Comparison operations
        if operation_type in ["==", "!=", ">", ">=", "<", "<="]:
            op_str: str = cast("str", operation_type)
            # Resolve left side: if col_value is a CaseWhen, evaluate it
            if hasattr(col_value, "evaluate") and hasattr(col_value, "conditions"):
                col_value = col_value.evaluate(row)
            # Resolve right side: if it's a Column/ColumnOperation, evaluate it;
            # if it's a literal value (str, int, float, etc.), use directly.
            right_val = operation.value
            if isinstance(right_val, (ColumnOperation, Column)):
                right_val = ConditionEvaluator._get_column_value(row, right_val)
            elif hasattr(right_val, "evaluate") and hasattr(right_val, "conditions"):
                right_val = right_val.evaluate(row)
            elif hasattr(right_val, "value") and not isinstance(
                right_val, (Column, ColumnOperation)
            ):
                right_val = right_val.value  # Unwrap Literal
            return ConditionEvaluator._evaluate_comparison(col_value, op_str, right_val)

        # String operations. A NULL operand makes the whole predicate NULL, not
        # FALSE -- the two are indistinguishable to `filter` but not to a
        # projection or to an enclosing NOT.
        if operation_type == "like":
            if col_value is None or operation.value is None:
                return None
            return ConditionEvaluator._evaluate_like_operation(
                col_value, operation.value
            )
        elif operation_type in ("startswith", "startsWith"):
            if col_value is None:
                return None
            return str(col_value).startswith(str(operation.value))
        elif operation_type in ("endswith", "endsWith"):
            if col_value is None:
                return None
            return str(col_value).endswith(str(operation.value))
        elif operation_type == "contains":
            if col_value is None:
                return None
            return str(operation.value) in str(col_value)
        elif operation_type == "rlike":
            if col_value is None or operation.value is None:
                return None
            import re

            return bool(re.search(str(operation.value), str(col_value)))
        elif operation_type == "isin":
            if operation.value is None:
                return None
            values = (
                list(operation.value)
                if isinstance(operation.value, (list, tuple))
                else [operation.value]
            )
            resolved = [
                ConditionEvaluator._get_column_value(row, item)
                if isinstance(item, (Column, ColumnOperation))
                else item.value
                if hasattr(item, "value")
                and hasattr(item, "name")
                and not isinstance(item, (Column, ColumnOperation))
                else item
                for item in values
            ]
            # SQL IN: NULL on the left is NULL; a match is TRUE; no match with
            # a NULL in the list is NULL; otherwise FALSE.
            if col_value is None:
                return None
            if ConditionEvaluator._evaluate_isin_operation(col_value, resolved):
                return True
            if any(item is None for item in resolved):
                return None
            return False
        elif operation_type == "between":
            if col_value is None or operation.value is None:
                return None
            return ConditionEvaluator._evaluate_between_operation(
                col_value, operation.value
            )

        # UDF operations (return value, evaluate as truthy/falsy in filter)
        if operation_type == "udf":
            result = ConditionEvaluator._evaluate_column_operation_value(row, operation)
            return bool(result) if result is not None else False

        # Function operations (hash, math, string functions)
        if operation_type in [
            "md5",
            "sha1",
            "crc32",
            "upper",
            "lower",
            "length",
            "trim",
            "abs",
            "round",
            "bround",
            "log10",
            "log",
            "log2",
            "concat",
            "split",
            "regexp_replace",
            "coalesce",
            "ceil",
            "floor",
            "sqrt",
            "exp",
            "sin",
            "cos",
            "tan",
            "asin",
            "acos",
            "atan",
            "sinh",
            "cosh",
            "tanh",
            "degrees",
            "radians",
            "sign",
            "greatest",
            "least",
            "when",
            "otherwise",
            "isnull",
            "isnotnull",
            "isnan",
            "nvl",
            "nvl2",
            "current_date",
            "current_timestamp",
            "to_date",
            "to_timestamp",
            "try_to_timestamp",
            "hour",
            "day",
            "dayofmonth",
            "month",
            "year",
            "dayofweek",
            "dayofyear",
            "weekofyear",
            "quarter",
            "minute",
            "second",
            "date_add",
            "date_sub",
            "datediff",
            "last_day",
            "trunc",
            "unix_timestamp",
            "from_unixtime",
        ]:
            # Evaluate through the value path rather than the scalar helper
            # `_evaluate_function_operation(col_value, ...)`, which only sees the
            # first operand and returns None whenever that operand is NULL. That
            # is wrong for null-tolerant / multi-argument functions:
            # `coalesce(NULL, False)` must be False, not NULL. The value path is
            # row-aware, handles every operand, and already covers a superset of
            # these operations (this mirrors the `udf` branch above).
            return ConditionEvaluator._to_sql_boolean(
                ConditionEvaluator._evaluate_column_operation_value(row, operation)
            )
        elif operation_type == "array_contains":
            # array_contains needs special handling - check if value is in array
            if col_value is None:
                return None
            search_value = ConditionEvaluator._resolve_operand(row, operation.value)
            if isinstance(col_value, (list, tuple)):
                return search_value in col_value
            return False
        elif operation_type == "size":
            # size returns the length of an array or map
            if col_value is None:
                return -1  # type: ignore[return-value]
            if isinstance(col_value, (list, tuple, dict)):
                return len(col_value)  # type: ignore[return-value]
            return -1  # type: ignore[return-value]
        elif operation_type in ("exists", "forall", "filter", "transform"):
            # Same evaluation as the value path: a higher-order function used
            # as a predicate (``df.filter(F.exists(...))``) must agree with the
            # same expression projected. The local implementation this replaced
            # called ``operation.value`` -- a MockLambdaExpression, which is not
            # callable -- and so returned the input array unchanged.
            return cast(
                "bool",
                ConditionEvaluator._evaluate_higher_order_operation(row, operation),
            )

        # Arithmetic operations
        if operation_type in ["+", "-", "*", "/", "%"]:
            left_value = ConditionEvaluator._get_column_value(row, operation.column)
            right_value = ConditionEvaluator._get_column_value(row, operation.value)

            if left_value is None or right_value is None:
                return None

            try:
                if operation_type == "+":
                    # PySpark compatibility: String concatenation with + operator returns None
                    # when DataFrame is cached. Check if both operands are strings.
                    is_string_concatenation = isinstance(
                        left_value, str
                    ) and isinstance(right_value, str)
                    if is_string_concatenation and get_row_value(
                        row, "__dataframe_is_cached__", False
                    ):
                        # Check if we're in a cached DataFrame context
                        return None
                    add_result: Any = left_value + right_value
                    return cast("Optional[bool]", add_result)
                elif operation_type == "-":
                    return cast("Optional[bool]", left_value - right_value)
                elif operation_type == "*":
                    return cast("bool", left_value * right_value)
                elif operation_type == "/":
                    if right_value == 0:
                        return None
                    return cast("bool", left_value / right_value)
                elif operation_type == "%":
                    if right_value == 0:
                        return None
                    return cast("bool", left_value % right_value)
            except (TypeError, ValueError, ZeroDivisionError):
                return None

        # Logical operations
        if operation_type in ["and", "&"]:
            left_result = ConditionEvaluator.evaluate_condition(row, operation.column)
            right_result = ConditionEvaluator.evaluate_condition(row, operation.value)
            return ConditionEvaluator._kleene_and(left_result, right_result)
        elif operation_type in ["or", "|"]:
            left_result = ConditionEvaluator.evaluate_condition(row, operation.column)
            right_result = ConditionEvaluator.evaluate_condition(row, operation.value)
            return ConditionEvaluator._kleene_or(left_result, right_result)
        elif operation_type in ["not", "!", "~"]:
            return ConditionEvaluator._kleene_not(
                ConditionEvaluator.evaluate_condition(row, operation.column)
            )

        # UDF operations - evaluate the UDF and return its boolean result
        if operation_type == "udf":
            result = ConditionEvaluator._evaluate_function_operation_value(
                row, operation
            )
            return bool(result) if result is not None else False

        return False

    @staticmethod
    def _evaluate_function_operation(
        value: Any, operation_type: str, scale: int = 0
    ) -> Any:
        """Evaluate function operations like md5, sha1, crc32, etc.

        Args:
            value: The input value to the function
            operation_type: The function name (md5, sha1, etc.)

        Returns:
            The result of the function operation, or None if input is None
        """
        # Handle null input - most functions return None for null input
        if value is None:
            return None

        # Hash functions
        if operation_type == "md5":
            import hashlib

            return hashlib.md5(str(value).encode()).hexdigest()
        elif operation_type == "sha1":
            import hashlib

            return hashlib.sha1(str(value).encode()).hexdigest()
        elif operation_type == "crc32":
            import zlib

            return zlib.crc32(str(value).encode()) & 0xFFFFFFFF

        # String functions
        elif operation_type == "upper":
            return str(value).upper()
        elif operation_type == "lower":
            return str(value).lower()
        elif operation_type == "length":
            return len(str(value))
        elif operation_type == "trim":
            return str(value).strip()

        # Math functions
        elif operation_type == "abs":
            return abs(float(value)) if value is not None else None
        elif operation_type == "round":
            return spark_round(value, scale)
        elif operation_type == "log10":
            import math

            return (
                math.log10(float(value))
                if value is not None and float(value) > 0
                else None
            )
        elif operation_type == "log":
            import math

            return (
                math.log(float(value))
                if value is not None and float(value) > 0
                else None
            )
        elif operation_type == "log2":
            import math

            return (
                math.log2(float(value))
                if value is not None and float(value) > 0
                else None
            )
        elif operation_type == "ceil":
            import math

            return math.ceil(float(value)) if value is not None else None
        elif operation_type == "floor":
            import math

            return math.floor(float(value)) if value is not None else None
        elif operation_type == "sqrt":
            import math

            return (
                math.sqrt(float(value))
                if value is not None and float(value) >= 0
                else None
            )
        elif operation_type == "exp":
            import math

            return math.exp(float(value)) if value is not None else None
        elif operation_type == "sin":
            import math

            return math.sin(float(value)) if value is not None else None
        elif operation_type == "cos":
            import math

            return math.cos(float(value)) if value is not None else None
        elif operation_type == "tan":
            import math

            return math.tan(float(value)) if value is not None else None
        elif operation_type == "asin":
            import math

            return (
                math.asin(float(value))
                if value is not None and -1 <= float(value) <= 1
                else None
            )
        elif operation_type == "acos":
            import math

            return (
                math.acos(float(value))
                if value is not None and -1 <= float(value) <= 1
                else None
            )
        elif operation_type == "atan":
            import math

            return math.atan(float(value)) if value is not None else None
        elif operation_type == "sinh":
            import math

            return math.sinh(float(value)) if value is not None else None
        elif operation_type == "cosh":
            import math

            return math.cosh(float(value)) if value is not None else None
        elif operation_type == "tanh":
            import math

            return math.tanh(float(value)) if value is not None else None
        elif operation_type == "degrees":
            import math

            return math.degrees(float(value)) if value is not None else None
        elif operation_type == "radians":
            import math

            return math.radians(float(value)) if value is not None else None
        elif operation_type == "sign":
            if value is None:
                return None
            val = float(value)
            if val > 0:
                return 1
            elif val < 0:
                return -1
            else:
                return 0

        # String functions
        elif operation_type == "concat":
            # For concat, we need to handle multiple arguments
            # This is a simplified version - in practice, concat might need special handling
            return str(value) if value is not None else None
        elif operation_type == "split":
            # For split, we need the delimiter - this is a simplified version
            return str(value).split() if value is not None else None
        elif operation_type == "regexp_replace":
            # For regexp_replace, we need pattern and replacement - this is a simplified version
            return str(value) if value is not None else None

        # Conditional functions
        elif operation_type == "coalesce":
            # For coalesce, we need multiple values - this is a simplified version
            return value if value is not None else None
        elif operation_type == "isnull":
            return value is None
        elif operation_type == "isnotnull":
            return value is not None
        elif operation_type == "isnan":
            if value is None:
                return False
            try:
                import math

                return math.isnan(float(value))
            except (ValueError, TypeError):
                return False
        elif operation_type == "nvl":
            # For nvl, we need a default value - this is a simplified version
            return value if value is not None else None
        elif operation_type == "nvl2":
            # For nvl2, we need two default values - this is a simplified version
            return value if value is not None else None

        # Comparison functions
        elif operation_type == "greatest":
            # For greatest, we need multiple values - this is a simplified version
            return value if value is not None else None
        elif operation_type == "least":
            # For least, we need multiple values - this is a simplified version
            return value if value is not None else None

        # Datetime functions
        elif operation_type == "current_date":
            from datetime import date

            return date.today()
        elif operation_type == "current_timestamp":
            from datetime import datetime

            return datetime.now()
        elif operation_type == "to_date":
            # For to_date, we need a format - this is a simplified version
            if value is None:
                return None
            try:
                from datetime import datetime

                return datetime.strptime(str(value), "%Y-%m-%d").date()
            except ValueError:
                return None
        elif operation_type in ("to_timestamp", "try_to_timestamp"):
            if value is None:
                return None
            try:
                from datetime import datetime

                # Default ISO-like formats (format from operation handled in _evaluate_function_operation_value)
                try:
                    return datetime.fromisoformat(str(value).replace(" ", "T"))
                except ValueError:
                    return datetime.strptime(str(value), "%Y-%m-%d %H:%M:%S")
            except (ValueError, TypeError):
                return None
        elif operation_type == "hour":
            if value is None:
                return None
            try:
                from datetime import datetime

                if isinstance(value, str):
                    try:
                        dt = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        dt = datetime.fromisoformat(value.replace(" ", "T"))
                else:
                    dt = value
                return dt.hour
            except (ValueError, AttributeError):
                return None
        elif operation_type == "day" or operation_type == "dayofmonth":
            if value is None:
                return None
            try:
                from datetime import datetime

                if isinstance(value, str):
                    try:
                        dt = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        dt = datetime.fromisoformat(value.replace(" ", "T"))
                else:
                    dt = value
                return dt.day
            except (ValueError, AttributeError):
                return None
        elif operation_type == "month":
            if value is None:
                return None
            try:
                from datetime import datetime

                if isinstance(value, str):
                    try:
                        dt = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        dt = datetime.fromisoformat(value.replace(" ", "T"))
                else:
                    dt = value
                return dt.month
            except (ValueError, AttributeError):
                return None
        elif operation_type == "year":
            if value is None:
                return None
            try:
                from datetime import datetime

                if isinstance(value, str):
                    try:
                        dt = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        dt = datetime.fromisoformat(value.replace(" ", "T"))
                else:
                    dt = value
                return dt.year
            except (ValueError, AttributeError):
                return None
        elif operation_type == "dayofweek":
            if value is None:
                return None
            try:
                from datetime import datetime

                if isinstance(value, str):
                    try:
                        dt = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        dt = datetime.fromisoformat(value.replace(" ", "T"))
                else:
                    dt = value
                return dt.isoweekday() % 7 + 1  # PySpark: Sun=1, Mon=2, ..., Sat=7
            except (ValueError, AttributeError):
                return None
        elif operation_type == "dayofyear":
            if value is None:
                return None
            try:
                from datetime import datetime

                if isinstance(value, str):
                    try:
                        dt = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        dt = datetime.fromisoformat(value.replace(" ", "T"))
                else:
                    dt = value
                return dt.timetuple().tm_yday
            except (ValueError, AttributeError):
                return None
        elif operation_type == "weekofyear":
            if value is None:
                return None
            try:
                from datetime import datetime

                if isinstance(value, str):
                    try:
                        dt = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        dt = datetime.fromisoformat(value.replace(" ", "T"))
                else:
                    dt = value
                return dt.isocalendar()[1]
            except (ValueError, AttributeError):
                return None
        elif operation_type == "quarter":
            if value is None:
                return None
            try:
                from datetime import datetime

                if isinstance(value, str):
                    try:
                        dt = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        dt = datetime.fromisoformat(value.replace(" ", "T"))
                else:
                    dt = value
                return (dt.month - 1) // 3 + 1
            except (ValueError, AttributeError):
                return None
        elif operation_type == "minute":
            if value is None:
                return None
            try:
                from datetime import datetime

                if isinstance(value, str):
                    try:
                        dt = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        dt = datetime.fromisoformat(value.replace(" ", "T"))
                else:
                    dt = value
                return dt.minute
            except (ValueError, AttributeError):
                return None
        elif operation_type == "second":
            if value is None:
                return None
            try:
                from datetime import datetime

                if isinstance(value, str):
                    try:
                        dt = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        dt = datetime.fromisoformat(value.replace(" ", "T"))
                else:
                    dt = value
                return dt.second
            except (ValueError, AttributeError):
                return None
        elif operation_type == "date_add":
            # For date_add, we need days to add
            if value is None:
                return None
            try:
                from datetime import datetime, timedelta, date as date_type

                # type: ignore[assignment]
                if isinstance(value, str):
                    try:  # type: ignore[assignment]
                        dt = datetime.strptime(value, "%Y-%m-%d").date()  # type: ignore[assignment]
                    except ValueError:  # type: ignore[assignment]
                        dt = datetime.fromisoformat(value.replace(" ", "T")).date()  # type: ignore[assignment]
                elif isinstance(value, datetime):  # type: ignore[assignment]
                    dt = value.date()  # type: ignore[assignment]
                elif isinstance(value, date_type):
                    dt = value  # type: ignore[assignment]
                else:
                    dt = value
                return dt + timedelta(
                    days=1
                )  # Simplified: always add 1 day in fallback
            except (ValueError, AttributeError):
                return None
        elif operation_type == "date_sub":
            # For date_sub, we need days to subtract
            if value is None:
                return None
            try:
                from datetime import datetime, timedelta, date as date_type

                # type: ignore[assignment]
                if isinstance(value, str):
                    try:  # type: ignore[assignment]
                        dt = datetime.strptime(value, "%Y-%m-%d").date()  # type: ignore[assignment]
                    except ValueError:  # type: ignore[assignment]
                        dt = datetime.fromisoformat(value.replace(" ", "T")).date()  # type: ignore[assignment]
                elif isinstance(value, datetime):  # type: ignore[assignment]
                    dt = value.date()  # type: ignore[assignment]
                elif isinstance(value, date_type):
                    dt = value  # type: ignore[assignment]
                else:
                    dt = value
                return dt - timedelta(
                    days=1
                )  # Simplified: always subtract 1 day in fallback
            except (ValueError, AttributeError):
                return None
        elif operation_type == "datediff":
            # For datediff, we need two dates - this is a simplified version
            if value is None:
                return None
            try:
                from datetime import datetime

                if isinstance(value, str):
                    dt = datetime.strptime(value, "%Y-%m-%d")
                else:
                    dt = value
                return (datetime.now() - dt).days
            except (ValueError, AttributeError):
                return None
        elif operation_type == "unix_timestamp":
            if value is None:
                return None
            try:
                from datetime import datetime

                if isinstance(value, str):
                    dt = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
                else:
                    dt = value
                return dt.timestamp()
            except (ValueError, AttributeError):
                return None
        elif operation_type == "from_unixtime":
            if value is None:
                return None
            try:
                from datetime import datetime

                return datetime.fromtimestamp(float(value))
            except (ValueError, AttributeError):
                return None

        # Default fallback
        return None

    @staticmethod
    def _get_column_value(row: Dict[str, Any], column: Union[Column, str, Any]) -> Any:
        """Get column value from row.

        Args:
            row: Data row.
            column: Column reference.

        Returns:
            Column value.
        """
        # Check ColumnOperation BEFORE Column since ColumnOperation is a subclass of Column
        # This ensures nested operations are properly evaluated, not treated as simple column references
        if isinstance(column, ColumnOperation):
            # Recursively evaluate the operation
            return ConditionEvaluator._evaluate_column_operation_value(row, column)
        # CaseWhen (F.when(...).otherwise(...)) is neither a Column nor a
        # ColumnOperation and exposes no ``.value``, so it used to reach the
        # ``return column`` fallback below and hand back the *unevaluated*
        # object. As an arithmetic operand that object then absorbed the
        # operator -- ``CaseWhen * 2.0`` dispatches to ``CaseWhen.__mul__``,
        # which builds a new ColumnOperation instead of multiplying -- so a
        # projected ``F.when(...) * F.lit(2.0)`` yielded a ColumnOperation
        # *as the cell value* rather than a number (BUG-054). Detected with
        # the same duck-type used by ``evaluate_condition`` above.
        elif hasattr(column, "evaluate") and hasattr(column, "conditions"):
            return column.evaluate(row)
        elif isinstance(column, Column):
            val = get_row_value(row, column.name)
            # Resolve dot-notation alias references (e.g. "r.id" -> "r_id")
            if val is None and "." in column.name:
                underscore_name = column.name.replace(".", "_", 1)
                val = get_row_value(row, underscore_name)
            return val
        elif isinstance(column, str):
            val = get_row_value(row, column)
            if val is None and "." in column:
                underscore_name = column.replace(".", "_", 1)
                val = get_row_value(row, underscore_name)
            return val
        elif hasattr(column, "value"):
            # Literal or similar object with a value attribute
            return column.value
        else:
            return column

    @staticmethod
    def _resolve_operand(row: Dict[str, Any], value: Any) -> Any:
        """Resolve a scalar operand that may be a column reference.

        Functions such as ``array_remove(col, x)`` accept either a plain Python
        literal or a ``Column``/``ColumnOperation``/``Literal``. A column-shaped
        operand must be resolved against the row before it is compared with
        array elements: comparing an element against an unresolved ``Column``
        invokes ``Column.__eq__``/``__ne__``, which returns a *truthy
        ColumnOperation* rather than a bool, so every comparison silently
        succeeds.

        A bare ``str`` is deliberately treated as a literal, not as a column
        name -- that is the PySpark signature for these functions.

        Args:
            row: Data row.
            value: The operand, either a literal or a column reference.

        Returns:
            The resolved value.
        """
        if isinstance(value, (Column, ColumnOperation)) or (
            not isinstance(value, (str, bytes)) and hasattr(value, "name")
        ):
            return ConditionEvaluator._get_column_value(row, value)
        return value

    @staticmethod
    def _element_key(item: Any) -> Any:
        """Build a hashable identity key for an array element.

        Mirrors the fallback used by ``array_union`` so that unhashable
        elements (lists, dicts) still participate in set operations.
        """
        if isinstance(item, (int, float, str, bool, type(None))):
            return item
        try:
            hash(item)
        except TypeError:
            return repr(item)
        return item

    @staticmethod
    def _coerce_for_comparison(left_val: Any, right_val: Any) -> Tuple[Any, Any]:
        """Coerce string to numeric for comparison if one is numeric and other is string.

        PySpark behavior: when comparing string with numeric, try to cast string to numeric.

        Args:
            left_val: Left value
            right_val: Right value

        Returns:
            Tuple of (coerced_left, coerced_right)
        """
        # Temporal pairs first: a date/datetime beside a string or beside the
        # other temporal type must be reconciled, or Python raises TypeError
        # and the comparison collapses to NULL. See core.datetime_utils.
        temporal_pair = coerce_temporal_pair(left_val, right_val)
        if temporal_pair is not None:
            return temporal_pair

        # Left is string, right is numeric: convert left to numeric
        if isinstance(left_val, str) and isinstance(right_val, (int, float)):
            try:
                if isinstance(right_val, int):
                    # Try integer first, then float
                    try:
                        left_num: Union[int, float] = int(float(left_val))
                    except (ValueError, TypeError):
                        left_num = float(left_val)
                else:
                    left_num = float(left_val)
                return left_num, right_val
            except (ValueError, TypeError):
                # Conversion failed, return original values
                return left_val, right_val
        # Right is string, left is numeric: convert right to numeric
        elif isinstance(right_val, str) and isinstance(left_val, (int, float)):
            try:
                if isinstance(left_val, int):
                    try:
                        right_num: Union[int, float] = int(float(right_val))
                    except (ValueError, TypeError):
                        right_num = float(right_val)
                else:
                    right_num = float(right_val)
                return left_val, right_num
            except (ValueError, TypeError):
                # Conversion failed, return original values
                return left_val, right_val
        # No coercion needed
        return left_val, right_val

    @staticmethod
    def _evaluate_comparison(
        col_value: Any, operation: str, condition_value: Any
    ) -> Optional[bool]:
        """Evaluate a scalar comparison under SQL three-valued logic.

        This is the single comparison kernel for this evaluator: both the
        predicate path (``_evaluate_column_operation``) and the value/projection
        path (``_evaluate_comparison_operation``) route through it, so a
        ``select`` and a ``filter`` can no longer disagree about the same
        expression.

        Args:
            col_value: Left-hand value (already resolved against the row).
            operation: Comparison operator, as a symbol (``==``, ``!=``, ``<``,
                ``<=``, ``>``, ``>=``) or its named alias (``eq``, ``ne``,
                ``lt``, ``le``, ``gt``, ``ge``).
            condition_value: Right-hand value (already resolved).

        Returns:
            The comparison result, or ``None`` when either operand is NULL --
            ``NULL > 5`` is NULL in SQL, not FALSE. The distinction is
            load-bearing: ``NOT (NULL > 5)`` is NULL, not TRUE, so collapsing
            to FALSE here flips the sign of any enclosing negation.
        """
        # NULL propagates: an unknown operand makes the whole comparison
        # unknown, `!=` included (`NULL != 9` is NULL, not TRUE).
        if col_value is None or condition_value is None:
            return None

        op = _COMPARISON_ALIASES.get(operation, operation)

        # Apply coercion if types are different
        coerced_left, coerced_right = ConditionEvaluator._coerce_for_comparison(
            col_value, condition_value
        )

        try:
            if op == "==":
                return bool(coerced_left == coerced_right)
            elif op == "!=":
                return bool(coerced_left != coerced_right)
            elif op == ">":
                return bool(coerced_left > coerced_right)
            elif op == ">=":
                return bool(coerced_left >= coerced_right)
            elif op == "<":
                return bool(coerced_left < coerced_right)
            elif op == "<=":
                return bool(coerced_left <= coerced_right)
        except TypeError:
            # Operand types that coercion could not reconcile. PySpark would
            # have rejected the plan at analysis time; NULL is the closest
            # runtime approximation and matches the withColumn path.
            return None

        return False

    @staticmethod
    def _evaluate_like_operation(col_value: Any, pattern: str) -> bool:
        """Evaluate LIKE operation (full string match).

        Args:
            col_value: Column value.
            pattern: LIKE pattern (% = any sequence, _ = one char).

        Returns:
            True if pattern matches entire string.
        """
        if col_value is None:
            return False

        import re

        value = str(col_value)
        # Convert SQL LIKE to regex: % -> .*, _ -> ., escape others, full match
        result = []
        for c in str(pattern):
            if c == "%":
                result.append(".*")
            elif c == "_":
                result.append(".")
            else:
                result.append(re.escape(c))
        regex_pattern = "^" + "".join(result) + "$"
        return bool(re.fullmatch(regex_pattern, value))

    @staticmethod
    def _evaluate_isin_operation(col_value: Any, values: List[Any]) -> bool:
        """Evaluate IN operation.

        Args:
            col_value: Column value.
            values: List of values to check against.

        Returns:
            True if value is in list.
        """
        if col_value is None:
            return False
        # Direct match first
        if col_value in values:
            return True
        # Type coercion: PySpark compares with type casting (e.g. string "1" matches int 1)
        for v in values:
            if v is None:
                continue
            # String col_value vs numeric list value
            if isinstance(col_value, str) and isinstance(v, (int, float)):
                try:
                    if isinstance(v, int):
                        if int(float(col_value)) == v:
                            return True
                    else:
                        if float(col_value) == v:
                            return True
                except (ValueError, TypeError):
                    continue
            # Numeric col_value vs string list value
            elif isinstance(col_value, (int, float)) and isinstance(v, str):
                try:
                    if isinstance(col_value, int):
                        if col_value == int(float(v)):
                            return True
                    else:
                        if col_value == float(v):
                            return True
                except (ValueError, TypeError):
                    continue
        return False

    @staticmethod
    def _evaluate_between_operation(col_value: Any, bounds: Tuple[Any, Any]) -> bool:
        """Evaluate BETWEEN operation.

        Args:
            col_value: Column value.
            bounds: Tuple of (lower, upper) bounds.

        Returns:
            True if value is between bounds.
        """
        if col_value is None:
            return False

        lower, upper = bounds
        # Unwrap Literal objects
        if hasattr(lower, "value") and not isinstance(lower, (Column, ColumnOperation)):
            lower = lower.value
        if hasattr(upper, "value") and not isinstance(upper, (Column, ColumnOperation)):
            upper = upper.value
        return bool(lower <= col_value <= upper)
