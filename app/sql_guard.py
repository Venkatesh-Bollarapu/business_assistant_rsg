"""Defence-in-depth SQL validation.

The primary protection is architectural: the LLM never writes SQL, it only
selects an *intent*, and each intent maps to a fixed, parameterised template.
This guard is a second, independent layer. Every statement is checked here
immediately before execution, and the connection itself is read-only.
"""

from __future__ import annotations

import re

FORBIDDEN_KEYWORDS = (
    "insert", "update", "delete", "drop", "alter", "create", "replace", "truncate",
    "attach", "detach", "pragma", "vacuum", "reindex", "grant", "revoke", "exec",
)
_FORBIDDEN_RE = re.compile(r"\b(" + "|".join(FORBIDDEN_KEYWORDS) + r")\b", re.IGNORECASE)
ALLOWED_TABLES = {"transactions"}
_TABLE_RE = re.compile(r"\b(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_]*)", re.IGNORECASE)


class UnsafeSQLError(ValueError):
    """Raised when a statement is not a single, read-only SELECT on an allowed table."""


def validate_read_only_sql(sql: str) -> str:
    """Return the SQL unchanged if it is safe to run, otherwise raise UnsafeSQLError."""
    if not isinstance(sql, str) or not sql.strip():
        raise UnsafeSQLError("Empty SQL statement.")

    statement = sql.strip()
    if statement.endswith(";"):
        statement = statement[:-1].rstrip()

    if ";" in statement:
        raise UnsafeSQLError("Multiple statements are not allowed.")
    if "--" in statement or "/*" in statement or "*/" in statement:
        raise UnsafeSQLError("SQL comments are not allowed.")
    if not re.match(r"^select\b", statement, re.IGNORECASE):
        raise UnsafeSQLError("Only SELECT statements are allowed.")

    forbidden = _FORBIDDEN_RE.search(statement)
    if forbidden:
        raise UnsafeSQLError(f"Forbidden keyword in SQL: {forbidden.group(1).upper()}")

    tables = {name.lower() for name in _TABLE_RE.findall(statement)}
    if not tables or not tables <= ALLOWED_TABLES:
        raise UnsafeSQLError(f"Query touches a table outside the allowlist: {sorted(tables)}")

    return sql
