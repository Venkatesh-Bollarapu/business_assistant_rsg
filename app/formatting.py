"""Small display helpers (kept separate so business logic stays free of formatting)."""

from __future__ import annotations

import calendar


def inr(value: float) -> str:
    """Format a number with Indian digit grouping, e.g. 1439200 -> ₹14,39,200."""
    negative = value < 0
    whole, fraction = f"{abs(value):.2f}".split(".")
    head, tail = whole[:-3], whole[-3:]
    groups: list[str] = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    formatted = ",".join(groups + [tail]) if groups else tail
    if fraction != "00":
        formatted += f".{fraction}"
    return f"{'-' if negative else ''}₹{formatted}"


def month_label(month: str) -> str:
    """'2026-02' -> 'February 2026'."""
    year, month_number = month.split("-")
    return f"{calendar.month_name[int(month_number)]} {year}"
