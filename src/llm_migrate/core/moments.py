"""Timezone normalization for host-supplied timestamps at API boundaries.

Internals compare aware UTC datetimes (session-overlay expiry, freshness).
Hosts, however, routinely pass naive ISO strings (``2026-09-22T10:00:00``)
through MCP and the CLI; mixing those with aware datetimes raises
``TypeError`` at comparison time. Every transport boundary funnels its
timestamps through :func:`utc_moment`: naive means UTC, aware is converted
to UTC, and internal comparisons stay aware everywhere.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import overload


@overload
def utc_moment(value: str | datetime) -> datetime: ...


@overload
def utc_moment(value: None) -> None: ...


def utc_moment(value: str | datetime | None) -> datetime | None:
    """Parse/normalize one host timestamp into an aware UTC datetime.

    ``None`` passes through so callers keep their own "no timestamp given"
    defaults. A naive timestamp is interpreted as UTC; an aware one is
    converted to UTC.
    """
    if value is None:
        return None
    moment = datetime.fromisoformat(value) if isinstance(value, str) else value
    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)
