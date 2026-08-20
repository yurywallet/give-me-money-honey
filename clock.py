"""Injectable clock.

Per engineering-foundations §1: any logic that branches on "now" (poll
cadence, job-posting staleness, next-run scheduling) must not read the wall
clock directly, or it can't be pinned to a known date in a test and any
calendar-edge bug (midnight rollover, DST) stays invisible until it happens
live. Production code takes a Clock; tests pass FixedClock.
"""
from __future__ import annotations

import datetime as dt
from typing import Protocol


class Clock(Protocol):
    def now(self) -> dt.datetime: ...


class SystemClock:
    def now(self) -> dt.datetime:
        return dt.datetime.now(dt.timezone.utc)


class FixedClock:
    def __init__(self, fixed: dt.datetime):
        if fixed.tzinfo is None:
            raise ValueError("FixedClock requires a timezone-aware datetime")
        self._fixed = fixed

    def now(self) -> dt.datetime:
        return self._fixed

    def advance(self, **kwargs) -> None:
        self._fixed = self._fixed + dt.timedelta(**kwargs)
