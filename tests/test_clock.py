import datetime as dt

import pytest

from clock import FixedClock


def test_fixed_clock_returns_pinned_time():
    t = dt.datetime(2026, 7, 14, 12, 0, tzinfo=dt.timezone.utc)
    clock = FixedClock(t)
    assert clock.now() == t


def test_fixed_clock_advance():
    t = dt.datetime(2026, 7, 14, 12, 0, tzinfo=dt.timezone.utc)
    clock = FixedClock(t)
    clock.advance(days=1, hours=1)
    assert clock.now() == dt.datetime(2026, 7, 15, 13, 0, tzinfo=dt.timezone.utc)


def test_fixed_clock_month_boundary():
    t = dt.datetime(2026, 1, 31, 23, 0, tzinfo=dt.timezone.utc)
    clock = FixedClock(t)
    clock.advance(hours=2)
    assert clock.now() == dt.datetime(2026, 2, 1, 1, 0, tzinfo=dt.timezone.utc)


def test_fixed_clock_requires_tz_aware():
    with pytest.raises(ValueError):
        FixedClock(dt.datetime(2026, 7, 14))
