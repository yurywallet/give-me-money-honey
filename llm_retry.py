"""Retry a rate-limited (429 / RESOURCE_EXHAUSTED) LLM call, honoring the
server's retry hint.

Used by both fit_summary and agent's Gemini paths so a PER-MINUTE free-tier
limit self-heals (Gemini free tier is 5 requests/min) instead of failing. A
PER-DAY limit (20/day) is NOT fixable by waiting a few seconds - retries
exhaust and the error propagates so the caller can stop (app.py fails fast on
it). This is deliberately provider-agnostic string-matching: the google-genai
ClientError carries "429"/"RESOURCE_EXHAUSTED" in its message, and matching the
text keeps this module free of any SDK import.
"""
from __future__ import annotations

import re
import time
from typing import Callable, TypeVar

T = TypeVar("T")

_RATE_LIMIT_MARKERS = ("429", "RESOURCE_EXHAUSTED")
# A per-DAY quota won't clear in the seconds a retry can wait - waiting on it
# just hangs the caller (e.g. blocks the dashboard's pre-gen spinner for the
# full retry budget before failing anyway). Only per-minute/second limits are
# worth retrying. Gemini's error carries the quota id, e.g.
# "GenerateRequestsPerDayPerProjectPerModel-FreeTier".
_DAILY_MARKERS = ("PerDay", "per day", "per-day", "daily", "GenerateRequestsPerDay")
# Matches both "Please retry in 44.28s" and "'retryDelay': '44s'".
_RETRY_HINT_RE = re.compile(r"(?:retry in|retryDelay['\"]?:?)\s*['\"]?([\d.]+)\s*s")


def _is_retryable_rate_limit(exc: Exception) -> bool:
    msg = str(exc)
    if not any(marker in msg for marker in _RATE_LIMIT_MARKERS):
        return False
    return not any(marker in msg for marker in _DAILY_MARKERS)


def call_with_rate_limit_retry(
    call: Callable[[], T],
    max_retries: int = 2,
    max_wait_s: float = 60.0,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run `call()`, retrying on a rate-limit error up to `max_retries` times.

    Waits the server's hinted delay (capped at `max_wait_s`) between attempts,
    else a linear backoff. Non-rate-limit errors re-raise immediately; a
    rate-limit error that survives all retries re-raises too (a daily quota
    won't clear in seconds). `sleep` is injectable so tests don't actually wait.
    """
    for attempt in range(max_retries + 1):
        try:
            return call()
        except Exception as exc:  # noqa: BLE001 - re-raised below unless retryable
            if not _is_retryable_rate_limit(exc) or attempt == max_retries:
                raise
            hint = _RETRY_HINT_RE.search(str(exc))
            wait = float(hint.group(1)) if hint else 10.0 * (attempt + 1)
            sleep(min(wait, max_wait_s))
    raise AssertionError("unreachable")  # loop returns or raises
