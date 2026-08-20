import pytest

from llm_retry import call_with_rate_limit_retry


def _fake_429(retry_hint: str = "Please retry in 44s") -> Exception:
    return RuntimeError(f"429 RESOURCE_EXHAUSTED. {retry_hint}")


def test_returns_immediately_on_success():
    sleeps = []
    assert call_with_rate_limit_retry(lambda: "ok", sleep=sleeps.append) == "ok"
    assert sleeps == []  # never waited


def test_retries_on_rate_limit_then_succeeds():
    calls = {"n": 0}
    sleeps = []

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise _fake_429()
        return "ok"

    assert call_with_rate_limit_retry(flaky, sleep=sleeps.append) == "ok"
    assert calls["n"] == 2
    assert len(sleeps) == 1  # waited once, between the two attempts


def test_non_rate_limit_error_reraises_immediately():
    sleeps = []

    def boom():
        raise ValueError("bad request")

    with pytest.raises(ValueError, match="bad request"):
        call_with_rate_limit_retry(boom, sleep=sleeps.append)
    assert sleeps == []  # not retried


def test_persistent_per_minute_limit_exhausts_and_reraises():
    sleeps = []

    def always_429():
        raise _fake_429("Please retry in 5s")  # per-minute style, no daily marker

    with pytest.raises(RuntimeError, match="429"):
        call_with_rate_limit_retry(always_429, max_retries=2, sleep=sleeps.append)
    assert len(sleeps) == 2  # retried max_retries times, then gave up


def test_daily_quota_fails_fast_without_waiting():
    # A per-DAY quota won't clear in seconds - don't hang the caller retrying it.
    sleeps = []

    def daily_429():
        raise RuntimeError(
            "429 RESOURCE_EXHAUSTED quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier"
        )

    with pytest.raises(RuntimeError, match="429"):
        call_with_rate_limit_retry(daily_429, max_retries=2, sleep=sleeps.append)
    assert sleeps == []  # never waited - failed fast


def test_honors_server_retry_hint_capped():
    sleeps = []
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise _fake_429("Please retry in 44.28s")
        return "ok"

    call_with_rate_limit_retry(flaky, max_wait_s=60, sleep=sleeps.append)
    assert sleeps == [pytest.approx(44.28)]  # parsed the hint, under the cap


def test_retry_hint_is_capped_at_max_wait():
    sleeps = []
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise _fake_429("retryDelay: 300s")
        return "ok"

    call_with_rate_limit_retry(flaky, max_wait_s=60, sleep=sleeps.append)
    assert sleeps == [60]  # capped, never waits the full 300s
