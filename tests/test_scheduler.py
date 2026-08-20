import datetime as dt

from clock import FixedClock
from config import HardCriteria, SearchConfig
from db import get_conn, init_db, list_top_jobs
from scheduler import Scheduler
from sources.mock_source import MockJobSource


def _scheduler(tmp_path):
    conn = get_conn(str(tmp_path / "test.db"))
    init_db(conn)
    # Pinned rather than left at the module default: mock_source.py's fixture
    # salaries are calibrated to this floor (one listing is designed to fail
    # on salary + hybrid location).
    cfg = SearchConfig(hard=HardCriteria(min_salary=200_000))
    clock = FixedClock(dt.datetime(2026, 7, 14, 12, 0, tzinfo=dt.timezone.utc))
    return Scheduler(cfg, [MockJobSource()], conn, clock=clock), conn


def test_run_once_stores_scored_jobs(tmp_path):
    sched, conn = _scheduler(tmp_path)
    summary = sched.run_once()

    assert summary["mock"]["found"] == 4
    assert summary["mock"]["new"] == 4
    # Of the 4 mock listings, only "Senior Analytics Engineer" (mock-4)
    # matches the default profile's target titles; the others are gated out
    # by title even though some would otherwise pass salary/location/type -
    # e.g. mock-2 "Staff Platform Engineer" is a different role entirely.
    assert summary["mock"]["hard_pass"] == 1

    top = list_top_jobs(conn, limit=10, only_passing=True)
    assert len(top) == 1
    assert all(j.hard_pass for j in top)


def test_run_once_summary_explains_why_jobs_missed(tmp_path):
    # Operator ask 2026-08-02: "no hard requirements met" alone doesn't say
    # WHICH requirement - the summary must break the misses down by reason.
    sched, _ = _scheduler(tmp_path)
    summary = sched.run_once()["mock"]

    assert "needs_verification" in summary
    reasons = summary["fail_reasons"]
    assert reasons, "a run with failing jobs must report why they failed"
    # Of the 4 mock listings 3 miss: two on too-few matched skills, one on
    # hybrid location + below-floor salary (see sources/mock_source.py).
    assert reasons == {"skill_match": 2, "location_mode": 1, "salary": 1}
    # Sorted most-common-first so the UI's top row is the dominant blocker.
    assert list(reasons.values()) == sorted(reasons.values(), reverse=True)


def test_run_once_twice_does_not_duplicate(tmp_path):
    sched, conn = _scheduler(tmp_path)
    sched.run_once()
    summary = sched.run_once()

    assert summary["mock"]["new"] == 0
    count = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    assert count == 4


def test_a_failing_source_does_not_crash_the_run(tmp_path):
    class BrokenSource:
        name = "broken"

        def search(self, keywords):
            raise RuntimeError("network down")

    conn = get_conn(str(tmp_path / "test.db"))
    init_db(conn)
    cfg = SearchConfig()
    sched = Scheduler(cfg, [BrokenSource(), MockJobSource()], conn)

    summary = sched.run_once()
    assert "error" in summary["broken"]
    assert summary["mock"]["found"] == 4
