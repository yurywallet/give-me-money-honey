"""Runs configured job sources on a poll interval: fetch -> score -> store.

Dedup is by (source, external_id) via db.upsert_job - a job seen again just
refreshes last_seen_at/score rather than creating a duplicate row.
"""
from __future__ import annotations

import asyncio
import logging
import sqlite3

from candidate_profile import CandidateProfile, DEFAULT_PROFILE
from clock import Clock, SystemClock
from config import SearchConfig
from db import finish_search_run, start_search_run, upsert_job
from scoring import score_job
from sources import JobSource

logger = logging.getLogger("give_me_money_honey.scheduler")


class Scheduler:
    def __init__(
        self,
        config: SearchConfig,
        sources: list[JobSource],
        conn: sqlite3.Connection,
        clock: Clock | None = None,
        profile: CandidateProfile = DEFAULT_PROFILE,
    ):
        self.config = config
        self.sources = sources
        self.conn = conn
        self.clock = clock or SystemClock()
        self.profile = profile
        self._task: asyncio.Task | None = None
        self._stop_event: asyncio.Event | None = None

    def run_once(self) -> dict[str, dict]:
        """Fetch, score, and store from every source once. Returns a per-source summary."""
        summary: dict[str, dict] = {}
        for source in self.sources:
            now_iso = self.clock.now().isoformat()
            run_id = start_search_run(
                self.conn, source.name, ",".join(self.config.keywords), now_iso,
                # Sources that only return a capped top-N can't support the
                # "missing => delisted" inference (§ DECISIONS.md 2026-08-02).
                # getattr default True: a source predating this attribute keeps
                # the original behaviour rather than silently changing it.
                enumerates_all=getattr(source, "enumerates_all_matches", True),
            )
            try:
                jobs = source.search(list(self.config.keywords))
            except Exception as exc:  # noqa: BLE001 - one bad source must not kill the run
                logger.warning("source %s failed: %s", source.name, exc)
                finish_search_run(self.conn, run_id, self.clock.now().isoformat(), 0, 0, error=str(exc))
                summary[source.name] = {"error": str(exc)}
                continue

            n_new = 0
            n_passed = 0
            n_verify = 0
            # Why the non-passing jobs failed, aggregated (operator ask
            # 2026-08-02: "no hard requirements met" alone doesn't say WHICH
            # requirement). Grouped by the leading token of each reason
            # string, matching db.hard_fail_reason_counts' convention.
            fail_reasons: dict[str, int] = {}
            now = self.clock.now()
            for job in jobs:
                score_job(job, self.config.hard, self.profile, now=now)
                if job.hard_pass:
                    n_passed += 1
                else:
                    if job.needs_verification:
                        n_verify += 1
                    for reason in job.score_breakdown.get("hard_fail_reasons", []):
                        key = reason.split("'")[0].strip() if "'" in reason else reason.split(" ")[0]
                        fail_reasons[key] = fail_reasons.get(key, 0) + 1
                _, is_new = upsert_job(self.conn, job, self.clock.now().isoformat(), run_id)
                n_new += int(is_new)

            finish_search_run(self.conn, run_id, self.clock.now().isoformat(), len(jobs), n_new)
            summary[source.name] = {
                "found": len(jobs),
                "new": n_new,
                "hard_pass": n_passed,
                "needs_verification": n_verify,
                "fail_reasons": dict(sorted(fail_reasons.items(), key=lambda kv: -kv[1])),
            }
        return summary

    async def run_forever(self) -> None:
        self._stop_event = asyncio.Event()
        interval_seconds = self.config.poll_interval_minutes * 60
        while not self._stop_event.is_set():
            try:
                self.run_once()
            except Exception:  # noqa: BLE001 - a periodic loop must never die silently
                logger.exception("scheduler run_once failed")
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=interval_seconds)
            except asyncio.TimeoutError:
                pass

    def start(self) -> bool:
        """Start the background poll loop. Returns False if already running."""
        if self._task is not None and not self._task.done():
            return False
        self._task = asyncio.create_task(self.run_forever())
        return True

    def stop(self) -> bool:
        """Stop the background poll loop. Returns False if it wasn't running."""
        if self._stop_event is None or self._task is None or self._task.done():
            return False
        self._stop_event.set()
        return True

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()
