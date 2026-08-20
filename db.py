"""SQLite storage for discovered jobs.

Concurrency model (engineering-foundations §1, decided up front rather than
after the first corruption incident): WAL mode + a busy_timeout, because more
than one writer can plausibly touch this file at once - the MCP server
answering a tool call while the background scheduler is mid-poll. WAL lets
readers and one writer coexist instead of one side silently losing a write.
This mirrors trader_joe's database.py precedent for the same reason.
"""
from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
import statistics
from dataclasses import dataclass, field
from typing import Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source          TEXT NOT NULL,
    external_id     TEXT NOT NULL,
    title           TEXT NOT NULL,
    company         TEXT NOT NULL,
    url             TEXT NOT NULL,
    description     TEXT NOT NULL DEFAULT '',
    salary_min      INTEGER,
    salary_max      INTEGER,
    work_type       TEXT,
    location_mode   TEXT,
    location_country TEXT,
    location_raw    TEXT,
    posted_at       TEXT,
    hard_pass       INTEGER NOT NULL DEFAULT 0,
    needs_verification INTEGER NOT NULL DEFAULT 0,
    score           REAL NOT NULL DEFAULT 0,
    score_breakdown TEXT NOT NULL DEFAULT '{}',
    benefits        TEXT NOT NULL DEFAULT '[]',
    equity_signals  TEXT NOT NULL DEFAULT '[]',
    matched_skills  TEXT NOT NULL DEFAULT '[]',
    missing_skills  TEXT NOT NULL DEFAULT '[]',
    fit_summaries   TEXT NOT NULL DEFAULT '{}',
    last_seen_run_id INTEGER,
    first_seen_at   TEXT NOT NULL,
    last_seen_at    TEXT NOT NULL,
    UNIQUE(source, external_id)
);

CREATE INDEX IF NOT EXISTS idx_jobs_score ON jobs(hard_pass, score DESC);

CREATE TABLE IF NOT EXISTS search_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    source      TEXT NOT NULL,
    query       TEXT NOT NULL,
    n_found     INTEGER NOT NULL DEFAULT 0,
    n_new       INTEGER NOT NULL DEFAULT 0,
    error       TEXT,
    -- Whether this source returns EVERY match or a capped top-N; gates
    -- annotate_active's delisted inference (§ DECISIONS.md 2026-08-02).
    enumerates_all INTEGER NOT NULL DEFAULT 1
);

-- One row per "Rerun & save" click on the Role Map tab (§ DECISIONS.md
-- 2026-08-02) - a dated snapshot of role_fit.compute_role_fit's output so
-- match-score trend over time (as skills/anchors are edited) is visible
-- later, not just the always-live, unsaved computation the tab shows by
-- default. results_json is the full serialized snapshot, not normalized
-- into columns - this is a point-in-time log, not something queried by
-- field the way jobs/search_runs are.
CREATE TABLE IF NOT EXISTS role_fit_snapshots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    computed_at TEXT NOT NULL,
    results_json TEXT NOT NULL
);
"""


@dataclass
class Job:
    source: str
    external_id: str
    title: str
    company: str
    url: str
    description: str = ""
    salary_min: Optional[int] = None
    salary_max: Optional[int] = None
    work_type: Optional[str] = None
    location_mode: Optional[str] = None
    location_country: Optional[str] = None
    location_raw: Optional[str] = None
    posted_at: Optional[str] = None
    # Populated by scoring.score_job() before storage, not by the source.
    hard_pass: bool = False
    # True if hard_pass=False PURELY because a field was unconfirmed by the
    # source (salary/location/work_type never stated), not a confirmed bad
    # fit (§ DECISIONS.md 2026-07-30 "needs verification" bucket). Distinct
    # from hard_pass so a $196k Faire "Analytics Engineer" missing only
    # work_type/location confirmation isn't hidden the same way a confirmed-
    # onsite or confirmed-below-floor job is.
    needs_verification: bool = False
    score: float = 0.0
    score_breakdown: dict = field(default_factory=dict)
    benefits: list = field(default_factory=list)
    # Equity/stage signals from the JD text (scoring.parse_equity_signals) -
    # the ranking evidence behind the equity pivot (§ DECISIONS.md 2026-07-25).
    equity_signals: list = field(default_factory=list)
    matched_skills: list = field(default_factory=list)
    # Skills the JD wants that the profile lacks (scoring.missing_skills) - the
    # "add these to lift your match score" gap. Shown as distinct-color chips.
    missing_skills: list = field(default_factory=list)
    # LLM fit summaries keyed by provider name, e.g. {"gemini": "...",
    # "ollama": "..."} - so a Gemini and a local (Ollama) summary can coexist
    # and the user can compare them / fall back to the local one when the cloud
    # quota is hit. Populated on demand (set_fit_summary), never by
    # score_job()/upsert_job() (expensive, orthogonal to the deterministic
    # score). Empty dict means "none generated yet".
    fit_summaries: dict = field(default_factory=dict)
    # Set by upsert_job() to the search_runs row that last confirmed this
    # listing; is_active is derived from it by annotate_active(), never
    # stored directly - see annotate_active()'s docstring for why.
    last_seen_run_id: Optional[int] = None
    # When this listing was last returned by a search. Used by
    # annotate_active for the staleness fallback on sources that can't
    # enumerate all matches (§ DECISIONS.md 2026-08-02).
    last_seen_at: Optional[str] = None
    is_active: bool = True
    id: Optional[int] = None
    db_conn: Optional[sqlite3.Connection] = field(default=None, repr=False, compare=False)

    @property
    def fit_summary(self) -> Optional[str]:
        """Backward-compatible single-summary accessor.

        Historically the code used a single `fit_summary` TEXT column. The
        newer model stores per-provider summaries in `fit_summaries` (a
        dict). Tests and UI still expect `job.fit_summary` to return the
        cached string when one exists (or None). Return the first provider's
        summary if present.
        """
        if not self.fit_summaries:
            return None
        # Return an arbitrary provider's summary (preserve existing behaviour)
        return next(iter(self.fit_summaries.values()))

    @fit_summary.setter
    def fit_summary(self, value: str | None) -> None:
        """Allow assigning a single summary to the Job instance for UI convenience.

        This updates the in-memory `fit_summaries` mapping under the default
        provider key ('gemini') so callers that do `job.fit_summary = s` after
        persisting via `set_fit_summary(...)` don't hit an AttributeError and
        see the new value immediately. Persistence to the DB is the
        responsibility of `set_fit_summary` and is not performed here.
        """
        if value is None:
            # remove any cached provider summaries
            self.fit_summaries = {}
            return
        if not isinstance(self.fit_summaries, dict):
            self.fit_summaries = {}
        self.fit_summaries["gemini"] = value
        if self.id is not None and self.db_conn is not None:
            set_fit_summary(self.db_conn, self.id, value)


def get_conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    _ensure_column(conn, "jobs", "needs_verification", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(conn, "jobs", "equity_signals", "TEXT NOT NULL DEFAULT '[]'")
    _ensure_column(conn, "jobs", "missing_skills", "TEXT NOT NULL DEFAULT '[]'")
    _ensure_column(conn, "jobs", "fit_summaries", "TEXT NOT NULL DEFAULT '{}'")
    _ensure_column(conn, "search_runs", "enumerates_all", "INTEGER NOT NULL DEFAULT 1")
    _backfill_enumerates_all(conn)
    _migrate_single_fit_summary(conn)
    conn.commit()


def _backfill_enumerates_all(conn: sqlite3.Connection) -> None:
    """Historical runs predate the column and defaulted to 1 (=complete),
    which is wrong for the capped LinkedIn source and would keep mislabelling
    its jobs "no longer accepting" until the next run overwrote it. Correct
    those rows once (§ DECISIONS.md 2026-08-02). Idempotent."""
    conn.execute(
        "UPDATE search_runs SET enumerates_all = 0 "
        "WHERE source = 'linkedin_apify' AND enumerates_all = 1"
    )


def _migrate_single_fit_summary(conn: sqlite3.Connection) -> None:
    """One-time carry-over: an older single `fit_summary` TEXT column (if the DB
    predates per-provider summaries) is folded into fit_summaries under a
    'gemini' key, since that's what generated them. Harmless/no-op on a DB that
    never had the column or already migrated."""
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(jobs)")}
    if "fit_summary" not in cols:
        return
    for row in conn.execute(
        "SELECT id, fit_summary FROM jobs WHERE fit_summary IS NOT NULL AND fit_summaries = '{}'"
    ).fetchall():
        conn.execute(
            "UPDATE jobs SET fit_summaries = ? WHERE id = ?",
            (json.dumps({"gemini": row["fit_summary"]}), row["id"]),
        )


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, coltype: str) -> None:
    """Additive migration for a table that may already exist from before this
    column was introduced - `CREATE TABLE IF NOT EXISTS` in SCHEMA doesn't
    touch an existing table's columns, only a fresh database gets the column
    from SCHEMA directly."""
    cols = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")


def upsert_job(conn: sqlite3.Connection, job: Job, now_iso: str, run_id: Optional[int] = None) -> tuple[int, bool]:
    """Insert or refresh a job. Returns (row_id, is_new).

    `run_id` (the search_runs row this fetch belongs to) is stored as
    last_seen_run_id - the presence check annotate_active() uses to tell a
    still-listed job from one that quietly dropped out of the source's
    results.
    """
    cur = conn.execute(
        "SELECT id FROM jobs WHERE source = ? AND external_id = ?",
        (job.source, job.external_id),
    )
    row = cur.fetchone()
    if row is None:
        cur = conn.execute(
            """
            INSERT INTO jobs (
                source, external_id, title, company, url, description,
                salary_min, salary_max, work_type, location_mode,
                location_country, location_raw, posted_at,
                hard_pass, needs_verification, score, score_breakdown, benefits,
                equity_signals, matched_skills, missing_skills, last_seen_run_id,
                first_seen_at, last_seen_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job.source, job.external_id, job.title, job.company, job.url,
                job.description, job.salary_min, job.salary_max, job.work_type,
                job.location_mode, job.location_country, job.location_raw,
                job.posted_at, int(job.hard_pass), int(job.needs_verification), job.score,
                json.dumps(job.score_breakdown), json.dumps(job.benefits),
                json.dumps(job.equity_signals), json.dumps(job.matched_skills),
                json.dumps(job.missing_skills), run_id, now_iso, now_iso,
            ),
        )
        conn.commit()
        return cur.lastrowid, True

    job_id = row["id"]
    conn.execute(
        """
        UPDATE jobs SET
            title = ?, company = ?, url = ?, description = ?,
            salary_min = ?, salary_max = ?, work_type = ?, location_mode = ?,
            location_country = ?, location_raw = ?, posted_at = ?,
            hard_pass = ?, needs_verification = ?, score = ?, score_breakdown = ?,
            benefits = ?, equity_signals = ?, matched_skills = ?, missing_skills = ?,
            last_seen_run_id = ?, last_seen_at = ?
        WHERE id = ?
        """,
        (
            job.title, job.company, job.url, job.description,
            job.salary_min, job.salary_max, job.work_type, job.location_mode,
            job.location_country, job.location_raw, job.posted_at,
            int(job.hard_pass), int(job.needs_verification), job.score,
            json.dumps(job.score_breakdown), json.dumps(job.benefits),
            json.dumps(job.equity_signals), json.dumps(job.matched_skills),
            json.dumps(job.missing_skills), run_id, now_iso, job_id,
        ),
    )
    conn.commit()
    return job_id, False


def _row_to_job(conn: sqlite3.Connection, row: sqlite3.Row) -> Job:
    # Defensive access: older DBs may lack newer columns; fall back to
    # sensible defaults instead of raising IndexError.
    cols = set(row.keys())

    def g(k, default=None):
        return row[k] if k in cols else default

    return Job(
        id=g("id"),
        source=g("source", ""),
        external_id=g("external_id", ""),
        title=g("title", ""),
        company=g("company", ""),
        url=g("url", ""),
        description=g("description", ""),
        salary_min=g("salary_min"),
        salary_max=g("salary_max"),
        work_type=g("work_type"),
        location_mode=g("location_mode"),
        location_country=g("location_country"),
        location_raw=g("location_raw"),
        posted_at=g("posted_at"),
        hard_pass=bool(g("hard_pass", 0)),
        needs_verification=bool(g("needs_verification", 0)),
        score=g("score", 0.0),
        score_breakdown=json.loads(g("score_breakdown") or "{}"),
        benefits=json.loads(g("benefits") or "[]"),
        equity_signals=json.loads(g("equity_signals") or "[]"),
        matched_skills=json.loads(g("matched_skills") or "[]"),
        missing_skills=json.loads(g("missing_skills") or "[]"),
        fit_summaries=json.loads(g("fit_summaries") or "{}"),
        last_seen_run_id=g("last_seen_run_id"),
        last_seen_at=g("last_seen_at"),
        db_conn=conn,
    )


def rescore_all(conn: sqlite3.Connection, profile, hard, score_fn, now: Optional[datetime] = None) -> int:
    """Re-run scoring over every stored job against the CURRENT profile and
    persist the result. Returns the number of rows changed.

    `now`, if given, is passed through to `score_fn` so freshness bonuses
    (§ DECISIONS.md 2026-08-18) get recomputed too - a job that was fresh
    when first scored goes stale over time even with nothing else about it
    changing, which every OTHER field re-scored here does not do on its own.

    Exists because scoring output is stored at scrape time, so editing the
    profile leaves every existing row stale (operator hit this: skills they
    had just added were still listed as "missing" on 450 of 1123 jobs).
    `score_fn` is injected rather than imported so db.py keeps no dependency
    on scoring.py - the same direction of dependency the rest of the module
    maintains.

    Persists EVERY scoring column, and the change check compares every one of
    them. An earlier ad-hoc re-score updated only salary/hard_pass/score and
    silently left matched_skills/missing_skills frozen at their scrape-time
    values; a first version of THIS function then repeated the same mistake in
    its change check, skipping rows whose score_breakdown had changed but
    whose score hadn't - which left `anchor_matches` listing anchors the
    profile no longer had (found 2026-08-11 right after narrowing
    anchor_tools). Compare and write the whole set, not a subset.
    """
    changed = 0
    for row in conn.execute("SELECT * FROM jobs").fetchall():
        job = _row_to_job(conn, row)
        before = (
            job.hard_pass, job.needs_verification, job.score,
            list(job.matched_skills), list(job.missing_skills),
            list(job.benefits), list(job.equity_signals), dict(job.score_breakdown),
        )
        # Re-parse salary from the description rather than trusting a stale
        # stored value - parse_salary has been fixed several times
        # (§ DECISIONS.md 2026-08-02) and old rows carry its old mistakes.
        # But RESTORE the stored value if the text yields nothing: some
        # sources (mock, manual inserts) set salary_min/max structurally
        # without repeating it in the description, and blindly clearing it
        # would delete real data on every re-score.
        stored_min, stored_max = job.salary_min, job.salary_max
        job.salary_min = None
        job.salary_max = None
        score_fn(job, hard, profile, now=now)
        if job.salary_min is None and job.salary_max is None and (stored_min or stored_max):
            job.salary_min, job.salary_max = stored_min, stored_max
            score_fn(job, hard, profile, now=now)
        after = (
            job.hard_pass, job.needs_verification, job.score,
            list(job.matched_skills), list(job.missing_skills),
            list(job.benefits), list(job.equity_signals), dict(job.score_breakdown),
        )
        if before == after and row["salary_min"] == job.salary_min:
            continue
        changed += 1
        conn.execute(
            """
            UPDATE jobs SET salary_min = ?, salary_max = ?, hard_pass = ?,
                needs_verification = ?, score = ?, score_breakdown = ?,
                benefits = ?, equity_signals = ?, matched_skills = ?, missing_skills = ?
            WHERE id = ?
            """,
            (
                job.salary_min, job.salary_max, int(job.hard_pass),
                int(job.needs_verification), job.score, json.dumps(job.score_breakdown),
                json.dumps(job.benefits), json.dumps(job.equity_signals),
                json.dumps(job.matched_skills), json.dumps(job.missing_skills),
                row["id"],
            ),
        )
    conn.commit()
    return changed


def set_fit_summary(conn: sqlite3.Connection, job_id: int, provider_or_summary: str, summary: str | None = None) -> None:
    """Cache a generated fit summary.

    New callers pass a `provider` and `summary` (provider_or_summary, summary).
    Older code and tests called this with just a single `summary` positional
    argument (provider_or_summary) - treat that as a legacy write under the
    'gemini' provider for backward compatibility.
    """
    if summary is None:
        provider = "gemini"
        summary_text = provider_or_summary
    else:
        provider = provider_or_summary
        summary_text = summary

    if summary_text is None:
        return

    row = conn.execute("SELECT fit_summaries FROM jobs WHERE id = ?", (job_id,)).fetchone()
    summaries = json.loads(row["fit_summaries"]) if row and row["fit_summaries"] else {}
    summaries[provider] = summary_text
    conn.execute("UPDATE jobs SET fit_summaries = ? WHERE id = ?", (json.dumps(summaries), job_id))
    conn.commit()


def latest_successful_run_id(conn: sqlite3.Connection, source: str) -> Optional[int]:
    """The most recent search_runs row for `source` that actually completed
    without erroring - a source outage must not be mistaken for every one
    of its jobs having been delisted."""
    row = conn.execute(
        """
        SELECT id FROM search_runs
        WHERE source = ? AND finished_at IS NOT NULL AND error IS NULL
        ORDER BY started_at DESC LIMIT 1
        """,
        (source,),
    ).fetchone()
    return row["id"] if row else None


def hard_fail_reason_counts(conn: sqlite3.Connection, source: str) -> dict[str, int]:
    """Why jobs from `source`'s most recent search run failed the hard gate,
    grouped by reason prefix (e.g. "salary", "location_country") -> count.

    Used by agent.py's reflection step to decide how to adjust the search
    strategy: a title-mismatch-heavy run calls for different keywords, a
    salary-heavy run doesn't.
    """
    row = conn.execute(
        "SELECT id FROM search_runs WHERE source = ? ORDER BY started_at DESC LIMIT 1",
        (source,),
    ).fetchone()
    if row is None:
        return {}
    run_id = row["id"]
    rows = conn.execute(
        "SELECT score_breakdown FROM jobs WHERE source = ? AND last_seen_run_id = ? AND hard_pass = 0",
        (source, run_id),
    ).fetchall()
    counts: dict[str, int] = {}
    for r in rows:
        breakdown = json.loads(r["score_breakdown"])
        for reason in breakdown.get("hard_fail_reasons", []):
            key = reason.split("'")[0].strip() if "'" in reason else reason.split(" ")[0]
            counts[key] = counts.get(key, 0) + 1
    return counts


def role_salary_stats(
    conn: sqlite3.Connection, title_pattern: str, min_sane: int = 30000, floor: int = 100_000
) -> Optional[dict]:
    """Real disclosed-salary stats for jobs whose title matches
    `title_pattern` (case-insensitive regex, see role_fit.ROLE_TITLE_PATTERNS)
    - the Role Map tab's "what does this actually pay" column, computed from
    real postings already in this DB, not a guess.

    `min_sane` filters out parse artifacts (e.g. an hourly rate that slipped
    past scoring.parse_salary's own filters) that would otherwise skew a
    median toward an impossible number - same discipline as the salary
    regex bug this was built right after finding (§ DECISIONS.md 2026-08-02).

    `clear_rate` (fraction of disclosed salary_min >= `floor`) is what
    actually informed dropping data analyst/bi engineer from the paid
    LinkedIn search (§ DECISIONS.md 2026-08-02) and what agent.py's
    reflection step sees - the median can look fine while the clear rate is
    near zero, which is the number that actually predicts hard_pass odds.

    Returns None if no disclosed-salary posting matches (not zeros - a role
    with zero data should read as "no data", not "pays $0").
    """
    pattern = re.compile(title_pattern, re.IGNORECASE)
    rows = conn.execute("SELECT title, salary_min, salary_max FROM jobs").fetchall()
    los, his = [], []
    for r in rows:
        if not pattern.search(r["title"] or ""):
            continue
        if r["salary_min"] and r["salary_min"] >= min_sane:
            los.append(r["salary_min"])
        if r["salary_max"] and r["salary_max"] >= min_sane:
            his.append(r["salary_max"])
    if not los:
        return None
    return {
        "n": len(los),
        "min": min(los),
        "max": max(his) if his else max(los),
        "median_min": int(statistics.median(los)),
        "median_max": int(statistics.median(his)) if his else int(statistics.median(los)),
        "clear_rate": sum(1 for s in los if s >= floor) / len(los),
    }


def tiered_salary_stats(
    conn: sqlite3.Connection, patterns: dict[str, str], min_sane: int = 30000, floor: int = 100_000
) -> dict[str, dict]:
    """Salary stats per SENIORITY TIER, with each job assigned to exactly ONE
    tier - the first pattern in `patterns` that matches its title (see
    role_fit.SENIORITY_TIER_PATTERNS, which is ordered most-senior first).

    Exclusive assignment is the whole point: role_salary_stats() applied
    per-tier would count "Staff Analytics Engineer" in BOTH the Staff bucket
    and the untiered "analytics engineer" bucket, which is exactly the
    conflation that produced a misleading aggregate (§ DECISIONS.md
    2026-08-02, corrected entry).

    Reports two rates, because they answer different questions:
      - `clear_rate`   - fraction whose salary_MIN meets the floor. Matches
                         what hard_filter actually gates on (§ 2026-07-14:
                         the low end is the conservative verifiable number).
      - `band_reach_rate` - fraction whose salary_MAX reaches the floor, i.e.
                         "could this role pay above the floor at all". A
                         posting whose min falls just short of the floor but
                         whose max clears it is plainly a real opportunity;
                         judging a keyword's worth on min alone systematically
                         understates it.
    """
    compiled = [(tier, re.compile(pat, re.IGNORECASE)) for tier, pat in patterns.items()]
    buckets: dict[str, list[tuple[int, Optional[int]]]] = {tier: [] for tier in patterns}
    for r in conn.execute("SELECT title, salary_min, salary_max FROM jobs").fetchall():
        title = r["title"] or ""
        if not r["salary_min"] or r["salary_min"] < min_sane:
            continue
        for tier, pattern in compiled:
            if pattern.search(title):
                buckets[tier].append((r["salary_min"], r["salary_max"]))
                break  # first (most senior) match wins - no double-counting
    out: dict[str, dict] = {}
    for tier, pairs in buckets.items():
        if not pairs:
            continue
        los = [lo for lo, _ in pairs]
        his = [hi for _, hi in pairs if hi]
        out[tier] = {
            "n": len(los),
            "median_min": int(statistics.median(los)),
            "median_max": int(statistics.median(his)) if his else None,
            "clear_rate": round(sum(1 for lo in los if lo >= floor) / len(los), 3),
            "band_reach_rate": round(
                sum(1 for _, hi in pairs if hi and hi >= floor) / len(pairs), 3
            ),
        }
    return out


# How long a listing from a NON-enumerating source (LinkedIn's capped search)
# may go unconfirmed before it stops being presented as open. Deliberately
# generous: a live job can miss many runs purely by ranking below the cap
# (that false-positive is what the 2026-08-02 Higharc fix was about), so this
# is a backstop against indefinitely-stale rows, not a delisting detector.
# Postings that really are gone tend to stay gone, so 10 days of silence is
# meaningful where 1-2 runs of silence is not.
_STALE_AFTER_DAYS = 10


def _is_stale(last_seen_at: Optional[str]) -> bool:
    """True if this listing hasn't been re-confirmed in _STALE_AFTER_DAYS.
    Unknown/unparseable timestamps are treated as NOT stale - the same
    "don't assert what the data can't support" rule used throughout."""
    if not last_seen_at:
        return False
    try:
        seen = datetime.fromisoformat(last_seen_at)
    except ValueError:
        return False
    if seen.tzinfo is None:
        seen = seen.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - seen).days >= _STALE_AFTER_DAYS


def annotate_active(conn: sqlite3.Connection, jobs: list[Job]) -> list[Job]:
    """A job is active if it was returned by its source's most recent
    successful search run - i.e. still listed there, not merely "not yet
    proven gone." No successful run yet for that source means we have no
    evidence either way, so it defaults to active rather than being hidden.

    CRITICAL (§ DECISIONS.md 2026-08-02): this reasoning only holds for a
    source that can enumerate every match. A source returning a ranked,
    capped top-N (LinkedIn via Apify, `maxJobs` per keyword) gives absence NO
    meaning - a still-open job that merely ranked below the cap this run is
    indistinguishable from one that was taken down. Verified live: a real
    Higharc "Analytics Engineer" posting (score 262) was labelled "no longer
    accepting" purely because a 15/15-truncated result set didn't include it;
    38% of all stored LinkedIn jobs were mislabelled the same way. So for
    such sources every job stays active, and the delisted question is simply
    left unanswered rather than answered wrongly - same discipline as the
    Zoox fix (2026-07-29): never assert a status the data can't support.
    """
    latest_by_source: dict[str, Optional[int]] = {}
    enumerates_by_source: dict[str, bool] = {}
    keywords_by_source: dict[str, list[str]] = {}
    for job in jobs:
        if job.source not in latest_by_source:
            latest_by_source[job.source] = latest_successful_run_id(conn, job.source)
            enumerates_by_source[job.source] = source_enumerates_all(conn, job.source)
            keywords_by_source[job.source] = latest_run_keywords(conn, job.source)
        if not enumerates_by_source[job.source]:
            # Absence from ONE truncated run proves nothing - but "never mark
            # inactive" was too absolute (operator hit this: two Netflix
            # listings last confirmed 11 days ago, since expired, still shown
            # as top "Open" matches). Repeated absence across many days IS
            # evidence, even from a capped source, so fall back to staleness.
            job.is_active = not _is_stale(job.last_seen_at)
            continue
        # Second way absence can be meaningless (also found 2026-08-02): the
        # latest run may not have LOOKED for this job. Greenhouse fetches whole
        # boards but filters client-side by keyword, so when the keyword list
        # changes, previously-found jobs silently drop out - 175 of 203 flagged
        # "no longer accepting" were really just out of the current search's
        # scope (e.g. "Machine Learning Engineer" found during an agent-widened
        # run). Not searched for != taken down.
        kws = keywords_by_source[job.source]
        if kws and not any(k.lower() in (job.title or "").lower() for k in kws):
            job.is_active = True
            continue
        latest = latest_by_source[job.source]
        job.is_active = latest is None or job.last_seen_run_id == latest
    return jobs


def latest_run_keywords(conn: sqlite3.Connection, source: str) -> list[str]:
    """Keywords the most recent successful run for `source` actually searched
    (search_runs.query is the comma-joined keyword list). Used by
    annotate_active to avoid calling a job delisted when the latest run never
    looked for it. Empty list = unknown, which disables the check."""
    row = conn.execute(
        """
        SELECT query FROM search_runs
        WHERE source = ? AND finished_at IS NOT NULL AND error IS NULL
        ORDER BY started_at DESC LIMIT 1
        """,
        (source,),
    ).fetchone()
    if row is None or not row["query"]:
        return []
    return [k.strip() for k in row["query"].split(",") if k.strip()]


def source_enumerates_all(conn: sqlite3.Connection, source: str) -> bool:
    """Whether `source`'s most recent successful run returned the FULL set of
    matching jobs (see sources/__init__.py's `enumerates_all_matches`).
    Defaults to True when unknown - a source that has never recorded the flag
    keeps the original behaviour rather than silently changing it."""
    row = conn.execute(
        """
        SELECT enumerates_all FROM search_runs
        WHERE source = ? AND finished_at IS NOT NULL AND error IS NULL
        ORDER BY started_at DESC LIMIT 1
        """,
        (source,),
    ).fetchone()
    if row is None or row["enumerates_all"] is None:
        return True
    return bool(row["enumerates_all"])


def get_job(conn: sqlite3.Connection, job_id: int) -> Optional[Job]:
    row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None:
        return None
    job = _row_to_job(conn, row)
    annotate_active(conn, [job])
    return job


def list_top_jobs(
    conn: sqlite3.Connection,
    limit: int = 10,
    only_passing: bool = True,
    only_active: bool = True,
) -> list[Job]:
    query = "SELECT * FROM jobs"
    if only_passing:
        # needs_verification jobs are ALSO returned here (not just hard_pass) -
        # § DECISIONS.md 2026-07-30 "needs verification" bucket: a job failing
        # only on unconfirmed fields should still be visible, just routed to
        # its own tab by the caller (app.py), not hidden the same way a
        # confirmed-bad job (hard_pass=0, needs_verification=0) is.
        query += " WHERE hard_pass = 1 OR needs_verification = 1"
    query += " ORDER BY score DESC"
    rows = conn.execute(query).fetchall()
    jobs = annotate_active(conn, [_row_to_job(conn, r) for r in rows])
    if only_active:
        jobs = [j for j in jobs if j.is_active]
    return jobs[:limit]


def start_search_run(
    conn: sqlite3.Connection, source: str, query: str, now_iso: str, enumerates_all: bool = True
) -> int:
    """`enumerates_all` records whether this source can see every matching job
    or only a capped slice - annotate_active refuses to mark jobs delisted
    for a source that can't (§ DECISIONS.md 2026-08-02)."""
    cur = conn.execute(
        "INSERT INTO search_runs (started_at, source, query, enumerates_all) VALUES (?, ?, ?, ?)",
        (now_iso, source, query, int(enumerates_all)),
    )
    conn.commit()
    return cur.lastrowid


def finish_search_run(
    conn: sqlite3.Connection,
    run_id: int,
    now_iso: str,
    n_found: int,
    n_new: int,
    error: Optional[str] = None,
) -> None:
    conn.execute(
        "UPDATE search_runs SET finished_at = ?, n_found = ?, n_new = ?, error = ? WHERE id = ?",
        (now_iso, n_found, n_new, error, run_id),
    )
    conn.commit()


def save_role_fit_snapshot(conn: sqlite3.Connection, now_iso: str, results: dict) -> int:
    """Persist one Role Map computation (§ DECISIONS.md 2026-08-02) - `results`
    is a plain dict, already JSON-serializable (role_fit.py doesn't produce
    one itself since it stays a pure comparison function, no I/O)."""
    cur = conn.execute(
        "INSERT INTO role_fit_snapshots (computed_at, results_json) VALUES (?, ?)",
        (now_iso, json.dumps(results)),
    )
    conn.commit()
    return cur.lastrowid


def get_latest_role_fit_snapshot(conn: sqlite3.Connection) -> Optional[dict]:
    """Most recently saved Role Map snapshot, or None if "Rerun & save" has
    never been clicked - `computed_at` + `results` (already-decoded dict)."""
    row = conn.execute(
        "SELECT computed_at, results_json FROM role_fit_snapshots ORDER BY computed_at DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return None
    return {"computed_at": row["computed_at"], "results": json.loads(row["results_json"])}
