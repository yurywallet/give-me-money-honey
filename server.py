"""MCP server: periodic job search with hard-gate filtering and scoring.

Tools exposed:
  - search_jobs_now(keywords?)        run one search pass immediately
  - start_periodic_search(interval?)  start the background poll loop
  - stop_periodic_search()            stop it
  - get_top_jobs(limit, only_passing, only_active) ranked results, active-only by default
  - get_job(job_id)                   full detail incl. description + score breakdown
  - get_criteria()                    current hard/soft criteria in effect

Run: `python server.py` (stdio transport - point your MCP client at this
file). Uses MockJobSource ONLY when APIFY_TOKEN/APIFY_ACTOR_ID aren't set (zero-
setup verification path); set them in .env to search real LinkedIn listings
(see sources/linkedin_apify_source.py for what that needs). Mock is never
ADDITIVE alongside the real source (§ DECISIONS.md 2026-07-28) - it used to
run unconditionally every search, continuously writing fake example.com jobs
into the same real jobs.db even with real credentials configured.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

from mcp.server.fastmcp import FastMCP

from candidate_profile import load_profile
from config import BENEFIT_WEIGHTS, EQUITY_WEIGHTS, STAGE_WEIGHTS, SearchConfig, load_scoring_overrides
from db import get_conn, get_job as db_get_job, init_db, list_top_jobs, set_fit_summary
from fit_summary import configured_providers, default_provider, generate_with
from scheduler import Scheduler
from sources.linkedin_selector import build_linkedin_source
from sources.mock_source import MockJobSource

mcp = FastMCP("give-me-money-honey")

# Loaded once at startup - reflects whatever was last saved via the
# Streamlit page (app.py). Restart the server to pick up a later edit.
# Same for the operator-edited role-family / excluded-industry lists: without
# this call the server would score against config.py's in-code defaults while
# the app scored against the saved ones.
load_scoring_overrides()
_profile = load_profile()
_config = SearchConfig(keywords=_profile.title_keywords)
_conn = get_conn(_config.db_path)
init_db(_conn)

_sources = []
_linkedin = build_linkedin_source()  # remote-filtered or legacy, per GMMH_LINKEDIN_ACTOR
if _linkedin is not None:
    _sources.append(_linkedin)
if os.getenv("GREENHOUSE_COMPANIES"):
    from sources.greenhouse_source import GreenhouseSource

    _sources.append(GreenhouseSource())
if not _sources:
    # Mock is never additive alongside a real source (§ DECISIONS.md
    # 2026-07-28) - generalized here to "no real source configured at all",
    # now that there are two possible real sources, not just one.
    _sources = [MockJobSource()]

_scheduler = Scheduler(_config, _sources, _conn, profile=_profile)


def _job_summary(job) -> dict:
    return {
        "id": job.id,
        "source": job.source,
        "title": job.title,
        "company": job.company,
        "url": job.url,
        "salary_min": job.salary_min,
        "salary_max": job.salary_max,
        "location_raw": job.location_raw,
        "hard_pass": job.hard_pass,
        "score": job.score,
        "equity_signals": job.equity_signals,
        "benefits": job.benefits,
        "matched_skills": job.matched_skills,
        "is_active": job.is_active,
    }


@mcp.tool()
def search_jobs_now(keywords: list[str] | None = None) -> dict:
    """Run one job search pass immediately across all configured sources."""
    original_keywords = _config.keywords
    if keywords:
        _config.keywords = tuple(keywords)
    try:
        return _scheduler.run_once()
    finally:
        _config.keywords = original_keywords


@mcp.tool()
def start_periodic_search(interval_minutes: int | None = None) -> str:
    """Start polling all sources on a background interval (default: config's poll_interval_minutes)."""
    if interval_minutes:
        _config.poll_interval_minutes = interval_minutes
    started = _scheduler.start()
    if not started:
        return "already running"
    return f"started, polling every {_config.poll_interval_minutes} minutes"


@mcp.tool()
def stop_periodic_search() -> str:
    """Stop the background poll loop, if running."""
    stopped = _scheduler.stop()
    return "stopped" if stopped else "was not running"


@mcp.tool()
def get_top_jobs(limit: int = 10, only_passing: bool = True, only_active: bool = True) -> list[dict]:
    """Return the highest-scored jobs found so far, ranked descending.

    only_active=True (default) excludes jobs that dropped out of their
    source's most recent successful search - i.e. likely delisted/filled.
    """
    jobs = list_top_jobs(_conn, limit=limit, only_passing=only_passing, only_active=only_active)
    return [_job_summary(j) for j in jobs]


@mcp.tool()
def get_job(job_id: int) -> dict:
    """Full detail for one job, including the description and score breakdown."""
    job = db_get_job(_conn, job_id)
    if job is None:
        return {"error": f"no job with id {job_id}"}
    detail = _job_summary(job)
    detail["description"] = job.description
    detail["score_breakdown"] = job.score_breakdown
    detail["fit_summaries"] = job.fit_summaries  # {} if never generated - see get_fit_summary
    return detail


@mcp.tool()
def get_fit_summary(job_id: int, provider: str | None = None, force_regenerate: bool = False) -> dict:
    """Qualitative fit explanation for one job: what the role needs, how the
    candidate's experience translates to it, and any real gaps - beyond the
    keyword-based score alone. `provider` picks a specific one ('gemini',
    'claude', 'ollama'); omit it to use whichever default_provider() resolves
    to. Generated once per provider and cached; set force_regenerate=True to
    redo it (e.g. after editing the profile)."""
    job = db_get_job(_conn, job_id)
    if job is None:
        return {"error": f"no job with id {job_id}"}
    resolved_provider = provider or default_provider()
    cached = (job.fit_summaries or {}).get(resolved_provider)
    if not force_regenerate and cached:
        return {"job_id": job_id, "provider": resolved_provider, "fit_summary": cached, "cached": True}
    summary = generate_with(resolved_provider, job, _profile)
    set_fit_summary(_conn, job_id, resolved_provider, summary)
    return {"job_id": job_id, "provider": resolved_provider, "fit_summary": summary, "cached": False}


@mcp.tool()
def list_llm_providers() -> dict:
    """Which LLM providers are configured (env key/model set) and can be
    passed as get_fit_summary's `provider` argument."""
    return {"configured": configured_providers()}


@mcp.tool()
def get_profile() -> dict:
    """The candidate profile jobs are matched against (title keywords + skills)."""
    return {
        "title_keywords": list(_profile.title_keywords),
        "skills": list(_profile.skills),
        "summary": _profile.summary,
    }


@mcp.tool()
def get_criteria() -> dict:
    """Current hard gates and soft scoring weights in effect."""
    return {
        "hard": {
            "min_salary": _config.hard.min_salary,
            "work_type": _config.hard.work_type,
            "location_mode": _config.hard.location_mode,
            "location_country": _config.hard.location_country,
            "min_matched_skills": _config.hard.min_matched_skills,
        },
        "soft_benefit_weights": BENEFIT_WEIGHTS,
        "equity_weights": EQUITY_WEIGHTS,
        "stage_weights": STAGE_WEIGHTS,
        "poll_interval_minutes": _config.poll_interval_minutes,
        "keywords": list(_config.keywords),
    }


if __name__ == "__main__":
    mcp.run()
