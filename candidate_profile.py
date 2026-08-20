"""The candidate profile jobs are matched against.

This is a soft signal like benefits (§ scoring.py) - a job with zero skill
overlap still passes the hard gates and gets ranked, it just scores lower.
Skill/title matching is the same keyword-spotting approach as benefit
detection: no ML, just checked against the JD text.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateProfile:
    # Narrowed 2026-08-12 to the two target families only (§ DECISIONS.md).
    # `data analyst` / `bi engineer` / `business intelligence engineer` are
    # gone: measured 0% clear rate against the floor across n=38
    # disclosed-salary postings, so they cost paid searches and produced
    # nothing.
    #
    # NO separate seniority gate despite the "senior+" framing - the floor
    # already does that work (87% of postings clearing it are senior-titled
    # vs 61% below it), and a keyword-based one would drop non-standard
    # senior naming like Netflix's "Analytics Engineer 5". The bare
    # "analytics engineer" / "ai engineer" entries are kept deliberately:
    # Greenhouse filters titles by SUBSTRING, so they subsume every tier and
    # catch those non-standard names, while the salary floor screens out the
    # genuinely junior ones.
    title_keywords: tuple[str, ...] = (
        "analytics engineer",
        "senior analytics engineer",
        "staff analytics engineer",
        "principal analytics engineer",
        "lead analytics engineer",
        "ai engineer",
        "senior ai engineer",
        "staff ai engineer",
        "principal ai engineer",
        "lead ai engineer",
        # Kept although it's a third family: measured the STRONGEST tier of
        # all (33% clear rate, 100% band-reach) - dropping the best performer
        # to tidy the taxonomy would be a regression.
        "ai product engineer",
    )
    skills: tuple[str, ...] = (
        "dbt",
        "lookml",
        "looker",
        "bi",
        "sql",
        "snowflake",
        "redshift",
        "bigquery",
        "postgresql",
        "python",
        "semantic layer",
        "experimentation",
        "llm",
        "prompt engineering",
        "ai agents",
        "rag",
        "gpt",
        "product roadmap",
        # Parsed from the operator's real LinkedIn skills export (2026-08-02) -
        # noise filtered out (endorsement counts, "N experiences at...", course/
        # cert names, generic soft skills like "Communication"/"Leadership" too
        # broad to be a useful JD-match signal, outdated hobby-project tech
        # like PHP/PPC/Web Design no longer part of the target roles).
        "a/b testing",
        "vector database",
        "langchain",
        "langgraph",
        "mcp",
        "agentic ai",
        "openai api",
        "data engineering",
        "analytics engineering",
        "data modeling",
        "dimensional modeling",
        "etl",
        "streamlit",
        "cursor",
        "data quality",
        "business intelligence",
        "data analysis",
        "stakeholder management",
        "financial modeling",
        "risk management",
        "financial risk",
    )
    summary: str = (
        "Senior Analytics Engineer working across dbt, LookML, Redshift, and "
        "Snowflake, with 15+ years at the intersection of data modeling and "
        "practical product thinking. Builds semantic layers, experimentation "
        "frameworks, and AI-powered tooling (production Slack agent, custom "
        "GPT workflows) that Product, Marketing, and GTM teams actually use "
        "daily. Master's in Mathematics, Ph.D. in Finance. 3+ years at "
        "Recharge (subscription management SaaS), 1+ year at Bolt Technology, "
        "6+ years prior in banking risk and credit modeling. "
        "Stack: dbt, LookML, SQL, BigQuery, Snowflake, PostgreSQL, Python, Cursor."
    )
    # Free-text, not keyword-matched by scoring.py (which stays pure/regex-only
    # per its own module contract) - these feed fit_summary.py's LLM prompt so
    # the qualitative "why this fits" narrative has real specifics to draw on,
    # instead of just the one-paragraph summary above.
    work_experience: str = ""
    personal_projects: str = ""
    # Degrees/institutions, free text like work_experience - some JDs state a
    # minimum degree requirement, and the LLM fit-summary can reason about
    # whether it's met far better than a regex parser guessing at every
    # phrasing ("Bachelor's required", "advanced degree preferred", "PhD in a
    # quantitative field") ever could - not a scoring.py hard gate.
    education: str = (
        "Higher School of Economics - Doctor of Philosophy (Ph.D.), Economics, "
        "2004-2008. Thesis: «Formation of rating for Russian banks».\n"
        "Lomonosov Moscow State University (MSU) - Master's degree, Applied "
        "Mathematics, 1999-2004."
    )
    # § DECISIONS.md 2026-08-01: the profile-driven fix for the "generic AI
    # keyword overlap hard-passes a genuinely different, deeper-expertise role"
    # false positive (Reddit "Machine Learning Engineer" JDs matching on bare
    # "python"/"llm" with none of the operator's actual differentiators).
    # scoring.hard_filter requires at least one of anchor_tools + anchor_skills
    # to appear in the JD text (in addition to the min_matched_skills count) -
    # operator-curated, not a project-level guess, because only the operator
    # can say what's genuinely differentiating vs. generic-sounding-but-shared
    # vocabulary for THEIR specific background.
    # `python`/`sql`/`api` deliberately NOT here (dropped 2026-08-11 with
    # operator approval): they appear in essentially every technical JD, so
    # they can't discriminate - which is the entire job of an anchor. Keeping
    # them let Machine Learning Engineer roles hard_pass on
    # anchor_matches=['python','api'] alone. Measured: dropping them took the
    # Open tab from 13 jobs (12 of them Reddit ML/DS) to 4.
    anchor_tools: tuple[str, ...] = (
        "dbt", "looker", "snowflake", "lookml", "cursor", "claude", "openai", "mcp",
    )
    anchor_skills: tuple[str, ...] = (
        "analytics", "dimensional modeling", "ab-testing", "dashboards",
        "storytelling", "performance metrics", "business impact",
        "impact analytics", "product analytics", "gtm",
    )


DEFAULT_PROFILE = CandidateProfile()

# Where the Streamlit page's edits are persisted. Deliberately NOT read at
# import time (load_profile() is an explicit call) - importing this module
# must stay pure/side-effect-free so tests and other modules can rely on
# DEFAULT_PROFILE without triggering disk I/O.
PROFILE_PATH = os.getenv("GMMH_PROFILE_PATH", "candidate_profile.json")


def load_profile(path: str = PROFILE_PATH) -> CandidateProfile:
    """The saved profile if one exists on disk, else the hardcoded default."""
    if not os.path.exists(path):
        return DEFAULT_PROFILE
    with open(path) as f:
        data = json.load(f)
    return CandidateProfile(
        title_keywords=tuple(data.get("title_keywords", DEFAULT_PROFILE.title_keywords)),
        skills=tuple(data.get("skills", DEFAULT_PROFILE.skills)),
        summary=data.get("summary", DEFAULT_PROFILE.summary),
        work_experience=data.get("work_experience", DEFAULT_PROFILE.work_experience),
        personal_projects=data.get("personal_projects", DEFAULT_PROFILE.personal_projects),
        education=data.get("education", DEFAULT_PROFILE.education),
        anchor_tools=tuple(data.get("anchor_tools", DEFAULT_PROFILE.anchor_tools)),
        anchor_skills=tuple(data.get("anchor_skills", DEFAULT_PROFILE.anchor_skills)),
    )


def save_profile(profile: CandidateProfile, path: str = PROFILE_PATH) -> None:
    with open(path, "w") as f:
        json.dump(
            {
                "title_keywords": list(profile.title_keywords),
                "skills": list(profile.skills),
                "summary": profile.summary,
                "work_experience": profile.work_experience,
                "personal_projects": profile.personal_projects,
                "education": profile.education,
                "anchor_tools": list(profile.anchor_tools),
                "anchor_skills": list(profile.anchor_skills),
            },
            f,
            indent=2,
        )
