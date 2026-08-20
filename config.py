"""Search criteria and scoring configuration.

Hard criteria are pass/fail gates - a job that fails any of them is excluded
entirely, never merely down-scored. Soft criteria only ever add to score.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from candidate_profile import DEFAULT_PROFILE


@dataclass(frozen=True)
class HardCriteria:
    # Generic placeholder default - set your own real floor via GMMH_MIN_SALARY
    # in .env, never commit your actual target.
    min_salary: int = int(os.getenv("GMMH_MIN_SALARY", "100000"))
    work_type: str = "fulltime"          # must match Job.work_type exactly
    location_mode: str = "remote"        # must match Job.location_mode exactly
    location_country: str = "USA"        # must match Job.location_country exactly
    # Role-relevance gate (§ DECISIONS.md 2026-07-25) - replaces the old exact
    # title-keyword hard gate. A job needs at least this many of the
    # profile's skills mentioned in its own text to hard_pass; title itself
    # is scoring-only now (scoring.PROFILE_TITLE_MATCH_BONUS).
    min_matched_skills: int = 2
    # In addition to the count above, at least one of profile.anchor_tools +
    # profile.anchor_skills must appear in the JD text (§ DECISIONS.md
    # 2026-08-01). Closes a false-positive the plain count missed: "Senior
    # Machine Learning Engineer" JDs matched on 2 purely generic AI-adjacent
    # terms (python + llm/gpt) with zero of the operator's actual
    # differentiators and hard_passed - a genuinely different, deeper-expertise
    # role (real ML/model-training depth, months to reach and still not
    # production-level per the operator), same failure class DECISIONS.md's
    # 2026-07-25 entry named for "analytics engineer vs. backend/frontend
    # engineer". Deliberately profile data, not a project-level constant -
    # only the operator can say what's genuinely differentiating vs.
    # generic-sounding-but-shared vocabulary for THEIR background (a first
    # attempt hardcoded here excluded a genuine AI Engineer match at Socure
    # for the same reason it excluded the false-positive ML Engineer roles).
    # require_anchor_skill=False lets a profile with no anchors set fall back
    # to the plain count instead of hard-gating everything out.
    require_anchor_skill: bool = True
    # Title must name one of the target ROLE FAMILIES (§ DECISIONS.md
    # 2026-08-12, operator directive: "anchor to ai engineer or analytics
    # engineer roles"). Measured need: after relaxing the location gate, Open
    # held 47 jobs of which only 7 were in those families - the rest were
    # Machine Learning Engineer (11), Data Scientist (10), Data Engineer (5),
    # Software Engineer (3). Narrowing title_keywords only affects FUTURE
    # searches, so it could not clean what was already stored.
    #
    # This re-introduces a title gate, which § 2026-07-25 deliberately
    # removed - but not the same mechanism. That one required an EXACT match
    # against a list of full titles and failed because "it can't know every
    # title a company might invent". This is a SUBSTRING family match, so
    # "Analytics Engineer 5", "Senior Analytics Engineer, Ads" and
    # "Principal AI Engineer" all match on their own. The 2026-07-25
    # objection was to exact-title brittleness, not to role scoping as such.
    require_target_role_family: bool = True


# Substring patterns for the target role families (HardCriteria.
# require_target_role_family). Matched case-insensitively against the job
# TITLE. Deliberately loose enough to catch seniority prefixes and suffixed
# variants without enumerating them.
TARGET_ROLE_FAMILY_PATTERNS: tuple[str, ...] = (
    r"analytics engineer",
    r"\bai engineer\b",
    r"ai product engineer",
    # "Data & AI Engineer", "AI/ML Engineer" style compounds that are still
    # the AI-engineering family rather than a research/ML-training role.
    r"\bai\s*/?\s*ml engineer\b",
    r"data\s*&\s*ai engineer",
)


# Soft benefit keyword -> score weight. "family_medical" is the one soft
# criterion the operator named explicitly, so it outweighs the rest.
# NOTE: "equity" deliberately lives in EQUITY_WEIGHTS below, not here - it's no
# longer treated as just another perk (§ DECISIONS.md 2026-07-25, equity pivot).
# Operator directive 2026-08-19, in this exact order: family_medical, then
# dental, then unlimited PTO - all three above the 401k match. Day-to-day
# usable benefits over a deferred one, same reasoning as the equity reversal:
# certain now beats contingent later.
BENEFIT_WEIGHTS: dict[str, float] = {
    "family_medical": 25.0,
    "dental": 12.0,
    "vision": 5.0,
    "401k_match": 8.0,
    "unlimited_pto": 10.0,
    "parental_leave": 6.0,
    "remote_stipend": 4.0,
    "wellness": 3.0,
}

# Equity/stage signals - now a TIEBREAKER, not the thesis (§ DECISIONS.md
# 2026-08-12, reversing the 2026-07-25 equity pivot). Operator: "in a lot of
# cases equity stays paper money that is never real money, so I do not want
# to have a low paid job for a paper money future." So equity still ranks a
# job UP among otherwise-comparable ones, but can no longer outweigh base
# salary (EQUITY_SCORE_CAP < SALARY_BONUS_CAP) and can no longer buy a job
# past the salary floor (see EQUITY_SALARY_FLOOR below). Weights are scaled
# down rather than deleted - a real % of ownership is still worth something,
# just not more than cash. Detected in scoring.parse_equity_signals.
EQUITY_WEIGHTS: dict[str, float] = {
    "equity": 4.0,             # any equity/options/RSU mention - near-universal, so near-worthless as a signal
    "meaningful_equity": 8.0,  # "meaningful/significant/substantial equity"
    "equity_pct": 14.0,        # an explicit % of ownership named - the only genuinely informative one
    "founding": 10.0,          # founding engineer / early employee
}
STAGE_WEIGHTS: dict[str, float] = {
    "early_stage": 4.0,
    "seed": 3.0,
    "series_a": 6.0,
    "series_b": 8.0,
    "series_c_plus": 8.0,      # later stage = likelier the paper becomes cash
    "pre_ipo": 10.0,           # the one stage where equity most reliably liquidates
    "well_funded": 4.0,
}
EQUITY_SCORE_CAP = 30.0        # deliberately << SALARY_BONUS_CAP now

# NO equity-relaxed salary floor any more (§ DECISIONS.md 2026-08-12). This
# used to be 160_000: a job showing genuine startup upside could clear a
# LOWER bar than HardCriteria.min_salary, which is precisely the "low paid
# job for a paper money future" the operator has now ruled out. Every job is
# held to min_salary regardless of its equity story. Kept as a named constant
# set to None (rather than deleted) so scoring.py's call site stays explicit
# about the fact that there is no relaxation, instead of silently dropping a
# concept the log still discusses.
EQUITY_SALARY_FLOOR = None

# Points per $1,000 of salary above min_salary, capped so one outlier salary
# can't dwarf every benefit signal combined.
SALARY_BONUS_PER_1K = 1.0
SALARY_BONUS_CAP = 100.0

BASE_SCORE_ON_PASS = 100.0

# Freshness bonus (operator ask 2026-08-18, during active search: an
# already-old above-floor posting has usually built a real applicant pool,
# so recency is worth surfacing and worth a small nudge in ranking). Purely
# additive - see scoring.freshness_bonus for why tiers, not a decay curve.
# (max_age_days, bonus), checked in order, first match wins. Modest relative
# to salary/equity on purpose: freshness is a "worth noticing" signal, not a
# fit signal - it shouldn't outrank a stronger but slightly older match.
FRESHNESS_BONUS_TIERS: tuple[tuple[int, float], ...] = (
    (3, 15.0),   # posted within 3 days
    (7, 10.0),   # within a week
    (14, 5.0),   # within two weeks
    (30, 2.0),   # within a month
)

# Profile-match bonus (§ profile.py, scoring.match_profile_skills): points per
# matched skill keyword found in the JD, capped, plus a flat bonus if the
# title itself matches one of the profile's target titles.
PROFILE_SKILL_WEIGHT = 4.0
PROFILE_SKILL_CAP = 40.0
PROFILE_TITLE_MATCH_BONUS = 15.0

# Analytics/data-engineering skill vocabulary for the missing-skills gap
# (scoring.missing_skills, operator ask 2026-07-27). A skill a JD mentions that
# is in this universe but NOT in the candidate profile is a "skill to add" -
# one that would start counting toward the match score the moment it's added to
# the profile (each matched skill = PROFILE_SKILL_WEIGHT points, up to the cap).
# Deliberately BROADER than the profile's own skills - surfacing gaps is the point.
SKILL_UNIVERSE: tuple[str, ...] = (
    # warehouses / storage
    "snowflake", "redshift", "bigquery", "databricks", "postgresql", "mysql", "duckdb", "clickhouse",
    # transformation / orchestration / integration
    "dbt", "airflow", "dagster", "prefect", "spark", "fivetran", "stitch", "airbyte", "census", "hightouch",
    # BI / semantic layer
    "looker", "lookml", "tableau", "power bi", "mode", "sigma", "metabase", "hex", "superset",
    "semantic layer", "cube",
    # languages / libs
    "sql", "python", "r", "scala", "java", "pandas", "numpy", "scikit-learn",
    # analytics / stats / ml / ai
    "experimentation", "a/b testing", "statistics", "machine learning", "forecasting", "data science",
    "llm", "rag", "gpt", "langchain", "langgraph", "vector database", "prompt engineering", "ai agents",
    "mcp", "agentic ai", "openai api", "chatgpt", "cursor ai",
    # cloud / infra / eng
    "aws", "gcp", "azure", "docker", "kubernetes", "terraform", "git", "ci/cd", "kafka", "api", "graphql", "saas",
    # data concepts
    "data modeling", "dimensional modeling", "etl", "elt", "reverse etl", "data warehouse", "data lake",
    "data quality", "great expectations", "data governance", "business intelligence", "e-commerce",
    # finance/quant domain (operator's Ph.D. Finance differentiator - tailored
    # to this single-user tool, not a generic vocabulary; see profile.py)
    "financial modeling", "risk management",
)

# Industries to HARD-exclude regardless of fit (operator directive 2026-07-27).
# A job whose title/company/description matches any of these is rejected like a
# salary/location miss (scoring.excluded_industry -> hard_filter). Deliberately
# industry-SPECIFIC phrases, NOT bare "health"/"medical" - those match benefits
# text ("medical insurance for your family"), i.e. the perk, not the employer's
# industry. Extend this tuple to exclude other industries.
EXCLUDE_INDUSTRY_KEYWORDS: tuple[str, ...] = (
    "hospital", "clinical", "clinician", "pharma", "pharmaceutical", "biotech",
    "biopharma", "medical device", "life sciences", "health system",
    "healthcare provider", "healthcare company", "healthcare data", "telehealth",
    "telemedicine", "digital health", "healthtech", "health tech", "clinical trial",
    "oncology", "medicaid", "medicare", "value-based care", "patient care",
    "electronic health record",
)


@dataclass
class SearchConfig:
    """Mutable: keywords/poll_interval are meant to be adjusted at runtime
    via MCP tools (search_jobs_now, start_periodic_search). `hard` is left
    frozen - changing salary/location/work-type gates is a deliberate config
    edit, not a per-call override."""

    keywords: tuple[str, ...] = field(default_factory=lambda: DEFAULT_PROFILE.title_keywords)
    hard: HardCriteria = field(default_factory=HardCriteria)
    poll_interval_minutes: int = int(os.getenv("GMMH_POLL_INTERVAL_MINUTES", "60"))
    db_path: str = os.getenv("GMMH_DB_PATH", "jobs.db")


# --- operator-editable lists (§ DECISIONS.md 2026-08-19) --------------------
# TARGET_ROLE_FAMILY_PATTERNS and EXCLUDE_INDUSTRY_KEYWORDS are the two
# scoring lists the operator adjusts while actually searching ("that industry
# too", "that title variant"), so they're editable from the Scoring tab and
# persisted here rather than requiring a code edit + restart. Everything else
# in this file stays code: weights are a scoring MODEL change, which belongs
# in a commit with a DECISIONS.md entry, not in a text box.
#
# Same no-import-side-effects rule as candidate_profile.py: loading is an
# explicit call, so importing config never touches the filesystem. Callers
# that score jobs (app.py, server.py) call load_scoring_overrides() at
# startup; scoring.py reads these names off the module at call time, so an
# override applies without a restart.
SCORING_OVERRIDES_PATH = os.getenv("GMMH_SCORING_OVERRIDES_PATH", "scoring_overrides.json")

# The in-code values, kept so "reset to defaults" is possible after an
# override has been saved.
DEFAULT_TARGET_ROLE_FAMILY_PATTERNS = TARGET_ROLE_FAMILY_PATTERNS
DEFAULT_EXCLUDE_INDUSTRY_KEYWORDS = EXCLUDE_INDUSTRY_KEYWORDS


def load_scoring_overrides(path: str = SCORING_OVERRIDES_PATH) -> bool:
    """Apply the saved overrides, if any, to this module's globals. Returns
    True if a file was found and applied. Absent keys keep the in-code
    default, so a file listing only excluded industries doesn't blank the
    role families."""
    global TARGET_ROLE_FAMILY_PATTERNS, EXCLUDE_INDUSTRY_KEYWORDS
    if not os.path.exists(path):
        return False
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if "target_role_family_patterns" in data:
        TARGET_ROLE_FAMILY_PATTERNS = tuple(data["target_role_family_patterns"])
    if "exclude_industry_keywords" in data:
        EXCLUDE_INDUSTRY_KEYWORDS = tuple(data["exclude_industry_keywords"])
    return True


def save_scoring_overrides(
    role_family_patterns: tuple[str, ...],
    exclude_industry_keywords: tuple[str, ...],
    path: str = SCORING_OVERRIDES_PATH,
) -> None:
    """Persist both lists and apply them immediately. Role-family entries are
    REGEXES (that's what the gate matches with), so each is compiled here -
    an invalid one raises re.error rather than being written and blowing up
    inside scoring on the next job."""
    for pattern in role_family_patterns:
        re.compile(pattern)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "target_role_family_patterns": list(role_family_patterns),
                "exclude_industry_keywords": list(exclude_industry_keywords),
            },
            fh,
            indent=2,
        )
    load_scoring_overrides(path)
