"""Reference map of typical skills per target role family, and the
candidate-vs-role gap analysis behind the "Role Map" tab (operator ask
2026-08-01).

`ROLE_SKILLS` was originally a guessed, manually-curated list (same
precedent as config.SKILL_UNIVERSE) - replaced 2026-08-02 with an
evidence-based version per operator ask ("do a proper research... not only
for the database you have"). Two independent sources, cross-checked against
each other:

1. **Real frequency data from this project's own jobs.db** (746 real
   postings, grouped by title-substring match, skill mentions counted
   against config.SKILL_UNIVERSE's vocabulary): Analytics Engineer n=130,
   Data Analyst n=88, AI Engineer n=62, BI Engineer n=17, AI Product
   Engineer n=17. The two n=17 roles are thin samples - weighted more
   toward source 2 below for those.
2. **External 2026 industry research** (WebSearch, 2026-08-02): dbt Labs'
   "What is analytics engineering?", KORE1's Analytics Engineer hiring
   guide, Glassdoor/Velvet Jobs/TekRecruiter BI Engineer job descriptions,
   Turing College's "Rise of the AI Product Engineer", and multiple 2026 AI
   Engineer hiring/roadmap guides (dataskew.io, ayautomate.com,
   technovids.com) converging on evaluation/RAG/agent-orchestration as the
   defining 2026 AI Engineer skills.

Where the two sources agreed, kept the skill. Where jobs.db was thin (BI
Engineer, AI Product Engineer) or a skill was clearly emerging in 2026
listings but underrepresented in this project's specific search history
(e.g. "evaluation" for both AI roles), weighted toward the external source.
Still not a source of truth - revisit if it drifts from what real JDs ask
for; re-run the jobs.db frequency query (see this module's git history for
the query) periodically as more real postings accumulate.

Role families collapse title_keywords variants that share one skill set
(the seniority tiers of "analytics engineer"; the two "bi engineer" naming
variants) into a single entry - the skill set doesn't change with seniority,
only the bar for how much of it you're expected to own.

Deliberately separate from scoring.py: this doesn't touch hard_pass or
score - it's an advisory comparison the operator reads, not a gate.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from candidate_profile import CandidateProfile

ROLE_SKILLS: dict[str, tuple[str, ...]] = {
    # jobs.db n=130. Ordered by real frequency: sql 96%, dbt 78%, python 65%,
    # snowflake 65%, data quality 60%, data modeling 53%, looker 38%, data
    # warehouse 37%, airflow 36%, etl 35%, ci/cd 34%, business intelligence
    # 34%. dbt Labs/KORE1 confirm dbt+SQL+data modeling as the core three.
    "Analytics Engineer": (
        "sql", "dbt", "python", "snowflake", "data quality", "data modeling",
        "looker", "data warehouse", "airflow", "etl", "ci/cd",
        "business intelligence", "bigquery", "redshift", "semantic layer",
        "dimensional modeling", "git",
    ),
    # jobs.db n=88. sql 91%, python 67%, tableau 48%, power bi 40%,
    # statistics 40%, data science 38%, r 34%, business intelligence 30%,
    # data quality 22%, data modeling 20%.
    "Data Analyst": (
        "sql", "python", "tableau", "power bi", "statistics", "data science",
        "r", "business intelligence", "data quality", "data modeling",
        "looker", "dbt", "etl", "data visualization", "dashboards",
        "a/b testing", "excel", "storytelling",
    ),
    # jobs.db n=17 (thin) + external (Glassdoor/Velvet Jobs/TekRecruiter):
    # ETL, SQL, Power BI, Azure, data modeling, data architecture, Tableau,
    # data governance as the recurring external list; jobs.db agrees on
    # sql 100%, business intelligence 88%, tableau 53%, looker 53%, etl 47%,
    # dbt 47%, power bi 41%, data governance 29%.
    "BI Engineer": (
        "sql", "business intelligence", "python", "data quality", "tableau",
        "looker", "etl", "dbt", "power bi", "snowflake", "data governance",
        "ci/cd", "data modeling", "data architecture", "data warehouse",
        "azure", "dashboards", "elt",
    ),
    # jobs.db n=17 (thin) + external (Turing College, recruiter.daily.dev):
    # "combines AI/software engineering, product management, UX/UI, data
    # analysis" - Python + APIs + cloud + context engineering/retrieval/
    # vector DBs/agent frameworks/evaluation as the technical layer, product
    # intuition and shipping-to-real-users as the product layer. jobs.db:
    # llm 65%, python 53%, ai agents 29%, api 24%, prompt engineering 18%.
    "AI Product Engineer": (
        "python", "llm", "ai agents", "api", "prompt engineering", "rag",
        "product management", "product roadmap", "mcp", "vector database",
        "langchain", "evaluation", "gtm", "agile", "experimentation",
    ),
    # jobs.db n=62, richest sample: rag 53%, python 52%, llm 47%, aws 44%,
    # machine learning 44%, langchain 34%, ai agents 29%, prompt engineering
    # 24%, agentic ai 24%, mcp present. External sources converge hard on
    # RAG + agent orchestration + evaluation as THE 2026 differentiators
    # ("the biggest skills hiring managers look for... RAG, agents, and
    # evaluation"; "evaluation is nearly every senior AI engineer JD's
    # requirement") - "evaluation" added on that basis despite jobs.db not
    # tracking it (not yet in config.SKILL_UNIVERSE's vocabulary).
    "AI Engineer": (
        "rag", "python", "llm", "machine learning", "langchain", "aws",
        "ai agents", "prompt engineering", "agentic ai", "mcp",
        "vector database", "api", "langgraph", "openai api", "evaluation",
        "azure", "gcp",
    ),
}

# Case-insensitive title-match regex per role family, for db.role_salary_stats
# to find real postings belonging to each ROLE_SKILLS entry - same patterns
# used for the jobs.db skill-frequency research behind ROLE_SKILLS itself
# (§ DECISIONS.md 2026-08-02), kept here as the one place both that research
# and the Role Map tab's salary-range column draw from.
ROLE_TITLE_PATTERNS: dict[str, str] = {
    "Analytics Engineer": r"\banalytics engineer\b",
    "Data Analyst": r"\bdata analyst\b",
    "BI Engineer": r"\b(?:bi|business intelligence) engineer\b",
    "AI Product Engineer": r"\bai product engineer\b",
    "AI Engineer": r"\bai engineer\b",
}

# SENIORITY-SPLIT patterns, for PAY analysis only (§ DECISIONS.md 2026-08-02,
# corrected). Deliberately separate from ROLE_TITLE_PATTERNS above, which is
# keyed to ROLE_SKILLS's role families: skills genuinely don't change with
# seniority (that's this module's own docstring), but PAY very much does, so
# one aggregate salary number per family is actively misleading.
#
# Why this exists: an initial clear-rate analysis reported "Analytics
# Engineer clears the floor only 4% of the time (n=56)" and nearly led to
# dropping the operator's PRIMARY target role. Operator pushed back - staff
# analytics engineer roles clear it comfortably - and the data agreed once
# split: plain AE and Senior median mins ran well below Staff/Principal/Lead
# postings, most of which cleared the floor. The aggregate buried a
# genuinely high-yield tier under 49 lower-tier postings.
SENIORITY_TIER_PATTERNS: dict[str, str] = {
    # Ordered most-senior first; _tier_of() below takes the first match, so
    # "Sr. Staff Analytics Engineer" classifies as Staff/Principal, not Senior.
    "Staff/Principal/Lead Analytics Engineer":
        r"\b(?:staff|principal|lead|head)\b.{0,20}\banalytics engineer\b",
    "Senior Analytics Engineer": r"\b(?:senior|sr\.?)\b.{0,20}\banalytics engineer\b",
    "Analytics Engineer (no tier)": r"\banalytics engineer\b",
    "Staff/Principal/Lead AI Engineer":
        r"\b(?:staff|principal|lead|head)\b.{0,20}\bai engineer\b",
    "Senior AI Engineer": r"\b(?:senior|sr\.?)\b.{0,20}\bai engineer\b",
    "AI Engineer (no tier)": r"\bai engineer\b",
    "AI Product Engineer": r"\bai product engineer\b",
    "Data Analyst": r"\bdata analyst\b",
    "BI Engineer": r"\b(?:bi|business intelligence) engineer\b",
}

# One-sentence "what this role actually is" (operator ask 2026-08-02) - same
# sources as ROLE_SKILLS's methodology note above (dbt Labs, Turing College,
# external 2026 AI-engineering research), summarized short on purpose.
# Deliberately draws the same boundary as scoring.py's anchor-skill gate
# (§ DECISIONS.md 2026-08-01): AI Engineer is scoped to building ON TOP OF
# models via APIs, explicitly NOT the deeper model-training discipline
# ("Machine Learning Engineer") that gate exists to keep separate.
ROLE_DESCRIPTIONS: dict[str, str] = {
    "Analytics Engineer": (
        "Transforms raw data into clean, tested, documented models inside the warehouse "
        "(the \"T\" in ELT) - distinct from data engineers, who own extract/load."
    ),
    "Data Analyst": (
        "Turns data into decisions for stakeholders via SQL/Python analysis, dashboards, "
        "and statistical rigor - less ownership of the underlying pipeline than an analytics engineer."
    ),
    "BI Engineer": (
        "Builds the pipelines, warehouses, and reporting infrastructure BI insights run on - "
        "closer to data engineering than a Data Analyst, owning ETL and data governance too."
    ),
    "AI Product Engineer": (
        "Specs, builds, and ships AI-powered product features end-to-end - one person combining "
        "product judgment, engineering (APIs/agents/RAG), and UX rather than handing off to others."
    ),
    "AI Engineer": (
        "Builds production systems on top of foundation models via APIs - RAG, agent orchestration, "
        "evaluation - not training models from scratch (that's Machine Learning Engineer)."
    ),
}


@dataclass
class RoleFit:
    role: str
    role_skills: tuple[str, ...]
    matched: list[str]     # candidate has it, role needs it
    missing: list[str]     # role needs it, candidate doesn't have it
    anchor_gap: list[str]  # matched, but NOT in profile.anchor_tools/anchor_skills
    match_pct: float

    @property
    def anchored(self) -> list[str]:
        """Matched skills that ARE already an anchor - the "already covered" set."""
        gap = set(self.anchor_gap)
        return [s for s in self.matched if s not in gap]


def compute_role_fit(profile: CandidateProfile) -> list[RoleFit]:
    """One RoleFit per ROLE_SKILLS entry, comparing the profile's own skills
    (and anchor_tools/anchor_skills) against each role's typical skill set."""
    anchors = {s.lower() for s in profile.anchor_tools + profile.anchor_skills}
    # Anchors count as "have" too. They are NOT a subset of profile.skills
    # (§ DECISIONS.md 2026-08-01: "anchors aren't always a subset of
    # profile.skills - e.g. storytelling"), so keying matched/missing off
    # skills alone reported a skill the profile plainly lists - "dashboards",
    # an anchor_skill - as a gap, and made it unreachable for the ⭐ tier,
    # which is by definition matched AND anchored (operator hit this
    # 2026-08-18).
    have = {s.lower() for s in profile.skills} | anchors
    results = []
    for role, role_skills in ROLE_SKILLS.items():
        matched = [s for s in role_skills if s.lower() in have]
        missing = [s for s in role_skills if s.lower() not in have]
        anchor_gap = [s for s in matched if s.lower() not in anchors]
        match_pct = len(matched) / len(role_skills) if role_skills else 0.0
        results.append(RoleFit(role, role_skills, matched, missing, anchor_gap, match_pct))
    return results


def rank_anchor_candidates(role_fits: list[RoleFit]) -> list[tuple[str, list[str]]]:
    """Skills the candidate already has, matched against a real role, but not
    yet in anchor_tools/anchor_skills - ranked by how many roles they'd help
    if promoted to an anchor. The "add these to Anchor skills" list."""
    skill_to_roles: dict[str, list[str]] = defaultdict(list)
    for rf in role_fits:
        for skill in rf.anchor_gap:
            skill_to_roles[skill].append(rf.role)
    return sorted(skill_to_roles.items(), key=lambda kv: -len(kv[1]))


def rank_complementary_skills(role_fits: list[RoleFit]) -> list[tuple[str, list[str]]]:
    """Skills the candidate does NOT have, ranked by how many target roles
    require them - "best uplift" = breadth of impact across the whole role
    portfolio, not just one role. The "learn these next" list."""
    skill_to_roles: dict[str, list[str]] = defaultdict(list)
    for rf in role_fits:
        for skill in rf.missing:
            skill_to_roles[skill].append(rf.role)
    return sorted(skill_to_roles.items(), key=lambda kv: -len(kv[1]))
