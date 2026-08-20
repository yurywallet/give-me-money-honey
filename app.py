"""Streamlit page: edit the candidate profile, set search criteria, run a
search, and see ranked results.

Run: `streamlit run app.py`

Layout: search criteria + the Search button live in the sidebar; the main
pane is a tab strip - Profile first (full bio/experience/projects/skills
editor), then the ranked results (a KPI row, then one card per job with a
score meter, skill/benefit chips, and the inline "why this fits" summary).
Chips/meter use theme-independent neutrals plus one brand-blue accent
(dataviz skill: sequential magnitude = one hue) so they read in both
Streamlit light and dark themes without runtime theme detection.
"""
from __future__ import annotations

import datetime as dt
import html
import os

import copy
import dataclasses
import re

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from candidate_profile import CandidateProfile, load_profile, save_profile
from clock import SystemClock
# `config` as a module, not `from config import ...`, for the two editable
# lists: they are rebound by load/save_scoring_overrides at runtime, and a
# from-import would freeze this page on the startup value.
import config
from config import (
    BASE_SCORE_ON_PASS,
    BENEFIT_WEIGHTS,
    EQUITY_SCORE_CAP,
    EQUITY_WEIGHTS,
    FRESHNESS_BONUS_TIERS,
    PROFILE_SKILL_CAP,
    PROFILE_SKILL_WEIGHT,
    PROFILE_TITLE_MATCH_BONUS,
    SALARY_BONUS_CAP,
    SALARY_BONUS_PER_1K,
    STAGE_WEIGHTS,
    HardCriteria,
    SearchConfig,
    load_scoring_overrides,
    save_scoring_overrides,
)
from db import (
    get_conn,
    get_latest_role_fit_snapshot,
    init_db,
    list_top_jobs,
    rescore_all,
    role_salary_stats,
    save_role_fit_snapshot,
    set_fit_summary,
)
from fit_summary import configured_providers, generate_with, provider_configured
from role_fit import (
    ROLE_DESCRIPTIONS,
    ROLE_TITLE_PATTERNS,
    compute_role_fit,
    rank_anchor_candidates,
    rank_complementary_skills,
)
from scheduler import Scheduler
from scoring import score_job
from sources.linkedin_selector import build_linkedin_source, selected_linkedin_mode
from sources.mock_source import MockJobSource

# Before anything scores: apply the operator's saved role-family / excluded-
# industry lists. Cheap (one small JSON read) and done every rerun so the
# Scoring tab's editor takes effect immediately, no restart.
load_scoring_overrides()

st.set_page_config(page_title="give_me_money_honey", layout="wide")

_BLUE = "#2a78d6"
_STYLE = """
<style>
.gmh-row { display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
.gmh-title { font-size:1.05rem; font-weight:600; text-decoration:none; color:inherit; }
.gmh-title:hover { text-decoration:underline; }
.gmh-rank { font-variant-numeric:tabular-nums; color:rgba(128,128,128,.9); font-weight:600; }
.gmh-company { color:rgba(128,128,128,1); font-size:0.9rem; }
.gmh-badge { margin-left:auto; background:%(blue)s; color:#fff; font-weight:600;
  font-size:0.85rem; padding:2px 10px; border-radius:999px; font-variant-numeric:tabular-nums; }
.gmh-meter { height:8px; border-radius:4px; background:rgba(128,128,128,.20);
  width:100%%; overflow:hidden; margin:8px 0 10px; }
.gmh-meter-fill { height:100%%; border-radius:4px; background:%(blue)s; }
.gmh-meta { color:rgba(128,128,128,1); font-size:0.88rem; margin-bottom:8px; }
.gmh-label { font-size:0.72rem; text-transform:uppercase; letter-spacing:.04em;
  color:rgba(128,128,128,.9); margin:6px 0 3px; }
.gmh-chip { display:inline-block; padding:2px 10px; margin:0 5px 5px 0; border-radius:999px;
  font-size:0.78rem; color:inherit; border:1px solid rgba(128,128,128,.35);
  background:rgba(128,128,128,.08); white-space:nowrap; }
.gmh-chip-skill { border-color:rgba(42,120,214,.55); background:rgba(42,120,214,.10); }
.gmh-chip-equity { border-color:rgba(237,161,0,.65); background:rgba(237,161,0,.14); font-weight:600; }
.gmh-chip-missing { border-color:rgba(74,58,167,.6); background:rgba(74,58,167,.14); }
.gmh-muted { color:rgba(128,128,128,.8); font-size:0.82rem; }
.gmh-closed { color:#d03b3b; font-weight:600; font-size:0.78rem; margin-left:6px; }
.gmh-source { font-size:0.72rem; color:rgba(128,128,128,.9); border:1px solid rgba(128,128,128,.3);
  border-radius:999px; padding:1px 8px; white-space:nowrap; }
</style>
""" % {"blue": _BLUE}
st.markdown(_STYLE, unsafe_allow_html=True)


@st.cache_resource
def _get_conn():
    conn = get_conn(SearchConfig().db_path)
    init_db(conn)
    return conn


conn = _get_conn()

if "profile_loaded" not in st.session_state:
    _p = load_profile()
    st.session_state.title_keywords_text = ", ".join(_p.title_keywords)
    st.session_state.skills_text = ", ".join(_p.skills)
    st.session_state.summary_text = _p.summary
    st.session_state.work_experience_text = _p.work_experience
    st.session_state.personal_projects_text = _p.personal_projects
    st.session_state.education_text = _p.education
    st.session_state.anchor_tools_text = ", ".join(_p.anchor_tools)
    st.session_state.anchor_skills_text = ", ".join(_p.anchor_skills)
    st.session_state.search_keywords_text = ", ".join(_p.title_keywords)
    st.session_state.profile_loaded = True

_WORK_TYPES = ["fulltime", "parttime", "contract"]
_LOCATION_MODES = ["remote", "hybrid", "onsite"]
_hard_defaults = HardCriteria()


def _current_profile() -> CandidateProfile:
    """The profile as currently shown in the form (not necessarily saved yet) -
    used both for searching and for fit-summary generation, so a summary is
    always written against what the candidate is looking at."""
    return CandidateProfile(
        title_keywords=tuple(s.strip() for s in st.session_state.title_keywords_text.split(",") if s.strip()),
        skills=tuple(s.strip() for s in st.session_state.skills_text.split(",") if s.strip()),
        summary=st.session_state.summary_text,
        work_experience=st.session_state.work_experience_text,
        personal_projects=st.session_state.personal_projects_text,
        education=st.session_state.education_text,
        anchor_tools=tuple(s.strip() for s in st.session_state.anchor_tools_text.split(",") if s.strip()),
        anchor_skills=tuple(s.strip() for s in st.session_state.anchor_skills_text.split(",") if s.strip()),
    )


def _format_local_timestamp(iso_str: str) -> str:
    """Stored timestamps are UTC (clock.SystemClock) - convert to the
    server's local timezone for display. This is a single-operator local
    app (the server IS the operator's machine), so "local tz" here is
    unambiguous, unlike a multi-user deployed service. `astimezone()` with
    no argument does the UTC -> local conversion via the system tz."""
    local = dt.datetime.fromisoformat(iso_str).astimezone()
    return local.strftime("%Y-%m-%d %H:%M:%S %Z")


def _freshness_label(job) -> str:
    """"Posted Xd/Xw/Xmo ago" from score_breakdown's posted_age_days
    (operator ask 2026-08-18, while actively applying: an old above-floor
    posting has usually already built a real applicant pool, so recency is worth
    seeing at a glance). Empty string - not "unknown"/"—" - when the row
    predates this feature or age couldn't be parsed, so the meta line just
    quietly drops the clause rather than claiming ignorance about something
    most cards WILL have (same "don't assert what isn't there" rule as
    everywhere else, applied to omission this time instead of a stated
    unconfirmed value)."""
    age = (job.score_breakdown or {}).get("posted_age_days")
    if age is None:
        return ""
    if age == 0:
        return "🕐 Posted today"
    if age < 7:
        return f"🕐 Posted {age}d ago"
    if age < 30:
        return f"🕐 Posted {round(age / 7)}w ago"
    if age < 365:
        return f"🕐 Posted {round(age / 30)}mo ago"
    return f"🕐 Posted {round(age / 365)}yr ago"


def _fmt_salary(lo, hi) -> str:
    def k(v):
        return f"${v / 1000:.0f}k" if v else None
    lo_s, hi_s = k(lo), k(hi)
    if lo_s and hi_s:
        return f"{lo_s}–{hi_s}"
    return lo_s or hi_s or "—"


_SIGNAL_LABELS = {
    "meaningful_equity": "meaningful equity",
    "equity_pct": "% ownership named",
    "founding": "founding / early",
    "early_stage": "early-stage",
    "series_a": "Series A",
    "series_b": "Series B",
    "series_c_plus": "Series C+",
    "pre_ipo": "pre-IPO",
    "well_funded": "venture-backed",
    "family_medical": "family medical",
    "401k_match": "401k match",
    "remote_stipend": "remote stipend",
    "unlimited_pto": "unlimited PTO",
    "parental_leave": "parental leave",
}


def _label(key: str) -> str:
    return _SIGNAL_LABELS.get(key, key.replace("_", " "))


def _chips(items, variant="") -> str:
    if not items:
        return '<span class="gmh-muted">none</span>'
    cls = f"gmh-chip gmh-chip-{variant}" if variant else "gmh-chip"
    return "".join(f'<span class="{cls}">{html.escape(_label(i))}</span>' for i in items)


def _meter(score, max_score) -> str:
    pct = 0 if max_score <= 0 else max(4, min(100, round(100 * score / max_score)))
    return f'<div class="gmh-meter"><div class="gmh-meter-fill" style="width:{pct}%"></div></div>'


# --- sidebar: search criteria + search ------------------------------------
with st.sidebar:
    st.header("Search criteria")
    min_salary = st.number_input("Minimum salary", value=_hard_defaults.min_salary, step=5000)
    work_type = st.selectbox("Work type", _WORK_TYPES, index=_WORK_TYPES.index(_hard_defaults.work_type))
    location_mode = st.selectbox(
        "Location mode", _LOCATION_MODES, index=_LOCATION_MODES.index(_hard_defaults.location_mode)
    )
    location_country = st.text_input("Location country", value=_hard_defaults.location_country)
    # text_area, not text_input: the keyword list is long enough that a
    # single-line box hides all but the first two entries in the narrow
    # sidebar. Commas still separate (parsing strips whitespace, so a
    # wrapped/newline-separated entry parses the same).
    st.text_area("Search keywords (comma-separated)", key="search_keywords_text", height=130)
    st.caption(
        "**This search only** - what the job boards are queried for right now. "
        "Not saved: widen it to probe the market without touching your profile "
        "or any score. Seeded from the profile's Target job titles on load; the "
        "profile field is the one that scores (Profile tab)."
    )

    st.caption(
        "🤖 **AI-refined search** (`agent.py`) runs a LangGraph agent instead of "
        "one fixed search: it searches, looks at *why* jobs failed the hard gate "
        "(salary/location/work-type), and reflects with an LLM on whether to "
        "broaden or adjust keywords - then repeats. Live-tested: on this project's "
        "own data it found 8 real hard-passing Reddit ML roles ($216K-$409K) that "
        "the static keyword list found zero of, by adding 'machine learning "
        "engineer' and 'data engineer' on its own. Uses whichever LLM provider "
        "fit_summary would use (Ollama > Claude > Gemini) - free if only Gemini "
        "is configured. Cost note: each round repeats the REAL source search too - "
        "if LinkedIn/Apify is configured, that's real spend per round, not just "
        "the LLM call."
    )
    use_search_agent = st.checkbox("Use AI-refined search (agent.py)", value=True)
    max_agent_iterations = (
        st.number_input(
            "Max refinement rounds (recommended: 2)", min_value=1, max_value=5, value=2, step=1,
            help="2 is agent.py's own documented default - enough to broaden keywords once and "
                 "see the effect, without compounding search cost/LLM calls too far.",
        )
        if use_search_agent else 2
    )
    # Which LinkedIn actor is live, and what it costs - the switch is an env
    # var (GMMH_LINKEDIN_ACTOR), so without this the mode is invisible and a
    # credit-saving switch back to `legacy` would silently change what the
    # results mean (legacy's remote filter is ignored - § DECISIONS.md
    # 2026-08-02).
    if os.getenv("APIFY_TOKEN"):
        if selected_linkedin_mode() == "legacy":
            st.caption(
                "🔌 LinkedIn: **legacy** actor (cheap, ~\\$0.05/search). Its remote/full-time "
                "filters are ignored by the actor, so results are unfiltered and land in "
                "Needs verification. Unset `GMMH_LINKEDIN_ACTOR` in .env for the filtered source."
            )
        else:
            st.caption(
                "🔌 LinkedIn: **remote-filtered** actor (~\\$0.12/search). Uses LinkedIn's own "
                "`f_WT=2` filter, which is verified to work. Set `GMMH_LINKEDIN_ACTOR=legacy` "
                "in .env to switch to the cheaper one if Apify credit runs low."
            )

    run_search = st.button("Search", type="primary", use_container_width=True)

# ONE HardCriteria for the whole page, built from the sidebar inputs above and
# used by every path that calls score_job: the Search, the re-score on profile
# save, and the Scoring tab that documents them. Previously the search built
# its own from the sidebar while the profile-save re-score passed a bare
# HardCriteria() - so searching at a sidebar min_salary the operator actually
# set, then saving the profile, silently re-scored those same rows against
# the code default instead.
hard = HardCriteria(
    min_salary=int(min_salary),
    work_type=work_type,
    location_mode=location_mode,
    location_country=location_country,
)

if run_search:
    keywords = tuple(s.strip() for s in st.session_state.search_keywords_text.split(",") if s.strip())
    cfg = SearchConfig(keywords=keywords, hard=hard)

    sources = []
    # Which LinkedIn source (remote-filtered vs cheaper legacy) is chosen by
    # GMMH_LINKEDIN_ACTOR - see sources/linkedin_selector.py.
    _linkedin = build_linkedin_source()
    if _linkedin is not None:
        sources.append(_linkedin)
    if os.getenv("GREENHOUSE_COMPANIES"):
        from sources.greenhouse_source import GreenhouseSource

        sources.append(GreenhouseSource())
    if not sources:
        st.warning(
            "No real source configured (APIFY_TOKEN or GREENHOUSE_COMPANIES). "
            "The app will not use mock/example jobs by default - set one of these in .env to search real listings."
        )

    if sources and use_search_agent:
        from agent import run_search_agent

        st.session_state.pop("last_agent_state", None)
        try:
            with st.spinner("Searching (AI-refined)..."):
                final_state = run_search_agent(
                    sources, conn, _current_profile(), hard,
                    keywords=list(keywords), max_iterations=int(max_agent_iterations),
                )
            st.session_state.last_search_summary = final_state["last_summary"]
            st.session_state.last_agent_state = final_state
        except Exception as exc:  # noqa: BLE001 - surface, don't crash the page
            st.error(
                f"AI-refined search failed: {exc}. Falling back to a single plain search - "
                "check OLLAMA_MODEL/ANTHROPIC_API_KEY/GEMINI_API_KEY if this keeps happening."
            )
            scheduler = Scheduler(cfg, sources, conn, profile=_current_profile())
            with st.spinner("Searching..."):
                st.session_state.last_search_summary = scheduler.run_once()
    elif sources:
        st.session_state.pop("last_agent_state", None)
        scheduler = Scheduler(cfg, sources, conn, profile=_current_profile())
        with st.spinner("Searching..."):
            st.session_state.last_search_summary = scheduler.run_once()
    else:
        st.info("No search performed because no LinkedIn/Apify credentials are configured.")

# Pre-generate summaries for at most this many top OPEN jobs on load, per
# configured provider. Bounded because a local model (Ollama) can take
# 30-120s each and cloud free tiers rate-limit - an unbounded pre-gen would
# hang the page or blow the quota. The rest generate on demand via each
# card's per-provider button.
_PREGEN_LIMIT = 5

_PROVIDER_LABELS = {"gemini": "Gemini", "claude": "Claude", "ollama": "Ollama (local)"}


def _provider_label(provider: str) -> str:
    return _PROVIDER_LABELS.get(provider, provider.title())


# Human-readable label per Job.source value - every source's `name` attribute
# (sources/__init__.py's JobSource protocol) should have an entry here so a
# new source doesn't show up as a raw internal slug in the UI.
_SOURCE_LABELS = {
    "linkedin_apify": "LinkedIn",
    "greenhouse": "Greenhouse",
    "lever": "Lever",
    "mock": "Mock",
    "manual": "Manual",
}


def _source_label(source: str) -> str:
    return _SOURCE_LABELS.get(source, source.title())


def _providers_to_show(job) -> list[str]:
    """Configured providers first (stable order), then any provider that has
    a CACHED summary but is no longer configured - so a summary generated
    earlier (e.g. before you turned off Ollama) doesn't just disappear."""
    configured = configured_providers()
    cached = list(getattr(job, "fit_summaries", None) or {})
    return configured + [p for p in cached if p not in configured]


# Raw hard_fail_reason token -> plain-English label (operator ask 2026-08-02:
# "no hard requirements met" doesn't say WHICH requirement). Keys match the
# leading token scheduler.run_once / db.hard_fail_reason_counts group by.
_FAIL_REASON_LABELS = {
    "salary": "💰 salary below your floor (or not stated)",
    "work_type": "🕒 not full-time (or not stated)",
    "location_mode": "🏢 not remote (or not stated)",
    "location_country": "🌍 outside your target country (or not stated)",
    "excluded_industry": "🚫 excluded industry",
    "skill_match": "🧩 too few of your skills in the JD",
    "anchor_skill": "⚓ none of your anchor tools/skills present",
    "title": "🏷️ title outside your target roles",
}


def _fail_reason_label(token: str) -> str:
    return _FAIL_REASON_LABELS.get(token, token.replace("_", " "))


def _reason_token(reason: str) -> str:
    """Leading token of a hard_fail_reason string - same grouping convention
    as scheduler.run_once and db.hard_fail_reason_counts."""
    return reason.split("'")[0].strip() if "'" in reason else reason.split(" ")[0]


def render_search_summary(summary: dict) -> None:
    """Overall 'what did the last search actually find, and why did the rest
    miss' readout (operator ask 2026-08-02). Replaces dumping the raw dict."""
    if not isinstance(summary, dict) or not summary:
        return
    with st.expander("📊 Last search — results and why jobs missed", expanded=False):
        for source, s in summary.items():
            label = _source_label(source)
            if "error" in s:
                st.markdown(f"**{label}** — ❌ failed: {s['error']}")
                continue
            st.markdown(
                f"**{label}** — {s.get('found', 0)} found · {s.get('new', 0)} new · "
                f"✅ {s.get('hard_pass', 0)} passing · 🔍 {s.get('needs_verification', 0)} need verification"
            )
            reasons = s.get("fail_reasons") or {}
            if not reasons:
                continue
            total_missed = s.get("found", 0) - s.get("hard_pass", 0)
            st.caption(f"Why the other {total_missed} missed (a job can miss on more than one):")
            st.dataframe(
                pd.DataFrame(
                    [{"Reason": _fail_reason_label(k), "Jobs": v} for k, v in reasons.items()]
                ),
                hide_index=True, use_container_width=True,
            )


def render_job_card(rank: int, job, max_score: float) -> None:
    """One result card - reused by the Open and Closed tabs (single renderer,
    no duplicated markup)."""
    with st.container(border=True):
        title = html.escape(job.title)
        company = html.escape(job.company or "")
        freshness = _freshness_label(job)
        freshness_meta = f"&nbsp;&nbsp;·&nbsp;&nbsp;{freshness}" if freshness else ""
        closed_flag = (
            '<span class="gmh-closed">⚠️ no longer accepting applications</span>'
            if not job.is_active else ""
        )
        st.markdown(
            f'<div class="gmh-row">'
            f'<span class="gmh-rank">#{rank}</span>'
            f'<a class="gmh-title" href="{html.escape(job.url)}" target="_blank">{title}</a>'
            f'<span class="gmh-company">{company}</span>'
            f'<span class="gmh-source">{html.escape(_source_label(job.source))}</span>'
            f"{closed_flag}"
            f'<span class="gmh-badge">{job.score:.0f}</span>'
            f"</div>"
            f"{_meter(job.score, max_score)}"
            f'<div class="gmh-meta">💰 {_fmt_salary(job.salary_min, job.salary_max)}'
            f"&nbsp;&nbsp;·&nbsp;&nbsp;📍 {html.escape(job.location_raw or '—')}{freshness_meta}</div>"
            f'<div class="gmh-label">📈 Equity &amp; stage</div>{_chips(getattr(job, "equity_signals", []), variant="equity")}'
            f'<div class="gmh-label">✅ Matched skills</div>{_chips(getattr(job, "matched_skills", []), variant="skill")}'
            f'<div class="gmh-label">🎯 Skills to add — in this JD, not on your profile</div>'
            f'{_chips(getattr(job, "missing_skills", []), variant="missing")}'
            f'<div class="gmh-label">Benefits</div>{_chips(getattr(job, "benefits", []))}',
            unsafe_allow_html=True,
        )

        # Per-role "what doesn't match" (operator ask 2026-08-02) - surfaced
        # inline rather than buried in the Score breakdown JSON, since for a
        # needs-verification/closed role this is the whole reason it's here.
        _reasons = (job.score_breakdown or {}).get("hard_fail_reasons") or []
        if _reasons:
            st.markdown(
                '<div class="gmh-label">⚠️ Not matching</div>'
                + "".join(
                    f'<span class="gmh-chip gmh-chip-missing">'
                    f"{html.escape(_fail_reason_label(_reason_token(r)))}</span>"
                    for r in _reasons
                ),
                unsafe_allow_html=True,
            )
            with st.expander("Exact mismatch detail"):
                for r in _reasons:
                    st.markdown(f"- {r}")

        providers = _providers_to_show(job)
        if not providers:
            st.markdown('<div class="gmh-label">Why this fits</div>', unsafe_allow_html=True)
            st.markdown('<span class="gmh-muted">No fit summary yet — configure a provider and reload.</span>',
                        unsafe_allow_html=True)
        else:
            configured = configured_providers()
            summary_cols = st.columns(len(providers)) if len(providers) > 1 else [st.container()]
            for col, provider in zip(summary_cols, providers):
                with col:
                    label = _provider_label(provider)
                    st.markdown(f'<div class="gmh-label">Why this fits — {label}</div>', unsafe_allow_html=True)
                    summary = (job.fit_summaries or {}).get(provider)
                    is_configured = provider in configured
                    if summary:
                        st.markdown(summary)
                    elif is_configured:
                        st.caption("Not generated yet — use the button below.")
                    else:
                        st.caption(f"{label} is no longer configured (showing nothing cached).")
                    if is_configured:
                        btn_label = f"Regenerate ({label})" if summary else f"Generate ({label})"
                        if st.button(btn_label, key=f"fit_summary_{provider}_{job.id}"):
                            with st.spinner(f"Generating with {label}..."):
                                try:
                                    new_summary = generate_with(provider, job, _current_profile())
                                    set_fit_summary(conn, job.id, provider, new_summary)
                                    st.rerun()
                                except Exception as exc:  # noqa: BLE001
                                    st.error(f"{label} generation failed: {exc}")

        with st.expander("Score breakdown"):
            st.json(job.score_breakdown)
        with st.expander("Full description"):
            st.write(job.description)


def render_profile_tab() -> None:
    """Full profile editor. Target titles, Skills, and Anchor tools/skills
    drive scoring.py's mechanical hard-gate/keyword match; Work experience,
    Personal projects, and Education are free text that only feed
    fit_summary.py's LLM prompt - scoring.py stays pure/regex-only per its own
    module contract, so those three never affect the score itself.

    LinkedIn auto-scan isn't offered here: WebFetch can't reliably read an
    authenticated LinkedIn profile page, so pasting your own content below is
    the supported path, not a fallback.
    """
    st.caption(
        "Paste from your resume/LinkedIn rather than linking to it - profile pages "
        "require login, so nothing here can scan LinkedIn automatically."
    )
    st.text_input("Target job titles (comma-separated)", key="title_keywords_text")
    st.caption(
        f"**Scores every job, saved with the profile** - a job whose title matches one of "
        f"these earns +{PROFILE_TITLE_MATCH_BONUS:.0f} (`config.PROFILE_TITLE_MATCH_BONUS`), and "
        "Save profile re-scores all stored jobs. Deliberately separate from the sidebar's "
        "Search keywords, which change one search and are never saved - so a throwaway "
        "exploratory query can't quietly become a permanent scoring bonus."
    )
    st.text_area("Summary / bio", key="summary_text", height=140)
    st.text_area(
        "Work experience", key="work_experience_text", height=200,
        placeholder="One entry per role - company, title, dates, and a few bullets on what you did.",
    )
    st.text_area(
        "Personal projects", key="personal_projects_text", height=160,
        placeholder="Side projects, OSS, portfolio work - anything not covered by Work experience.",
    )
    st.text_area(
        "Education", key="education_text", height=100,
        placeholder="Institution - degree, field, dates. One entry per line.",
    )
    st.text_area("Skills (comma-separated)", key="skills_text", height=110)
    st.caption(
        "**Anchor tools/skills** (§ DECISIONS.md 2026-08-01): a job must mention at least "
        "one of these to hard-pass, in addition to the skill-count floor. Fixes a real false "
        "positive - JDs for genuinely different, deeper-expertise roles (e.g. Machine Learning "
        "Engineer) matched on generic shared vocabulary (python, llm) with zero of your actual "
        "differentiators and were hard-passing anyway. Pick terms that are YOURS specifically, "
        "not generic tech buzzwords everyone's JD uses - a term so common it also appears in "
        "the roles you want excluded won't filter them out."
    )
    st.text_input("Anchor tools (comma-separated)", key="anchor_tools_text")
    st.text_input("Anchor skills (comma-separated)", key="anchor_skills_text")
    st.caption(
        "Skills and Anchor tools/skills above drive scoring's keyword match and hard gate. "
        "Work experience, Personal projects, and Education feed the LLM fit-summary narrative only."
    )
    if st.button("Save profile"):
        _profile = _current_profile()
        save_profile(_profile)
        # Scoring output is STORED per job at scrape time, so editing the
        # profile leaves every existing row stale - the operator hit exactly
        # this: skills they had just added were still listed as "missing" on
        # 450 of 1123 jobs. Re-score on save so the results always reflect
        # the profile that's actually on screen.
        with st.spinner("Profile saved — re-scoring stored jobs against it..."):
            n = rescore_all(conn, _profile, hard, score_job, now=SystemClock().now())
        st.success(f"Profile saved. Re-scored {n} stored job(s) against it.")


def render_role_map_tab(profile: CandidateProfile) -> None:
    """Skill-gap analysis against role_fit.ROLE_SKILLS (operator ask
    2026-08-01, evidence-based per 2026-08-02 revision) - a skill map per
    target role family, compared to the profile's own Skills/Anchor
    tools/skills. Advisory only: this tab never touches scoring.py's
    hard_pass/score.
    """
    st.caption(
        "Skill map per target role family, built from real skill-mention frequency across "
        "this project's own job search results (746 postings) plus external 2026 industry "
        "research - see role_fit.ROLE_SKILLS for sources and methodology. Compares against "
        "your current Skills to show match strength per role, which of your existing skills "
        "should be promoted to Anchor tools/skills, and which missing skills would help the "
        "most roles if you learned them."
    )

    role_fits = compute_role_fit(profile)

    snap_col1, snap_col2 = st.columns([1, 3])
    with snap_col1:
        rerun_clicked = st.button("🔄 Rerun & save")
    if rerun_clicked:
        # Recompute fresh (the tab below already reflects the live profile on
        # every render - this is the explicit, deliberate save point, not a
        # different computation) and persist a dated snapshot so match-score
        # trend over time is visible later, not just the always-live view.
        role_fits = compute_role_fit(profile)
        snapshot = {
            "roles": [
                {
                    "role": rf.role,
                    "match_pct": rf.match_pct,
                    "matched": rf.matched,
                    "missing": rf.missing,
                    "anchor_gap": rf.anchor_gap,
                    "anchored": rf.anchored,
                }
                for rf in role_fits
            ],
            "anchor_candidates": [{"skill": s, "roles": roles} for s, roles in rank_anchor_candidates(role_fits)],
            "complementary_skills": [{"skill": s, "roles": roles} for s, roles in rank_complementary_skills(role_fits)],
        }
        now_iso = SystemClock().now().isoformat()
        save_role_fit_snapshot(conn, now_iso, snapshot)

    # Read the caption AFTER the save so it reflects the click without a
    # st.rerun(): rerunning inside st.tabs() resets the selection to the
    # first tab, so the operator got bounced to Profile and never saw the
    # updated timestamp - the save worked, the feedback didn't (2026-08-18).
    last_snapshot = get_latest_role_fit_snapshot(conn)
    with snap_col2:
        if last_snapshot:
            st.caption(f"Last saved: {_format_local_timestamp(last_snapshot['computed_at'])}")
        else:
            st.caption("Not saved yet - click Rerun & save to record a snapshot.")
    if rerun_clicked:
        st.toast("Role Map snapshot saved.")

    st.subheader("What each role actually is")
    for role in role_fits:
        st.markdown(f"**{role.role}** — {ROLE_DESCRIPTIONS.get(role.role, '')}")

    st.subheader("Match score by role")

    def _fmt_salary_range(stats: dict | None) -> str:
        if stats is None:
            return "no disclosed-salary data yet"
        return f"${stats['min']:,}–${stats['max']:,} (n={stats['n']})"

    df_scores = pd.DataFrame(
        [
            {
                "Role": rf.role,
                # ProgressColumn's "%.0f%%" format applies to the raw cell
                # value, not a derived percentage of min/max - a 0-1 fraction
                # rounds to "0%"/"1%" regardless of fill. Scale to 0-100 so
                # the printed number matches what the bar shows (verified
                # live 2026-08-02: this was wrong before the fix).
                "Match": round(rf.match_pct * 100),
                "Matched": len(rf.matched),
                "Total skills": len(rf.role_skills),
                # Real disclosed salaries from this project's own jobs.db, not
                # a guess - see db.role_salary_stats (§ DECISIONS.md 2026-08-02,
                # built right after finding/fixing a real parse_salary bug).
                "Salary range": _fmt_salary_range(
                    role_salary_stats(conn, ROLE_TITLE_PATTERNS[rf.role], floor=hard.min_salary)
                ),
            }
            for rf in role_fits
        ]
    ).sort_values("Match", ascending=False)
    st.dataframe(
        df_scores,
        column_config={
            # Sequential magnitude = one hue (dataviz skill) - the app's
            # single accent color, set via .streamlit/config.toml so this
            # (and every other Streamlit-native widget) matches the custom
            # HTML meters/chips elsewhere instead of Streamlit's default red.
            "Match": st.column_config.ProgressColumn("Match %", min_value=0, max_value=100, format="%.0f%%"),
        },
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Skill coverage by role")
    st.caption("⭐ anchored · ✅ you have it, not anchored yet · ➕ role needs it, gap in your Skills")
    all_skills = sorted({s for rf in role_fits for s in rf.role_skills})
    matrix_rows = []
    for skill in all_skills:
        row = {"Skill": skill}
        for rf in role_fits:
            if skill not in rf.role_skills:
                row[rf.role] = ""
            elif skill in rf.anchored:
                row[rf.role] = "⭐"
            elif skill in rf.matched:
                row[rf.role] = "✅"
            else:
                row[rf.role] = "➕"
        matrix_rows.append(row)
    st.dataframe(pd.DataFrame(matrix_rows), hide_index=True, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Add to Anchor skills")
        st.caption("Skills you already have, matched to a real role, not yet anchored.")
        anchor_candidates = rank_anchor_candidates(role_fits)
        if anchor_candidates:
            st.dataframe(
                pd.DataFrame(
                    [{"Skill": s, "Helps N roles": len(roles), "Roles": ", ".join(roles)} for s, roles in anchor_candidates]
                ),
                hide_index=True, use_container_width=True,
            )
        else:
            st.caption("Every matched skill relevant to these roles is already anchored.")

    with col2:
        st.subheader("Complementary skills (best uplift)")
        st.caption("Skills you don't have yet, ranked by how many target roles need them.")
        complementary = rank_complementary_skills(role_fits)
        if complementary:
            st.dataframe(
                pd.DataFrame(
                    [{"Skill": s, "Helps N roles": len(roles), "Roles": ", ".join(roles)} for s, roles in complementary[:10]]
                ),
                hide_index=True, use_container_width=True,
            )
        else:
            st.caption("You already have every skill in the role map.")


# One-line meaning per HardCriteria field. Rendered by iterating the DATACLASS,
# not this dict, so a gate added to HardCriteria shows up in the tab (with a
# blank note) instead of silently going undocumented.
_HARD_GATE_NOTES: dict[str, str] = {
    "min_salary": "Below this, the job scores 0 and is excluded. No equity relaxation (§ DECISIONS.md 2026-08-12).",
    "work_type": "Must match exactly. Unstated by the source -> 'needs verification', not a confirmed fail.",
    "location_mode": "Must match exactly. Unstated by the source -> 'needs verification'.",
    "location_country": "Must match exactly. Unstated by the source -> 'needs verification'.",
    "min_matched_skills": "Profile skills that must appear in the JD text. Falling short still computes a full score - it just isn't a pass.",
    "require_anchor_skill": "At least one of YOUR anchor tools/skills must appear too, not just generic shared vocabulary.",
    "require_target_role_family": "Title must name one of the target role families below (AI / Analytics Engineer).",
}


def render_scoring_tab(profile: CandidateProfile, hard: HardCriteria) -> None:
    """How a job gets its number, read live from config.py and the same
    HardCriteria object the Search passes to scoring.score_job - never a
    written-down copy of the weights. Editing config.py or the sidebar moves
    this tab and the next search together, which is the whole point of it
    (operator ask 2026-08-19).
    """
    st.caption(
        "Every number here is read from `config.py` and from the criteria set in the sidebar - "
        "the same `HardCriteria` object handed to `scoring.score_job()` by Search and by the "
        "re-score on Save profile. Nothing on this tab is a hand-written copy, so it cannot "
        "drift from what actually scored your jobs."
    )

    st.subheader("Hard gates - pass/fail, never a deduction")
    st.caption(
        "Salary, work type and location zero a job out entirely. Everything else below still "
        "computes a full, visible score - it just doesn't set `hard_pass`, so a close fit stays "
        "distinguishable from an unrelated role."
    )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Gate": f.name,
                    # The default is only worth screen space when it has been
                    # overridden - the sidebar is seeded from these defaults,
                    # so a separate column is identical on almost every row.
                    "In effect now": (
                        str(getattr(hard, f.name)) if getattr(hard, f.name) == f.default
                        else f"{getattr(hard, f.name)}  (default {f.default})"
                    ),
                    "What it does": _HARD_GATE_NOTES.get(f.name, ""),
                }
                for f in dataclasses.fields(hard)
            ]
        ),
        hide_index=True, use_container_width=True,
    )
    with st.expander(
        f"Edit target role families ({len(config.TARGET_ROLE_FAMILY_PATTERNS)} title patterns) "
        f"and hard-excluded industries ({len(config.EXCLUDE_INDUSTRY_KEYWORDS)} keywords)"
    ):
        st.caption(
            "One per line. Saved to `scoring_overrides.json` and applied immediately - these two "
            "lists are the ones that change while you're actually searching, so they're editable "
            "here instead of needing a code edit. The weights above are not: a weight is a scoring "
            "MODEL change and belongs in a commit."
        )
        ed1, ed2 = st.columns(2)
        with ed1:
            st.markdown("**Target role families** — matched against the job TITLE (regex, case-insensitive)")
            families_text = st.text_area(
                "Target role families", value="\n".join(config.TARGET_ROLE_FAMILY_PATTERNS),
                height=180, label_visibility="collapsed",
            )
        with ed2:
            st.markdown("**Hard-excluded industries** — matched as whole words in title + company + JD")
            excludes_text = st.text_area(
                "Hard-excluded industries", value="\n".join(config.EXCLUDE_INDUSTRY_KEYWORDS),
                height=180, label_visibility="collapsed",
            )
        save_col, reset_col = st.columns([1, 3])
        with save_col:
            save_lists = st.button("💾 Save lists")
        with reset_col:
            reset_lists = st.button("Reset both to the in-code defaults")
        if save_lists or reset_lists:
            if reset_lists:
                families = config.DEFAULT_TARGET_ROLE_FAMILY_PATTERNS
                excludes = config.DEFAULT_EXCLUDE_INDUSTRY_KEYWORDS
            else:
                families = tuple(s.strip() for s in families_text.splitlines() if s.strip())
                excludes = tuple(s.strip().lower() for s in excludes_text.splitlines() if s.strip())
            try:
                # Role families are regexes, so a typo like "ai engineer(" is
                # a real possibility - save_scoring_overrides compiles each
                # one first and raises rather than writing a file that would
                # crash scoring on the next job.
                save_scoring_overrides(families, excludes)
            except re.error as exc:
                st.error(f"Not saved - invalid regex in the role-family list: {exc}")
            else:
                st.success(
                    f"Saved {len(families)} role-family pattern(s) and {len(excludes)} excluded "
                    "industry keyword(s). Stored job scores still reflect the old lists until "
                    "re-scored - see the drift check below."
                )

    st.subheader("Soft signals - only ever add")
    max_fresh = max(b for _, b in FRESHNESS_BONUS_TIERS)
    benefits_total = sum(BENEFIT_WEIGHTS.values())
    soft_rows = [
        ("Base, for clearing every hard gate", f"{BASE_SCORE_ON_PASS:g}", f"{BASE_SCORE_ON_PASS:g}", "BASE_SCORE_ON_PASS"),
        ("Salary above the floor",
         f"+{SALARY_BONUS_PER_1K:g} per $1k over ${hard.min_salary:,}", f"{SALARY_BONUS_CAP:g}",
         "SALARY_BONUS_PER_1K / SALARY_BONUS_CAP"),
        ("Your skills found in the JD", f"+{PROFILE_SKILL_WEIGHT:g} each", f"{PROFILE_SKILL_CAP:g}",
         "PROFILE_SKILL_WEIGHT / PROFILE_SKILL_CAP"),
        ("Title matches a target job title", f"+{PROFILE_TITLE_MATCH_BONUS:g}", f"{PROFILE_TITLE_MATCH_BONUS:g}",
         "PROFILE_TITLE_MATCH_BONUS"),
        ("Equity + funding-stage signals", "summed, see below", f"{EQUITY_SCORE_CAP:g}",
         "EQUITY_WEIGHTS / STAGE_WEIGHTS / EQUITY_SCORE_CAP"),
        ("Benefits", "summed, see below", f"{benefits_total:g} (uncapped)", "BENEFIT_WEIGHTS"),
        ("Freshness of the posting", ", ".join(f"<={d}d +{b:g}" for d, b in FRESHNESS_BONUS_TIERS),
         f"{max_fresh:g}", "FRESHNESS_BONUS_TIERS"),
    ]
    st.dataframe(
        pd.DataFrame(soft_rows, columns=["Signal", "Points", "Most it can add", "Constant in config.py"]),
        hide_index=True, use_container_width=True,
    )
    ceiling = (
        BASE_SCORE_ON_PASS + SALARY_BONUS_CAP + PROFILE_SKILL_CAP + PROFILE_TITLE_MATCH_BONUS
        + EQUITY_SCORE_CAP + benefits_total + max_fresh
    )
    st.caption(
        f"Ceiling: **{ceiling:g}** if every signal maxes out at once. Cash outranks paper - the "
        f"salary cap ({SALARY_BONUS_CAP:g}) is deliberately larger than the equity cap "
        f"({EQUITY_SCORE_CAP:g}), and equity can no longer buy a job past the salary floor "
        "(§ DECISIONS.md 2026-08-12)."
    )

    col_e, col_b = st.columns(2)
    with col_e:
        st.markdown("**Equity / stage weights**")
        st.dataframe(
            pd.DataFrame(
                [{"Signal": k, "Points": v, "Kind": kind}
                 for kind, d in (("equity", EQUITY_WEIGHTS), ("stage", STAGE_WEIGHTS))
                 for k, v in d.items()]
            ).sort_values("Points", ascending=False),
            hide_index=True, use_container_width=True,
        )
    with col_b:
        st.markdown("**Benefit weights**")
        st.dataframe(
            pd.DataFrame([{"Benefit": k, "Points": v} for k, v in BENEFIT_WEIGHTS.items()])
            .sort_values("Points", ascending=False),
            hide_index=True, use_container_width=True,
        )

    st.subheader("Are the stored scores actually this configuration?")
    st.caption(
        "Scores are computed once, at scrape time, and stored on the row - so changing the "
        "sidebar criteria or the profile leaves stored scores describing the OLD rules until "
        "something re-scores them. This re-runs `score_job` on the loaded jobs with the settings "
        "above and reports the difference. Freshness is excluded from the comparison (it changes "
        "on its own as a posting ages, which isn't drift)."
    )
    drift = []
    for job in jobs:
        fresh = float((job.score_breakdown or {}).get("freshness_bonus", 0.0))
        # Shallow copy, not deepcopy: rows carry a live `db_conn` attribute,
        # which deepcopy can't pickle. score_job only ever ASSIGNS attributes
        # (never mutates a list/dict in place), so a shallow copy leaves the
        # displayed row untouched. now=None skips freshness entirely
        # (score_job's contract), so the stored freshness is added back
        # rather than recomputed - a posting aging past a tier boundary is
        # not configuration drift.
        recomputed = score_job(copy.copy(job), hard, profile, now=None)
        expected = round(recomputed.score + fresh, 2)
        if abs(expected - job.score) > 0.01 or recomputed.hard_pass != job.hard_pass:
            drift.append({
                "Title": job.title, "Company": job.company,
                "Stored": job.score, "With current settings": expected,
                "Stored pass": job.hard_pass, "Now passes": recomputed.hard_pass,
            })
    if not drift:
        st.success(f"Aligned - all {len(jobs)} loaded job(s) score the same under the settings above.")
    else:
        st.warning(
            f"{len(drift)} of {len(jobs)} loaded job(s) were scored under different settings. "
            "Search or Save profile re-scores everything; the button below does it without either."
        )
        st.dataframe(pd.DataFrame(drift), hide_index=True, use_container_width=True)
        if st.button("Re-score stored jobs with these settings"):
            with st.spinner("Re-scoring..."):
                n = rescore_all(conn, profile, hard, score_job, now=SystemClock().now())
            st.success(f"Re-scored {n} stored job(s).")


# --- main pane: profile + ranked results -----------------------------------
st.title("give_me_money_honey")

# only_active=False so closed jobs stay queryable; they go in their own tab
# rather than the Open list. The Apify source exposes no true "accepts
# applications" field, so is_active (still present in the most recent
# successful search) is the honest proxy for "still open".
jobs = list_top_jobs(conn, limit=25, only_passing=True, only_active=False)

active_jobs = sorted([j for j in jobs if j.is_active], key=lambda j: -j.score)
closed_jobs = sorted([j for j in jobs if not j.is_active], key=lambda j: -j.score)

# Split active jobs into confirmed passes and needs-verification (failed the
# hard gate ONLY on fields the source never stated - e.g. salary is fine but
# work_type/location weren't in the listing). These get a real computed score
# (see scoring.needs_verification) so a genuine near-match like a $220k
# analytics-engineer role isn't invisible just because the source didn't say
# "remote" - but they stay out of the confirmed "Open" list since they aren't
# verified fits yet.
open_jobs = [j for j in active_jobs if j.hard_pass]
verify_jobs = [j for j in active_jobs if j.needs_verification]
max_score = max((j.score for j in jobs), default=0)

tab_labels = ["👤 Profile", "🗺️ Role Map", "⚖️ Scoring", f"✅ Open ({len(open_jobs)})"]
if verify_jobs:
    tab_labels.append(f"🔍 Needs verification ({len(verify_jobs)})")
if closed_jobs:
    tab_labels.append(f"⚠️ No longer accepting ({len(closed_jobs)})")

tabs = st.tabs(tab_labels)

with tabs[0]:
    render_profile_tab()

with tabs[1]:
    render_role_map_tab(_current_profile())

with tabs[2]:
    render_scoring_tab(_current_profile(), hard)

with tabs[3]:
    if not jobs:
        st.info("No matching jobs yet - fill in your profile above, set search criteria in the sidebar, then Search.")
    else:
        # KPI row (dataviz: a headline number is a stat tile, not a chart).
        sal_los = [j.salary_min for j in open_jobs if j.salary_min]
        sal_his = [j.salary_max for j in open_jobs if j.salary_max]
        equity_open = sum(1 for j in open_jobs if j.equity_signals)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Open matches", len(open_jobs), delta=(f"-{len(closed_jobs)} closed" if closed_jobs else None),
                  delta_color="off")
        k2.metric("With equity", f"{equity_open}/{len(open_jobs)}")
        k3.metric("Top score", f"{max_score:.0f}")
        k4.metric(
            "Open salary range",
            # Escape $ so st.metric doesn't parse it as a LaTeX math delimiter.
            _fmt_salary(min(sal_los) if sal_los else None, max(sal_his) if sal_his else None).replace("$", "\\$"),
        )
        if "last_search_summary" in st.session_state:
            render_search_summary(st.session_state.last_search_summary)
        _agent_state = st.session_state.get("last_agent_state")
        if _agent_state:
            with st.expander(f"AI-refined search: {_agent_state['iteration']} round(s)"):
                for i, round_ in enumerate(_agent_state["history"], start=1):
                    st.markdown(f"**Round {i}** — keywords: `{', '.join(round_['keywords'])}`")
                    st.caption(f"Found/failed: {round_['summary']} — reasons: {round_['fail_reason_counts']}")
                    st.write(round_["reasoning"])
                st.markdown(f"**Final keywords:** `{', '.join(_agent_state['keywords'])}`")

        _has_provider = provider_configured()
        _providers = configured_providers()
        if not _has_provider:
            st.warning(
                "No LLM provider configured, so fit summaries can't be generated. Set OLLAMA_MODEL "
                "(local, unlimited), GEMINI_API_KEY (free tier), or ANTHROPIC_API_KEY in .env, then reload."
            )
        elif len(_providers) > 1:
            st.caption(
                f"Generating with {len(_providers)} providers: {', '.join(_provider_label(p) for p in _providers)}."
            )

        # Pre-generate summaries for the top few OPEN jobs, per configured
        # provider (see _PREGEN_LIMIT). Closed jobs never get an API call - you
        # can't apply to them. A dead provider (quota/rate/key error) is skipped
        # for the REST of this batch, but other providers keep going - a Gemini
        # quota hit must not also block Ollama, and vice versa.
        if _has_provider:
            top_jobs = active_jobs[:_PREGEN_LIMIT]
            to_pregen = [
                (job, provider)
                for job in top_jobs
                for provider in _providers
                if provider not in (job.fit_summaries or {})
            ]
            if to_pregen:
                dead_providers: set[str] = set()
                with st.spinner(f"Generating fit summaries for the top {len(top_jobs)} open role(s)..."):
                    for job, provider in to_pregen:
                        if provider in dead_providers:
                            continue
                        try:
                            summary = generate_with(provider, job, _current_profile())
                            set_fit_summary(conn, job.id, provider, summary)
                            job.fit_summaries[provider] = summary
                        except Exception as exc:  # noqa: BLE001 - surface, don't crash the page
                            # Almost always provider-wide (quota / rate limit /
                            # key), not job-specific - stop THIS provider, not
                            # the others.
                            dead_providers.add(provider)
                            st.warning(
                                f"Stopped generating with {_provider_label(provider)} — {exc}. "
                                "Cached summaries still show; other provider(s) continue; use each role's button or reload."
                            )

        st.divider()
        for rank, job in enumerate(open_jobs, start=1):
            render_job_card(rank, job, max_score)

# Profile, Role Map, Scoring, Open = 0-3; the optional tabs start after them.
_tab_idx = 4
if verify_jobs:
    with tabs[_tab_idx]:
        st.caption(
            "These roles failed the fit gate only on fields the source never "
            "stated (e.g. work arrangement or location weren't listed) — not "
            "on a confirmed bad fit. Score reflects the same computation as "
            "an Open match; check the listing directly to confirm."
        )
        for rank, job in enumerate(verify_jobs, start=1):
            render_job_card(rank, job, max_score)
    _tab_idx += 1

if closed_jobs:
    with tabs[_tab_idx]:
        st.caption(
            "These roles are no longer in the latest search — likely filled or closed. "
            "Re-run Search to refresh which are still open."
        )
        for rank, job in enumerate(closed_jobs, start=1):
            render_job_card(rank, job, max_score)
