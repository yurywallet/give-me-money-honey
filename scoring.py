"""Hard-gate filtering and soft-signal scoring.

Hard criteria (§ config.HardCriteria) are pass/fail - a job failing any one
of them never gets a score above zero and is flagged hard_pass=False, full
stop. Soft criteria (benefits, salary above the floor) only ever ADD to
score; there is no mechanism here for a soft signal to reject a job, by
design - that's what makes it "soft."
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from functools import lru_cache
from typing import Optional

# The two operator-editable lists (role families, excluded industries) are
# read off the MODULE at call time, not bound at import: they can be changed
# and saved from the Scoring tab while the app is running, and a `from config
# import X` binding would keep serving the value from startup. Everything
# else below is a from-import because it only changes with a code edit.
import config
from candidate_profile import CandidateProfile, DEFAULT_PROFILE
from config import (
    BASE_SCORE_ON_PASS,
    BENEFIT_WEIGHTS,
    EQUITY_SALARY_FLOOR,
    EQUITY_SCORE_CAP,
    EQUITY_WEIGHTS,
    FRESHNESS_BONUS_TIERS,
    PROFILE_SKILL_CAP,
    PROFILE_SKILL_WEIGHT,
    PROFILE_TITLE_MATCH_BONUS,
    SALARY_BONUS_CAP,
    SALARY_BONUS_PER_1K,
    SKILL_UNIVERSE,
    STAGE_WEIGHTS,
    HardCriteria,
)

# Which equity/stage signals count as GENUINE startup upside strong enough to
# relax the salary floor (§ DECISIONS.md 2026-07-25). Deliberately excludes bare
# "equity" (every big company grants RSUs), "well_funded", and late-stage
# (series_c_plus/pre_ipo) - those companies pay market base and shouldn't get
# the sub-floor pass. Founding/early + explicit % + meaningful + seed/A/B do.
_STRONG_EQUITY_SIGNALS = frozenset(
    {"founding", "equity_pct", "meaningful_equity", "early_stage", "seed", "series_a", "series_b"}
)


def has_strong_equity_upside(equity_signals: list[str]) -> bool:
    """True if the detected signals justify the relaxed salary floor."""
    return any(s in _STRONG_EQUITY_SIGNALS for s in equity_signals)


@lru_cache(maxsize=8)
def _exclude_industry_res(keywords: tuple[str, ...]) -> tuple[re.Pattern, ...]:
    """Compiled once per distinct keyword list - the cache key IS the list, so
    editing it in the Scoring tab recompiles, and not editing it costs a dict
    lookup per job rather than a recompile."""
    return tuple(re.compile(rf"\b{re.escape(k)}\b", re.IGNORECASE) for k in keywords)

# --- posting freshness (§ DECISIONS.md 2026-08-18) --------------------------
# Three DIFFERENT formats across real sources, all verified against live
# jobs.db data before writing this: LinkedIn's legacy actor returns relative
# English ("2 weeks ago", "1 day ago", already pre-bucketed by LinkedIn
# itself); the remote-filtered actor and Greenhouse return absolute dates
# ("2026-08-14" or full ISO8601 with a UTC offset). All three parse through
# one function so callers never need to know which source a job came from.
_RELATIVE_AGE_RE = re.compile(r"(\d+)\+?\s*(minute|hour|day|week|month|year)s?\s+ago", re.IGNORECASE)
_DAYS_PER_UNIT = {
    "minute": 1 / 1440, "hour": 1 / 24, "day": 1.0, "week": 7.0, "month": 30.0, "year": 365.0,
}


def parse_posted_age_days(posted_at: Optional[str], now: datetime) -> Optional[int]:
    """Days since a job was posted, from whatever format the source gave -
    relative English or an absolute date/ISO8601 timestamp. None if
    `posted_at` is empty or in a format neither branch recognizes (never
    guesses - same "don't assert what the data can't support" discipline as
    location/salary parsing elsewhere in this module)."""
    if not posted_at:
        return None
    text = posted_at.strip()

    m = _RELATIVE_AGE_RE.search(text)
    if m:
        n = int(m.group(1))
        unit = m.group(2).lower()
        return round(n * _DAYS_PER_UNIT[unit])

    try:
        posted_dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if posted_dt.tzinfo is None:
        posted_dt = posted_dt.replace(tzinfo=timezone.utc)
    now_aware = now if now.tzinfo is not None else now.replace(tzinfo=timezone.utc)
    return max(0, (now_aware - posted_dt).days)


def freshness_bonus(age_days: Optional[int]) -> float:
    """Points for a recently-posted listing, from FRESHNESS_BONUS_TIERS
    (config.py) - tiered, not a linear decay, because the practical
    difference is coarse ("this week" vs "this month" vs "old"), and coarse
    tiers are robust to the source-dependent precision loss above (a
    relative "2 weeks ago" is already bucketed by LinkedIn itself, so a
    fine-grained formula would imply precision that isn't there). 0.0 (not a
    penalty) for an old or unknown-age posting - same "soft criteria only
    ever add" rule as every other signal in this module; an old posting
    never loses points, it just doesn't gain this one."""
    if age_days is None:
        return 0.0
    for max_days, bonus in FRESHNESS_BONUS_TIERS:
        if age_days <= max_days:
            return bonus
    return 0.0


def excluded_industry(job: Job) -> Optional[str]:
    """The excluded-industry keyword this job matches (title/company/description),
    or None. A match hard-rejects the job (§ config.EXCLUDE_INDUSTRY_KEYWORDS)."""
    text = f"{job.title}\n{job.company}\n{job.description}"
    for pattern in _exclude_industry_res(tuple(config.EXCLUDE_INDUSTRY_KEYWORDS)):
        m = pattern.search(text)
        if m:
            return m.group(0).lower()
    return None


# --- onsite-schedule cross-check (operator-reported bug 2026-07-28) --------
# A job can be STORED as location_mode="remote" (the source's field mapping
# defaulted to the requested filter - see linkedin_apify_source.py's own
# docstring, "workplaceType sometimes null") while its own JD text plainly
# says otherwise, e.g. Pilot.com: "San Francisco, CA (3 days/week in office -
# Mondays, Tuesdays, and Thursdays)" - no literal "hybrid"/"onsite" word, just
# a day count + "in office", which sources/linkedin_apify_source.py's existing
# _HYBRID_LOCATION_CONTEXT (requires the word "hybrid") and
# _ONSITE_LOCATION_CONTEXT (requires "five days a week"/"mon-fri"/etc, not a
# partial count) both miss. This check is independent of the stored
# location_mode field - it re-examines the description directly, so it also
# corrects already-stored rows on the next re-score, not just new fetches.
_ONSITE_SCHEDULE_RE = re.compile(
    r"\b([1-5])(?:-\d)?\s*days?\s*(?:a|per|/)?\s*week\b[^.\n]{0,60}\b(office|onsite|on-site|in[-\s]?person)\b"
    r"|\b(office|onsite|on-site|in[-\s]?person)\b[^.\n]{0,60}\b([1-5])(?:-\d)?\s*days?\s*(?:a|per|/)?\s*week\b",
    re.IGNORECASE,
)


def describes_onsite_schedule(description: str) -> bool:
    """True if the JD text itself states a partial or full in-office schedule
    (e.g. "3 days/week in office", "2-3 days a week onsite") - reusable by
    both the hard-gate cross-check below and any source's own field inference."""
    return bool(_ONSITE_SCHEDULE_RE.search(description))


# Bare onsite/hybrid keyword signals - the ones the day-count check above
# doesn't cover (operator-reported 2026-07-28: Crunchyroll's "We are
# considering applicants for the location of Los Angeles, CA (onsite)" has no
# day count at all, just the bare word). Moved here from
# sources/linkedin_apify_source.py (which had its own near-duplicate copies)
# so there is exactly ONE place that decides "does this JD text contradict
# remote" - reused by both the hard-gate cross-check (already-stored rows)
# and the source's own field inference (new fetches), so they can't drift.
_ONSITE_KEYWORD_RE = re.compile(
    r"\b(in[-\s]?person|in\s+our\s+office|work together in person|in\s+the\s+office|"
    r"office[-\s]?based|on[-\s]?site|onsite|five\s+days?\s+a\s+week|mon\.?-?fri\.?|"
    r"monday[-\s]?to[-\s]?friday)\b",
    re.IGNORECASE,
)
_HYBRID_KEYWORD_RE = re.compile(
    r"\bhybrid\b.*\b(office|days?|week|onsite|in[-\s]?office)\b"
    r"|\b(office|days?|week|onsite|in[-\s]?office)\b.*\bhybrid\b",
    re.IGNORECASE,
)


def describes_onsite_location(description: str) -> bool:
    """True if the JD text states a bare onsite/in-person/full-week-in-office
    arrangement (not necessarily paired with a day count)."""
    return bool(_ONSITE_KEYWORD_RE.search(description))


def describes_hybrid_location(description: str) -> bool:
    """True if the JD text states a hybrid arrangement - either the word
    "hybrid" near an office/schedule word, or a day-count schedule."""
    return bool(_HYBRID_KEYWORD_RE.search(description)) or describes_onsite_schedule(description)


def contradicts_remote(description: str) -> bool:
    """True if the JD text describes ANY non-remote arrangement (onsite or
    hybrid) - used where the caller only needs "not actually remote", not
    which specific arrangement (the hard-gate cross-check below)."""
    return describes_onsite_location(description) or describes_hybrid_location(description)
from db import Job

# --- salary parsing -------------------------------------------------------

_CURRENCY = r"(?:\$|USD\s*)?"  # Only accept USD or no-currency (US listings)
# Two UNAMBIGUOUS alternatives, not two overlapping ones (found 2026-08-02):
# the old `\d{1,3}(?:,\d{3})?|\d+(?:\.\d+)?` let a comma-grouped number like
# "157,200" match via EITHER branch (first as "157,200" via the comma
# branch, or as bare "157" then fail downstream and backtrack into matching
# only "200" via the second `\d+` branch) - when something later in the
# pattern (the "per annum/pa" suffix alternation) failed to match starting
# from position 0, the engine slid forward and silently re-matched from
# WITHIN the number ("$157,200.00/yr..." -> captured "200.00", not
# "157,200"), undercounting the parsed salary by 100x+ with no error. Now
# the comma branch requires AT LEAST ONE comma group (`+` not `?`), so it
# can never partially overlap with the plain `\d+` branch - a comma-bearing
# number can ONLY match the first alternative in full, never a truncated
# substring of it.
_NUMBER = r"(\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
# "per\s+annum" must come BEFORE bare "per" - Python re tries alternatives
# left-to-right and stops at the first that matches, with no preference for
# a longer overall match; "per" is a strict prefix of "per annum", so with
# the old order the engine matched just "per" and left "annum" outside the
# captured span entirely (found 2026-07-30 via _has_currency_signal below
# failing to see "per annum" in m.group(0) despite it being right there in
# the source text - the match span itself was silently truncated).
# Trailing "USD" as well as leading (found 2026-08-02 on a real posting:
# "Base Pay Range (CA Only): $101,600 USD - $127,000 USD" parsed to nothing,
# because _CURRENCY only matches a PREFIX and the suffix "USD" then blocked
# the engine from reaching the dash).
_PER_YEAR = r"(?:\s*(?:USD|/|per\s+annum|per|a|pa|p\.a\.)\s*(?:yr|year|annual)?)?"

# Range like "$120,000 - $150,000", "120k-150k", "£60k – £70k", with optional
# currency symbols and optional 'per year' suffixs.
_SALARY_RANGE_RE = re.compile(
    rf"{_CURRENCY}\s*{_NUMBER}\s*([kK])?{_PER_YEAR}\s*(?:-|to|–|—)\s*{_CURRENCY}\s*{_NUMBER}\s*([kK])?{_PER_YEAR}",
    re.IGNORECASE,
)

# 'Up to' style (max only)
_SALARY_SINGLE_RE = re.compile(
    rf"(?:up to|up-to)\s*{_CURRENCY}\s*{_NUMBER}\s*([kK])?{_PER_YEAR}", re.IGNORECASE
)

# Plus / open-ended min (e.g. "$220,000+", "220k+")
_SALARY_PLUS_RE = re.compile(rf"{_CURRENCY}\s*{_NUMBER}\s*([kK])?\s*\+", re.IGNORECASE)

# 'From' or 'starting at' indicates a minimum only
_SALARY_FROM_RE = re.compile(rf"(?:from|starting at|minimum of)\s*{_CURRENCY}\s*{_NUMBER}\s*([kK])?{_PER_YEAR}", re.IGNORECASE)

# All four patterns above allow $/k/per-year to be fully ABSENT (they're all
# optional), so a bare number range/plus/from with no financial marker at all
# matches too - verified live (2026-07-30) against a real Greenhouse posting:
# "Minimum requirements 5+ years of experience" parsed as salary_min=5, since
# "5+" alone satisfies _SALARY_PLUS_RE with no currency/k-suffix required.
# This checks the ACTUALLY MATCHED text (not just what the pattern allows)
# for a real financial marker, and rejects the match otherwise - "5+" fails,
# "$220,000+"/"220k+" pass.
_CURRENCY_SIGNAL_RE = re.compile(r"\$|usd|\d[kK]\b|/\s*(?:yr|year)|per\s+(?:year|annum)|p\.?a\.?\b", re.IGNORECASE)


def _has_currency_signal(matched_text: str) -> bool:
    return bool(_CURRENCY_SIGNAL_RE.search(matched_text))


# Hourly/monthly pay is out of scope (annual-only). Applied NEAR a matched
# figure rather than to the whole posting - see _is_non_annual_rate. Bare
# "hour"/"month" are deliberately excluded: they match interview durations
# ("1 hour"), reporting cadence ("monthly reporting"), and perks ("monthly
# stipend"), none of which describe how the ROLE is paid.
_NON_ANNUAL_RATE_RE = re.compile(
    r"\b(per\s+hour|hourly|/\s*hr\b|an\s+hour|per\s+month|monthly|/\s*mo(?:nth)?\b|a\s+month)\b",
    re.IGNORECASE,
)
# Tight window ON PURPOSE. A pay cadence attaches to the figure it prices
# ("$60 per hour", "hourly rate of $60"), so it sits immediately around the
# match. A wider window re-introduces the bug this replaced: "Build automated
# monthly reporting. The pay range is $116,400 - $194,000" has 'monthly' only
# ~35 chars from the figure while describing a job duty, not the pay.
_RATE_WINDOW = 22


def _to_int(number: str, k_suffix: Optional[str]) -> int:
    # Support decimal numbers like 120.5
    has_comma = "," in number
    val = float(number.replace(",", ""))
    # A comma-grouped number (e.g. "105,000") is already a full thousands
    # value - a trailing "k" there is a JD-authoring artifact (found
    # 2026-07-30: a real posting read "105,000k to 125,000k annually",
    # parsing to $105,000,000), never a real x1000 multiplier. Only apply
    # the k-suffix to bare numbers like "105k".
    if k_suffix and not has_comma:
        val *= 1000
    return int(val)


def _is_non_annual_rate(text: str, match: re.Match) -> bool:
    """Whether an hourly/monthly cadence applies to THIS matched figure.

    Checked in a window around the match, not across the whole posting
    (found 2026-08-02). The old whole-text guard discarded the salary of any
    posting that merely contained the words anywhere - real examples that
    lost a genuine annual range: "Technical Evaluation in Domain (1 hour)"
    (an interview stage) and "Build automated monthly reporting processes"
    (a job duty). Same proximity discipline as _has_family_medical.
    """
    # Includes the match itself: "per" is often consumed INSIDE the match by
    # _PER_YEAR ("$50 - $60 per" + " hour"), so scanning only outside it would
    # split "per hour" in half and miss a genuine hourly rate.
    window = text[max(0, match.start() - _RATE_WINDOW): match.end() + _RATE_WINDOW]
    return bool(_NON_ANNUAL_RATE_RE.search(window))


def _first_valid(pattern: re.Pattern, text: str):
    """First match that actually looks like annual pay - i.e. carries a real
    currency signal and isn't an hourly/monthly figure.

    Iterates rather than taking `search()`'s first hit (found 2026-08-02): a
    real posting read "...(10-30 minutes)... base salary range for this role
    is $150K - $250K". The bare "10-30" matched first, was correctly rejected
    as non-currency, and the old code then gave up on ranges entirely instead
    of looking further - so a clearly-stated salary was read as "not stated".
    """
    for m in pattern.finditer(text):
        if _has_currency_signal(m.group(0)) and not _is_non_annual_rate(text, m):
            return m
    return None


def parse_salary(text: str) -> tuple[Optional[int], Optional[int]]:
    """Best-effort salary range extraction from free text. Returns (min, max)."""
    m = _first_valid(_SALARY_RANGE_RE, text)
    if m:
        lo = _to_int(m.group(1), m.group(2))
        hi = _to_int(m.group(3), m.group(4))
        # If one side used 'k' and the other didn't (e.g. "50-100k"),
        # normalize the smaller one.
        if m.group(2) is None and m.group(4) and lo < 1000:
            lo *= 1000
        return min(lo, hi), max(lo, hi)

    m = _first_valid(_SALARY_SINGLE_RE, text)
    if m:
        return None, _to_int(m.group(1), m.group(2))

    m = _first_valid(_SALARY_PLUS_RE, text)
    if m:
        return _to_int(m.group(1), m.group(2)), None

    m = _first_valid(_SALARY_FROM_RE, text)
    if m:
        return _to_int(m.group(1), m.group(2)), None

    return None, None


# --- benefit keyword detection -------------------------------------------

_FAMILY_WORDS = re.compile(r"\b(family|spouse|dependents?|domestic partner)\b", re.IGNORECASE)
_MEDICAL_WORDS = re.compile(r"\b(medical|health)\s*(insurance|coverage|plan|care)?\b", re.IGNORECASE)

_BENEFIT_PATTERNS: dict[str, re.Pattern] = {
    "dental": re.compile(r"\bdental\b", re.IGNORECASE),
    "vision": re.compile(r"\bvision\b", re.IGNORECASE),
    "401k_match": re.compile(r"\b401\s?\(?k\)?\s*(match|matching)?\b", re.IGNORECASE),
    "unlimited_pto": re.compile(r"\bunlimited\s+(pto|vacation|time off)\b", re.IGNORECASE),
    "parental_leave": re.compile(r"\b(parental|maternity|paternity)\s+leave\b", re.IGNORECASE),
    "remote_stipend": re.compile(r"\b(home[- ]office|remote)\s+stipend\b", re.IGNORECASE),
    "wellness": re.compile(r"\bwellness\b", re.IGNORECASE),
}

# --- equity / stage signal detection (§ DECISIONS.md 2026-07-25) ----------
# Ordered so the base "equity" mention is checked first; the richer signals
# (meaningful/pct/founding) stack on top of it in the score, since a JD that
# says "meaningful equity, 0.5%, founding engineer" is a stronger upside bet
# than one that just says "equity".
_EQUITY_PATTERNS: dict[str, re.Pattern] = {
    "equity": re.compile(r"\b(equity|stock\s+options?|\brsus?\b|\bisos?\b)\b", re.IGNORECASE),
    "meaningful_equity": re.compile(
        r"\b(meaningful|significant|substantial|generous|competitive)\s+equity\b", re.IGNORECASE
    ),
    # A % named next to equity/ownership, in either order, within a short window.
    "equity_pct": re.compile(
        r"(\d+(?:\.\d+)?\s*%[^.\n]{0,40}(equity|ownership)|(equity|ownership)[^.\n]{0,40}\d+(?:\.\d+)?\s*%)",
        re.IGNORECASE,
    ),
    "founding": re.compile(
        r"\b(founding\s+(engineer|team|member)|early\s+employee|ground\s+floor|"
        r"first\s+(engineer|few\s+engineers|\d+\s+engineers)|employee\s+#?\d{1,2}\b)",
        re.IGNORECASE,
    ),
}
_STAGE_PATTERNS: dict[str, re.Pattern] = {
    "early_stage": re.compile(r"\bearly[-\s]stage\b", re.IGNORECASE),
    "seed": re.compile(r"\b(pre[-\s]?seed|seed[-\s](stage|round|funded|funding))\b", re.IGNORECASE),
    "series_a": re.compile(r"\bseries[-\s]a\b", re.IGNORECASE),
    "series_b": re.compile(r"\bseries[-\s]b\b", re.IGNORECASE),
    "series_c_plus": re.compile(r"\bseries[-\s][c-f]\b", re.IGNORECASE),
    "pre_ipo": re.compile(r"\bpre[-\s]?ipo\b", re.IGNORECASE),
    "well_funded": re.compile(
        r"\b(venture[-\s]backed|vc[-\s]backed|backed\s+by|well[-\s]funded|"
        r"raised\s+\$\d|\$\d+(?:\.\d+)?\s*(m|million|b|billion)\b)",
        re.IGNORECASE,
    ),
}

@lru_cache(maxsize=8)
def _target_role_family_res(patterns: tuple[str, ...]) -> tuple[re.Pattern, ...]:
    """Same call-time compile-and-cache as _exclude_industry_res above."""
    return tuple(re.compile(p, re.IGNORECASE) for p in patterns)

_PROXIMITY_WINDOW = 120  # chars; keeps "family" and "medical" from matching across unrelated paragraphs


def _skill_pattern(skill: str) -> re.Pattern:
    # \b on both ends so short skills like "bi" or "sql" match as whole
    # words only ("bi" must not match inside "big").
    return re.compile(rf"\b{re.escape(skill)}\b", re.IGNORECASE)


def _has_family_medical(text: str) -> bool:
    for fam_match in _FAMILY_WORDS.finditer(text):
        window = text[max(0, fam_match.start() - _PROXIMITY_WINDOW): fam_match.end() + _PROXIMITY_WINDOW]
        if _MEDICAL_WORDS.search(window):
            return True
    return False


def match_profile_skills(description: str, profile: CandidateProfile) -> list[str]:
    """Which of the profile's skills are mentioned in the JD text."""
    return [s for s in profile.skills if _skill_pattern(s).search(description)]


def matches_target_role_family(title: str) -> bool:
    """Whether the job TITLE names one of the target role families
    (config.TARGET_ROLE_FAMILY_PATTERNS). Substring/regex, not exact-title -
    see HardCriteria.require_target_role_family for why that distinction
    matters relative to the 2026-07-25 removal of the old title gate."""
    return any(
        p.search(title or "")
        for p in _target_role_family_res(tuple(config.TARGET_ROLE_FAMILY_PATTERNS))
    )


def match_anchor_signals(description: str, profile: CandidateProfile) -> list[str]:
    """Which of the profile's anchor_tools/anchor_skills are mentioned in the
    JD text - independent of match_profile_skills, since anchors are often
    NOT in profile.skills at all (e.g. "storytelling", "gtm", "cursor").
    Used by hard_filter's require_anchor_skill gate (§ DECISIONS.md 2026-08-01)."""
    anchors = profile.anchor_tools + profile.anchor_skills
    return [a for a in anchors if _skill_pattern(a).search(description)]


def missing_skills(description: str, profile: CandidateProfile) -> list[str]:
    """Skills this JD mentions that are in SKILL_UNIVERSE but NOT in the
    candidate profile - the 'add these to your profile to lift the match score'
    list (operator ask 2026-07-27). Each one, once added, would be counted by
    match_profile_skills and earn PROFILE_SKILL_WEIGHT points."""
    have = {s.lower() for s in profile.skills}
    return [
        skill for skill in SKILL_UNIVERSE
        if skill.lower() not in have and _skill_pattern(skill).search(description)
    ]


def title_matches_profile(title: str, profile: CandidateProfile) -> bool:
    """Whether the job title matches one of the profile's target titles."""
    return any(_skill_pattern(t).search(title) for t in profile.title_keywords)


def parse_benefits(description: str) -> list[str]:
    """Keyword-spot the benefits list from a job description's free text."""
    found = []
    if _has_family_medical(description):
        found.append("family_medical")
    for name, pattern in _BENEFIT_PATTERNS.items():
        if pattern.search(description):
            found.append(name)
    return found


def parse_equity_signals(description: str) -> list[str]:
    """Equity-upside and company-stage signals in the JD text (§ DECISIONS.md
    2026-07-25). Returns detected keys from EQUITY_WEIGHTS + STAGE_WEIGHTS -
    the ranking-time evidence that a role is an ownership bet, not just a
    paycheck. Multiple can (and should) stack: "founding" implies "equity",
    both count."""
    found = []
    for name, pattern in _EQUITY_PATTERNS.items():
        if pattern.search(description):
            found.append(name)
    for name, pattern in _STAGE_PATTERNS.items():
        if pattern.search(description):
            found.append(name)
    return found


# --- hard gate -------------------------------------------------------------

def hard_filter(job: Job, hard: HardCriteria, min_salary_override: Optional[int] = None) -> tuple[bool, list[str]]:
    """Returns (passes, reasons_failed). Any non-empty reasons list means reject.

    `min_salary_override`, when set, replaces `hard.min_salary` for the salary
    gate only - used by score_job to apply the equity-relaxed floor to roles
    with genuine startup upside (§ DECISIONS.md 2026-07-25)."""
    reasons: list[str] = []

    excluded = excluded_industry(job)
    if excluded:
        reasons.append(f"excluded_industry {excluded}")

    # Cross-check the job's OWN "remote" field claim against its own JD text -
    # see contradicts_remote's docstring for why (Pilot.com bug 2026-07-24,
    # Crunchyroll bug 2026-07-28 - a bare "(onsite)" with no day count, which
    # describes_onsite_schedule alone didn't catch). Scoped to only fire when
    # location_mode itself says "remote" (not e.g. already-correct "hybrid"),
    # so a job that's already gated by the plain field-mismatch check below
    # isn't ALSO flagged here for the same underlying issue under a different,
    # ungrouped reason string. No quote char before "location_mode" so
    # db.hard_fail_reason_counts groups this under the same "location_mode"
    # key as the plain field mismatch.
    if (
        hard.location_mode.lower() == "remote"
        and (job.location_mode or "").lower() == "remote"
        and contradicts_remote(job.description)
    ):
        reasons.append("location_mode 'remote' contradicted by description (states an onsite/hybrid arrangement)")

    floor = min_salary_override if min_salary_override is not None else hard.min_salary

    salary_min = job.salary_min
    if salary_min is None:
        parsed_min, parsed_max = parse_salary(job.description)
        salary_min = parsed_min if parsed_min is not None else parsed_max

    # Every check below distinguishes "the source never stated this" (None)
    # from "confirmed something other than what's required" (a real,
    # conflicting value) using the literal word "unconfirmed" - consistently,
    # not just for location_mode (2026-07-29's Zoox fix). This is what lets
    # needs_verification() classify a job's failure reasons without re-deriving
    # field values itself: a reason is unconfirmed-only iff every one of them
    # contains "unconfirmed" (§ DECISIONS.md 2026-07-30, "needs verification"
    # bucket - missing data isn't the same claim as a confirmed bad fit).
    if salary_min is None or salary_min < floor:
        if salary_min is None:
            reasons.append(f"salary unconfirmed by source, required {floor}")
        else:
            reasons.append(f"salary {salary_min} < required {floor}")

    if (job.work_type or "").lower() != hard.work_type.lower():
        if job.work_type is None:
            reasons.append(f"work_type unconfirmed by source, required {hard.work_type}")
        else:
            reasons.append(f"work_type '{job.work_type}' != required '{hard.work_type}'")

    # Location mode is "must not be CONFIRMED hybrid/onsite", not "must be
    # confirmed remote" (§ DECISIONS.md 2026-08-12, operator directive).
    # Measured on real data: of the above-floor jobs this gate was blocking,
    # 113 were merely UNCONFIRMED (the source never stated an arrangement)
    # versus only 30 genuinely confirmed onsite/hybrid. Treating silence as
    # disqualifying buried ~113 well-paid roles - including Netflix at
    # $380k-$610k - for a fact nobody ever asserted. Silence is now allowed
    # through; a stated onsite/hybrid arrangement still rejects, and the
    # separate contradicts_remote check above still catches a "remote" field
    # that the JD text disproves.
    #
    # Deliberately NOT applied to work_type/location_country below: the
    # operator scoped this to location mode, and those gates are cheap to
    # satisfy from JD text, so silence there still routes to Needs
    # verification rather than passing outright.
    if job.location_mode is not None and job.location_mode.lower() != hard.location_mode.lower():
        reasons.append(f"location_mode '{job.location_mode}' != required '{hard.location_mode}'")

    if (job.location_country or "").upper() != hard.location_country.upper():
        if job.location_country is None:
            reasons.append(f"location_country unconfirmed by source, required {hard.location_country}")
        else:
            reasons.append(f"location_country '{job.location_country}' != required '{hard.location_country}'")

    return (len(reasons) == 0, reasons)


def needs_verification(reasons: list[str]) -> bool:
    """True if EVERY hard-fail reason is due to a source-unconfirmed field
    (salary/work_type/location_mode/location_country never stated) rather
    than a confirmed conflicting value or a positive-signal exclusion
    (excluded_industry, a remote claim contradicted by the JD's own text -
    both are genuine bad-fit signals, never worded as "unconfirmed"). Used to
    route a job to a 'needs verification' bucket (operator ask 2026-07-30)
    instead of hiding it the same way a confirmed-bad job is hidden - a
    $196k Faire "Senior Analytics Engineer" failing only because the source
    didn't state work_type/location isn't the same claim as a job confirmed
    onsite or confirmed below the salary floor."""
    return bool(reasons) and all("unconfirmed" in r for r in reasons)


# --- scoring ----------------------------------------------------------------

def score_job(
    job: Job,
    hard: HardCriteria,
    profile: CandidateProfile = DEFAULT_PROFILE,
    now: Optional[datetime] = None,
) -> Job:
    """Mutates and returns job with hard_pass, score, score_breakdown, benefits,
    and matched_skills set.

    `now`: if given, adds a freshness bonus (scoring.freshness_bonus) based on
    `job.posted_at` and stamps `posted_age_days` in the breakdown for display.
    Optional and defaults to None (no freshness scoring) so this stays a pure
    function of its arguments for every existing caller/test that doesn't
    pass a clock - scoring.py's module docstring guarantee ("no I/O, no
    network... exhaustively unit-testable") extends to time: nothing in here
    calls datetime.now() itself.

    Title is scoring-only now (§ DECISIONS.md 2026-07-25) - it never gates
    `hard_pass`. Analytics-adjacent titles (data analyst, data scientist,
    analytics engineer, ...) share most of the same tool vocabulary just in
    a different proportion, and a company can call the same role almost
    anything - an exact title keyword list is the wrong mechanism to tell
    "different role, different skills" (e.g. backend/frontend engineer)
    apart from "same skillset, different title". `hard.min_matched_skills`
    does that job instead: it gates on actual tool/skill overlap in the JD
    text, which degrades gracefully to whatever title a company happens to
    use and doesn't require enumerating every synonym in advance.

    A job that fails salary/work_type/location is zeroed out completely -
    those are fundamentally wrong fits regardless of skill overlap. A job
    that clears those but falls short of `min_matched_skills` still gets a
    fully computed score (informational, same principle as before): visible
    and rankable, just not `hard_pass=True`.

    Equity no longer relaxes the salary floor (§ DECISIONS.md 2026-08-12,
    reversing the 2026-07-25 pivot). It used to: a role with genuine startup
    upside cleared a lower bar than hard.min_salary. That is exactly the "low
    paid job for a paper money future" the operator ruled out, so every job
    is now held to min_salary regardless of its equity story. Equity survives
    only as a capped soft score (EQUITY_SCORE_CAP << SALARY_BONUS_CAP).
    """
    equity_signals = parse_equity_signals(job.description)
    # EQUITY_SALARY_FLOOR is None by design - see config.py. Left as an
    # explicit `or` rather than deleted so the relaxation's absence is
    # visible here, not just implied by its removal.
    salary_floor = (
        EQUITY_SALARY_FLOOR
        if (EQUITY_SALARY_FLOOR is not None and has_strong_equity_upside(equity_signals))
        else hard.min_salary
    )

    parsed_min, parsed_max = parse_salary(job.description)
    if job.salary_min is None and parsed_min is not None:
        job.salary_min = parsed_min
    if job.salary_max is None and parsed_max is not None:
        job.salary_max = parsed_max

    benefits = parse_benefits(job.description)
    matched_skills = match_profile_skills(job.description, profile)
    missing = missing_skills(job.description, profile)

    passes, reasons = hard_filter(job, hard, min_salary_override=salary_floor)

    job.equity_signals = equity_signals
    job.benefits = benefits
    job.matched_skills = matched_skills
    job.missing_skills = missing

    # "Needs verification" (§ DECISIONS.md 2026-07-30): a job failing ONLY on
    # unconfirmed fields (the source never stated salary/location/work_type)
    # gets a real, rankable score below like any other visible job - it's
    # not the same claim as a CONFIRMED bad fit (wrong industry, a definite
    # below-floor salary, a confirmed onsite role), which still zeroes out
    # completely just below. hard_pass stays False either way.
    job.needs_verification = (not passes) and needs_verification(reasons)

    if not passes and not job.needs_verification:
        job.hard_pass = False
        job.score = 0.0
        job.score_breakdown = {"hard_fail_reasons": reasons}
        return job

    breakdown: dict = {"base": BASE_SCORE_ON_PASS}

    effective_salary = job.salary_min
    if effective_salary is None:
        parsed_min, parsed_max = parse_salary(job.description)
        effective_salary = parsed_min if parsed_min is not None else parsed_max

    salary_bonus = 0.0
    if effective_salary and effective_salary > hard.min_salary:
        salary_bonus = min(
            (effective_salary - hard.min_salary) / 1000 * SALARY_BONUS_PER_1K,
            SALARY_BONUS_CAP,
        )
    breakdown["salary_bonus"] = round(salary_bonus, 2)

    # Equity is the heaviest soft axis now (§ DECISIONS.md 2026-07-25 pivot):
    # summed across every detected equity/stage signal, capped so a keyword-
    # stuffed JD can't run away, but the cap is >= the salary cap on purpose.
    equity_raw = 0.0
    for sig in equity_signals:
        weight = EQUITY_WEIGHTS.get(sig, STAGE_WEIGHTS.get(sig, 0.0))
        equity_raw += weight
        breakdown[f"equity:{sig}"] = weight
    equity_score = min(equity_raw, EQUITY_SCORE_CAP)
    breakdown["equity_score"] = round(equity_score, 2)

    benefit_score = 0.0
    for b in benefits:
        weight = BENEFIT_WEIGHTS.get(b, 0.0)
        benefit_score += weight
        breakdown[f"benefit:{b}"] = weight

    skill_bonus = min(len(matched_skills) * PROFILE_SKILL_WEIGHT, PROFILE_SKILL_CAP)
    breakdown["profile_skill_bonus"] = round(skill_bonus, 2)

    title_bonus = PROFILE_TITLE_MATCH_BONUS if title_matches_profile(job.title, profile) else 0.0
    breakdown["profile_title_bonus"] = title_bonus

    fresh_bonus = 0.0
    if now is not None:
        age_days = parse_posted_age_days(job.posted_at, now)
        breakdown["posted_age_days"] = age_days
        fresh_bonus = freshness_bonus(age_days)
        breakdown["freshness_bonus"] = fresh_bonus

    job.score = round(
        BASE_SCORE_ON_PASS + salary_bonus + equity_score + benefit_score
        + skill_bonus + title_bonus + fresh_bonus,
        2,
    )
    job.score_breakdown = breakdown

    if job.needs_verification:
        # Real, rankable score already computed above; hard_pass stays False
        # and the ORIGINAL hard_filter reasons (all "unconfirmed by source")
        # are preserved so the UI can explain why - distinct from the
        # min_matched_skills reason below, which is a different, computed
        # judgment, not missing data.
        job.hard_pass = False
        breakdown["hard_fail_reasons"] = reasons
        return job

    # No quote characters in either message below on purpose: db.hard_fail_reason_counts
    # groups reasons by splitting on the first "'" (else first space) - a quoted
    # skill list would break that into an unstable, per-job key instead of one
    # consistent bucket.
    if len(matched_skills) < hard.min_matched_skills:
        job.hard_pass = False
        matched_str = ", ".join(matched_skills) if matched_skills else "none"
        breakdown["hard_fail_reasons"] = [
            f"skill_match {len(matched_skills)} < required {hard.min_matched_skills} (matched: {matched_str})"
        ]
        return job

    anchor_matches = match_anchor_signals(job.description, profile)
    if hard.require_anchor_skill and profile.anchor_tools + profile.anchor_skills and not anchor_matches:
        # Enough matches by COUNT, but none of the operator's own
        # differentiators showed up (§ DECISIONS.md 2026-08-01) - e.g. bare
        # "python" + "llm", which a Machine Learning Engineer JD matches just
        # as easily as a genuine fit, with none of the operator-named overlap.
        job.hard_pass = False
        matched_str = ", ".join(matched_skills) if matched_skills else "none"
        breakdown["hard_fail_reasons"] = [
            f"anchor_skill matched ({matched_str}) but none are an anchor tool/skill from the profile"
        ]
        return job
    breakdown["anchor_matches"] = anchor_matches

    if hard.require_target_role_family and not matches_target_role_family(job.title):
        # Scope gate (§ DECISIONS.md 2026-08-12): the JD can share plenty of
        # vocabulary with the profile while being a different discipline -
        # Machine Learning Engineer and Data Scientist roles were dominating
        # Open on shared analytics language alone. Title is the only place
        # the DISCIPLINE is actually stated.
        job.hard_pass = False
        breakdown["hard_fail_reasons"] = [
            f"role_family title ({job.title}) is outside the target families"
        ]
        return job

    job.hard_pass = True
    return job
