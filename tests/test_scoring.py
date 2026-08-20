import datetime as dt
import re

import pytest

from candidate_profile import CandidateProfile, DEFAULT_PROFILE
from config import EQUITY_SCORE_CAP, HardCriteria
from db import Job
from scoring import (
    contradicts_remote,
    excluded_industry,
    freshness_bonus,
    hard_filter,
    match_profile_skills,
    missing_skills,
    needs_verification,
    parse_benefits,
    parse_equity_signals,
    parse_posted_age_days,
    parse_salary,
    score_job,
    title_matches_profile,
)
from sources.linkedin_apify_source import LinkedInApifySource


def _job(**overrides) -> Job:
    base = dict(
        source="test",
        external_id="1",
        title="Engineer",
        company="Acme",
        url="https://example.com/1",
        description="",
        salary_min=None,
        salary_max=None,
        work_type="fulltime",
        location_mode="remote",
        location_country="USA",
    )
    base.update(overrides)
    return Job(**base)


# --- parse_salary ---------------------------------------------------------

def test_parse_salary_dollar_range():
    assert parse_salary("Base pay $200,000 - $240,000 per year") == (200000, 240000)


def test_parse_salary_k_range():
    assert parse_salary("Compensation $200k-$240k") == (200000, 240000)


def test_parse_salary_up_to():
    assert parse_salary("Pay up to $250,000 depending on experience") == (None, 250000)


def test_parse_salary_plus():
    assert parse_salary("$220,000+ base") == (220000, None)


def test_parse_salary_comma_formatted_number_with_redundant_k_suffix_is_not_multiplied():
    # Real Dynamo Technologies posting (found 2026-07-30): "105,000k to
    # 125,000k annually" is a JD-authoring typo for $105,000-$125,000, not
    # $105 million - a comma-grouped number is already a full thousands
    # value, so a trailing "k" there must not multiply it again.
    assert parse_salary("105,000k to 125,000k annually") == (105000, 125000)


def test_parse_salary_none_found():
    assert parse_salary("Competitive salary, benefits included") == (None, None)


def test_parse_salary_does_not_truncate_comma_grouped_numbers():
    # Regression lock (2026-08-02): a real LinkedIn posting's
    # "$157,200.00/yr - $214,800.00/yr" parsed to (200, 214800), not
    # (157200, 214800) - the old `_NUMBER` regex's two overlapping
    # alternatives let the engine backtrack into matching only the last 3
    # digits after the comma when the "per annum/pa" suffix pattern failed
    # to match starting from the number's true start. Silently undercounted
    # real postings by 100x+ with no error - found via a `jobs.db` salary
    # query returning obviously-impossible values like salary_min=200.
    assert parse_salary("$157,200.00/yr - $214,800.00/yr") == (157200, 214800)
    assert parse_salary("$85,000.00/yr - $95,000.00/yr") == (85000, 95000)
    assert parse_salary("$1,234,567 - $2,000,000") == (1234567, 2000000)


def test_parse_salary_k_range_no_currency():
    assert parse_salary("Compensation 120k-150k") == (120000, 150000)


def test_parse_salary_per_annum_and_en_dash():
    assert parse_salary("Salary 120,000–150,000 per annum") == (120000, 150000)


def test_parse_salary_from_minimum():
    assert parse_salary("From $100k") == (100000, None)


def test_parse_salary_ignores_years_of_experience_plus_pattern():
    # The real bug (2026-07-30, found live against a Greenhouse posting):
    # "Minimum requirements 5+ years of experience" parsed as salary_min=5 -
    # _SALARY_PLUS_RE's "N+" pattern has no currency requirement at all, so
    # "5+" alone (from "5+ years") matched. A bare number+"+" with no $/k/
    # per-year marker must not be treated as a salary.
    assert parse_salary("Minimum requirements 5+ years of experience in credit analysis.") == (None, None)


def test_parse_salary_ignores_bare_number_range_with_no_currency_marker():
    # Same bug class applied to the range pattern - "3-5 years" or "10-15
    # people" must not parse as a salary range just because two bare numbers
    # separated by a dash happen to appear.
    assert parse_salary("Team of 10-15 people, 3-5 years of experience required.") == (None, None)


def test_parse_salary_still_accepts_a_real_range_with_currency_marker():
    # Guard against overcorrecting: a genuine salary must still parse.
    assert parse_salary("Compensation: $150,000 - $180,000 per year.") == (150000, 180000)


def test_parse_salary_range_with_k_and_label():
    assert parse_salary("Compensation Range: $170K - $210K") == (170000, 210000)


def test_parse_salary_other_currency():
    # Non-USD currency listings are out of scope for US-only parsing.
    assert parse_salary("£60k - £70k") == (None, None)


# --- posting freshness (2026-08-18) -----------------------------------------

_NOW = dt.datetime(2026, 8, 18, tzinfo=dt.timezone.utc)


def test_parse_posted_age_days_relative_english():
    # LinkedIn's legacy actor - already bucketed by LinkedIn itself.
    assert parse_posted_age_days("1 day ago", _NOW) == 1
    assert parse_posted_age_days("2 weeks ago", _NOW) == 14
    assert parse_posted_age_days("1 week ago", _NOW) == 7
    assert parse_posted_age_days("5 months ago", _NOW) == 150
    assert parse_posted_age_days("3 hours ago", _NOW) == 0
    assert parse_posted_age_days("43 minutes ago", _NOW) == 0
    assert parse_posted_age_days("30+ days ago", _NOW) == 30


def test_parse_posted_age_days_absolute_date():
    # curious_coder (the remote-filtered LinkedIn actor) - date only, no time.
    assert parse_posted_age_days("2026-08-14", _NOW) == 4


def test_parse_posted_age_days_full_iso8601_with_offset():
    # Greenhouse - full timestamp with a UTC offset. 2026-08-16T10:00:00-04:00
    # is 2026-08-16T14:00:00 UTC, i.e. 1 full day before _NOW (2026-08-18
    # midnight UTC) - the offset must be applied before diffing, not ignored.
    assert parse_posted_age_days("2026-08-16T10:00:00-04:00", _NOW) == 1


def test_parse_posted_age_days_none_when_unavailable_or_unparseable():
    assert parse_posted_age_days(None, _NOW) is None
    assert parse_posted_age_days("", _NOW) is None
    assert parse_posted_age_days("Reposted", _NOW) is None


def test_freshness_bonus_tiers():
    assert freshness_bonus(0) == 15.0
    assert freshness_bonus(3) == 15.0
    assert freshness_bonus(4) == 10.0
    assert freshness_bonus(7) == 10.0
    assert freshness_bonus(8) == 5.0
    assert freshness_bonus(14) == 5.0
    assert freshness_bonus(15) == 2.0
    assert freshness_bonus(30) == 2.0
    assert freshness_bonus(31) == 0.0
    assert freshness_bonus(365) == 0.0


def test_freshness_bonus_none_age_is_zero_not_a_penalty():
    # Soft criteria only ever add - unknown age must not be treated as "old"
    # (which would be a penalty), just as "no bonus".
    assert freshness_bonus(None) == 0.0


_FRESHNESS_FIXTURE = dict(
    title="Senior Analytics Engineer",
    salary_min=210000,
    description="Build our semantic layer in dbt and LookML, query Snowflake with SQL.",
)


def test_score_job_without_now_has_no_freshness_signal():
    # Default backward-compatible behavior: no `now` means score_job stays a
    # pure function of its other arguments, same as every call site before
    # this feature existed.
    job = _job(posted_at="1 day ago", **_FRESHNESS_FIXTURE)
    scored = score_job(job, HardCriteria(), DEFAULT_PROFILE)
    assert "freshness_bonus" not in scored.score_breakdown
    assert "posted_age_days" not in scored.score_breakdown


def test_score_job_with_now_adds_freshness_bonus_to_the_total():
    job_fresh = _job(external_id="a", posted_at="1 day ago", **_FRESHNESS_FIXTURE)
    job_old = _job(external_id="b", posted_at="6 months ago", **_FRESHNESS_FIXTURE)

    fresh = score_job(job_fresh, HardCriteria(), DEFAULT_PROFILE, now=_NOW)
    old = score_job(job_old, HardCriteria(), DEFAULT_PROFILE, now=_NOW)

    assert fresh.score_breakdown["posted_age_days"] == 1
    assert fresh.score_breakdown["freshness_bonus"] == 15.0
    assert old.score_breakdown["freshness_bonus"] == 0.0
    assert fresh.score - old.score == 15.0, "freshness must move the TOTAL score, not just the breakdown"


def test_score_job_freshness_does_not_affect_hard_pass():
    # Purely additive - an old posting must still hard_pass if everything
    # else qualifies (soft criteria never gate).
    job = _job(posted_at="1 year ago", **_FRESHNESS_FIXTURE)
    scored = score_job(job, HardCriteria(), DEFAULT_PROFILE, now=_NOW)
    assert scored.hard_pass is True


# --- parse_benefits --------------------------------------------------------

def test_family_medical_detected_when_nearby():
    text = "We provide medical insurance for you and your family at no cost."
    assert "family_medical" in parse_benefits(text)


def test_family_medical_not_detected_when_unrelated():
    text = (
        "Medical device experience is a plus. " + ("filler " * 40) +
        "We host a family picnic every summer."
    )
    assert "family_medical" not in parse_benefits(text)


def test_other_benefits_detected():
    text = "We offer dental, vision, a 401k match, and unlimited PTO."
    found = parse_benefits(text)
    assert set(found) == {"dental", "vision", "401k_match", "unlimited_pto"}


def test_no_benefits_in_plain_text():
    assert parse_benefits("Great opportunity to join our team.") == []


def test_equity_no_longer_counted_as_a_benefit():
    # equity moved out of BENEFIT_WEIGHTS into the equity system (no double-count).
    assert "equity" not in parse_benefits("Competitive salary and equity offered.")


# --- parse_equity_signals -------------------------------------------------

def test_equity_signals_detects_base_equity():
    assert "equity" in parse_equity_signals("We offer competitive pay and stock options.")
    assert "equity" in parse_equity_signals("You'll receive RSUs on a 4-year vest.")


def test_equity_signals_detects_meaningful_equity_and_percent():
    found = parse_equity_signals("Meaningful equity — 0.5% ownership for the founding team.")
    assert "meaningful_equity" in found
    assert "equity_pct" in found
    assert "founding" in found


def test_equity_signals_detects_stage():
    found = parse_equity_signals("We are a Series B, venture-backed startup.")
    assert "series_b" in found
    assert "well_funded" in found


def test_equity_signals_empty_when_none_present():
    assert parse_equity_signals("A stable enterprise role at a large public company.") == []


# --- hard_filter -----------------------------------------------------------

def test_hard_filter_passes_when_all_criteria_met():
    hard = HardCriteria()
    job = _job(salary_min=210000)
    passes, reasons = hard_filter(job, hard)
    assert passes
    assert reasons == []


def test_hard_filter_rejects_low_salary():
    hard = HardCriteria(min_salary=200_000)
    job = _job(salary_min=150000)
    passes, reasons = hard_filter(job, hard)
    assert not passes
    assert any("salary" in r for r in reasons)


def test_hard_filter_rejects_hybrid_location():
    hard = HardCriteria()
    job = _job(salary_min=210000, location_mode="hybrid")
    passes, reasons = hard_filter(job, hard)
    assert not passes
    assert any("location_mode" in r for r in reasons)


def test_hard_filter_rejects_remote_claim_contradicted_by_day_count_schedule():
    # The exact real-world bug (2026-07-28): Pilot.com's listing had no literal
    # "hybrid"/"onsite" word anywhere - just a day count + "in office" - and the
    # source defaulted its location_mode field to "remote" (the request filter).
    # This must be caught at the hard gate regardless of the stored field.
    hard = HardCriteria()
    job = _job(
        salary_min=220000,
        location_mode="remote",  # what the (buggy) source field claimed
        description="San Francisco, CA. 3 days/week in office - Mondays, Tuesdays, and Thursdays.",
    )
    passes, reasons = hard_filter(job, hard)
    assert not passes
    assert any("location_mode" in r for r in reasons)


def test_hard_filter_rejects_remote_claim_contradicted_by_bare_onsite_keyword():
    # The exact real-world bug (2026-07-29): Crunchyroll's listing had no day
    # count at all - just "We are considering applicants for the location of
    # Los Angeles, CA (onsite)." The day-count-only check (previous test)
    # doesn't catch this; contradicts_remote's bare-keyword detection must.
    hard = HardCriteria()
    job = _job(
        salary_min=220000,
        location_mode="remote",
        description="We are considering applicants for the location of Los Angeles, CA (onsite).",
    )
    passes, reasons = hard_filter(job, hard)
    assert not passes
    assert any("location_mode" in r for r in reasons)


def test_contradicts_remote_true_for_bare_onsite_and_hybrid_keywords():
    assert contradicts_remote("Los Angeles, CA (onsite).") is True
    assert contradicts_remote("This is a hybrid role based in our office.") is True
    assert contradicts_remote("Fully remote, work from anywhere.") is False


def test_hard_filter_does_not_double_flag_an_already_correct_hybrid():
    # If location_mode is already correctly "hybrid" (not falsely "remote"),
    # the day-count text check must not ALSO fire - one reason, not two.
    hard = HardCriteria()
    job = _job(
        salary_min=220000,
        location_mode="hybrid",
        description="3 days/week in office.",
    )
    passes, reasons = hard_filter(job, hard)
    assert not passes
    location_reasons = [r for r in reasons if "location_mode" in r]
    assert len(location_reasons) == 1


def test_hard_filter_allows_genuinely_remote_job_with_no_schedule_language():
    hard = HardCriteria()
    job = _job(salary_min=220000, location_mode="remote", description="Fully remote, work from anywhere.")
    passes, _ = hard_filter(job, hard)
    assert passes


def test_hard_filter_day_count_check_only_applies_when_remote_is_required():
    # If the operator's own criteria allow hybrid, a day-count schedule isn't a
    # conflict at all - only relevant when hard.location_mode is "remote".
    hard = HardCriteria(location_mode="hybrid")
    job = _job(
        salary_min=220000,
        location_mode="hybrid",
        description="3 days/week in office.",
    )
    passes, _ = hard_filter(job, hard)
    assert passes


def test_hard_filter_rejects_part_time():
    hard = HardCriteria()
    job = _job(salary_min=210000, work_type="parttime")
    passes, reasons = hard_filter(job, hard)
    assert not passes
    assert any("work_type" in r for r in reasons)


def test_linkedin_apify_source_skips_low_yield_keywords(monkeypatch):
    # § DECISIONS.md 2026-08-02: data analyst/bi engineer/business
    # intelligence engineer have a real, measured ~0% $200K clear rate
    # (db.role_salary_stats against 746 real postings) - skipped for this
    # PAID source specifically so a Search click doesn't spend Apify credit
    # on keywords that essentially never produce a hard_pass.
    source = LinkedInApifySource(token="fake", actor_id="fake")
    called_with = []

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return []

    def fake_post(url, params=None, json=None, timeout=None):
        called_with.append(json["searchQuery"])
        return FakeResponse()

    monkeypatch.setattr("sources.linkedin_apify_source.httpx.post", fake_post)
    source.search([
        "analytics engineer", "data analyst", "bi engineer",
        "business intelligence engineer", "ai engineer",
    ])

    assert called_with == ["analytics engineer", "ai engineer"]


def test_linkedin_apify_source_infers_hybrid_from_description():
    source = LinkedInApifySource(token="fake", actor_id="fake")
    item = {
        "id": "pilot-123",
        "title": "Senior Analytics Engineer",
        "companyName": "Pilot.com",
        "url": "https://example.com/jobs/pilot-123",
        "descriptionText": "Hybrid role, 3 days/week in our San Francisco office. $210,000-$230,000.",
        "location": "San Francisco, CA (Hybrid)",
        "workplaceType": None,
        "employmentType": "F",
    }
    job = source._map_item(item)

    assert job.location_mode == "hybrid"
    assert job.location_raw == "San Francisco, CA (Hybrid)"


def test_linkedin_apify_source_infers_hybrid_from_day_count_alone_no_hybrid_word():
    # The exact real Pilot.com bug: no literal "hybrid"/"onsite" word anywhere,
    # location has no "(Hybrid)" suffix either - just a day count + "in office".
    source = LinkedInApifySource(token="fake", actor_id="fake")
    item = {
        "id": "pilot-real",
        "title": "Senior Analytics Engineer",
        "companyName": "Pilot.com",
        "url": "https://example.com/jobs/pilot-real",
        "descriptionText": (
            "San Francisco, CA (3 days/week in office - Mondays, Tuesdays, and Thursdays). "
            "$210,000-$230,000."
        ),
        "location": "San Francisco, CA",
        "workplaceType": None,
        "employmentType": "F",
    }
    job = source._map_item(item)

    assert job.location_mode == "hybrid"


def test_linkedin_apify_source_infers_onsite_from_description():
    source = LinkedInApifySource(token="fake", actor_id="fake")
    item = {
        "id": "local-1",
        "title": "AI Engineer",
        "companyName": "Edra",
        "url": "https://linkedin.example/jobs/1",
        "descriptionText": "We work together in person five days a week from our offices in the West Village (New York).",
        "location": "New York, NY",
        "workplaceType": None,
        "employmentType": "F",
    }
    job = source._map_item(item)

    assert job.location_mode == "onsite"
    assert "New York" in job.location_raw


def test_linkedin_apify_source_returns_none_when_workplace_type_and_text_both_uninformative():
    # The real Zoox bug (2026-07-29): workplaceType null, description has ZERO
    # location/schedule signal anywhere (confirmed against the live LinkedIn
    # page via screenshot - it shows a "Hybrid" pill, but that's a UI badge
    # this source's data never carries). Must NOT default to the requested
    # filter ("remote") - that's exactly the false assumption that shipped a
    # hybrid role as a confirmed remote match. None means "can't confirm."
    source = LinkedInApifySource(token="fake", actor_id="fake")
    item = {
        "id": "zoox-like",
        "title": "Lead Analytics Engineer",
        "companyName": "SomeCo",
        "url": "https://linkedin.example/jobs/2",
        "descriptionText": "Own the data foundation. 10+ years experience. Python and SQL required.",
        "location": "Foster City, CA",
        "workplaceType": None,
        "employmentType": "F",
    }
    job = source._map_item(item)
    assert job.location_mode is None


def test_hard_filter_allows_unconfirmed_location_mode():
    # REVERSED 2026-08-12 (§ DECISIONS.md, operator directive): the gate is
    # now "must not be CONFIRMED hybrid/onsite", not "must be confirmed
    # remote". Measured on real data: 113 of the >=$200k jobs this blocked
    # were merely unconfirmed vs ~30 genuinely onsite/hybrid - silence was
    # burying well-paid roles (Netflix $380k-$610k) over a fact nobody
    # asserted.
    hard = HardCriteria()
    job = _job(salary_min=220000, location_mode=None)
    passes, reasons = hard_filter(job, hard)
    assert passes
    assert not any("location_mode" in r for r in reasons)


def test_hard_filter_still_rejects_a_confirmed_hybrid_or_onsite_role():
    # The other half of the directive: stated hybrid/onsite still rejects.
    hard = HardCriteria()
    for mode in ("hybrid", "onsite"):
        passes, reasons = hard_filter(_job(salary_min=220000, location_mode=mode), hard)
        assert not passes
        assert any("location_mode" in r for r in reasons)


def test_hard_filter_falls_back_to_description_salary():
    hard = HardCriteria()
    job = _job(salary_min=None, description="Pay up to $250,000 annually")
    passes, _ = hard_filter(job, hard)
    assert passes


# --- needs_verification / score_job branching ------------------------------

def test_needs_verification_true_when_every_reason_is_unconfirmed():
    reasons = [
        "work_type unconfirmed by source, required fulltime",
        "location_mode unconfirmed by source, required remote",
    ]
    assert needs_verification(reasons) is True


def test_needs_verification_false_when_any_reason_is_confirmed_wrong():
    reasons = [
        "salary 150000 < required 200000",
        "location_mode unconfirmed by source, required remote",
    ]
    assert needs_verification(reasons) is False


def test_needs_verification_false_when_no_reasons():
    assert needs_verification([]) is False


def test_score_job_unconfirmed_only_gets_real_score_not_zero():
    # Faire-style case: salary confirmed and above floor, but work_type and
    # location fields are simply absent from the source, not contradicted.
    hard = HardCriteria()
    profile = CandidateProfile(
        title_keywords=("analytics engineer",), skills=("sql", "dbt"), summary="x"
    )
    job = _job(
        title="Senior Analytics Engineer",
        description="Build dbt models with SQL and Python.",
        salary_min=220000,
        work_type=None,
        location_mode=None,
        location_country=None,
    )
    score_job(job, hard, profile)
    assert job.needs_verification is True
    assert job.hard_pass is False
    assert job.score > 0
    reasons = job.score_breakdown["hard_fail_reasons"]
    assert all("unconfirmed" in r for r in reasons)


def test_score_job_confirmed_wrong_still_zeroes_score():
    hard = HardCriteria()
    profile = CandidateProfile(
        title_keywords=("analytics engineer",), skills=("sql", "dbt"), summary="x"
    )
    job = _job(
        title="Senior Analytics Engineer",
        description="Join our telehealth company. dbt SQL.",
        salary_min=220000,
    )
    score_job(job, hard, profile)
    assert job.needs_verification is False
    assert job.hard_pass is False
    assert job.score == 0.0


def test_score_job_confirmed_low_salary_is_not_needs_verification_even_with_other_unconfirmed():
    hard = HardCriteria(min_salary=200_000)
    profile = CandidateProfile(
        title_keywords=("analytics engineer",), skills=("sql", "dbt"), summary="x"
    )
    job = _job(
        title="Senior Analytics Engineer",
        description="Build dbt models with SQL.",
        salary_min=150000,
        work_type=None,
        location_mode=None,
        location_country=None,
    )
    score_job(job, hard, profile)
    assert job.needs_verification is False
    assert job.score == 0.0


# --- excluded industry (healthcare) ---------------------------------------

def test_excluded_industry_detects_healthcare():
    job = _job(description="Join our telehealth company building clinical data pipelines.")
    assert excluded_industry(job) is not None


def test_excluded_industry_matches_company_or_title():
    assert excluded_industry(_job(company="Acme Health System")) == "health system"
    assert excluded_industry(_job(title="Analytics Engineer, Oncology")) == "oncology"


def test_medical_insurance_benefit_is_not_an_excluded_industry():
    # The key false-positive to avoid: a benefits blurb naming medical/health
    # coverage must NOT read as a healthcare EMPLOYER.
    job = _job(
        title="Analytics Engineer",
        description="Great fintech role. Medical and health insurance for you and your family, plus dental.",
    )
    assert excluded_industry(job) is None


def test_healthcare_job_hard_fails_even_if_otherwise_qualified():
    hard = HardCriteria()
    job = _job(
        title="Senior Analytics Engineer",
        salary_min=220000,
        description="Series B healthtech company. Build dbt and SQL models on patient care data.",
    )
    scored = score_job(job, hard, DEFAULT_PROFILE)
    assert scored.hard_pass is False
    assert any("excluded_industry" in r for r in scored.score_breakdown["hard_fail_reasons"])


# --- score_job ---------------------------------------------------------

def test_score_job_zero_when_hard_fail():
    hard = HardCriteria(min_salary=200_000)
    job = _job(salary_min=100000)
    scored = score_job(job, hard)
    assert scored.hard_pass is False
    assert scored.score == 0.0
    assert scored.benefits == []


def test_score_job_adds_salary_and_benefit_bonus():
    hard = HardCriteria(min_salary=200_000)
    job = _job(
        title="Analytics Engineer",
        salary_min=210000,
        description=(
            "Medical insurance for family, plus dental and a 401k match. "
            "You'll write SQL and use dbt daily."
        ),
    )
    scored = score_job(job, hard)
    assert scored.hard_pass is True
    # base 100 + salary bonus 10 ((210000-200000)/1000) + family_medical 25 +
    # dental 12 + 401k_match 8 + skill_bonus 8 (sql, dbt) + title_bonus 15
    assert scored.score == 178.0
    assert set(scored.benefits) == {"family_medical", "dental", "401k_match"}


def test_score_job_salary_bonus_is_capped():
    hard = HardCriteria()
    job = _job(title="Analytics Engineer", salary_min=500000)
    scored = score_job(job, hard)
    assert scored.score_breakdown["salary_bonus"] == 100.0


# --- profile matching -------------------------------------------------

def test_match_profile_skills_finds_mentioned_skills():
    text = "You'll write SQL against Snowflake and Redshift, and build in dbt and LookML."
    found = match_profile_skills(text, DEFAULT_PROFILE)
    assert set(found) == {"sql", "snowflake", "redshift", "dbt", "lookml"}


def test_match_profile_skills_short_skill_is_whole_word_only():
    # "bi" must not match inside "big"
    assert "bi" not in match_profile_skills("We are a big company.", DEFAULT_PROFILE)
    assert "bi" in match_profile_skills("Strong BI background required.", DEFAULT_PROFILE)


def test_missing_skills_are_jd_skills_not_on_the_profile():
    # airflow + fivetran are in SKILL_UNIVERSE and this JD, but not in the
    # default profile -> they're "skills to add". dbt IS on the profile -> not missing.
    text = "You'll orchestrate with Airflow, sync via Fivetran, and build dbt models."
    found = missing_skills(text, DEFAULT_PROFILE)
    assert "airflow" in found
    assert "fivetran" in found
    assert "dbt" not in found  # already on the profile


def test_missing_skills_ignores_words_outside_the_universe():
    # A random non-skill word must not show up as a missing skill.
    assert missing_skills("We value curiosity and teamwork.", DEFAULT_PROFILE) == []


def test_score_job_records_missing_skills():
    hard = HardCriteria()
    job = _job(
        title="Analytics Engineer",
        salary_min=210000,
        description="Build dbt models, write SQL, orchestrate with Airflow and Dagster.",
    )
    scored = score_job(job, hard, DEFAULT_PROFILE)
    assert "airflow" in scored.missing_skills
    assert "dagster" in scored.missing_skills
    assert "airflow" not in scored.matched_skills  # missing != matched


def test_title_matches_profile():
    assert title_matches_profile("Senior Analytics Engineer", DEFAULT_PROFILE)
    assert title_matches_profile("Staff AI Engineer", DEFAULT_PROFILE)
    assert not title_matches_profile("Backend Engineer", DEFAULT_PROFILE)
    # "data analyst" was dropped from the target families 2026-08-12
    # (measured 0% $200k clear rate) - it must no longer earn a title bonus.
    assert not title_matches_profile("Data Analyst", DEFAULT_PROFILE)


def test_score_job_adds_profile_skill_and_title_bonus():
    hard = HardCriteria()
    job = _job(
        title="Senior Analytics Engineer",
        salary_min=210000,
        description="Build our semantic layer in dbt and LookML, query Snowflake and Redshift with SQL.",
    )
    scored = score_job(job, hard, DEFAULT_PROFILE)
    assert scored.hard_pass is True
    assert set(scored.matched_skills) == {"dbt", "lookml", "snowflake", "redshift", "sql", "semantic layer"}
    assert scored.score_breakdown["profile_title_bonus"] == 15.0
    # 6 matched skills * 4.0/skill = 24, under the 40-point cap
    assert scored.score_breakdown["profile_skill_bonus"] == 24.0


def test_score_job_title_never_gates_hard_pass():
    # A "Staff Platform Engineer" title alone must not exclude a job anymore
    # - title is scoring-only. This one still fails hard_pass, but because it
    # has too few matched skills (0 < the default floor of 2), not because of
    # its title.
    hard = HardCriteria(min_salary=200_000)
    job = _job(title="Staff Platform Engineer", salary_min=260000)
    scored = score_job(job, hard, DEFAULT_PROFILE)
    assert scored.hard_pass is False
    # base 100 + salary bonus 60 ((260000-200000)/1000, under the 100 cap);
    # no skills mentioned in the (empty) description, no title bonus either
    assert scored.score == 160.0
    assert any("skill_match" in r for r in scored.score_breakdown["hard_fail_reasons"])


def test_score_job_records_skills_for_hard_failed_jobs():
    hard = HardCriteria()
    job = _job(
        title="Backend Engineer",
        salary_min=190000,
        description="Build our semantic layer in dbt and SQL on Snowflake.",
    )
    scored = score_job(job, hard, DEFAULT_PROFILE)
    assert scored.hard_pass is False
    assert set(scored.matched_skills) == {"dbt", "snowflake", "sql", "semantic layer"}
    assert scored.missing_skills == []


def test_score_job_close_family_title_with_skill_overlap_no_longer_passes():
    # REVERSED 2026-08-12 (§ DECISIONS.md, operator directive to scope to the
    # AI/Analytics Engineer families). This used to assert the opposite and
    # was the core justification for dropping the title gate on 2026-07-25:
    # "different title, same skillset" should pass.
    #
    # The trade-off is real and worth stating: a genuinely well-matched role
    # under an unusual title is now excluded. It was accepted because the
    # measured cost of NOT scoping was larger - after the location gate was
    # relaxed, Open held 47 jobs of which only 7 were in the target families;
    # the rest were Machine Learning Engineer / Data Scientist / Data
    # Engineer roles passing on shared analytics vocabulary alone.
    # Title is the only field that states the DISCIPLINE.
    hard = HardCriteria()
    job = _job(
        title="Senior Backend Engineer",
        salary_min=210000,
        description="Write SQL against Snowflake and Redshift, build in dbt and LookML.",
    )
    scored = score_job(job, hard, DEFAULT_PROFILE)
    assert scored.hard_pass is False
    assert any("role_family" in r for r in scored.score_breakdown["hard_fail_reasons"])
    # Still fully scored and visible - only the pass flag changes, same
    # informational principle as every other non-salary gate.
    assert set(scored.matched_skills) == {"sql", "snowflake", "redshift", "dbt", "lookml"}
    assert scored.score > 0


def test_score_job_rejects_generic_ai_overlap_with_no_anchor_match():
    # The real false positive this was built for (2026-08-01): a "Machine
    # Learning Engineer" JD clears min_matched_skills on purely generic,
    # widely-shared AI vocabulary (llm, gpt) - present in profile.skills but
    # NOT in DEFAULT_PROFILE's anchor_tools/anchor_skills - with none of the
    # operator's actual differentiators. Must not hard_pass even though the
    # plain skill count is satisfied.
    hard = HardCriteria()
    job = _job(
        title="Senior Machine Learning Engineer, GenAI Platform",
        salary_min=280000,
        description="Build and train large-scale LLM systems; experience with GPT-style "
                     "architectures and distributed model training required.",
    )
    scored = score_job(job, hard, DEFAULT_PROFILE)
    assert set(scored.matched_skills) >= {"llm", "gpt"}
    assert scored.hard_pass is False
    assert any("anchor_skill" in r for r in scored.score_breakdown["hard_fail_reasons"])


def test_score_job_passes_generic_ai_overlap_plus_one_anchor_match():
    # Same generic overlap as above, but the JD also mentions a real anchor
    # ("dbt") - now a genuine signal of fit, not just shared buzzwords.
    hard = HardCriteria()
    job = _job(
        title="AI Engineer",
        salary_min=220000,
        description="Build LLM-powered agents; own the dbt models feeding the agent's context.",
    )
    scored = score_job(job, hard, DEFAULT_PROFILE)
    assert scored.hard_pass is True
    assert "dbt" in scored.score_breakdown["anchor_matches"]


def test_score_job_anchor_gate_is_a_noop_when_profile_has_no_anchors():
    # A profile that never set anchor_tools/anchor_skills must fall back to
    # the plain skill-count gate, not hard-exclude everything.
    hard = HardCriteria()
    profile = CandidateProfile(skills=("llm", "gpt"), anchor_tools=(), anchor_skills=())
    job = _job(title="AI Engineer", salary_min=220000, description="Build LLM systems using GPT models.")
    scored = score_job(job, hard, profile)
    assert scored.hard_pass is True


def test_score_job_below_skill_floor_still_scores_informationally():
    # A title-matching job that clears salary/location/type but mentions
    # none of the profile's skills should still fail hard_pass (too generic
    # a description to call it a real match) while remaining visible/ranked,
    # not silently zeroed like a salary/location failure.
    hard = HardCriteria(min_salary=200_000)
    job = _job(title="Senior Analytics Engineer", salary_min=210000, description="Great team, great mission.")
    scored = score_job(job, hard, DEFAULT_PROFILE)
    assert scored.hard_pass is False
    assert scored.matched_skills == []
    # base 100 + salary bonus 10 + title bonus 15 (title does match); no skills, no benefits
    assert scored.score == 125.0


def test_score_job_profile_skill_bonus_is_capped():
    hard = HardCriteria()
    skills = tuple(f"skill{i}" for i in range(15))  # 15 * 4.0 = 60, over the 40 cap
    profile = CandidateProfile(skills=skills)
    job = _job(title="Analytics Engineer", salary_min=210000, description=" ".join(skills))
    scored = score_job(job, hard, profile)
    assert scored.score_breakdown["profile_skill_bonus"] == 40.0


# --- equity scoring (the pivot) -------------------------------------------

def test_score_job_adds_equity_score_and_records_signals():
    hard = HardCriteria()
    job = _job(
        title="Analytics Engineer",
        salary_min=210000,
        description="Series A startup. Meaningful equity for early hires. Build in dbt with SQL.",
    )
    scored = score_job(job, hard, DEFAULT_PROFILE)
    assert scored.hard_pass is True
    assert "series_a" in scored.equity_signals
    assert "meaningful_equity" in scored.equity_signals
    assert scored.score_breakdown["equity_score"] > 0


def test_salary_outranks_equity():
    # REVERSED 2026-08-12 (§ DECISIONS.md): this used to assert the opposite -
    # that above the floor an equity-rich job beats a higher-paid one. Operator
    # ruled that out ("equity stays paper money that is never real money"), so
    # cash now wins and EQUITY_SCORE_CAP << SALARY_BONUS_CAP enforces it.
    hard = HardCriteria(min_salary=200_000)
    high_salary_no_equity = _job(
        external_id="a", title="Analytics Engineer", salary_min=280000,
        description="Large public company. Write SQL and dbt models.",
    )
    lower_salary_rich_equity = _job(
        external_id="b", title="Analytics Engineer", salary_min=205000,
        description=(
            "Series B, venture-backed. Founding data hire with meaningful equity, "
            "0.5% ownership. Build our dbt and SQL stack."
        ),
    )
    a = score_job(high_salary_no_equity, hard, DEFAULT_PROFILE)
    b = score_job(lower_salary_rich_equity, hard, DEFAULT_PROFILE)
    assert a.score > b.score, "a $75k higher base must beat an equity story"
    # Equity still counts for SOMETHING - it's a tiebreaker, not deleted.
    assert b.score_breakdown["equity_score"] > 0


def test_equity_no_longer_buys_a_role_past_the_salary_floor():
    # REVERSED 2026-08-12: a $175k founding/Series A role used to clear the
    # relaxed $160k equity floor. That IS the "low paid job for a paper money
    # future" the operator ruled out, so every job is held to min_salary now.
    hard = HardCriteria(min_salary=200_000)
    job = _job(
        title="Founding Analytics Engineer",
        salary_min=175000,
        description="Series A. Founding data hire, meaningful equity. Build our dbt and SQL stack.",
    )
    scored = score_job(job, hard, DEFAULT_PROFILE)
    assert scored.hard_pass is False
    assert any("salary" in r for r in scored.score_breakdown["hard_fail_reasons"])


def test_same_sub_floor_role_without_equity_still_fails():
    hard = HardCriteria(min_salary=200_000)
    job = _job(
        title="Analytics Engineer",
        salary_min=175000,
        description="Established company. Build dbt models and write SQL.",
    )
    scored = score_job(job, hard, DEFAULT_PROFILE)
    assert scored.hard_pass is False
    assert any("salary" in r for r in scored.score_breakdown["hard_fail_reasons"])


def test_bare_equity_mention_does_not_relax_the_floor():
    # Generic "equity"/RSU (standard big-company comp) must NOT drop the floor -
    # only genuine startup upside does.
    hard = HardCriteria(min_salary=200_000)
    job = _job(
        title="Analytics Engineer",
        salary_min=175000,
        description="Large public company. Equity and RSUs offered. dbt and SQL.",
    )
    scored = score_job(job, hard, DEFAULT_PROFILE)
    assert scored.hard_pass is False


def test_equity_role_below_even_the_relaxed_floor_still_fails():
    # $150k is below the $160k equity floor - even strong upside can't save it.
    hard = HardCriteria(min_salary=200_000)
    job = _job(
        title="Founding Analytics Engineer",
        salary_min=150000,
        description="Seed stage. Founding engineer, meaningful equity. dbt and SQL.",
    )
    scored = score_job(job, hard, DEFAULT_PROFILE)
    assert scored.hard_pass is False


def test_equity_score_is_capped():
    hard = HardCriteria()
    job = _job(
        title="Analytics Engineer",
        salary_min=205000,
        description=(
            "Pre-seed, seed round, Series A, Series B, Series C, pre-IPO, venture-backed. "
            "Founding engineer, early employee, ground floor. Meaningful significant equity, "
            "2% ownership. Stock options and RSUs. dbt and SQL."
        ),
    )
    scored = score_job(job, hard, DEFAULT_PROFILE)
    assert scored.score_breakdown["equity_score"] == EQUITY_SCORE_CAP


# --- parse_salary false negatives found on real postings (2026-08-02) -------

def test_parse_salary_accepts_trailing_usd():
    # Real posting: "Base Pay Range (CA Only): $101,600 USD - $127,000 USD".
    # _CURRENCY only matched a PREFIX, so the trailing USD blocked the dash.
    assert parse_salary("Base Pay Range (CA Only): $101,600 USD - $127,000 USD") == (101600, 127000)


def test_parse_salary_ignores_hourly_monthly_words_far_from_the_figure():
    # The old guard scanned the WHOLE posting for hour/monthly and discarded
    # the salary of any job that merely contained them. Real losses: an
    # interview stage ("1 hour") and a job duty ("monthly reporting").
    assert parse_salary(
        "Technical Evaluation in Domain (1 hour). The US base salary range "
        "for this full-time position is $160,000 to $210,000 + equity"
    ) == (160000, 210000)
    assert parse_salary(
        "Build automated monthly reporting processes. The pay range for this "
        "position is $116,400.00 - $194,000.00 Annual (USD)"
    ) == (116400, 194000)


def test_parse_salary_still_rejects_genuine_hourly_and_monthly_pay():
    # The cadence must still win when it actually prices the figure.
    assert parse_salary("This role pays $60 per hour") == (None, None)
    assert parse_salary("Pay: $50 - $60 per hour") == (None, None)
    assert parse_salary("Compensation is $8,000 per month") == (None, None)
    assert parse_salary("$5,000 monthly stipend for wellness") == (None, None)


def test_parse_salary_skips_a_non_currency_range_before_the_real_one():
    # Real posting: "(10-30 minutes) ... base salary range ... $150K - $250K".
    # search() returned the bare "10-30" first; it was correctly rejected as
    # non-currency, but the old code then gave up on ranges entirely.
    assert parse_salary(
        "Screening call (10-30 minutes). The base salary range for this role "
        "is $150K - $250K, depending on experience"
    ) == (150000, 250000)


def test_scoring_overrides_apply_without_reimport(tmp_path, monkeypatch):
    """The Scoring tab edits these two lists while the app is running, so
    scoring must read them off config at CALL time, not bind them at import
    (§ DECISIONS.md 2026-08-19)."""
    import config
    from scoring import excluded_industry, matches_target_role_family

    assert matches_target_role_family("Senior Analytics Engineer") is True
    assert matches_target_role_family("Revenue Operations Manager") is False

    path = tmp_path / "overrides.json"
    original_families = config.TARGET_ROLE_FAMILY_PATTERNS
    original_excludes = config.EXCLUDE_INDUSTRY_KEYWORDS
    try:
        config.save_scoring_overrides(
            (r"revenue operations",), ("gambling",), path=str(path)
        )
        assert matches_target_role_family("Revenue Operations Manager") is True
        assert matches_target_role_family("Senior Analytics Engineer") is False
        job = _job(title="Revenue Operations Manager", description="A gambling company.")
        assert excluded_industry(job) == "gambling"
        # a previously-excluded industry no longer excludes
        assert excluded_industry(_job(title="X", description="A biotech company.")) is None
    finally:
        config.TARGET_ROLE_FAMILY_PATTERNS = original_families
        config.EXCLUDE_INDUSTRY_KEYWORDS = original_excludes


def test_save_scoring_overrides_rejects_an_invalid_regex(tmp_path):
    """A role-family entry is a regex typed by hand - an unbalanced paren must
    fail at save time, not inside scoring on the next job."""
    import config

    path = tmp_path / "overrides.json"
    with pytest.raises(re.error):
        config.save_scoring_overrides(("ai engineer(",), (), path=str(path))
    assert not path.exists()
