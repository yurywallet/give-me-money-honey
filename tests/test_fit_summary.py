import pytest

from candidate_profile import DEFAULT_PROFILE
from db import Job
import fit_summary
from fit_summary import (
    _prompt,
    configured_providers,
    default_generate_fn,
    default_provider,
    generate_fit_summary,
    generate_with,
    provider_configured,
)


def _job(**overrides) -> Job:
    base = dict(
        source="mock",
        external_id="1",
        title="Senior Analytics Engineer",
        company="Acme",
        url="https://example.com/1",
        description="Build our semantic layer in dbt and LookML.",
        matched_skills=["dbt", "lookml"],
    )
    base.update(overrides)
    return Job(**base)


def test_generate_fit_summary_uses_injected_generate_fn():
    calls = []

    def fake_generate(job, profile):
        calls.append((job.title, profile.summary))
        return "fake fit summary"

    result = generate_fit_summary(_job(), DEFAULT_PROFILE, generate_fn=fake_generate)

    assert result == "fake fit summary"
    assert calls == [("Senior Analytics Engineer", DEFAULT_PROFILE.summary)]


def test_generate_fit_summary_passes_the_actual_job_and_profile_through():
    # Not just any job/profile - the exact objects, so a real generate_fn can
    # read job.description, job.matched_skills, profile.skills, etc.
    received = {}

    def fake_generate(job, profile):
        received["job"] = job
        received["profile"] = profile
        return "irrelevant"

    job = _job(description="Needs Snowflake and Python.")
    generate_fit_summary(job, DEFAULT_PROFILE, generate_fn=fake_generate)

    assert received["job"] is job
    assert received["profile"] is DEFAULT_PROFILE


# --- default_generate_fn provider dispatch ---------------------------------

def test_default_generate_fn_raises_clearly_when_no_provider_configured(monkeypatch):
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="No LLM provider configured"):
        default_generate_fn(_job(), DEFAULT_PROFILE)


def test_default_generate_fn_prefers_gemini_over_claude_when_both_configured(monkeypatch):
    # § DECISIONS.md 2026-08-02: Anthropic is LAST in the chain. The operator
    # has no Anthropic access, and the old Anthropic-first order made
    # Gemini-only operation accidental - a stray non-empty ANTHROPIC_API_KEY
    # from any source would silently reroute every call to a dead provider.
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    calls = []
    monkeypatch.setattr(fit_summary, "claude_generate_fn", lambda j, p: calls.append("claude") or "ok")
    monkeypatch.setattr(fit_summary, "gemini_generate_fn", lambda j, p: calls.append("gemini") or "ok")

    default_generate_fn(_job(), DEFAULT_PROFILE)

    assert calls == ["gemini"]


def test_default_generate_fn_still_uses_claude_when_it_is_the_only_provider(monkeypatch):
    # Anthropic being last must not mean unreachable - this stays a real
    # multi-provider picker for anyone who only has an Anthropic key.
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    calls = []
    monkeypatch.setattr(fit_summary, "claude_generate_fn", lambda j, p: calls.append("claude") or "ok")

    default_generate_fn(_job(), DEFAULT_PROFILE)

    assert calls == ["claude"]


def test_default_generate_fn_falls_back_to_gemini_when_only_that_is_set(monkeypatch):
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    calls = []
    monkeypatch.setattr(fit_summary, "claude_generate_fn", lambda j, p: calls.append("claude") or "ok")
    monkeypatch.setattr(fit_summary, "gemini_generate_fn", lambda j, p: calls.append("gemini") or "ok")

    default_generate_fn(_job(), DEFAULT_PROFILE)

    assert calls == ["gemini"]


def test_default_generate_fn_prefers_ollama_when_set(monkeypatch):
    # Local model is the deliberate opt-out of API rate limits - it wins even
    # when a cloud key is also present.
    monkeypatch.setenv("OLLAMA_MODEL", "mistral-nemo")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    calls = []
    monkeypatch.setattr(fit_summary, "ollama_generate_fn", lambda j, p: calls.append("ollama") or "ok")
    monkeypatch.setattr(fit_summary, "gemini_generate_fn", lambda j, p: calls.append("gemini") or "ok")
    default_generate_fn(_job(), DEFAULT_PROFILE)
    assert calls == ["ollama"]


def test_prompt_includes_work_experience_and_projects_when_present():
    profile = DEFAULT_PROFILE.__class__(
        title_keywords=DEFAULT_PROFILE.title_keywords,
        skills=DEFAULT_PROFILE.skills,
        summary=DEFAULT_PROFILE.summary,
        work_experience="Data Analyst at Acme, 2020-2023.",
        personal_projects="Built a job-search MCP server.",
    )
    p = _prompt(_job(), profile)
    assert "Work experience:" in p
    assert "Data Analyst at Acme" in p
    assert "Personal projects:" in p
    assert "job-search MCP server" in p


def test_prompt_includes_education_when_present():
    # DEFAULT_PROFILE has real education text by default (§ DECISIONS.md 2026-08-01).
    p = _prompt(_job(), DEFAULT_PROFILE)
    assert "Education:" in p
    assert "Higher School of Economics" in p


def test_prompt_omits_empty_sections():
    profile = DEFAULT_PROFILE.__class__(
        title_keywords=DEFAULT_PROFILE.title_keywords,
        skills=DEFAULT_PROFILE.skills,
        summary=DEFAULT_PROFILE.summary,
        work_experience="",
        personal_projects="",
        education="",
    )
    p = _prompt(_job(), profile)
    assert "Work experience:" not in p
    assert "Personal projects:" not in p
    assert "Education:" not in p


def test_prompt_asks_for_ranked_missing_skills():
    # #2 + #3: the prompt must instruct a ranked list of JD-required skills the
    # candidate lacks, and must feed in the candidate's own skills so the model
    # can compute the gap.
    p = _prompt(_job(), DEFAULT_PROFILE)
    assert "Missing skills" in p
    assert "ranked" in p.lower()
    # the candidate's skills are in the prompt so "missing" is computable
    assert DEFAULT_PROFILE.skills[0] in p


def test_provider_configured_reflects_env(monkeypatch):
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert provider_configured() is False

    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    assert provider_configured() is True


# --- multi-provider (configured_providers / generate_with / default_provider) ---

def test_configured_providers_lists_every_set_provider_in_order(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "mistral-nemo")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    assert configured_providers() == ["gemini", "ollama"]


def test_configured_providers_empty_when_none_set(monkeypatch):
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert configured_providers() == []


def test_generate_with_calls_the_named_providers_generate_fn(monkeypatch):
    # Regression lock (2026-07-28): generate_with must resolve the provider's
    # generate_fn DYNAMICALLY (via globals()), not via a dict of function
    # references captured once at import time - a dict of direct references
    # would silently ignore monkeypatch.setattr(fit_summary, "gemini_generate_fn",
    # fake) and fall through to the REAL function, making a live API call (this
    # actually happened and hung retrying a real rate limit - see DECISIONS.md).
    calls = []
    monkeypatch.setattr(fit_summary, "gemini_generate_fn", lambda j, p: calls.append("gemini") or "g-result")
    monkeypatch.setattr(fit_summary, "ollama_generate_fn", lambda j, p: calls.append("ollama") or "o-result")

    assert generate_with("gemini", _job(), DEFAULT_PROFILE) == "g-result"
    assert generate_with("ollama", _job(), DEFAULT_PROFILE) == "o-result"
    assert calls == ["gemini", "ollama"]


def test_default_provider_matches_default_generate_fn_choice(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "mistral-nemo")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    assert default_provider() == "ollama"

    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    assert default_provider() == "gemini"


def test_default_provider_raises_when_none_configured(monkeypatch):
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="No LLM provider configured"):
        default_provider()
