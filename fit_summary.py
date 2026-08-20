"""LLM-generated qualitative fit explanation, on top of scoring.py's
deterministic score (§ DECISIONS.md 2026-07-25).

Deliberately NOT part of scoring.py or the scheduler's automatic run_once
pass - scoring.py is a pure, dependency-free function by design (see
DECISIONS.md's family-medical entry), and generating a summary for every
job on every search cycle would call the API for jobs nobody ever looks at.
This is opt-in, on-demand (server.get_fit_summary), and cached in
jobs.fit_summary so a given job is only ever summarized once unless
explicitly regenerated - same "cache aggressively, only pay for what's
actually looked at" principle discussed for the (not-yet-built) comp-lookup
agent.

Same injectable-generate_fn pattern as agent.py's reflect_fn: the graph/
function is fully testable with a fake, only a live call to a real provider
needs its API key and costs real tokens.

Two providers, not just one (2026-07-25): the operator's account doesn't
have Anthropic API access (a Claude subscription is not the same product as
an ANTHROPIC_API_KEY), so `default_generate_fn` picks whichever of
ANTHROPIC_API_KEY / GEMINI_API_KEY is actually set rather than hard-requiring
one specific provider - Claude preferred if both happen to be set, purely
for consistency with agent.py's reflect step, not because Gemini is worse
for this task.
"""
from __future__ import annotations

import os
from typing import Callable

from candidate_profile import CandidateProfile
from db import Job

GenerateFn = Callable[[Job, CandidateProfile], str]

_PROMPT_TEMPLATE = """You are helping a job candidate evaluate how well a specific job posting fits their background. Be direct and honest, not promotional - the candidate needs this to make a real decision, not to feel good.

Candidate background:
{summary}
{extra_sections}
Candidate's skills: {skills}

Job: {title} at {company}

Job description:
{description}

Skills already keyword-matched between the profile and this JD: {matched_skills}

Write a concise fit summary in exactly four short parts, each under its own bold header. Start directly with part 1's header — no preamble, no opening line like "Here is a summary".
1. **What the role needs** - synthesize the JD's real requirements in your own words, don't just copy phrases from it.
2. **How the candidate fits** - how their specific experience translates to those requirements; be concrete, reference specific things from the background above, not generic reassurance.
3. **Real gaps** - stretch areas stated plainly; if this is a weak fit, say so, do not oversell.
4. **Missing skills (ranked by impact)** - skills this role clearly REQUIRES that are NOT already in the candidate's skills list above, ranked most-critical/highest-impact first. Number them. For each: the skill, then a few words on why it matters for THIS specific role. Only genuine requirements, not nice-to-haves. If the candidate already has every key skill, say "No critical skill gaps." Do not repeat skills the candidate already has.

Keep the whole thing under 220 words."""


def _prompt(job: Job, profile: CandidateProfile) -> str:
    # Optional, free-text sections - included only when the candidate actually
    # filled them in, so an empty profile field doesn't leave a labeled blank
    # section in the prompt.
    extra = []
    if profile.work_experience.strip():
        extra.append(f"Work experience:\n{profile.work_experience.strip()}")
    if profile.personal_projects.strip():
        extra.append(f"Personal projects:\n{profile.personal_projects.strip()}")
    if profile.education.strip():
        # Some JDs state a minimum degree requirement - included so the model
        # can reason about whether it's met, not just list skills.
        extra.append(f"Education:\n{profile.education.strip()}")
    extra_sections = ("\n\n".join(extra) + "\n") if extra else ""

    return _PROMPT_TEMPLATE.format(
        summary=profile.summary,
        extra_sections=extra_sections,
        skills=", ".join(profile.skills),
        title=job.title,
        company=job.company,
        description=job.description,
        matched_skills=", ".join(job.matched_skills) if job.matched_skills else "none",
    )


def claude_generate_fn(job: Job, profile: CandidateProfile) -> str:
    """Requires ANTHROPIC_API_KEY. Swappable via generate_fit_summary(...,
    generate_fn=...) - tests inject a fake so this is exercised without a
    real API call or cost."""
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-opus-5",
        # max_tokens caps thinking + response text COMBINED, and Opus 5 runs
        # with thinking ON by default (Opus 4.8 did not). The previous 400 was
        # sized for response text alone; left unchanged it would let thinking
        # consume the whole budget and truncate or starve the summary itself.
        # The response target is short (~150 words); the headroom is for
        # thinking, and output billing is per token actually produced, so a
        # ceiling this size costs nothing when the model doesn't use it.
        max_tokens=2000,
        messages=[{"role": "user", "content": _prompt(job, profile)}],
    )
    return response.content[0].text


def gemini_generate_fn(job: Job, profile: CandidateProfile) -> str:
    """Requires GEMINI_API_KEY. Model configurable via GEMINI_MODEL (default
    a fast/cheap tier - this is a short synthesis task, not one that needs
    the largest model available)."""
    from google import genai

    from llm_retry import call_with_rate_limit_retry

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    response = call_with_rate_limit_retry(
        lambda: client.models.generate_content(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            contents=_prompt(job, profile),
        )
    )
    return response.text


def ollama_generate_fn(job: Job, profile: CandidateProfile) -> str:
    """Local model via Ollama - no API key, no rate limits, runs on your machine.
    Requires a running Ollama server (OLLAMA_HOST, default http://localhost:11434)
    and a pulled model (OLLAMA_MODEL, e.g. 'mistral-nemo')."""
    import httpx

    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "mistral-nemo")
    resp = httpx.post(
        f"{host}/api/generate",
        json={"model": model, "prompt": _prompt(job, profile), "stream": False},
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def provider_configured() -> bool:
    """Whether any LLM provider is configured - lets the UI pre-generate summaries
    only when a real call can actually succeed, instead of crashing on one."""
    return bool(
        os.getenv("OLLAMA_MODEL") or os.getenv("ANTHROPIC_API_KEY") or os.getenv("GEMINI_API_KEY")
    )


def default_provider() -> str:
    """Name of the provider default_generate_fn would use: OLLAMA_MODEL (local,
    unlimited) first if set - the deliberate opt-out of API rate limits - then
    GEMINI_API_KEY, then ANTHROPIC_API_KEY last. Callers that need to LABEL a
    summary correctly (db.set_fit_summary's provider key) should call this
    instead of guessing - see the 2026-07-28 mislabeling bug this replaced,
    where callers hardcoded 'gemini' regardless of which provider actually ran.

    Anthropic moved to last 2026-08-02, matching agent.default_reflect_fn -
    see that docstring for why (the operator has no Anthropic access, and the
    old ordering made Gemini-only operation accidental rather than
    guaranteed)."""
    if os.getenv("OLLAMA_MODEL"):
        return "ollama"
    if os.getenv("GEMINI_API_KEY"):
        return "gemini"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "claude"
    raise RuntimeError(
        "No LLM provider configured - set OLLAMA_MODEL, ANTHROPIC_API_KEY, or "
        "GEMINI_API_KEY in .env to generate a fit summary."
    )


def default_generate_fn(job: Job, profile: CandidateProfile) -> str:
    """Picks a provider via default_provider() and generates with it."""
    return generate_with(default_provider(), job, profile)


def generate_fit_summary(
    job: Job, profile: CandidateProfile, generate_fn: GenerateFn = default_generate_fn
) -> str:
    return generate_fn(job, profile)


# --- multi-provider support (operator ask 2026-07-28: see both side by side,
# so a local Ollama summary stays available when the Gemini quota is hit) ---
# Maps provider name -> the generate_fn's NAME (a string), not the function
# object itself. Resolved via globals() at CALL time in generate_with, not
# captured once at import time - a dict of direct function references would
# freeze in the real functions permanently, so monkeypatch.setattr(fit_summary,
# "gemini_generate_fn", fake) in a test would silently miss (this exact bug
# shipped once: a test's fake was bypassed, the real Gemini API got called,
# and it hung in llm_retry's real time.sleep on a live rate limit - see
# DECISIONS.md 2026-07-28).
_PROVIDER_GENERATE_FN_NAMES = {
    "gemini": "gemini_generate_fn",
    "claude": "claude_generate_fn",
    "ollama": "ollama_generate_fn",
}
_PROVIDER_ENV = {"gemini": "GEMINI_API_KEY", "claude": "ANTHROPIC_API_KEY", "ollama": "OLLAMA_MODEL"}
_PROVIDER_ORDER = ("gemini", "claude", "ollama")  # display/generation order


def configured_providers() -> list[str]:
    """Every provider with a key/model set, in display order - used to
    generate/show one summary per configured provider, not just the default."""
    return [p for p in _PROVIDER_ORDER if os.getenv(_PROVIDER_ENV[p])]


def generate_with(provider: str, job: Job, profile: CandidateProfile) -> str:
    """Generate a summary using one specific provider by name (not the
    priority-ordered default picker) - so a caller can request 'gemini' and
    'ollama' independently and keep both results. Looks the function up by
    name in this module's globals at call time - see the comment above."""
    fn = globals()[_PROVIDER_GENERATE_FN_NAMES[provider]]
    return fn(job, profile)
