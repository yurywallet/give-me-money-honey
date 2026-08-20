import datetime as dt

import pytest

import agent
from agent import ReflectionDecision, default_reflect_fn, run_search_agent
from clock import FixedClock
from config import HardCriteria
from candidate_profile import DEFAULT_PROFILE
from db import get_conn, init_db, hard_fail_reason_counts
from sources.mock_source import MockJobSource


def _conn(tmp_path):
    conn = get_conn(str(tmp_path / "test.db"))
    init_db(conn)
    return conn


def _clock():
    return FixedClock(dt.datetime(2026, 7, 23, 12, 0, tzinfo=dt.timezone.utc))


def test_agent_stops_after_one_iteration_when_reflect_says_done(tmp_path):
    def always_done(keywords, summary, fail_reason_counts, history, role_base_rates):
        return ReflectionDecision(done=True, keywords=keywords, reasoning="good enough")

    conn = _conn(tmp_path)
    final = run_search_agent(
        [MockJobSource()], conn, DEFAULT_PROFILE, HardCriteria(min_salary=200_000),
        clock=_clock(), reflect_fn=always_done, max_iterations=5,
    )

    assert final["iteration"] == 1
    assert final["done"] is True
    assert len(final["history"]) == 1
    assert final["last_summary"]["mock"]["found"] == 4


def test_agent_loops_and_stops_at_max_iterations(tmp_path):
    calls = []

    def always_refine(keywords, summary, fail_reason_counts, history, role_base_rates):
        calls.append(keywords)
        return ReflectionDecision(done=False, keywords=keywords, reasoning="try again")

    conn = _conn(tmp_path)
    final = run_search_agent(
        [MockJobSource()], conn, DEFAULT_PROFILE, HardCriteria(min_salary=200_000),
        clock=_clock(), reflect_fn=always_refine, max_iterations=3,
    )

    assert final["iteration"] == 3
    assert final["done"] is True
    assert len(calls) == 3
    assert len(final["history"]) == 3


def test_agent_evaluate_step_reports_why_jobs_failed(tmp_path):
    def always_done(keywords, summary, fail_reason_counts, history, role_base_rates):
        return ReflectionDecision(done=True, keywords=keywords, reasoning=str(fail_reason_counts))

    conn = _conn(tmp_path)
    final = run_search_agent(
        [MockJobSource()], conn, DEFAULT_PROFILE, HardCriteria(min_salary=200_000),
        clock=_clock(), reflect_fn=always_done, max_iterations=1,
    )

    # Of the 4 mock listings, only "Senior Analytics Engineer" passes; the other
    # 3 fail on too-few matched skills (mock-1, mock-2 - real salary/location
    # matches, but neither JD mentions any profile skill) or hybrid location +
    # below-floor salary, both on mock-3 - see test_scheduler.py and
    # sources/mock_source.py.
    expected = {"skill_match": 2, "location_mode": 1, "salary": 1}
    assert final["fail_reason_counts"] == expected
    # Cross-check directly against the db helper the agent's evaluate step uses.
    assert hard_fail_reason_counts(conn, "mock") == expected


def test_agent_passes_seniority_split_base_rates_to_reflection(tmp_path):
    # § DECISIONS.md 2026-08-02 (corrected): the reflection step must see
    # CROSS-RUN hit rates, split BY SENIORITY TIER. An aggregate over
    # "analytics engineer" hid a genuinely strong Staff tier behind a mass of
    # lower-tier postings and nearly justified dropping the operator's primary
    # role - so this locks in that the tiers stay separated, and that a Staff
    # posting is NOT also counted in the untiered bucket.
    from db import upsert_job, Job

    received = {}

    def capture(keywords, summary, fail_reason_counts, history, role_base_rates):
        received["base_rates"] = role_base_rates
        return ReflectionDecision(done=True, keywords=keywords, reasoning="ok")

    conn = _conn(tmp_path)
    seed = [
        ("Data Analyst", 90_000, 110_000),
        ("Data Analyst", 110_000, 130_000),
        # Two plain AE well under the floor, one Staff AE clearly over it.
        ("Analytics Engineer", 120_000, 150_000),
        ("Analytics Engineer", 130_000, 160_000),
        ("Staff Analytics Engineer", 210_000, 260_000),
    ]
    for i, (title, lo, hi) in enumerate(seed):
        upsert_job(conn, Job(source="mock", external_id=f"seed-{i}", title=title, company="X",
                             url="https://example.com", description="d",
                             salary_min=lo, salary_max=hi), "t0")

    run_search_agent(
        [MockJobSource()], conn, DEFAULT_PROFILE, HardCriteria(min_salary=200_000),
        clock=_clock(), reflect_fn=capture, max_iterations=1,
    )

    rates = received["base_rates"]
    staff = rates["Staff/Principal/Lead Analytics Engineer"]
    plain = rates["Analytics Engineer (no tier)"]

    # The Staff posting counts ONCE, in the Staff bucket only.
    assert staff["n"] == 1 and staff["clear_rate"] == 1.0
    assert plain["n"] == 2 and plain["clear_rate"] == 0.0

    # The whole point: aggregating these would report 1/3 = 33% and bury the
    # fact that the Staff tier is 100% and the untiered tier is 0%.
    assert staff["clear_rate"] > plain["clear_rate"]

    # band_reach_rate is reported alongside clear_rate - a posting whose MAX
    # reaches the floor is a real opportunity even if its MIN doesn't.
    assert rates["Data Analyst"]["band_reach_rate"] == 0.0
    assert staff["band_reach_rate"] == 1.0


def test_reflect_prompt_includes_base_rates_when_present():
    from agent import _reflect_prompt

    prompt = _reflect_prompt(
        ["data analyst"], {}, {}, [],
        {"Data Analyst": {"n_with_disclosed_salary": 28, "clear_rate": 0.0, "median_min": 85727}},
    )
    assert "clear_rate" in prompt
    assert "Data Analyst" in prompt

    # ...and stays silent (no empty/misleading section) when there's no data.
    assert "clear_rate" not in _reflect_prompt(["data analyst"], {}, {}, [], {})


def test_agent_keeps_refined_keywords_from_reflection(tmp_path):
    def refine_once(keywords, summary, fail_reason_counts, history, role_base_rates):
        if not history:
            return ReflectionDecision(done=False, keywords=["bi engineer"], reasoning="broaden")
        return ReflectionDecision(done=True, keywords=keywords, reasoning="stop")

    conn = _conn(tmp_path)
    final = run_search_agent(
        [MockJobSource()], conn, DEFAULT_PROFILE, HardCriteria(min_salary=200_000),
        clock=_clock(), reflect_fn=refine_once, max_iterations=5,
    )

    assert final["iteration"] == 2
    assert final["history"][0]["keywords"] == list(DEFAULT_PROFILE.title_keywords)
    assert final["keywords"] == ["bi engineer"]


# --- default_reflect_fn provider dispatch (mirrors test_fit_summary) --------

def test_default_reflect_fn_raises_when_no_provider(monkeypatch):
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="No LLM provider configured"):
        default_reflect_fn([], {}, {}, [], {})


def test_default_reflect_fn_prefers_gemini_over_claude_when_both(monkeypatch):
    # § DECISIONS.md 2026-08-02: Anthropic is LAST. A stray non-empty
    # ANTHROPIC_API_KEY must not silently reroute the search's reflection
    # step to a provider the operator has no access to.
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    calls = []
    monkeypatch.setattr(agent, "claude_reflect_fn", lambda *a: calls.append("claude"))
    monkeypatch.setattr(agent, "gemini_reflect_fn", lambda *a: calls.append("gemini"))
    default_reflect_fn([], {}, {}, [], {})
    assert calls == ["gemini"]


def test_default_reflect_fn_still_uses_claude_when_it_is_the_only_provider(monkeypatch):
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    calls = []
    monkeypatch.setattr(agent, "claude_reflect_fn", lambda *a: calls.append("claude"))
    default_reflect_fn([], {}, {}, [], {})
    assert calls == ["claude"]


def test_default_reflect_fn_falls_back_to_gemini(monkeypatch):
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    calls = []
    monkeypatch.setattr(agent, "claude_reflect_fn", lambda *a: calls.append("claude"))
    monkeypatch.setattr(agent, "gemini_reflect_fn", lambda *a: calls.append("gemini"))
    default_reflect_fn([], {}, {}, [], {})
    assert calls == ["gemini"]


def test_default_reflect_fn_prefers_ollama_when_set(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "mistral-nemo")
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    calls = []
    monkeypatch.setattr(agent, "ollama_reflect_fn", lambda *a: calls.append("ollama"))
    monkeypatch.setattr(agent, "gemini_reflect_fn", lambda *a: calls.append("gemini"))
    default_reflect_fn([], {}, {}, [], {})
    assert calls == ["ollama"]
