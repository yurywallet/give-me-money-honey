"""LangGraph search-strategy agent: plan -> search -> evaluate -> reflect -> loop/finalize.

Real, production LangGraph experience (not course-only) - this closes a specific
gap flagged against a Staff AI Engineer JD requiring "deep hands-on experience
with... at least one agentic framework": the LangGraph/MCP evidence on file was
course-based ("Introduction to agent skills", "Building Ambient Agents with
LangGraph"), not a shipped system. This is the shipped system.

Design: a stateful graph that runs a real search via the existing Scheduler,
evaluates *why* jobs failed the hard gate (db.hard_fail_reason_counts), and
reflects (an LLM call) on whether to refine keywords or stop - the
plan -> execute -> observe -> iteratively improve loop, not a fixed script.

The reflection step is injected as `reflect_fn` (same pattern as clock.py's
Clock injection, per engineering-foundations) so the graph is fully testable
without a real API key or network call - only a live run of the default
`default_reflect_fn` needs a provider key (ANTHROPIC_API_KEY or GEMINI_API_KEY,
mirroring fit_summary.default_generate_fn).

Cost note: each iteration runs a REAL search via Scheduler.run_once(). If the
configured source is LinkedInApifySource, that's real Apify spend per
iteration on top of the reflection LLM call - see that module's own cost
note. Keep max_iterations small (default 2) while iterating on this.
"""
from __future__ import annotations

import json
import os
from typing import Callable, Optional, TypedDict

from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from candidate_profile import CandidateProfile
from clock import Clock
from config import HardCriteria, SearchConfig
from db import hard_fail_reason_counts, tiered_salary_stats
from role_fit import SENIORITY_TIER_PATTERNS
from scheduler import Scheduler
from sources import JobSource


class ReflectionDecision(BaseModel):
    done: bool
    keywords: list[str]
    reasoning: str


class SearchState(TypedDict):
    keywords: list[str]
    iteration: int
    max_iterations: int
    last_summary: dict
    fail_reason_counts: dict
    # Historical clear-rate per role family across ALL previously collected
    # postings, not just this session (§ DECISIONS.md 2026-08-02) - lets the
    # reflection step see "this role family never clears the floor" instead
    # of only reacting to the current run's fail counts.
    role_base_rates: dict
    history: list[dict]
    done: bool


ReflectFn = Callable[[list[str], dict, dict, list[dict], dict], ReflectionDecision]


def _reflect_prompt(
    keywords: list[str],
    summary: dict,
    fail_reason_counts: dict,
    history: list[dict],
    role_base_rates: dict,
) -> str:
    """Shared prompt for both provider reflect fns - a single source so the
    Claude and Gemini paths can never drift apart."""
    base_rate_note = ""
    if role_base_rates:
        # Historical hit-rate per role family, from real postings already
        # collected (§ DECISIONS.md 2026-08-02). Without this the model can
        # only react to THIS session's fail counts, so it keeps re-proposing
        # role families that have never once cleared the salary floor across
        # hundreds of prior postings - a pattern only visible across runs.
        base_rate_note = (
            "\nHistorical hit rate per role AND SENIORITY TIER, measured across every "
            "posting this project has already collected. Each posting is counted in "
            "exactly one tier (most senior match wins):\n"
            f"{json.dumps(role_base_rates, indent=2)}\n"
            "  - clear_rate: fraction whose salary MINIMUM met the floor (what the hard "
            "gate actually tests).\n"
            "  - band_reach_rate: fraction whose salary MAXIMUM reaches the floor, i.e. "
            "the role can pay above it even if the posted minimum doesn't. Judging a "
            "keyword on clear_rate alone understates roles posted as wide bands.\n"
            "  - n: how many postings back the numbers - a small n is a weak signal.\n"
            "Weigh this carefully. SENIORITY TIER often matters more than role family: "
            "the same role can be dead at one tier and strong at another, so prefer "
            "re-targeting a weak keyword to a more senior phrasing of the SAME role "
            "before abandoning that role entirely. Only drop a role outright when BOTH "
            "rates are near zero across a large n at every tier.\n"
        )
    return (
        "You are tuning a job-search agent's LinkedIn keyword strategy.\n\n"
        f"Current keywords: {keywords}\n"
        f"Latest run summary (per source: found/new/hard_pass counts): {json.dumps(summary)}\n"
        f"Why jobs failed the hard gate this run (reason -> count): {json.dumps(fail_reason_counts)}\n"
        f"History of prior iterations this session: {json.dumps(history)}\n"
        f"{base_rate_note}\n"
        "Decide whether the current keyword set is adequate or should be refined "
        "to surface more passing jobs. If refining, propose a revised keyword list "
        "(don't drop keywords that are working; add or adjust ones that aren't "
        "surfacing anything). Set done=true once further refinement is unlikely "
        "to help, or returns are diminishing."
    )


def claude_reflect_fn(
    keywords: list[str], summary: dict, fail_reason_counts: dict, history: list[dict],
    role_base_rates: dict,
) -> ReflectionDecision:
    """Reflection step via Claude. Requires ANTHROPIC_API_KEY. Swappable via
    `build_graph(..., reflect_fn=...)` - tests inject a fake to exercise the
    graph without a real API call."""
    import anthropic

    client = anthropic.Anthropic()
    prompt = _reflect_prompt(keywords, summary, fail_reason_counts, history, role_base_rates)
    response = client.messages.parse(
        model="claude-opus-5",
        # Already thinking-enabled, so this call didn't have fit_summary.py's
        # "thinking suddenly turned on" problem — adaptive is still valid on
        # Opus 5 and equivalent to its default. Budget raised anyway because
        # Opus 5 thinks more readily, and here thinking shares max_tokens with
        # a STRUCTURED output: if thinking exhausts the budget the parse has
        # nothing to return, which fails harder than a truncated paragraph.
        max_tokens=2048,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": prompt}],
        output_format=ReflectionDecision,
    )
    return response.parsed_output


def gemini_reflect_fn(
    keywords: list[str], summary: dict, fail_reason_counts: dict, history: list[dict],
    role_base_rates: dict,
) -> ReflectionDecision:
    """Reflection step via Gemini (structured output → ReflectionDecision).
    Requires GEMINI_API_KEY. Mirrors fit_summary.gemini_generate_fn so the
    whole project runs on one key when only Gemini is configured."""
    from google import genai

    from llm_retry import call_with_rate_limit_retry

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    response = call_with_rate_limit_retry(
        lambda: client.models.generate_content(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            contents=_reflect_prompt(keywords, summary, fail_reason_counts, history, role_base_rates),
            config={"response_mime_type": "application/json", "response_schema": ReflectionDecision},
        )
    )
    # google-genai parses the JSON into the pydantic model on .parsed; fall
    # back to manual parse if the SDK returns only text.
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, ReflectionDecision):
        return parsed
    return ReflectionDecision(**json.loads(response.text))


def ollama_reflect_fn(
    keywords: list[str], summary: dict, fail_reason_counts: dict, history: list[dict],
    role_base_rates: dict,
) -> ReflectionDecision:
    """Reflection step via a local Ollama model (structured output via Ollama's
    `format` = JSON schema). No API key, no rate limits. Mirrors
    fit_summary.ollama_generate_fn (OLLAMA_HOST / OLLAMA_MODEL)."""
    import httpx

    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "mistral-nemo")
    resp = httpx.post(
        f"{host}/api/generate",
        json={
            "model": model,
            "prompt": _reflect_prompt(keywords, summary, fail_reason_counts, history, role_base_rates),
            "stream": False,
            "format": ReflectionDecision.model_json_schema(),
        },
        timeout=180,
    )
    resp.raise_for_status()
    return ReflectionDecision(**json.loads(resp.json()["response"]))


def default_reflect_fn(
    keywords: list[str], summary: dict, fail_reason_counts: dict, history: list[dict],
    role_base_rates: dict,
) -> ReflectionDecision:
    """Picks a provider: OLLAMA_MODEL (local, unlimited) first if set, then
    GEMINI_API_KEY, then ANTHROPIC_API_KEY last. Mirrors
    fit_summary.default_generate_fn. Raises clearly if none is configured.

    Anthropic is LAST deliberately (§ DECISIONS.md 2026-08-02): the operator
    has no Anthropic API access, and the original Anthropic-before-Gemini
    order was an arbitrary tiebreak "purely for consistency ... not because
    Gemini is worse" (§ 2026-07-25). That order made Gemini-only operation
    accidental rather than guaranteed - a non-empty ANTHROPIC_API_KEY from
    any source (a stray shell export, another project's .env) would silently
    reroute every reflection call to a provider with no credits and fail the
    whole search. Anthropic still works when it's the ONLY thing configured,
    so this stays a real multi-provider picker, not a removal."""
    if os.getenv("OLLAMA_MODEL"):
        return ollama_reflect_fn(keywords, summary, fail_reason_counts, history, role_base_rates)
    if os.getenv("GEMINI_API_KEY"):
        return gemini_reflect_fn(keywords, summary, fail_reason_counts, history, role_base_rates)
    if os.getenv("ANTHROPIC_API_KEY"):
        return claude_reflect_fn(keywords, summary, fail_reason_counts, history, role_base_rates)
    raise RuntimeError(
        "No LLM provider configured - set OLLAMA_MODEL, ANTHROPIC_API_KEY, or "
        "GEMINI_API_KEY in .env to run the search agent's reflection step."
    )


def build_graph(
    sources: list[JobSource],
    conn,
    profile: CandidateProfile,
    hard: HardCriteria,
    clock: Optional[Clock] = None,
    reflect_fn: ReflectFn = default_reflect_fn,
):
    """Compile the run_search -> evaluate -> reflect graph.

    `sources` is reused as-is across iterations (same instances, e.g. an
    already-configured LinkedInApifySource) - only the search `keywords` in
    state change between iterations.
    """

    def run_search(state: SearchState) -> SearchState:
        cfg = SearchConfig(keywords=tuple(state["keywords"]), hard=hard)
        scheduler = Scheduler(cfg, sources, conn, clock=clock, profile=profile)
        state["last_summary"] = scheduler.run_once()
        return state

    def evaluate(state: SearchState) -> SearchState:
        counts: dict[str, int] = {}
        for source in sources:
            for reason, n in hard_fail_reason_counts(conn, source.name).items():
                counts[reason] = counts.get(reason, 0) + n
        state["fail_reason_counts"] = counts

        # Cross-run context: how often each role/seniority tier has EVER
        # cleared the configured salary floor, across every posting already
        # collected. Uses hard.min_salary (not a hardcoded literal) so this
        # tracks whatever floor the search is actually running with.
        #
        # Split BY SENIORITY TIER, not just role family (§ DECISIONS.md
        # 2026-08-02, corrected): an aggregate over "analytics engineer"
        # reported a 4% clear rate and nearly justified dropping the
        # operator's primary role, because 49 plain/senior postings buried 8
        # Staff/Principal/Lead ones that actually run $165k-$236k.
        state["role_base_rates"] = tiered_salary_stats(
            conn, SENIORITY_TIER_PATTERNS, floor=hard.min_salary
        )
        return state

    def reflect(state: SearchState) -> SearchState:
        decision = reflect_fn(
            state["keywords"], state["last_summary"], state["fail_reason_counts"],
            state["history"], state["role_base_rates"],
        )
        state["history"].append(
            {
                "keywords": state["keywords"],
                "summary": state["last_summary"],
                "fail_reason_counts": state["fail_reason_counts"],
                "reasoning": decision.reasoning,
            }
        )
        state["iteration"] += 1
        state["done"] = decision.done or state["iteration"] >= state["max_iterations"]
        if not state["done"]:
            state["keywords"] = decision.keywords
        return state

    def route(state: SearchState) -> str:
        return END if state["done"] else "run_search"

    graph = StateGraph(SearchState)
    graph.add_node("run_search", run_search)
    graph.add_node("evaluate", evaluate)
    graph.add_node("reflect", reflect)
    graph.set_entry_point("run_search")
    graph.add_edge("run_search", "evaluate")
    graph.add_edge("evaluate", "reflect")
    graph.add_conditional_edges("reflect", route, {"run_search": "run_search", END: END})
    return graph.compile()


def run_search_agent(
    sources: list[JobSource],
    conn,
    profile: CandidateProfile,
    hard: HardCriteria,
    keywords: Optional[list[str]] = None,
    max_iterations: int = 2,
    clock: Optional[Clock] = None,
    reflect_fn: ReflectFn = default_reflect_fn,
) -> SearchState:
    """Run the search-strategy agent to completion and return its final state."""
    graph = build_graph(sources, conn, profile, hard, clock=clock, reflect_fn=reflect_fn)
    initial: SearchState = {
        "keywords": list(keywords or profile.title_keywords),
        "iteration": 0,
        "max_iterations": max_iterations,
        "last_summary": {},
        "fail_reason_counts": {},
        "role_base_rates": {},
        "history": [],
        "done": False,
    }
    return graph.invoke(initial)
