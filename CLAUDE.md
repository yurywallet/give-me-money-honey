# CLAUDE.md — give_me_money_honey

> **Note on this snapshot:** `docs/DECISIONS.md`, referenced throughout this
> file as the project's research log, is excluded here — it contains the
> operator's real search data (specific postings, salary bands, clear
> rates). It exists in the private working copy this snapshot was taken
> from. References to it below describe the project's actual practice.

## GOAL

Find the operator a **well-paid senior AI Engineer or Analytics Engineer
role**: remote, US, full-time, with a real base salary above the floor.
Equity is a tiebreaker between otherwise-comparable offers — never a reason
to accept a lower base.

Operator's reasoning (2026-08-12): *"in a lot of cases equity stays paper
money that is never real money, so I do not want to have a low paid job for
a paper money future."* Cash is the certain part of the offer; equity is a
lottery ticket priced by someone else.

**This reverses the 2026-07-25 "equity pivot"** (docs/DECISIONS.md), which
said the opposite — that salary above the floor is a commodity and startup
ownership is the asymmetric bet. That framing is dead. Concretely: there is
no longer an equity-relaxed salary floor (a job below `min_salary` does not
get in on the strength of its equity story), and the equity score can no
longer outweigh salary. A change that surfaces lower-paid equity-rich
early-stage roles at the expense of high-base senior ones is now a
regression, not an improvement.

Scope is also narrower than "any data role": **AI Engineer and Analytics
Engineer families, senior and above.** Data Analyst / BI Engineer / Data
Scientist are out — measured 0% clear rate against the configured salary
floor on the first two, and the third is a different discipline.

## Where the truth lives — read before re-deriving anything

| Question | File |
|---|---|
| What does this do, how do I run it, what tools does the MCP server expose? | `README.md` |
| Why is it built this way? Why was X rejected? | `docs/DECISIONS.md` |
| How do I add a new job board? | `docs/ADDING_A_JOB_SOURCE.md` |
| What are the exact weights/floors? | `config.py` (values), `scoring.py` (logic) |

`docs/DECISIONS.md` is the single most important file here. Several decisions
have been made, reversed, and re-made (the title gate is the clearest case:
added as a hard gate, softened, then dropped entirely for a matched-skill
floor). **Check it before changing scoring behavior** — and when a decision
genuinely closes, append an entry. A "why not" that isn't written down gets
re-derived from scratch six weeks later.

Current state — how many sources are live, what's passing, which providers are
configured — is deliberately NOT recorded in this file. It goes stale the day
it's written. Read `docs/DECISIONS.md`'s latest entries and the actual `.env`.

## Design contracts

These span multiple files, so a change to one side silently breaks the other.

**Hard criteria gate; soft criteria only ever add.** `hard_filter()` decides
pass/fail. Soft signals (benefits, salary-above-floor, skill match, equity)
only add points and must never gate. This has been violated once already — a
title hard-gate crept in undocumented and made "different title, same
skillset" indistinguishable from a genuine non-match. Adding a new gate is a
`HardCriteria` change with a DECISIONS.md entry, never a quiet `return 0` in a
scoring helper.

**A hard-gate miss still computes a full, visible score.** Only
salary/work_type/location zero a job out. Anything else (below the
matched-skill floor, title mismatch) still gets a real score and breakdown, so
a close-fit role with strong skill overlap is distinguishable from an
unrelated one. Both looking like `0.0` is the bug that motivated this.

**`scoring.py` is pure and dependency-free** — no I/O, no network, no LLM
calls. That's what makes it exhaustively unit-testable, and it's why the
qualitative summary lives in `fit_summary.py` instead. Keep it that way.

**Job sources return the FULL job description text**, never a title or
snippet. Every soft signal (benefits, equity, stage, skills) is parsed out of
that text — a source that returns a truncated JD silently starves the whole
scoring model rather than failing loudly. This is the current bottleneck, not
a hypothetical: see DECISIONS.md's last entry.

**LLM calls are on-demand and cached, never in the poll loop.** `run_once`
scores every job on every cycle; calling a model there would bill for jobs
nobody opens. `fit_summary` is generated on request and cached on the row —
and `upsert_job` must never overwrite that column (there is a test for this
specifically; a silent wipe on the next poll is worse than never generating
one).

**Adding a job source touches more than one file.** Follow
`docs/ADDING_A_JOB_SOURCE.md` end to end in the same change — a half-wired
source that imports cleanly but never reaches the scheduler is the failure
mode that checklist exists to prevent.

## Stack

Python 3.12, stdlib-heavy. SQLite (WAL) via `db.py`. MCP server (`server.py`,
stdio) + Streamlit UI (`app.py`) over the same `Scheduler`. Deps pinned with
`==` in `requirements.txt`.

LLM providers are injectable (`generate_fn` / `reflect_fn`) so tests run with
fakes and no key or cost. `default_generate_fn` picks whichever of
`ANTHROPIC_API_KEY` / `GEMINI_API_KEY` is actually set — check which one is
live before assuming a code path runs at all.

## Verifying a change

`.venv/bin/python -m pytest -q` — the full suite, auto-discovered.

Note what the suite does *not* cover: every LLM call site is exercised through
an injected fake, so a green run says nothing about whether a real request
body is correct (model ID, `max_tokens`, thinking budget). Those need either a
live call or careful reading. Say which one you did.

`MockJobSource` runs the whole pipeline end-to-end with zero credentials — use
it to actually exercise a change rather than reasoning about it.

## Working rules

**Deliver what was asked, at the scope intended.** Make routine judgment calls
yourself; check in only when different readings of the request lead to
materially different work. If the request looks mistaken or a better approach
exists, say so in a sentence and continue as asked — don't quietly narrow,
widen, or transform it.

**No speculative additions.** No config knobs, abstractions, or error handlers
for cases that can't occur. This project's own precedent: keyword proximity was
chosen over an NLP model, and mock-first over required credentials — start
cheap, revisit only if reality shows it matters.

**When investigating, read the raw thing.** A score that looks wrong, a job
that won't pass, a source returning nothing — read the actual JD text, the
actual row, the actual response payload. Summarizing evidence before
understanding it is how a wrong diagnosis gets confirmed. (Filtering *streams*
you already understand is fine; filtering *evidence* is not.)

**Report every finding, then rank.** When auditing or debugging, surface
everything found including low-confidence items, and sort by severity at the
end. Don't pre-filter to "only what looks important" while still looking.

**Match written length to substance.** DECISIONS.md entries and commit bodies
should carry the reasoning and the numbers, then stop — no filler or restated
summaries. An entry a future session can act on beats a longer one.

**Commits:** `type(scope): imperative summary`, body explains *why*. Every
commit leaves the suite green.
