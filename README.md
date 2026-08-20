# give_me_money_honey

An MCP server that periodically searches for jobs, reads the full job
description, and ranks each listing against a hard/soft criteria model.

## Criteria

**Hard** (a job failing any of these is excluded entirely, score = 0):
- Salary ≥ your configured floor (`HardCriteria.min_salary`, set via
  `GMMH_MIN_SALARY` in `.env` — the shipped default is a generic placeholder)
- Work type: full-time
- Location: remote, USA

**Soft** (only ever adds to score, never gates):
- Benefits mentioned in the JD text — family medical insurance weighted
  highest, plus dental/vision/401k match/equity/parental leave/etc.
- Salary above your configured floor — more salary adds more score (capped
  so one outlier salary can't dwarf every benefit signal combined).
- **Profile match** — how well the JD lines up with the candidate profile in
  `candidate_profile.py` (title: Analytics Engineer / Data Analyst; skills:
  dbt, LookML, Looker, BI, SQL, Snowflake, Redshift, BigQuery, PostgreSQL,
  Python, semantic layer, experimentation). Points per matched skill keyword
  found in the JD (capped), plus a flat bonus if the job title itself matches
  one of the profile's target titles. Search keywords default to the
  profile's own title keywords.

See `config.py` for the exact weights and `scoring.py` for how the JD text
is parsed (salary ranges, benefit keywords, family+medical proximity match,
profile skill/title matching).

## Architecture

```
sources/          pluggable job-board integrations (JobSource protocol)
  mock_source.py            fake data, no credentials — default, always works
  linkedin_apify_source.py  real LinkedIn via Apify's managed proxy (needs your own token)
candidate_profile.py  the candidate profile jobs are matched against (title + skills)
scoring.py        hard_filter() + score_job() — pure functions, fully unit-tested
db.py             SQLite storage (WAL mode) — jobs + search_runs tables
scheduler.py      fetch → score → store, once or on a poll loop
clock.py          injectable clock so scheduling logic is testable
server.py         the MCP server — wires the above into tools
app.py            Streamlit page — edit profile, set criteria, search, see ranked results
```

Every job source returns the **full job description text**, not a
title/snippet — the scoring model depends on actually reading it.

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # optional — only needed for the real LinkedIn source
```

Runs immediately with zero config using `MockJobSource` (three fake
listings, one intentionally hybrid so you can see the hard gate reject it).

## Running

As an MCP server (stdio transport):

```bash
.venv/bin/python server.py
```

Point your MCP client at it, e.g. in Claude Desktop's config:

```json
{
  "mcpServers": {
    "give-me-money-honey": {
      "command": "/absolute/path/to/give_me_money_honey/.venv/bin/python",
      "args": ["/absolute/path/to/give_me_money_honey/server.py"]
    }
  }
}
```

### Tools exposed

| Tool | Purpose |
|---|---|
| `search_jobs_now(keywords?)` | Run one search pass immediately across all configured sources |
| `start_periodic_search(interval_minutes?)` | Start the background poll loop |
| `stop_periodic_search()` | Stop it |
| `get_top_jobs(limit, only_passing, only_active)` | Ranked results, highest score first. `only_active=True` (default) hides jobs that dropped out of their source's most recent successful search pass — i.e. likely delisted/filled. |
| `get_job(job_id)` | Full detail: description + score breakdown |
| `get_criteria()` | Current hard gates + soft weights in effect |
| `get_profile()` | The candidate profile jobs are matched against |

## Streamlit page

```bash
.venv/bin/streamlit run app.py
```

- **Profile** — prepopulated from `candidate_profile.py`'s default bio; edit
  target titles/skills/summary and click **Save profile** to persist to
  `candidate_profile.json` (gitignored - it's your personal data, not repo
  state). `server.py` picks up a saved edit on its next restart.
- **Job details** — the hard-gate search criteria (min salary, work type,
  location mode/country) plus search keywords, prepopulated from
  `config.HardCriteria()` defaults. These are per-session, not persisted.
- **Search** — runs one search pass through the same `Scheduler` the MCP
  server uses (`MockJobSource` by default, plus the real LinkedIn source if
  `APIFY_TOKEN`/`APIFY_ACTOR_ID` are set), scored against whatever profile
  and criteria are currently in the form.
- **Results (ranked)** — highest score first, each with a clickable link to
  the posting, salary, benefits, matched skills, and expandable score
  breakdown + full description.

## Adding a real job source

`MockJobSource` proves the pipeline end-to-end but obviously isn't real
data. Two ways to get real LinkedIn listings (both from the "managed proxy"
family — your own LinkedIn session/cookies are never used, avoiding the ban
risk of direct scraping):

- **`sources/linkedin_apify_source.py`** is already scaffolded — set
  `APIFY_TOKEN` and `APIFY_ACTOR_ID` in `.env` (pick any LinkedIn-jobs actor
  from the Apify Store) and `server.py` picks it up automatically. **Read
  the module docstring first** — actor output schemas vary, and the field
  mapping needs a one-time check against a real sample before trusting
  scores computed from it.
- To add a different source (Indeed, a different Apify actor, a direct API),
  follow `docs/ADDING_A_JOB_SOURCE.md` — a numbered checklist so nothing
  gets wired halfway (a lesson trader_joe's CLAUDE.md learned the hard way
  about "new instance of a repeating concept" style additions).

## Testing

```bash
.venv/bin/python -m pytest -q
```

Auto-discovers every `tests/test_*.py` — no hand-maintained file list to
go stale as sources are added.

## Design notes

`docs/DECISIONS.md` normally lives here — why the salary gate uses the
conservative end of a range, why benefit-detection is keyword-proximity
rather than an LLM call, why hard/soft criteria are strictly separated. It's
excluded from this shared snapshot because it's a running research log full
of the operator's real search data (specific postings, salary bands, clear
rates); kept privately alongside the full git history.
