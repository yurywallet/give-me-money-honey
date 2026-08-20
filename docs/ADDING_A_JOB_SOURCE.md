# Adding a new job source — wiring checklist

Written after the second source (`linkedin_apify_source.py`) was added,
following `engineering-foundations` §2: the moment a repeating concept gets
its second instance, write down every place a third one will need to touch.
Every future source is "done" only when every item below is checked.

1. **`sources/<name>_source.py`** — implement the `JobSource` protocol
   (`sources/__init__.py`): a `name: str` attribute and
   `search(keywords: list[str]) -> list[db.Job]`. The returned `Job.description`
   **must** be the full job description text, not a title/snippet — scoring
   depends on reading it (salary language, benefit keywords).
2. **Field mapping verified against a real sample**, not assumed. Pull a
   handful of actual raw items from the new source and manually confirm
   `salary_min`/`salary_max`/`work_type`/`location_mode`/`location_country`
   map correctly before trusting any score computed from them
   (engineering-foundations §3 — a mapping that "usually" works is a live
   bug until checked against raw data).
3. **`server.py`** — import and append an instance to `_sources`, gated on
   whatever credentials/config that source needs being present (see how the
   Apify source is only added when `APIFY_TOKEN`/`APIFY_ACTOR_ID` are set —
   the server must still run on `MockJobSource` alone with zero config).
4. **`.env.example`** — add the new source's required env vars with blank
   defaults and a one-line comment on what they're for.
5. **`tests/test_<name>_source.py`** — at minimum, a test that maps one
   realistic raw item (captured from step 2) into the expected `Job` fields.
   Do not skip this because "it just calls an API" — the mapping logic is
   exactly the part that's project-specific and worth pinning down.
6. **`requirements.txt`** — pin any new dependency's exact installed version
   (`==`, matching this repo's convention), not a range.
7. **De-dup key** — confirm the source's `external_id` is stable across
   repeated fetches of the same listing (job board IDs are usually stable;
   a URL that includes a session/tracking token is not — check before using
   it as the dedup key, or every poll will re-insert the same job as "new").

Sources never need their own DB table or scoring logic — `db.Job` and
`scoring.score_job()` are shared across every source by design.
