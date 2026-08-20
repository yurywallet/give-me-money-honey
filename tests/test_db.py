from db import (
    Job,
    finish_search_run,
    get_conn,
    get_job,
    get_latest_role_fit_snapshot,
    init_db,
    latest_successful_run_id,
    list_top_jobs,
    role_salary_stats,
    save_role_fit_snapshot,
    set_fit_summary,
    start_search_run,
    upsert_job,
)


def _conn(tmp_path):
    conn = get_conn(str(tmp_path / "test.db"))
    init_db(conn)
    return conn


def _job(**overrides) -> Job:
    base = dict(
        source="mock",
        external_id="1",
        title="Engineer",
        company="Acme",
        url="https://example.com/1",
        description="desc",
    )
    base.update(overrides)
    return Job(**base)


def test_wal_mode_enabled(tmp_path):
    conn = _conn(tmp_path)
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


def test_upsert_job_inserts_new(tmp_path):
    conn = _conn(tmp_path)
    job = _job()
    job_id, is_new = upsert_job(conn, job, "2026-07-14T00:00:00")
    assert is_new is True
    stored = get_job(conn, job_id)
    assert stored.title == "Engineer"
    assert stored.source == "mock"
    assert stored.external_id == "1"


def test_upsert_job_dedups_by_source_and_external_id(tmp_path):
    conn = _conn(tmp_path)
    job = _job(title="Engineer v1")
    id1, is_new1 = upsert_job(conn, job, "2026-07-14T00:00:00")

    updated = _job(title="Engineer v2", score=42.0)
    id2, is_new2 = upsert_job(conn, updated, "2026-07-15T00:00:00")

    assert id1 == id2
    assert is_new1 is True
    assert is_new2 is False

    stored = get_job(conn, id1)
    assert stored.title == "Engineer v2"
    assert stored.score == 42.0

    count = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    assert count == 1


def test_list_top_jobs_orders_by_score_and_filters_hard_pass(tmp_path):
    conn = _conn(tmp_path)
    upsert_job(conn, _job(external_id="a", score=10.0, hard_pass=True), "t")
    upsert_job(conn, _job(external_id="b", score=90.0, hard_pass=True), "t")
    upsert_job(conn, _job(external_id="c", score=99.0, hard_pass=False), "t")

    passing = list_top_jobs(conn, limit=10, only_passing=True)
    assert [j.external_id for j in passing] == ["b", "a"]

    everyone = list_top_jobs(conn, limit=10, only_passing=False)
    assert [j.external_id for j in everyone] == ["c", "b", "a"]


def test_search_run_lifecycle(tmp_path):
    conn = _conn(tmp_path)
    run_id = start_search_run(conn, "mock", "engineer", "2026-07-14T00:00:00")
    finish_search_run(conn, run_id, "2026-07-14T00:05:00", n_found=3, n_new=2)

    row = conn.execute("SELECT * FROM search_runs WHERE id = ?", (run_id,)).fetchone()
    assert row["n_found"] == 3
    assert row["n_new"] == 2
    assert row["finished_at"] == "2026-07-14T00:05:00"
    assert row["error"] is None


# --- role_salary_stats -------------------------------------------------------

def test_role_salary_stats_none_when_no_matching_titles(tmp_path):
    conn = _conn(tmp_path)
    upsert_job(conn, _job(title="Analytics Engineer", salary_min=210000, salary_max=250000), "t0")
    assert role_salary_stats(conn, r"\bai engineer\b") is None


def test_role_salary_stats_computes_real_stats(tmp_path):
    conn = _conn(tmp_path)
    upsert_job(conn, _job(external_id="1", title="Analytics Engineer", salary_min=150000, salary_max=190000), "t0")
    upsert_job(conn, _job(external_id="2", title="Senior Analytics Engineer", salary_min=200000, salary_max=250000), "t0")

    stats = role_salary_stats(conn, r"\banalytics engineer\b")
    assert stats["n"] == 2
    assert stats["min"] == 150000
    assert stats["max"] == 250000
    assert stats["median_min"] == 175000
    assert stats["median_max"] == 220000


def test_role_salary_stats_filters_out_values_below_min_sane(tmp_path):
    # A parse artifact (e.g. an hourly rate) must not corrupt the stats.
    conn = _conn(tmp_path)
    upsert_job(conn, _job(external_id="1", title="Analytics Engineer", salary_min=45, salary_max=200000), "t0")
    upsert_job(conn, _job(external_id="2", title="Analytics Engineer", salary_min=210000, salary_max=250000), "t0")

    stats = role_salary_stats(conn, r"\banalytics engineer\b", min_sane=30000)
    assert stats["n"] == 1  # the salary_min=45 row is excluded
    assert stats["min"] == 210000


# --- role_fit snapshots -----------------------------------------------------

def test_get_latest_role_fit_snapshot_returns_none_when_never_saved(tmp_path):
    conn = _conn(tmp_path)
    assert get_latest_role_fit_snapshot(conn) is None


def test_save_and_get_latest_role_fit_snapshot_round_trips(tmp_path):
    conn = _conn(tmp_path)
    results = {"roles": [{"role": "Analytics Engineer", "match_pct": 0.83}]}
    save_role_fit_snapshot(conn, "2026-08-02T00:00:00", results)

    snap = get_latest_role_fit_snapshot(conn)
    assert snap["computed_at"] == "2026-08-02T00:00:00"
    assert snap["results"] == results


def test_get_latest_role_fit_snapshot_returns_the_most_recent_one(tmp_path):
    conn = _conn(tmp_path)
    save_role_fit_snapshot(conn, "2026-08-01T00:00:00", {"roles": ["old"]})
    save_role_fit_snapshot(conn, "2026-08-02T00:00:00", {"roles": ["new"]})

    snap = get_latest_role_fit_snapshot(conn)
    assert snap["computed_at"] == "2026-08-02T00:00:00"
    assert snap["results"] == {"roles": ["new"]}


# --- is_active -------------------------------------------------------------

def test_job_is_active_when_seen_in_latest_successful_run(tmp_path):
    conn = _conn(tmp_path)
    run_id = start_search_run(conn, "mock", "q", "t0")
    upsert_job(conn, _job(external_id="a"), "t0", run_id)
    finish_search_run(conn, run_id, "t1", n_found=1, n_new=1)

    jobs = list_top_jobs(conn, only_passing=False, only_active=False)
    assert jobs[0].is_active is True


def test_job_is_inactive_when_missing_from_latest_successful_run(tmp_path):
    # Query matches the job title ("Engineer") on purpose - the delisted
    # inference only applies to jobs the latest run actually searched for.
    conn = _conn(tmp_path)
    run1 = start_search_run(conn, "mock", "engineer", "t0")
    upsert_job(conn, _job(external_id="a"), "t0", run1)
    upsert_job(conn, _job(external_id="b"), "t0", run1)
    finish_search_run(conn, run1, "t1", n_found=2, n_new=2)

    # Second poll only finds "b" - "a" has dropped out of the source's results.
    run2 = start_search_run(conn, "mock", "engineer", "t2")
    upsert_job(conn, _job(external_id="b"), "t2", run2)
    finish_search_run(conn, run2, "t3", n_found=1, n_new=0)

    everyone = {j.external_id: j.is_active for j in list_top_jobs(conn, only_passing=False, only_active=False)}
    assert everyone == {"a": False, "b": True}

    active_only = list_top_jobs(conn, only_passing=False, only_active=True)
    assert [j.external_id for j in active_only] == ["b"]


def test_job_stays_active_when_source_cannot_enumerate_all_matches(tmp_path):
    # § DECISIONS.md 2026-08-02: a capped/ranked source (LinkedIn via Apify)
    # gives absence no meaning - a live job that merely ranked below the cap
    # is indistinguishable from a delisted one. Real case: a Higharc
    # "Analytics Engineer" posting (score 262) was labelled "no longer
    # accepting" off a 15/15-truncated result set while still being open.
    conn = _conn(tmp_path)
    run1 = start_search_run(conn, "mock", "engineer", "t0", enumerates_all=False)
    upsert_job(conn, _job(external_id="a"), "t0", run1)
    upsert_job(conn, _job(external_id="b"), "t0", run1)
    finish_search_run(conn, run1, "t1", n_found=2, n_new=2)

    run2 = start_search_run(conn, "mock", "engineer", "t2", enumerates_all=False)
    upsert_job(conn, _job(external_id="b"), "t2", run2)
    finish_search_run(conn, run2, "t3", n_found=1, n_new=0)

    everyone = {j.external_id: j.is_active for j in list_top_jobs(conn, only_passing=False, only_active=False)}
    assert everyone == {"a": True, "b": True}, "a truncated result set must not imply delisting"


def test_job_stays_active_when_latest_run_did_not_search_for_it(tmp_path):
    # The second way absence is meaningless: the latest run never LOOKED for
    # this job (its title matches no current keyword). Greenhouse filters a
    # full board client-side by keyword, so changing the keyword list silently
    # dropped 175 still-open jobs into "no longer accepting".
    conn = _conn(tmp_path)
    run1 = start_search_run(conn, "mock", "engineer,scientist", "t0")
    upsert_job(conn, _job(external_id="eng", title="Analytics Engineer"), "t0", run1)
    upsert_job(conn, _job(external_id="sci", title="Data Scientist"), "t0", run1)
    finish_search_run(conn, run1, "t1", n_found=2, n_new=2)

    # Keywords narrowed to "engineer" only - "Data Scientist" wasn't searched.
    run2 = start_search_run(conn, "mock", "engineer", "t2")
    upsert_job(conn, _job(external_id="eng", title="Analytics Engineer"), "t2", run2)
    finish_search_run(conn, run2, "t3", n_found=1, n_new=0)

    everyone = {j.external_id: j.is_active for j in list_top_jobs(conn, only_passing=False, only_active=False)}
    assert everyone == {"eng": True, "sci": True}, "out-of-scope != delisted"


def test_failed_run_does_not_mark_jobs_inactive(tmp_path):
    conn = _conn(tmp_path)
    run1 = start_search_run(conn, "mock", "q", "t0")
    upsert_job(conn, _job(external_id="a"), "t0", run1)
    finish_search_run(conn, run1, "t1", n_found=1, n_new=1)

    # A source outage must not be read as "every job it ever found is gone."
    run2 = start_search_run(conn, "mock", "q", "t2")
    finish_search_run(conn, run2, "t3", n_found=0, n_new=0, error="network down")

    assert latest_successful_run_id(conn, "mock") == run1
    jobs = list_top_jobs(conn, only_passing=False, only_active=True)
    assert [j.external_id for j in jobs] == ["a"]


def test_no_search_runs_yet_defaults_to_active(tmp_path):
    conn = _conn(tmp_path)
    upsert_job(conn, _job(external_id="a"), "t0")  # no run_id at all
    jobs = list_top_jobs(conn, only_passing=False, only_active=True)
    assert [j.external_id for j in jobs] == ["a"]


# --- fit_summary -------------------------------------------------------------

def test_new_job_has_no_fit_summary_by_default(tmp_path):
    conn = _conn(tmp_path)
    job_id, _ = upsert_job(conn, _job(), "t0")
    assert get_job(conn, job_id).fit_summary is None


def test_set_fit_summary_roundtrips(tmp_path):
    conn = _conn(tmp_path)
    job_id, _ = upsert_job(conn, _job(), "t0")
    set_fit_summary(conn, job_id, "Strong fit on dbt/SQL, gap on Kubernetes.")
    assert get_job(conn, job_id).fit_summary == "Strong fit on dbt/SQL, gap on Kubernetes."


def test_equity_signals_roundtrip(tmp_path):
    conn = _conn(tmp_path)
    job = _job(equity_signals=["equity", "series_b", "founding"])
    job_id, _ = upsert_job(conn, job, "t0")
    assert get_job(conn, job_id).equity_signals == ["equity", "series_b", "founding"]


def test_missing_skills_roundtrip(tmp_path):
    conn = _conn(tmp_path)
    job = _job(missing_skills=["airflow", "fivetran"])
    job_id, _ = upsert_job(conn, job, "t0")
    assert get_job(conn, job_id).missing_skills == ["airflow", "fivetran"]


def test_equity_signals_default_empty(tmp_path):
    conn = _conn(tmp_path)
    job_id, _ = upsert_job(conn, _job(), "t0")
    assert get_job(conn, job_id).equity_signals == []


def test_rescoring_a_job_does_not_wipe_a_cached_fit_summary(tmp_path):
    # The scheduler re-scores (upsert_job) the same job every time it's seen
    # again in a later search pass - that must never silently erase an
    # already-generated summary, since generating a new one costs real money
    # and the candidate may have already read the cached one.
    conn = _conn(tmp_path)
    job_id, _ = upsert_job(conn, _job(external_id="a", score=100.0), "t0")
    set_fit_summary(conn, job_id, "Good fit summary.")

    upsert_job(conn, _job(external_id="a", score=150.0), "t1")

    stored = get_job(conn, job_id)
    assert stored.score == 150.0  # the re-score itself did take effect
    assert stored.fit_summary == "Good fit summary."  # but this survived it


# --- staleness fallback for non-enumerating sources (2026-08-02) ------------

def test_stale_job_from_non_enumerating_source_is_marked_inactive(tmp_path):
    # "Never mark inactive for a capped source" was too absolute: two Netflix
    # listings last confirmed 11 days ago (since expired) were still shown as
    # top Open matches. One truncated run proves nothing; 10+ days of silence
    # is real evidence.
    import datetime as _dt
    conn = _conn(tmp_path)
    old = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=11)).isoformat()
    fresh = _dt.datetime.now(_dt.timezone.utc).isoformat()

    run = start_search_run(conn, "mock", "engineer", "t0", enumerates_all=False)
    upsert_job(conn, _job(external_id="stale"), old, run)
    upsert_job(conn, _job(external_id="fresh"), fresh, run)
    finish_search_run(conn, run, "t1", n_found=2, n_new=2)

    got = {j.external_id: j.is_active
           for j in list_top_jobs(conn, only_passing=False, only_active=False)}
    assert got == {"stale": False, "fresh": True}


def test_recently_seen_job_from_capped_source_stays_active(tmp_path):
    # The Higharc case must NOT regress: missing from the latest truncated
    # run, but seen recently, so still open.
    import datetime as _dt
    conn = _conn(tmp_path)
    recent = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=2)).isoformat()

    run1 = start_search_run(conn, "mock", "engineer", "t0", enumerates_all=False)
    upsert_job(conn, _job(external_id="a"), recent, run1)
    finish_search_run(conn, run1, "t1", n_found=1, n_new=1)
    run2 = start_search_run(conn, "mock", "engineer", "t2", enumerates_all=False)
    finish_search_run(conn, run2, "t3", n_found=0, n_new=0)

    jobs = list_top_jobs(conn, only_passing=False, only_active=False)
    assert jobs[0].is_active is True


# --- rescore_all persists the SKILL columns too -----------------------------

def test_rescore_all_refreshes_skills_against_the_current_profile(tmp_path):
    # The real bug: an ad-hoc re-score updated salary/hard_pass/score but not
    # matched_skills/missing_skills, so 450 of 1123 jobs listed skills the
    # operator had since ADDED to their profile as still "missing".
    from candidate_profile import CandidateProfile
    from config import HardCriteria
    from db import rescore_all
    from scoring import score_job

    conn = _conn(tmp_path)
    upsert_job(conn, _job(external_id="j1", description="We use dbt and tableau daily."), "t0")
    # Simulate the stale state: tableau recorded as missing.
    conn.execute("UPDATE jobs SET missing_skills = ?", ('["tableau"]',))
    conn.commit()

    # ...but the profile now HAS tableau.
    profile = CandidateProfile(skills=("dbt", "tableau"), anchor_tools=("dbt",), anchor_skills=())
    rescore_all(conn, profile, HardCriteria(), score_job)

    job = list_top_jobs(conn, only_passing=False, only_active=False)[0]
    assert "tableau" not in job.missing_skills, "a skill on the profile must not be reported missing"
    assert "tableau" in job.matched_skills


def test_rescore_all_persists_breakdown_changes_even_when_score_is_unchanged(tmp_path):
    # Found 2026-08-11: rescore_all's change check compared only
    # score/hard_pass/skills, so a row whose score_breakdown changed but whose
    # score didn't was SKIPPED - leaving anchor_matches listing anchors the
    # profile no longer had. Same partial-persistence class of bug the
    # function exists to prevent, repeated inside the function itself.
    from candidate_profile import CandidateProfile
    from config import HardCriteria
    from db import rescore_all
    from scoring import score_job

    conn = _conn(tmp_path)
    upsert_job(conn, _job(
        external_id="j1",
        description="We use dbt, snowflake and python for analytics. Base pay $250,000 - $300,000 per year.",
        work_type="fulltime",
        location_mode="remote", location_country="USA",
    ), "t0")

    wide = CandidateProfile(skills=("dbt", "snowflake", "python"),
                            anchor_tools=("dbt", "python"), anchor_skills=())
    rescore_all(conn, wide, HardCriteria(), score_job)
    before = list_top_jobs(conn, only_passing=False, only_active=False)[0]
    assert "python" in before.score_breakdown["anchor_matches"]
    score_before = before.score

    # Narrow the anchors. dbt still matches, so hard_pass and the SCORE are
    # unchanged - only the breakdown differs. It must still be persisted.
    narrow = CandidateProfile(skills=("dbt", "snowflake", "python"),
                              anchor_tools=("dbt",), anchor_skills=())
    rescore_all(conn, narrow, HardCriteria(), score_job)
    after = list_top_jobs(conn, only_passing=False, only_active=False)[0]

    assert after.score == score_before, "precondition: score unchanged, so only the breakdown differs"
    assert "python" not in after.score_breakdown["anchor_matches"]
    assert after.score_breakdown["anchor_matches"] == ["dbt"]


def test_rescore_all_keeps_a_structurally_set_salary_not_present_in_the_text(tmp_path):
    # rescore_all clears salary to force a re-parse (parse_salary has been
    # fixed repeatedly, so old rows carry old mistakes). But some sources -
    # mock_source, and manually inserted rows - set salary_min/max
    # structurally WITHOUT repeating it in the description. Clearing it
    # unconditionally deleted real data on every re-score.
    from candidate_profile import CandidateProfile
    from config import HardCriteria
    from db import rescore_all
    from scoring import score_job

    conn = _conn(tmp_path)
    upsert_job(conn, _job(
        external_id="structural",
        description="Great team. No pay figure anywhere in this text.",
        salary_min=250000, salary_max=300000,
        work_type="fulltime", location_mode="remote", location_country="USA",
    ), "t0")

    profile = CandidateProfile(skills=("dbt",), anchor_tools=("dbt",), anchor_skills=())
    rescore_all(conn, profile, HardCriteria(), score_job)

    job = list_top_jobs(conn, only_passing=False, only_active=False)[0]
    assert job.salary_min == 250000, "a structurally-provided salary must survive a re-score"
    assert job.salary_max == 300000
