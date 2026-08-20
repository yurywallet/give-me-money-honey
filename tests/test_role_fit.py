from candidate_profile import CandidateProfile
from role_fit import ROLE_SKILLS, compute_role_fit, rank_anchor_candidates, rank_complementary_skills


def test_role_skills_map_is_non_empty_for_every_role():
    for role, skills in ROLE_SKILLS.items():
        assert skills, f"{role} has no skills defined"


def test_compute_role_fit_splits_matched_and_missing():
    profile = CandidateProfile(
        skills=("sql", "dbt", "python"),
        anchor_tools=(), anchor_skills=(),
    )
    fits = {rf.role: rf for rf in compute_role_fit(profile)}
    ae = fits["Analytics Engineer"]
    assert "sql" in ae.matched and "dbt" in ae.matched and "python" in ae.matched
    assert "snowflake" in ae.missing
    assert ae.match_pct == len(ae.matched) / len(ae.role_skills)


def test_anchor_gap_excludes_already_anchored_skills():
    profile = CandidateProfile(
        skills=("sql", "dbt"),
        anchor_tools=("dbt",), anchor_skills=(),
    )
    fits = {rf.role: rf for rf in compute_role_fit(profile)}
    ae = fits["Analytics Engineer"]
    assert "dbt" not in ae.anchor_gap  # already anchored
    assert "sql" in ae.anchor_gap      # matched, not yet anchored
    assert "dbt" in ae.anchored
    assert "sql" not in ae.anchored


def test_rank_anchor_candidates_orders_by_breadth_across_roles():
    profile = CandidateProfile(
        skills=("sql", "python"),  # both appear in multiple ROLE_SKILLS entries
        anchor_tools=(), anchor_skills=(),
    )
    fits = compute_role_fit(profile)
    ranked = rank_anchor_candidates(fits)
    ranked_skills = [s for s, _ in ranked]
    assert "sql" in ranked_skills
    assert "python" in ranked_skills
    # both should out-rank any single-role-only anchor candidate
    counts = dict(ranked)
    assert len(counts["sql"]) >= 1
    assert len(counts["python"]) >= 1


def test_rank_complementary_skills_orders_by_number_of_roles_needing_it():
    profile = CandidateProfile(skills=(), anchor_tools=(), anchor_skills=())
    fits = compute_role_fit(profile)
    ranked = rank_complementary_skills(fits)
    # sorted descending by breadth
    counts = [len(roles) for _, roles in ranked]
    assert counts == sorted(counts, reverse=True)
    # a skill needed by every role should be at/near the top
    top_skill, top_roles = ranked[0]
    assert len(top_roles) >= 2


def test_rank_complementary_skills_excludes_skills_the_candidate_already_has():
    profile = CandidateProfile(skills=("python",), anchor_tools=(), anchor_skills=())
    fits = compute_role_fit(profile)
    ranked_skills = [s for s, _ in rank_complementary_skills(fits)]
    assert "python" not in ranked_skills


def test_anchor_only_skill_counts_as_matched_and_anchored():
    # "dashboards" listed as an anchor_skill but not under skills: anchors
    # aren't a subset of skills, so it must not read as a gap (2026-08-18).
    profile = CandidateProfile(
        skills=("sql",), anchor_tools=(), anchor_skills=("dashboards",),
    )
    da = next(rf for rf in compute_role_fit(profile) if rf.role == "Data Analyst")
    assert "dashboards" in da.matched
    assert "dashboards" not in da.missing
    assert "dashboards" in da.anchored
    assert "dashboards" not in da.anchor_gap
