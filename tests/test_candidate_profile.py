import json
import os

from candidate_profile import DEFAULT_PROFILE, CandidateProfile, load_profile, save_profile


def test_load_profile_returns_default_when_no_file(tmp_path):
    path = str(tmp_path / "missing.json")
    assert load_profile(path) == DEFAULT_PROFILE


def test_save_then_load_round_trips(tmp_path):
    path = str(tmp_path / "profile.json")
    custom = CandidateProfile(
        title_keywords=("data analyst", "bi engineer"),
        skills=("sql", "tableau"),
        summary="A custom bio.",
        work_experience="Data Analyst at Acme, 2020-2023.",
        personal_projects="Built a job-search MCP server.",
        education="State University - B.S. Statistics, 2016-2020.",
        anchor_tools=("dbt", "looker"),
        anchor_skills=("dimensional modeling", "gtm"),
    )
    save_profile(custom, path)

    loaded = load_profile(path)
    assert loaded == custom


def test_load_profile_defaults_new_fields_when_missing_from_saved_json(tmp_path):
    # A profile saved before work_experience/personal_projects/education/anchor_*
    # existed must still load cleanly - backward compatibility for existing
    # candidate_profile.json files.
    path = str(tmp_path / "old_profile.json")
    with open(path, "w") as f:
        json.dump({"title_keywords": ["data analyst"], "skills": ["sql"], "summary": "Old bio."}, f)

    loaded = load_profile(path)
    assert loaded.work_experience == DEFAULT_PROFILE.work_experience
    assert loaded.personal_projects == DEFAULT_PROFILE.personal_projects
    assert loaded.education == DEFAULT_PROFILE.education
    assert loaded.anchor_tools == DEFAULT_PROFILE.anchor_tools
    assert loaded.anchor_skills == DEFAULT_PROFILE.anchor_skills


def test_save_profile_writes_json_file(tmp_path):
    path = str(tmp_path / "profile.json")
    save_profile(DEFAULT_PROFILE, path)
    assert os.path.exists(path)
