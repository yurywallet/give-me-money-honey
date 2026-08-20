import sources.linkedin_selector as selector
from sources.linkedin_remote_source import LinkedInRemoteApifySource
from sources.linkedin_selector import LEGACY, REMOTE, build_linkedin_source, selected_linkedin_mode


def _source():
    return LinkedInRemoteApifySource(token="fake", actor_id="fake~actor")


# Real item captured live 2026-08-02 from curious_coder/linkedin-jobs-scraper
# via a LinkedIn search URL carrying f_WT=2, trimmed to the fields _map_item
# reads. Note applyUrl is an empty string and `link` holds the real URL.
_REAL_ITEM = {
    "id": "4404289613",
    "title": "Senior / Staff Data Platform Engineer",
    "companyName": "Radar",
    "link": "https://www.linkedin.com/jobs/view/senior-staff-data-platform-engineer-4404289613",
    "applyUrl": "",
    "location": "United States",
    "employmentType": "Full-time",
    "salary": "",
    "postedAt": "2026-08-06",
    "descriptionText": "This role can be either in our NYC HQ or remote in the US. "
                       "You will build our dbt models and own the semantic layer.",
}


def test_maps_real_sample_correctly():
    job = _source()._map_item(_REAL_ITEM)
    assert job.source == "linkedin_apify"  # same namespace as the legacy source
    assert job.external_id == "4404289613"
    assert job.title == "Senior / Staff Data Platform Engineer"
    assert job.company == "Radar"
    assert job.url.endswith("4404289613")  # `link`, not the empty applyUrl
    assert job.work_type == "fulltime"
    assert job.location_country == "USA"
    assert "dbt models" in job.description


def test_search_url_carries_linkedins_own_remote_filter():
    # The whole premise of this source: the ACTOR's remote param is ignored,
    # LinkedIn's own f_WT=2 URL parameter is not (§ DECISIONS.md 2026-08-02).
    url = _source()._search_url("staff analytics engineer")
    assert "f_WT=2" in url          # Remote
    assert "f_JT=F" in url          # Full-time
    assert "staff+analytics+engineer" in url
    assert "United+States" in url


def test_location_mode_defaults_to_remote_from_the_verified_filter():
    # Unlike the legacy source (whose filter is ignored, so it must return
    # None/"unconfirmed"), membership in a VERIFIED remote-filtered result
    # set is real evidence - that's the point of this source existing.
    job = _source()._map_item(dict(_REAL_ITEM, descriptionText="Great team, great mission."))
    assert job.location_mode == "remote"


def test_explicit_onsite_or_hybrid_text_still_overrides_the_filter():
    # Text beats the filter - a posting that states an in-office schedule is
    # not remote just because it came back from a remote-filtered search.
    onsite = _source()._map_item(
        dict(_REAL_ITEM, descriptionText="This role is fully onsite in our Austin office.")
    )
    assert onsite.location_mode == "onsite"

    hybrid = _source()._map_item(
        dict(_REAL_ITEM, descriptionText="Hybrid role, 3 days/week in our SF office.")
    )
    assert hybrid.location_mode == "hybrid"


def test_foreign_location_is_not_marked_usa():
    job = _source()._map_item(dict(_REAL_ITEM, location="Toronto, Ontario, Canada"))
    assert job.location_country is None


def test_salary_text_is_prepended_so_parse_salary_can_read_it():
    job = _source()._map_item(dict(_REAL_ITEM, salary="$210,000.00/yr - $260,000.00/yr"))
    from scoring import parse_salary

    assert parse_salary(job.description) == (210000, 260000)


def test_source_cannot_enumerate_all_matches():
    # Still a capped search, so absence must not imply delisted.
    assert LinkedInRemoteApifySource.enumerates_all_matches is False


# --- selector (the switch-back mechanism) ----------------------------------

def test_selector_defaults_to_remote(monkeypatch):
    monkeypatch.delenv("GMMH_LINKEDIN_ACTOR", raising=False)
    assert selected_linkedin_mode() == REMOTE


def test_selector_honours_legacy(monkeypatch):
    monkeypatch.setenv("GMMH_LINKEDIN_ACTOR", "legacy")
    assert selected_linkedin_mode() == LEGACY


def test_selector_falls_back_to_default_on_a_typo(monkeypatch):
    # A typo in .env must not take the whole search offline.
    monkeypatch.setenv("GMMH_LINKEDIN_ACTOR", "lgeacy")
    assert selected_linkedin_mode() == REMOTE


def test_build_returns_none_without_a_token(monkeypatch):
    monkeypatch.delenv("APIFY_TOKEN", raising=False)
    assert build_linkedin_source() is None


def test_build_returns_remote_source_by_default(monkeypatch):
    monkeypatch.setenv("APIFY_TOKEN", "fake")
    monkeypatch.delenv("GMMH_LINKEDIN_ACTOR", raising=False)
    assert isinstance(build_linkedin_source(), LinkedInRemoteApifySource)


def test_build_returns_legacy_source_when_selected(monkeypatch):
    from sources.linkedin_apify_source import LinkedInApifySource

    monkeypatch.setenv("APIFY_TOKEN", "fake")
    monkeypatch.setenv("APIFY_ACTOR_ID", "fake~actor")
    monkeypatch.setenv("GMMH_LINKEDIN_ACTOR", "legacy")
    assert isinstance(build_linkedin_source(), LinkedInApifySource)


def test_build_returns_none_for_legacy_without_its_actor_id(monkeypatch):
    # The legacy source targets one specific actor and can't default it.
    monkeypatch.setenv("APIFY_TOKEN", "fake")
    monkeypatch.delenv("APIFY_ACTOR_ID", raising=False)
    monkeypatch.setenv("GMMH_LINKEDIN_ACTOR", "legacy")
    assert build_linkedin_source() is None


def test_both_sources_share_a_name_so_dedup_survives_a_switch(monkeypatch):
    # Verified live 2026-08-02: three job IDs appeared in BOTH actors'
    # results for the same query, i.e. one LinkedIn ID namespace.
    from sources.linkedin_apify_source import LinkedInApifySource

    assert LinkedInRemoteApifySource.name == LinkedInApifySource.name
