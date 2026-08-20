from sources.greenhouse_source import GreenhouseSource


def _source():
    return GreenhouseSource(company_tokens=["stripe"])


# Real item structure captured live 2026-07-30 from
# boards-api.greenhouse.io/v1/boards/instacart/jobs (Account Manager),
# trimmed to the fields _map_item actually reads.
_REAL_INSTACART_ITEM = {
    "id": 6088765,
    "title": "Account Manager",
    "company_name": "Instacart",
    "absolute_url": "https://instacart.careers/job/account-manager/",
    "content": "&lt;p&gt;We are looking for an &lt;strong&gt;Account Manager&lt;/strong&gt; "
    "to join our Ads team. Experience with SQL and data analysis required.&lt;/p&gt;",
    "location": {"name": "United States - Remote"},
    "metadata": [
        {"name": "Employment Type", "value": "Regular"},
        {"name": "Time Type", "value": "Full time"},
    ],
    "first_published": "2026-06-01T00:00:00Z",
}


def test_maps_real_sample_correctly():
    job = GreenhouseSource()._map_item(_REAL_INSTACART_ITEM)
    assert job.source == "greenhouse"
    assert job.external_id == "6088765"
    assert job.title == "Account Manager"
    assert job.company == "Instacart"
    assert job.location_mode == "remote"
    assert job.location_country == "USA"
    assert job.work_type == "fulltime"
    # HTML unescaped and tags stripped, not passed through raw
    assert "<p>" not in job.description and "&lt;" not in job.description
    assert "Account Manager" in job.description
    assert "SQL" in job.description


def test_double_escaped_salary_dash_is_parseable():
    # Regression lock (2026-08-01): real Datadog postings disclose salary as
    # "$272,000 &mdash; $340,000 USD" - the leading & gets escaped AGAIN by
    # Greenhouse's JSON layer ("&amp;mdash;"), so a single unescape() pass
    # left the literal string "&mdash;" in the description, which
    # scoring.parse_salary()'s dash regex doesn't recognize - real $272k-$340k
    # roles were silently failing the hard gate as "salary unconfirmed".
    item = dict(
        _REAL_INSTACART_ITEM,
        content="&lt;p&gt;The yearly salary for this role is: $272,000 &amp;mdash; $340,000 USD&lt;/p&gt;",
    )
    job = GreenhouseSource()._map_item(item)
    assert "&mdash;" not in job.description
    assert "$272,000 — $340,000" in job.description

    from scoring import parse_salary
    assert parse_salary(job.description) == (272000, 340000)


def test_location_mode_hybrid_from_location_raw():
    item = dict(_REAL_INSTACART_ITEM, location={"name": "San Francisco, CA (Hybrid)"})
    job = GreenhouseSource()._map_item(item)
    assert job.location_mode == "hybrid"


def test_location_mode_does_not_fall_back_to_description_text():
    # Regression lock (2026-07-30): a job whose location_raw is silent about
    # arrangement must stay unconfirmed even if the DESCRIPTION happens to
    # mention "hybrid"/"in-person" in an unrelated sense (a required skill,
    # or the company's own product) - verified false positives on real Airbnb
    # postings ("experience leading hybrid teams", "in-person experiences and
    # services" - a product line, not the job's own location).
    item = dict(
        _REAL_INSTACART_ITEM,
        location={"name": "United States"},
        content="Experience leading hybrid teams (onsite/remote) and in-person customer experiences required.",
    )
    job = GreenhouseSource()._map_item(item)
    assert job.location_mode is None


def test_location_country_excludes_foreign_locations():
    item = dict(_REAL_INSTACART_ITEM, location={"name": "Toronto, Canada"})
    job = GreenhouseSource()._map_item(item)
    assert job.location_country is None


def test_location_country_recognizes_bare_us_token():
    item = dict(_REAL_INSTACART_ITEM, location={"name": "CHI, SF, NYC, SEA, US Remote"})
    job = GreenhouseSource()._map_item(item)
    assert job.location_country == "USA"


def test_work_type_falls_back_to_description_when_metadata_silent():
    item = dict(_REAL_INSTACART_ITEM, metadata=[], content="This is a full-time role based in the US.")
    job = GreenhouseSource()._map_item(item)
    assert job.work_type == "fulltime"


def test_work_type_none_when_neither_metadata_nor_text_confirm():
    item = dict(_REAL_INSTACART_ITEM, metadata=[], content="A great opportunity to join our team.")
    job = GreenhouseSource()._map_item(item)
    assert job.work_type is None


def test_search_filters_by_keyword_in_title(monkeypatch):
    import sources.greenhouse_source as gh_module

    calls = []

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "jobs": [
                    dict(_REAL_INSTACART_ITEM, id=1, title="Data Analyst"),
                    dict(_REAL_INSTACART_ITEM, id=2, title="Software Engineer, Backend"),
                ]
            }

    def fake_get(url, params=None, timeout=None):
        calls.append(url)
        return FakeResponse()

    monkeypatch.setattr(gh_module.httpx, "get", fake_get)

    jobs = GreenhouseSource(company_tokens=["stripe"]).search(["data"])

    assert len(jobs) == 1
    assert jobs[0].title == "Data Analyst"
    assert calls == ["https://boards-api.greenhouse.io/v1/boards/stripe/jobs"]


def test_search_returns_empty_when_no_companies_configured(monkeypatch):
    monkeypatch.delenv("GREENHOUSE_COMPANIES", raising=False)
    assert GreenhouseSource(company_tokens=[]).search(["data"]) == []


def test_search_skips_a_company_whose_board_errors_without_killing_the_rest(monkeypatch):
    import httpx

    import sources.greenhouse_source as gh_module

    class FakeResponse:
        def __init__(self, ok):
            self.ok = ok

        def raise_for_status(self):
            if not self.ok:
                raise httpx.HTTPError("boom")

        def json(self):
            return {"jobs": [dict(_REAL_INSTACART_ITEM, id=99, title="Data Analyst")]}

    def fake_get(url, params=None, timeout=None):
        return FakeResponse(ok="broken-co" not in url)

    monkeypatch.setattr(gh_module.httpx, "get", fake_get)

    jobs = GreenhouseSource(company_tokens=["broken-co", "stripe"]).search(["data"])

    assert len(jobs) == 1  # broken-co's error didn't kill stripe's results
