"""LinkedIn jobs via Apify, using a search URL with LinkedIn's OWN remote
filter (`f_WT=2`) instead of an actor-specific parameter.

Why this exists alongside linkedin_apify_source.py (§ DECISIONS.md
2026-08-02): that source's actor silently IGNORES its `workplaceType` /
`jobType` parameters - verified by A/B, `workplaceType=2` and
`workplaceType=all` return byte-identical result sets, same IDs in the same
order - and never populates a workplace-type output field, so every search
came back unfiltered with no remote signal to read. `valig`'s actor has the
same problem (its `remote` param is ignored too; its `workType` field is job
FUNCTION, "Information Technology", not work arrangement).

What actually works is passing LinkedIn's own query parameter in the search
URL. Verified live 2026-08-02 against `curious_coder/linkedin-jobs-scraper`
(which accepts `urls`): `f_WT=2` vs no filter returned only 4/15 overlapping
jobs, and the filtered set's JDs really do say "This role has been
categorized as a Remote position", "remote in the US", "open to remote
candidates across the United States" - while the unfiltered set was all
Boston/SF/Santa Clara with no remote language.

THE REMOTE-TRUST RULE, and why it doesn't contradict the Zoox fix:
`_infer_location_mode` here defaults to "remote" when the JD says nothing,
which the Zoox fix (2026-07-29) explicitly forbade for the other source.
The difference is evidence, not preference. That rule exists because
defaulting to an UNVERIFIED filter's intent invents a fact; here the filter
is verified to actually work, so membership in the result set IS evidence.
An explicit onsite/hybrid statement in the JD still overrides it - text
beats the filter, same precedence as the other source.

Cost: ~$0.001/job (~2.4x linkedin_apify_source) - measured over the
2026-08-02 trial runs. Selected via GMMH_LINKEDIN_ACTOR (see
sources/linkedin_selector.py); switch back to the cheaper source if the
Apify credit runs low.
"""
from __future__ import annotations

import os
from urllib.parse import quote_plus

import httpx

from db import Job
from scoring import describes_hybrid_location, describes_onsite_location
from sources.linkedin_apify_source import (
    _FOREIGN_COUNTRY_HINTS,
    _LOW_YIELD_KEYWORDS,
    APIFY_BASE_URL,
)

# LinkedIn's own search-URL filters (not the actor's). f_WT=2 is Remote,
# f_JT=F is Full-time - these are what the actor's own equivalents failed to
# apply. Verified live 2026-08-02; if LinkedIn renames them this source's
# whole premise breaks, so re-run the A/B (filtered vs unfiltered result
# sets must differ) before trusting it again.
_LINKEDIN_SEARCH_URL = "https://www.linkedin.com/jobs/search/"
_REMOTE_PARAM = "f_WT=2"
_FULLTIME_PARAM = "f_JT=F"


class LinkedInRemoteApifySource:
    name = "linkedin_apify"  # same namespace as the legacy source on purpose:
    # both return LinkedIn job IDs (verified 2026-08-02 - three IDs appeared in
    # both actors' results for the same query), so dedup by
    # (source, external_id) keeps working across a switch and history is
    # continuous rather than forking into a second "source".

    # Still a capped search - absence from a result set proves nothing
    # (§ DECISIONS.md 2026-08-02, annotate_active).
    enumerates_all_matches = False

    def __init__(self, token: str | None = None, actor_id: str | None = None):
        self.token = token or os.getenv("APIFY_TOKEN")
        self.actor_id = actor_id or os.getenv(
            "GMMH_LINKEDIN_REMOTE_ACTOR_ID", "curious_coder~linkedin-jobs-scraper"
        )
        if not self.token:
            raise RuntimeError("APIFY_TOKEN must be set before LinkedInRemoteApifySource can run.")
        self.location = "United States"
        self.max_jobs_per_keyword = 15  # cost cap, same as the legacy source

    def _search_url(self, keyword: str) -> str:
        return (
            f"{_LINKEDIN_SEARCH_URL}?keywords={quote_plus(keyword)}"
            f"&location={quote_plus(self.location)}&{_REMOTE_PARAM}&{_FULLTIME_PARAM}"
        )

    def search(self, keywords: list[str]) -> list[Job]:
        seen_ids: set[str] = set()
        jobs: list[Job] = []
        url = f"{APIFY_BASE_URL}/acts/{self.actor_id}/run-sync-get-dataset-items"
        for keyword in keywords:
            if keyword.strip().lower() in _LOW_YIELD_KEYWORDS:
                continue
            resp = httpx.post(
                url,
                params={"token": self.token},
                json={
                    "urls": [self._search_url(keyword)],
                    "limitPerSource": self.max_jobs_per_keyword,
                    "scrapeCompany": False,
                },
                timeout=300,
            )
            resp.raise_for_status()
            for item in resp.json():
                job = self._map_item(item)
                if not job.external_id or job.external_id in seen_ids:
                    continue
                seen_ids.add(job.external_id)
                jobs.append(job)
        return jobs

    def _map_item(self, item: dict) -> Job:
        # Field names verified against real items (2026-08-02): id, title,
        # companyName, link, applyUrl, location, employmentType, salary,
        # postedAt, descriptionText. `applyUrl` is often an empty string and
        # `link` carries the real URL, so `link` is preferred.
        description = item.get("descriptionText") or item.get("descriptionHtml") or ""
        salary_text = item.get("salary")
        if salary_text:
            # Prepend so scoring.parse_salary's text fallback can read it -
            # same approach as the legacy source.
            description = f"{salary_text}\n{description}"

        employment_type = item.get("employmentType")
        work_type = (
            "fulltime"
            if employment_type and "full" in str(employment_type).lower()
            else (employment_type or "fulltime")  # the URL filtered to f_JT=F
        )

        location_raw = item.get("location") or ""
        return Job(
            source=self.name,
            external_id=str(item.get("id") or ""),
            title=item.get("title", ""),
            company=item.get("companyName", ""),
            url=item.get("link") or item.get("applyUrl") or "",
            description=description,
            salary_min=None,
            salary_max=None,
            work_type=work_type,
            location_mode=self._infer_location_mode(location_raw, description),
            location_country=self._infer_location_country(location_raw),
            location_raw=location_raw,
            posted_at=item.get("postedAt"),
        )

    def _infer_location_mode(self, location_raw: str, description: str) -> str | None:
        """Remote unless the JD explicitly says otherwise - see the
        REMOTE-TRUST RULE in this module's docstring. Text still wins over
        the filter, so a posting that says "3 days/week in office" is hybrid
        even though it came back from a remote-filtered search."""
        if describes_onsite_location(location_raw) or describes_onsite_location(description):
            return "onsite"
        if describes_hybrid_location(location_raw) or describes_hybrid_location(description):
            return "hybrid"
        return "remote"

    def _infer_location_country(self, location_raw: str) -> str | None:
        text = location_raw.lower()
        if any(hint in text for hint in _FOREIGN_COUNTRY_HINTS):
            return None
        if "united states" in text or "usa" in text:
            return "USA"
        # The search URL already pins location=United States; a bare US
        # city/state ("Austin, TX") carries no country token of its own.
        return "USA" if location_raw.strip() else None
