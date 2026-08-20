"""Real LinkedIn job source via an Apify actor (the "managed proxy" route -
Apify's proxy/browser pool does the scraping, this process never touches
your own LinkedIn session or credentials).

Calibrated against automation-lab/linkedin-jobs-scraper's real input schema
and a live sample item (engineering-foundations §3: a lookup/mapping that
"usually" matches is a live bug until verified against a raw sample - this
one has been). Real output fields seen: id, title, url, companyName,
location, postedAt, salary, workplaceType, employmentType, descriptionText,
descriptionHtml, benefits. If you swap in a different actor, this mapping
must be re-verified - field names vary per actor and change over time.

Two things NOT yet confirmed against a real example (the one live sample
pulled had neither): the exact free-text format of the `salary` field, and
whether `workplaceType`/`employmentType` are reliably populated per item
rather than sometimes null. Until then, this source treats its own request
filters (workplaceType="2"/remote, jobType="F"/fulltime) as authoritative
for a returned item's location_mode/work_type when those output fields
come back empty, since the actor is expected to only return matches for the
filters it was given.

Setup:
  1. Create an Apify account, get an API token.
  2. Pick a LinkedIn jobs actor from the Apify Store (search "linkedin jobs").
  3. Set APIFY_TOKEN and APIFY_ACTOR_ID in .env (format: "username~actor-name").
  4. Run scheduler.run_once() once and inspect the stored `description` and
     `benefits` fields on a couple of jobs before trusting the scores -
     especially if you're using a different actor than the one above.

Cost note: search() runs one actor invocation per keyword in
CandidateProfile.title_keywords (see that file - each is a separate targeted
search, not one joined query) MINUS _LOW_YIELD_KEYWORDS below (skipped here
only - Greenhouse/free sources still get the full list), at
max_jobs_per_keyword each. On Apify's Free plan ($5/month credit,
~$0.0006/job on this actor), 6 keywords x 15 jobs is ~$0.05 per manual
Search click - cheap. It adds up fast under periodic polling, though: at the
default 60-minute interval that's ~$1.2/day, enough to exhaust the monthly
$5 credit in under 5 days. Prefer manual searches over
scheduler.start()/start_periodic_search() while on the free plan, or raise
GMMH_POLL_INTERVAL_MINUTES substantially.
"""
from __future__ import annotations

import os

import httpx

from db import Job
from scoring import describes_hybrid_location, describes_onsite_location

APIFY_BASE_URL = "https://api.apify.com/v2"

_WORKPLACE_TYPE_NAMES = {"1": "onsite", "2": "remote", "3": "hybrid"}
_JOB_TYPE_NAMES = {"F": "fulltime", "P": "parttime", "C": "contract", "T": "temporary", "I": "internship"}

# Real `location` values are often just a US city/state ("New York, NY",
# "Los Angeles Metropolitan Area") with no literal country name - a bare
# substring check for "united states" misses nearly all of them. Since the
# actor's own search is already filtered to location="United States", treat
# that request as authoritative unless the location text itself names a
# different country.
_FOREIGN_COUNTRY_HINTS = (
    "canada", "united kingdom", " uk", "india", "germany", "france",
    "australia", "mexico", "brazil", "netherlands", "ireland", "spain",
    "italy", "poland", "philippines", "singapore", "japan", "china",
)

# Keywords with a real, measured ~0% $200K-floor clear rate (db.role_salary_stats
# against 746 real postings, § DECISIONS.md 2026-08-02: 0/28 Data Analyst,
# 0/11 BI Engineer postings with disclosed salary ever cleared $200K, vs.
# 30-33% for AI Engineer/AI Product Engineer). Skipped for THIS source
# specifically, since it's the only PAID one - each keyword is a separate
# billed Apify run. Not removed from CandidateProfile.title_keywords itself:
# GreenhouseSource (free) and the Role Map tab still use the full list.
_LOW_YIELD_KEYWORDS = {"data analyst", "bi engineer", "business intelligence engineer"}


class LinkedInApifySource:
    name = "linkedin_apify"
    # LinkedIn is a RANKED search capped at max_jobs_per_keyword - we never
    # see past the cap, so a missing job may simply have ranked lower this
    # run. Verified live 2026-08-02: a plain "analytics engineer" search
    # returned exactly 15/15 (i.e. truncated) and omitted a Higharc posting
    # that was still open, which the old logic then labelled "no longer
    # accepting". See sources/__init__.py for what this flag gates.
    enumerates_all_matches = False

    def __init__(self, token: str | None = None, actor_id: str | None = None):
        self.token = token or os.getenv("APIFY_TOKEN")
        self.actor_id = actor_id or os.getenv("APIFY_ACTOR_ID")
        if not self.token or not self.actor_id:
            raise RuntimeError(
                "APIFY_TOKEN and APIFY_ACTOR_ID must be set (see this file's docstring) "
                "before LinkedInApifySource can run."
            )
        self._workplace_type = "2"  # Remote - kept in sync with run()'s run_input
        self._job_type = "F"  # Full-time - kept in sync with run()'s run_input
        self._location = "United States"  # kept in sync with search()'s run_input
        self.max_jobs_per_keyword = 15  # cost cap; see module docstring

    def search(self, keywords: list[str]) -> list[Job]:
        # One actor run per keyword, not one run on a joined blob string -
        # "analytics engineer data analyst senior analytics engineer" as a
        # single LinkedIn search query is a worse, more diluted match than
        # three separate targeted searches. Dedup by external_id since the
        # same posting can legitimately surface under more than one keyword.
        seen_ids: set[str] = set()
        jobs: list[Job] = []
        url = f"{APIFY_BASE_URL}/acts/{self.actor_id}/run-sync-get-dataset-items"
        for keyword in keywords:
            if keyword.strip().lower() in _LOW_YIELD_KEYWORDS:
                continue
            run_input = {
                "searchQuery": keyword,
                "location": self._location,
                "jobType": self._job_type,
                "workplaceType": self._workplace_type,
                "maxJobs": self.max_jobs_per_keyword,
            }
            resp = httpx.post(url, params={"token": self.token}, json=run_input, timeout=180)
            resp.raise_for_status()
            for item in resp.json():
                job = self._map_item(item)
                if job.external_id in seen_ids:
                    continue
                seen_ids.add(job.external_id)
                jobs.append(job)
        return jobs

    def _map_item(self, item: dict) -> Job:
        description = item.get("descriptionText") or item.get("descriptionHtml") or ""
        salary_text = item.get("salary")
        if salary_text:
            # Feed the raw salary text into the description so the existing
            # parse_salary() fallback in scoring.py can extract it - format
            # unconfirmed (see module docstring), so no numeric value here.
            description = f"{salary_text}\n{description}"

        employment_type = item.get("employmentType")
        work_type = (
            "fulltime" if employment_type and "full" in str(employment_type).lower()
            else _JOB_TYPE_NAMES.get(self._job_type) if not employment_type
            else employment_type
        )

        workplace_type = item.get("workplaceType")
        location_raw = item.get("location", "")
        location_mode = self._infer_location_mode(workplace_type, location_raw, description)

        location_country = self._infer_location_country(location_raw)

        return Job(
            source=self.name,
            external_id=str(item.get("id") or item.get("url")),
            title=item.get("title", ""),
            company=item.get("companyName", ""),
            url=item.get("url", item.get("applyUrl", "")),
            description=description,
            salary_min=None,
            salary_max=None,
            work_type=work_type,
            location_mode=location_mode,
            location_country=location_country,
            location_raw=location_raw,
            posted_at=item.get("postedAt"),
        )

    def _infer_location_mode(self, workplace_type: str | None, location_raw: str, description: str) -> str | None:
        # If the JD explicitly says the role is in-person, prefer that over
        # whatever workplaceType claims - text evidence beats a raw field.
        if self._is_onsite_listing(location_raw, description):
            return "onsite"
        if self._is_hybrid_listing(location_raw, description):
            return "hybrid"
        if str(workplace_type) in _WORKPLACE_TYPE_NAMES:
            return _WORKPLACE_TYPE_NAMES[str(workplace_type)]
        # workplaceType came back empty/null AND no text signal confirms or
        # contradicts remote either (operator-reported 2026-07-29: Zoox's
        # actual LinkedIn page shows a "Hybrid" pill - confirmed via
        # screenshot - but neither workplaceType nor the JD text state it
        # anywhere this source receives; a live WebFetch of the same URL
        # missed it too, so it's a genuine UI-chip field, not scrapeable
        # from any text this source has). Previously defaulted to the
        # REQUESTED filter (e.g. "remote") on the assumption the actor only
        # returns exact matches for what it was asked - proven false here.
        # Return None ("unconfirmed") instead: this fails hard_filter's
        # exact-match requirement like any other non-remote job, rather than
        # silently asserting a match the source cannot actually back up.
        return None

    def _is_hybrid_listing(self, location_raw: str, description: str) -> bool:
        if "hybrid" in location_raw.lower():
            return True
        # Do not mark hybrid if description explicitly says onsite.
        if describes_onsite_location(description):
            return False
        # describes_hybrid_location (scoring.py) is the single canonical
        # onsite/hybrid text detector - reused, not re-implemented, so the
        # source's field inference and the hard-gate cross-check (scoring.py's
        # contradicts_remote) can't drift apart (Pilot.com bug 2026-07-24,
        # Crunchyroll bug 2026-07-28).
        return describes_hybrid_location(description)

    def _is_onsite_listing(self, location_raw: str, description: str) -> bool:
        if any(k in (location_raw or "").lower() for k in ("office", "onsite", "on-site")):
            return True
        return describes_onsite_location(description)

    def _infer_location_country(self, location_raw: str) -> str | None:
        text = location_raw.lower()
        if any(hint in text for hint in _FOREIGN_COUNTRY_HINTS):
            return None
        if "united states" in text or "usa" in text or self._location.strip().lower() == "united states":
            return "USA"
        return None
