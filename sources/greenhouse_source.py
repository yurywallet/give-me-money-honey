"""Greenhouse public job-board API (§ docs/ADDING_A_JOB_SOURCE.md checklist).

Public, free, no-auth JSON API, ONE BOARD PER COMPANY:
boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true

Verified live 2026-07-30 against stripe (540 jobs), airbnb (189), instacart
(125), aurorainnovation (Aurora's actual board token - NOT "aurora", board
tokens don't always match the obvious brand name). netflix/google confirmed
NOT on Greenhouse (or Lever, checked too) under any reasonable token guess -
they run proprietary in-house career platforms with no public API equivalent,
so this source structurally cannot cover every company.

No cross-company keyword search exists here, unlike LinkedIn's platform-wide
crawl - each call fetches ONE company's full job list, filtered client-side
by keyword. Company list is config-driven (GREENHOUSE_COMPANIES env var,
comma-separated board tokens) - the whole point of this source is naming
companies you trust, not a platform-wide search.

Field mapping, verified against a real sample (Instacart's "Account Manager"):
- location_mode: `location.name` states it in plain text (e.g. "United
  States - Remote", "SF (Hybrid)") - trusted ONLY from this label field,
  checked as bare "hybrid"/"remote" substrings (scoring.describes_hybrid_
  location requires "hybrid" to co-occur with a second word, right for
  scanning JD prose, wrong for an already-unambiguous label); "onsite" still
  goes through scoring.describes_onsite_location, which has no such gap.
  Deliberately does NOT fall back to scanning the `content` body - verified
  live (2026-07-30) that doing so produces real false positives on
  Greenhouse's longer, more narrative postings (Airbnb's "in-person
  experiences and services" describing the COMPANY'S PRODUCT, and
  "experience leading hybrid teams (onsite/remote)" describing a required
  SKILL - neither describes the role's own arrangement). If location_raw is
  silent, the result is None ("unconfirmed") - same discipline as
  linkedin_apify_source's Zoox fix (2026-07-29): never default to a passing
  value just because nothing contradicts it.
- work_type: `metadata` is a list of PER-COMPANY-CONFIGURABLE custom fields -
  field NAMES vary (seen: "Time Type" = "Full time" on Instacart's board), so
  this searches by name pattern, not a fixed key, falling back to description
  text, else None.
- salary: no structured field in this API at all - relies entirely on
  scoring.parse_salary()'s text fallback (many US postings state it inline
  for pay-transparency-law compliance, e.g. CA/NY/CO).
"""
from __future__ import annotations

import html as html_lib
import os
import re

import httpx

from db import Job
from scoring import describes_onsite_location

GREENHOUSE_BASE_URL = "https://boards-api.greenhouse.io/v1/boards"

_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_WORK_TYPE_METADATA_NAMES = ("time type", "employment type", "job type")
_US_STATE_ABBR = (
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA",
    "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT",
    "VA", "WA", "WV", "WI", "WY", "DC",
)
_US_STATE_RE = re.compile(r",\s*(" + "|".join(_US_STATE_ABBR) + r")\b")
# Case-sensitive on purpose: bare "US" (as in "US Remote", "SF, US") is a
# common Greenhouse convention, but lowercase "us" collides with the common
# pronoun - requiring the capitalized form avoids that false-positive class.
_BARE_US_RE = re.compile(r"\bUS\b")
_FOREIGN_COUNTRY_HINTS = (
    "canada", "united kingdom", " uk", "india", "germany", "france",
    "australia", "mexico", "brazil", "netherlands", "ireland", "spain",
    "italy", "poland", "philippines", "singapore", "japan", "china",
)


def _clean_html(raw: str) -> str:
    """`content` is HTML, double-escaped in the JSON - one unescape() pass
    only resolves the outer layer, leaving entities like "&mdash;" as that
    literal string rather than an actual em dash. Verified live (2026-08-01):
    real salary disclosures ("$272,000 &mdash; $340,000 USD") were silently
    unparseable by scoring.parse_salary() because of this - a single-pass
    unescape leaves a dash the regex doesn't recognize, so genuinely
    above-floor jobs were failing the hard gate as "salary unconfirmed"."""
    unescaped = html_lib.unescape(html_lib.unescape(raw or ""))
    text = _TAG_RE.sub(" ", unescaped)
    return _WHITESPACE_RE.sub(" ", text).strip()


class GreenhouseSource:
    name = "greenhouse"
    # Each call fetches a company's ENTIRE public board and filters
    # client-side, so if a job is gone from the response it really was
    # taken down - absence is meaningful here, unlike LinkedIn's capped
    # search. See sources/__init__.py.
    enumerates_all_matches = True

    def __init__(self, company_tokens: list[str] | None = None):
        self.company_tokens = company_tokens or [
            t.strip() for t in os.getenv("GREENHOUSE_COMPANIES", "").split(",") if t.strip()
        ]

    def search(self, keywords: list[str]) -> list[Job]:
        if not self.company_tokens:
            return []
        keyword_patterns = [re.compile(re.escape(k), re.IGNORECASE) for k in keywords]
        jobs: list[Job] = []
        for token in self.company_tokens:
            try:
                resp = httpx.get(
                    f"{GREENHOUSE_BASE_URL}/{token}/jobs", params={"content": "true"}, timeout=30
                )
                resp.raise_for_status()
            except httpx.HTTPError:
                # One company's board being down/renamed must not kill the
                # whole search - per-company isolation, one level deeper than
                # scheduler.py's per-source isolation.
                continue
            for item in resp.json().get("jobs", []):
                title = item.get("title", "")
                if keyword_patterns and not any(p.search(title) for p in keyword_patterns):
                    continue
                jobs.append(self._map_item(item))
        return jobs

    def _map_item(self, item: dict) -> Job:
        description = _clean_html(item.get("content", ""))
        location_raw = (item.get("location") or {}).get("name", "")

        return Job(
            source=self.name,
            external_id=str(item["id"]),
            title=item.get("title", ""),
            company=item.get("company_name", ""),
            url=item.get("absolute_url", ""),
            description=description,
            salary_min=None,
            salary_max=None,
            work_type=self._infer_work_type(item.get("metadata") or [], description),
            location_mode=self._infer_location_mode(location_raw),
            location_country=self._infer_location_country(location_raw),
            location_raw=location_raw,
            posted_at=item.get("first_published"),
        )

    def _infer_location_mode(self, location_raw: str) -> str | None:
        """Deliberately does NOT fall back to scanning the description body -
        verified live (2026-07-30) that this produces real false positives on
        Greenhouse's longer, more narrative postings: Airbnb's "in-person
        experiences and services" describes the COMPANY'S PRODUCT (its
        Experiences marketplace), and "experience leading hybrid teams
        (onsite/remote)" describes a required SKILL (managing a mixed team),
        neither describes the role's OWN work arrangement - both matched the
        same detector that correctly reads LinkedIn's typically terser text.
        Greenhouse's own convention states work arrangement directly in
        location_raw (e.g. "United States - Remote", "SF (Hybrid)"); if
        that's silent, the honest answer is unconfirmed, not a guess pulled
        from a much longer, unrelated document."""
        # "hybrid"/"remote" are checked as bare substrings, not via scoring's
        # describes_hybrid_location - that helper requires "hybrid" to
        # co-occur with a second word (office/days/week), which is right for
        # scanning JD prose (bare "hybrid" alone is ambiguous - "hybrid
        # cloud", "hybrid car") but wrong for a location LABEL field, where a
        # bare "(Hybrid)" tag is already unambiguous on its own. onsite still
        # goes through describes_onsite_location, since "onsite"/"on-site"
        # are already standalone (not proximity-gated) alternatives there.
        raw_lower = location_raw.lower()
        if "hybrid" in raw_lower:
            return "hybrid"
        if "remote" in raw_lower:
            return "remote"
        if describes_onsite_location(location_raw):
            return "onsite"
        return None

    def _infer_work_type(self, metadata: list[dict], description: str) -> str | None:
        for field in metadata:
            name = str(field.get("name", "")).lower()
            if any(hint in name for hint in _WORK_TYPE_METADATA_NAMES):
                value = str(field.get("value", "")).lower()
                if "full" in value:
                    return "fulltime"
                if "part" in value:
                    return "parttime"
                if "contract" in value or "temp" in value:
                    return "contract"
                if "intern" in value:
                    return "internship"
        text = description.lower()
        if "full-time" in text or "full time" in text:
            return "fulltime"
        if "part-time" in text or "part time" in text:
            return "parttime"
        return None

    def _infer_location_country(self, location_raw: str) -> str | None:
        text = location_raw.lower()
        if any(hint in text for hint in _FOREIGN_COUNTRY_HINTS):
            return None
        if (
            "united states" in text
            or "usa" in text
            or _US_STATE_RE.search(location_raw)
            or _BARE_US_RE.search(location_raw)
        ):
            return "USA"
        return None
