"""A fully-working fake job source - no credentials, no network.

Exists so the scheduler, DB, scoring, and MCP tools can all be exercised
end to end (per engineering-foundations: "verify a change by actually
exercising it, not just green tests") without needing a real LinkedIn/Apify
account. Swap in linkedin_apify_source.LinkedInApifySource (or a new source)
for real data once credentials are configured - see docs/ADDING_A_JOB_SOURCE.md.
"""
from __future__ import annotations

from db import Job

_FAKE_LISTINGS = [
    dict(
        external_id="mock-1",
        title="Senior Backend Engineer",
        company="Nimbus Data",
        url="https://example.com/jobs/mock-1",
        description=(
            "Fully remote (US only) full-time role. Base salary $210,000-$240,000 "
            "plus bonus. We offer medical, dental, and vision insurance for you and "
            "your family, a 401k match, and unlimited PTO."
        ),
        salary_min=210_000,
        salary_max=240_000,
        work_type="fulltime",
        location_mode="remote",
        location_country="USA",
        location_raw="Remote - United States",
        posted_at="2026-07-10",
    ),
    dict(
        external_id="mock-2",
        title="Staff Platform Engineer",
        company="Ridgeline Systems",
        url="https://example.com/jobs/mock-2",
        description=(
            "Remote, full-time. Compensation up to $260k. Benefits include equity, "
            "parental leave, and a remote stipend. Health coverage available for "
            "employees; family dependents can be added at employee cost."
        ),
        salary_min=None,
        salary_max=260_000,
        work_type="fulltime",
        location_mode="remote",
        location_country="USA",
        location_raw="US Remote",
        posted_at="2026-07-11",
    ),
    dict(
        external_id="mock-3",
        title="Backend Engineer (Hybrid)",
        company="Local Foundry",
        url="https://example.com/jobs/mock-3",
        description="Hybrid role, 3 days/week in our Austin office. $190,000-$205,000.",
        salary_min=190_000,
        salary_max=205_000,
        work_type="fulltime",
        location_mode="hybrid",
        location_country="USA",
        location_raw="Austin, TX (Hybrid)",
        posted_at="2026-07-09",
    ),
    dict(
        external_id="mock-4",
        title="Senior Analytics Engineer",
        company="Beacon Analytics",
        url="https://example.com/jobs/mock-4",
        description=(
            "Fully remote, full-time, US-based. $215,000-$235,000 base. You will "
            "own our semantic layer built in dbt and LookML, write SQL against "
            "Snowflake and Redshift, and partner with BI stakeholders across "
            "Product and Marketing. We're a Series B, venture-backed company "
            "offering meaningful equity — 0.25%-0.5% ownership for early hires. "
            "Medical insurance for you and your family, dental, and a 401k match included."
        ),
        salary_min=215_000,
        salary_max=235_000,
        work_type="fulltime",
        location_mode="remote",
        location_country="USA",
        location_raw="Remote - US",
        posted_at="2026-07-12",
    ),
]


class MockJobSource:
    name = "mock"

    def search(self, keywords: list[str]) -> list[Job]:
        return [Job(source=self.name, **listing) for listing in _FAKE_LISTINGS]
