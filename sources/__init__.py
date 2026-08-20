"""Pluggable job-source interface.

Every source returns a list of db.Job with `description` populated with the
FULL job description text - the whole point of this project is scoring off
the real JD (benefits, salary language), not just a title/snippet, so a
source that can't fetch the full description is not done yet.

See docs/ADDING_A_JOB_SOURCE.md before adding a new one - it's the §2 wiring
checklist (config, scheduler registration, tests) written after the second
source (linkedin_apify_source) was added, per engineering-foundations.
"""
from __future__ import annotations

from typing import Protocol

from db import Job


class JobSource(Protocol):
    name: str

    # Whether search() returns EVERY currently-listed job matching the
    # keywords, or only a truncated slice of them (§ DECISIONS.md 2026-08-02).
    # This is what makes "job X wasn't in the latest results" mean something:
    #
    #   True  - the source enumerates the full matching set (e.g. Greenhouse
    #           returns a company's entire board, filtered client-side), so a
    #           job that vanishes really was delisted.
    #   False - the source returns a ranked, capped top-N (e.g. LinkedIn via
    #           Apify, maxJobs per keyword), so absence proves NOTHING: a
    #           still-live job that merely slipped below the cap this run
    #           looks identical to one that was taken down.
    #
    # db.annotate_active refuses to mark jobs inactive for a source that
    # can't enumerate - same discipline as the Zoox fix (2026-07-29): never
    # assert a status the data doesn't actually support.
    enumerates_all_matches: bool = True

    def search(self, keywords: list[str]) -> list[Job]:
        """Fetch current listings matching keywords, JD included. No scoring here."""
        ...
