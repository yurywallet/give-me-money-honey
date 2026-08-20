"""Which LinkedIn source to use - the remote-filtered one, or the cheaper
legacy one (operator ask 2026-08-02: "so if credit is fully used can switch
back").

Two LinkedIn sources exist for a real reason, not as duplication:

  remote (default) - sources/linkedin_remote_source.py. Passes LinkedIn's own
      `f_WT=2` filter in a search URL, which is VERIFIED to actually filter
      (A/B: only 4/15 overlap vs unfiltered). Costs ~2.4x more per job.
  legacy           - sources/linkedin_apify_source.py. Cheaper, but its
      actor silently ignores the remote/full-time parameters, so results are
      unfiltered and carry no work-arrangement signal.

Switch with GMMH_LINKEDIN_ACTOR=legacy when the Apify credit runs low.
Both report `name = "linkedin_apify"` and return LinkedIn job IDs, so
switching does not fork history or re-insert existing jobs as new.
"""
from __future__ import annotations

import os

REMOTE = "remote"
LEGACY = "legacy"


def selected_linkedin_mode() -> str:
    """`GMMH_LINKEDIN_ACTOR`, defaulting to the remote-filtered source.
    Anything unrecognised falls back to the default rather than raising -
    a typo in .env should not take the whole search offline."""
    mode = (os.getenv("GMMH_LINKEDIN_ACTOR") or REMOTE).strip().lower()
    return LEGACY if mode == LEGACY else REMOTE


def build_linkedin_source():
    """The configured LinkedIn source, or None if Apify isn't set up.

    The legacy source additionally needs APIFY_ACTOR_ID (it targets one
    specific actor); the remote source defaults its own actor id, so a token
    alone is enough.
    """
    token = os.getenv("APIFY_TOKEN")
    if not token:
        return None

    if selected_linkedin_mode() == LEGACY:
        if not os.getenv("APIFY_ACTOR_ID"):
            return None
        from sources.linkedin_apify_source import LinkedInApifySource

        return LinkedInApifySource()

    from sources.linkedin_remote_source import LinkedInRemoteApifySource

    return LinkedInRemoteApifySource()
