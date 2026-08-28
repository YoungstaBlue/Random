"""Live smoke test for claude_legal/analysis/citator.py.

Run this from a machine with real internet access — it makes genuine HTTPS
calls to CourtListener. It could NOT be run from the sandbox this pipeline was
built in (that environment's egress policy blocks courtlistener.com), so the
citator's request/response field-mapping has not yet been verified against the
live API. This script exists to close that gap.

Usage:
    export CLAUDE_LEGAL_COURTLISTENER_API_TOKEN=<your token>   # optional but recommended
    python examples/smoke_test_citator.py

    # or pass the token directly:
    python examples/smoke_test_citator.py --token <your token>

    # test specific citations instead of the defaults:
    python examples/smoke_test_citator.py --citation "410 U.S. 113" --citation "999 Made.Up 1"

What it checks, and how to read the output:
    - A well-known citation (Brown v. Board, 347 U.S. 483) SHOULD resolve —
      if it doesn't, the request/response shape in citator.py needs fixing
      against CourtListener's actual current API.
    - An invented citation SHOULD NOT resolve — confirms the no-match path
      doesn't false-positive.
    - The raw JSON response is printed for the first citation so you can
      diff CourtListener's real field names against what citator.py expects
      (see the ASSUMED SHAPE section below) if anything looks wrong.

This is an existence check only — a "RESOLVED" result means CourtListener has
an opinion on file matching the citation, NOT that the case is still good law.
"""
from __future__ import annotations

import argparse
import json
import sys

sys.path.insert(0, __file__.rsplit("/", 2)[0])  # allow running without `pip install -e .`

from claude_legal.analysis.citator import CourtListenerCitator  # noqa: E402
from claude_legal.config import Settings  # noqa: E402
from claude_legal.schemas import Citation  # noqa: E402

DEFAULT_CITATIONS = [
    "347 U.S. 483",   # Brown v. Board of Education — should resolve
    "410 U.S. 113",   # Roe v. Wade — should resolve
    "999 Made.Up 1",  # not a real reporter — should NOT resolve
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--token", help="CourtListener API token (else reads "
                         "CLAUDE_LEGAL_COURTLISTENER_API_TOKEN from env/.env)")
    parser.add_argument("--base-url", default="https://www.courtlistener.com")
    parser.add_argument("--citation", action="append", dest="citations",
                         help="citation text to test; repeatable. Defaults to "
                              "a known-good, a known-good, and a bogus citation.")
    args = parser.parse_args()

    settings_kwargs = {"courtlistener_enabled": True, "courtlistener_base_url": args.base_url}
    if args.token:
        settings_kwargs["courtlistener_api_token"] = args.token
    settings = Settings(**settings_kwargs)

    if not settings.courtlistener_api_token:
        print("NOTE: no token set — running against CourtListener's anonymous rate limit.\n"
              "      Pass --token or set CLAUDE_LEGAL_COURTLISTENER_API_TOKEN for a real run.\n")

    citator = CourtListenerCitator(settings)
    citations = args.citations or DEFAULT_CITATIONS

    print(f"Testing {len(citations)} citation(s) against {settings.courtlistener_base_url} ...\n")

    had_error = False
    raw_shown = False
    for text in citations:
        citation = Citation(raw=text, normalized=text)

        # Also capture the raw response once, for shape verification.
        if not raw_shown:
            try:
                raw = citator._lookup(text)  # noqa: SLF001 - intentional, for diagnostics
                print("--- raw CourtListener response for the first citation ---")
                print(json.dumps(raw, indent=2)[:4000])
                print("--- end raw response ---\n")
            except Exception as exc:
                print(f"[raw fetch failed: {exc}]\n")
                had_error = True
            raw_shown = True

        citator.resolve([citation])

        status = "RESOLVED" if citation.resolved else (
            "NO MATCH" if citation.resolved is False else "ERROR"
        )
        print(f"[{status}] {text}")
        if citation.resolved:
            print(f"    case_name         : {citation.case_name}")
            print(f"    cluster_id        : {citation.cluster_id}")
            print(f"    courtlistener_url : {citation.courtlistener_url}")
            print(f"    citation_count    : {citation.citation_count}")
        if citation.notes:
            print(f"    notes             : {citation.notes}")
            if citation.resolved is None:
                had_error = True
        print()

    print("Summary: check that the known-good citations show RESOLVED with a")
    print("sensible case_name/URL, and the bogus one shows NO MATCH. If the raw")
    print("response above uses different field names than citator.py expects")
    print("(id/case_name/absolute_url/citation_count under 'clusters'), update")
    print("the .get() calls in CourtListenerCitator._resolve_one accordingly.")

    return 1 if had_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
