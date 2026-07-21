#!/usr/bin/env python3
"""Query the Azure Artifacts PyPI feed: what is actually published?

After uploading, the release flow asks the feed itself whether the version is
being served, rather than concluding "published" from twine's exit code. That is
what makes a green Release run mean the package is installable.

This replaces a ``curl | grep`` that had never been run against the real feed.
The shape it assumed is now pinned by tests against a captured production
response -- see ``tests/unit/test_release_feed.py`` and its fixture.

The index is a PEP 503 "simple" page. Verified against the real
``solya-azure-devops/sparkless`` feed: ``GET`` with basic auth ``_:<PAT>``
returns 200 and a single-line HTML document whose anchors carry the artifact
filename as their text::

    <a href="https://pkgs.dev.azure.com/.../sparkless-4.2.2-py3-none-any.whl#sha256=..."
       data-requires-python="&gt;=3.9">sparkless-4.2.2-py3-none-any.whl</a>

Usage::

    python3 scripts/release/feed.py filenames
    python3 scripts/release/feed.py has --version 4.2.4 [--wait]

Requires ``AZURE_DEVOPS_PAT``. If it is missing this exits 2 rather than
reporting "not published" -- "could not check" and "is not there" are different
answers and must not be conflated.
"""

import argparse
import base64
import os
import re
import sys
import time
import urllib.error
import urllib.request
from typing import List, Optional

DEFAULT_ORG = "solya-azure-devops"
DEFAULT_PROJECT = "sparkless"
DEFAULT_FEED = "sparkless"
PACKAGE = "sparkless"

# Anchor text is the artifact filename. Matching the anchor *text* rather than
# the href matters: Azure normalises the version inside the download URL
# (".../sparkless/4.2/sparkless-4.2.0-py3-none-any.whl"), so the href is not a
# reliable place to read a version from.
ARTIFACT = re.compile(r">\s*([A-Za-z0-9._+-]+\.(?:whl|tar\.gz))\s*</a>")


def parse_filenames(html: str) -> List[str]:
    """Extract artifact filenames from a PEP 503 simple index page."""
    return ARTIFACT.findall(html)


def artifacts_for_version(filenames: List[str], version: str) -> List[str]:
    """Filenames belonging to *version*, matched exactly.

    Exact matching matters: a substring test for "sparkless-4.2" would happily
    accept "sparkless-4.2.10-py3-none-any.whl" as proof that 4.2.1 shipped.
    """
    wheel_prefix = f"{PACKAGE}-{version}-"
    sdist = f"{PACKAGE}-{version}.tar.gz"
    return [
        name
        for name in filenames
        if (name.startswith(wheel_prefix) and name.endswith(".whl")) or name == sdist
    ]


def simple_index_url(org: str, project: str, feed: str) -> str:
    return (
        f"https://pkgs.dev.azure.com/{org}/{project}/_packaging/{feed}"
        f"/pypi/simple/{PACKAGE}/"
    )


def fetch_index(url: str, pat: str, timeout: int = 30) -> str:
    """GET the simple index using basic auth ``_:<PAT>``."""
    token = base64.b64encode(f"_:{pat}".encode()).decode("ascii")
    request = urllib.request.Request(url)
    request.add_header("Authorization", f"Basic {token}")
    request.add_header("Accept", "text/html")
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        charset = response.headers.get_content_charset() or "utf-8"
        body: str = response.read().decode(charset, errors="replace")
    return body


def _pat() -> str:
    pat = os.environ.get("AZURE_DEVOPS_PAT", "")
    if not pat:
        sys.stderr.write(
            "error: AZURE_DEVOPS_PAT is unset; cannot query the feed.\n"
            "Refusing to report 'not published' when the real answer is "
            "'could not check'.\n"
        )
        raise SystemExit(2)
    return pat


def _load(args: argparse.Namespace) -> List[str]:
    url = simple_index_url(args.org, args.project, args.feed)
    try:
        return parse_filenames(fetch_index(url, _pat()))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            # The package has never been published to this feed at all.
            return []
        sys.stderr.write(f"error: {url} returned HTTP {exc.code} {exc.reason}\n")
        raise SystemExit(2) from exc
    except urllib.error.URLError as exc:
        sys.stderr.write(f"error: could not reach {url}: {exc.reason}\n")
        raise SystemExit(2) from exc


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--org", default=os.environ.get("AZURE_DEVOPS_ORG", DEFAULT_ORG)
    )
    parser.add_argument(
        "--project", default=os.environ.get("AZURE_DEVOPS_PROJECT", DEFAULT_PROJECT)
    )
    parser.add_argument(
        "--feed", default=os.environ.get("AZURE_DEVOPS_FEED", DEFAULT_FEED)
    )

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("filenames", help="print every artifact filename on the feed")

    has = sub.add_parser("has", help="check whether a version is served")
    has.add_argument("--version", required=True)
    has.add_argument(
        "--wait",
        action="store_true",
        help="poll for up to a minute (the feed is eventually consistent "
        "after an upload)",
    )

    args = parser.parse_args(argv)

    if args.command == "filenames":
        for name in _load(args):
            print(name)
        return 0

    attempts = 6 if args.wait else 1
    for attempt in range(1, attempts + 1):
        found = artifacts_for_version(_load(args), args.version)
        if found:
            print(f"feed serves {PACKAGE} {args.version}: {', '.join(sorted(found))}")
            return 0
        if attempt < attempts:
            print(
                f"{PACKAGE} {args.version} not visible yet "
                f"(attempt {attempt}/{attempts}); waiting 10s...",
                flush=True,
            )
            time.sleep(10)

    sys.stderr.write(f"error: feed does not serve {PACKAGE} {args.version}\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
