#!/usr/bin/env python3
"""Print the CHANGELOG.md section for one version, for use as release notes.

Changesets writes a ``## <version>`` heading per release. This pulls that
section out so the GitHub release body matches the changelog exactly.

Exits non-zero if the section is missing -- a release whose changelog entry was
never written is a broken release, not a cosmetic problem. (The v4.2.2 release
made under the previous tooling bumped both version files and wrote no
changelog entry at all; nobody noticed because nothing checked.)

Usage::

    python3 scripts/release/extract_changelog.py 4.2.3
"""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CHANGELOG = REPO_ROOT / "CHANGELOG.md"


def extract(text: str, version: str) -> "str | None":
    # Changesets emits "## 4.2.3". The older semantic-release entries look like
    # "## [4.2.1](https://...compare/v4.2.0...v4.2.1) (2026-04-13)", so accept
    # arbitrary trailing text. The (?![\d.]) guard stops "4.2.1" from matching a
    # "## 4.2.10" heading.
    escaped = re.escape(version)
    heading = re.compile(
        r"^(?P<hashes>#{1,3})[ \t]+\[?" + escaped + r"\]?(?![\d.]).*$",
        re.MULTILINE,
    )
    match = heading.search(text)
    if match is None:
        return None

    body_start = match.end()
    level = len(match.group("hashes"))

    # The section ends at the next heading of the same or shallower depth.
    next_heading = re.compile(r"^#{1," + str(level) + r"}[ \t]+\S", re.MULTILINE)
    following = next_heading.search(text, body_start)
    body_end = following.start() if following else len(text)

    return text[body_start:body_end].strip()


def main(argv: "list[str]") -> int:
    if len(argv) != 1:
        sys.stderr.write("usage: extract_changelog.py <version>\n")
        return 2
    version = argv[0]

    if not CHANGELOG.is_file():
        sys.stderr.write(f"error: {CHANGELOG} is missing\n")
        return 1

    section = extract(CHANGELOG.read_text(encoding="utf-8"), version)
    if section is None:
        sys.stderr.write(
            f"error: no CHANGELOG.md section found for version {version}.\n"
            "A release with no changelog entry means the version step did not "
            "run properly -- refusing to release blind.\n"
        )
        return 1

    print(section)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
