#!/usr/bin/env python3
"""Assert the three version declarations agree.

Sparkless declares its version in three places:

- ``package.json``           (the Changesets shim -- what the tooling bumps)
- ``pyproject.toml``         (what the wheel is built with)
- ``sparkless/_version.py``  (the runtime fallback when the package metadata
                              is unavailable)

They must always agree. This runs in CI on every PR, so drift is caught long
before a release rather than being discovered from a wrongly-tagged artifact.

Usage::

    python3 scripts/release/check_version_consistency.py          # check
    python3 scripts/release/check_version_consistency.py --print  # emit version
"""

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

PACKAGE_JSON = REPO_ROOT / "package.json"
PYPROJECT = REPO_ROOT / "pyproject.toml"
VERSION_PY = REPO_ROOT / "sparkless" / "_version.py"

PYPROJECT_VERSION = re.compile(r'^version = "([^"]+)"$', re.MULTILINE)
VERSION_PY_VERSION = re.compile(r'^[ \t]*__version__ = "([^"]+)"$', re.MULTILINE)


def _sole_match(path: Path, pattern: "re.Pattern[str]", label: str) -> str:
    if not path.is_file():
        sys.stderr.write(f"error: {path} is missing\n")
        raise SystemExit(1)
    matches = pattern.findall(path.read_text(encoding="utf-8"))
    if len(matches) != 1:
        sys.stderr.write(
            f"error: expected exactly 1 {label} version literal in {path}, found {len(matches)}\n"
        )
        raise SystemExit(1)
    return matches[0]


def collect() -> "dict[str, str]":
    package_json = json.loads(PACKAGE_JSON.read_text(encoding="utf-8"))
    return {
        "package.json": str(package_json.get("version")),
        "pyproject.toml": _sole_match(PYPROJECT, PYPROJECT_VERSION, "pyproject.toml"),
        "sparkless/_version.py": _sole_match(
            VERSION_PY, VERSION_PY_VERSION, "_version.py"
        ),
    }


def main(argv: "list[str]") -> int:
    versions = collect()
    distinct = set(versions.values())

    if len(distinct) != 1:
        sys.stderr.write("error: version files disagree:\n")
        for name, value in versions.items():
            sys.stderr.write(f"  {name:<24} {value}\n")
        sys.stderr.write(
            "\nRun `pnpm run version` (or fix by hand) so all three match.\n"
        )
        return 1

    version = distinct.pop()
    if "--print" in argv:
        print(version)
    else:
        print(f"version {version} consistent across {len(versions)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
