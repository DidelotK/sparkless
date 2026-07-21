#!/usr/bin/env python3
"""Propagate the Changesets-bumped version into the Python version files.

Changesets bumps ``package.json`` (the release shim). This script copies that
version into the two files that actually define the published artifact:

- ``pyproject.toml``  -> ``[project] version``
- ``sparkless/_version.py`` -> the ``__version__`` fallback literal

This replaces the two ``sed -i`` calls that used to live in ``.releaserc.json``.
The point of the rewrite is that ``sed`` exits 0 when its pattern matches
nothing: a rename or a reformat would silently produce a release whose version
files were never touched. Every step here asserts its own effect and exits
non-zero if the file did not end up on the expected version.

Usage::

    python3 scripts/release/sync_version.py
"""

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

PACKAGE_JSON = REPO_ROOT / "package.json"
PYPROJECT = REPO_ROOT / "pyproject.toml"
VERSION_PY = REPO_ROOT / "sparkless" / "_version.py"

SEMVER = re.compile(r"^\d+\.\d+\.\d+$")

# `version = "..."` inside the [project] table, anchored to the start of a line.
PYPROJECT_VERSION = re.compile(r'^version = "(?P<version>[^"]+)"$', re.MULTILINE)

# The hardcoded fallback in _version.py. Deliberately does not match the
# `__version__ = version("sparkless")` line above it (no opening quote there).
VERSION_PY_VERSION = re.compile(
    r'^(?P<indent>[ \t]*)__version__ = "(?P<version>[^"]+)"$', re.MULTILINE
)


def fail(message: str) -> "None":
    sys.stderr.write(f"error: {message}\n")
    raise SystemExit(1)


def read_target_version() -> str:
    if not PACKAGE_JSON.is_file():
        fail(f"{PACKAGE_JSON} is missing; Changesets has nothing to version")
    try:
        data = json.loads(PACKAGE_JSON.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"{PACKAGE_JSON} is not valid JSON: {exc}")
    version = data.get("version")
    if not isinstance(version, str) or not SEMVER.match(version):
        fail(f"{PACKAGE_JSON} has no usable semver 'version' field (got {version!r})")
    return version


def rewrite(path: Path, pattern: "re.Pattern[str]", version: str, label: str) -> bool:
    """Rewrite the single version literal in *path*. Returns True if changed."""
    if not path.is_file():
        fail(f"{path} is missing")

    original = path.read_text(encoding="utf-8")
    matches = pattern.findall(original)
    if len(matches) == 0:
        fail(
            f"no {label} version literal found in {path}. The file's layout changed and "
            "this script no longer matches it -- fix the pattern in "
            "scripts/release/sync_version.py rather than releasing a stale "
            "version."
        )
    if len(matches) > 1:
        fail(
            f"{path} contains {len(matches)} {label} version literals; expected exactly 1. Refusing "
            "to guess which one is authoritative."
        )

    def _replace(match: "re.Match[str]") -> str:
        return match.group(0).replace(
            '"{}"'.format(match.group("version")), f'"{version}"'
        )

    updated = pattern.sub(_replace, original)

    if updated != original:
        path.write_text(updated, encoding="utf-8")

    # Read back from disk and assert the effect, rather than trusting the write.
    confirmed = pattern.search(path.read_text(encoding="utf-8"))
    if confirmed is None or confirmed.group("version") != version:
        fail(f"{path} still does not report version {version} after rewriting")

    return updated != original


def main() -> int:
    version = read_target_version()

    changed_pyproject = rewrite(PYPROJECT, PYPROJECT_VERSION, version, "pyproject.toml")
    changed_version_py = rewrite(VERSION_PY, VERSION_PY_VERSION, version, "_version.py")

    print(f"package.json version: {version}")
    print(
        "  pyproject.toml       {}".format(
            "updated" if changed_pyproject else "already correct"
        )
    )
    print(
        "  sparkless/_version.py {}".format(
            "updated" if changed_version_py else "already correct"
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
