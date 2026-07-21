#!/usr/bin/env bash
#
# Build and publish the sparkless wheel + sdist to the Azure Artifacts PyPI feed,
# then tag the commit and create the GitHub release.
#
# This is the `publish` command handed to changesets/action. The action runs it
# when there are no pending changesets on `main` -- i.e. after a "Version
# Packages" PR is merged, when `pyproject.toml` already carries the new version.
#
# DESIGN NOTE -- every failure is fatal.
#
# The setup this replaces reported success while publishing nothing, twice over:
#
#   `twine upload ... || echo "Warning: upload of $f failed"` treated every
#   upload error -- auth, network, rejected metadata -- as benign.
#
#   The publish/skip decision was made by grepping semantic-release's console
#   prose, and a miss *skipped* the publish job rather than failing it.
#
# So there is no `|| true` here and no output parsing. The one thing this script
# does not do is guess: after uploading it asks the feed whether the version is
# actually served (scripts/release/feed.py), and only tags once the feed says
# yes. A tag therefore always means "this version is installable".
#
# KNOWN LIMITATION -- re-uploading an existing version fails.
#
# There is deliberately no `twine --skip-existing`: twine 6.x gates that flag
# behind a hard URL allowlist of PyPI and TestPyPI
# (twine/settings.py::verify_feature_capability), so against this Azure feed it
# raises UnsupportedConfiguration *before uploading anything*. That is how the
# 4.2.3 publish failed. Consequently a re-run that tries to upload a version
# already on the feed will fail loudly. That is the correct default -- versions
# are immutable -- and the recovery is to cut a new version rather than to
# re-push an old one. See docs/release-process.md.

set -euo pipefail

AZURE_DEVOPS_ORG="${AZURE_DEVOPS_ORG:-solya-azure-devops}"
AZURE_DEVOPS_PROJECT="${AZURE_DEVOPS_PROJECT:-sparkless}"
AZURE_DEVOPS_FEED="${AZURE_DEVOPS_FEED:-sparkless}"
export AZURE_DEVOPS_ORG AZURE_DEVOPS_PROJECT AZURE_DEVOPS_FEED

FEED_BASE="https://pkgs.dev.azure.com/${AZURE_DEVOPS_ORG}/${AZURE_DEVOPS_PROJECT}/_packaging/${AZURE_DEVOPS_FEED}/pypi"
UPLOAD_URL="${FEED_BASE}/upload/"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

die() {
  echo "::error::$*" >&2
  exit 1
}

group() { echo "::group::$*"; }
endgroup() { echo "::endgroup::"; }

# ---------------------------------------------------------------------------
# 1. What are we releasing, and has it shipped already?
# ---------------------------------------------------------------------------

VERSION="$(python3 scripts/release/check_version_consistency.py --print)" \
  || die "version files disagree; refusing to publish (see the error above)"
TAG="v${VERSION}"

echo "Release candidate: ${TAG}"

# The tag is pushed only after the feed has confirmed it serves the wheel, so
# its presence means the publish genuinely succeeded. Asking the remote (rather
# than fetching tags locally) keeps this independent of local tag state.
remote_tag="$(git ls-remote --tags origin "refs/tags/${TAG}")" \
  || die "could not query tags on origin; cannot tell whether ${TAG} was already released"

if [ -n "${remote_tag}" ]; then
  echo "Tag ${TAG} already exists on origin -- version ${VERSION} was already released."
  echo "Nothing to do. (This is the 'no release was due' outcome, not a failure.)"
  exit 0
fi

# ---------------------------------------------------------------------------
# 2. Preconditions. Fail before doing any work rather than half-way through.
# ---------------------------------------------------------------------------

[ -n "${AZURE_DEVOPS_PAT:-}" ] \
  || die "AZURE_DEVOPS_PAT is unset or empty. Add it as a repository secret (Azure DevOps PAT with Packaging read & write). Refusing to attempt an unauthenticated publish."
[ -n "${GITHUB_TOKEN:-}" ] \
  || die "GITHUB_TOKEN is unset; cannot tag or create the GitHub release."

# ---------------------------------------------------------------------------
# 3. Install, test, build.
# ---------------------------------------------------------------------------

group "Install build and publish tooling"
python3 -m pip install --upgrade pip
# Pinned deliberately: an unpinned twine is what broke the 4.2.3 publish. twine
# 6.1 introduced the --skip-existing repository allowlist and CI picked it up
# silently on the next run.
python3 -m pip install -e ".[dev]" "build>=1,<2" "twine>=6.2,<7"
endgroup

group "Run tests before publishing"
TZ="America/New_York" python3 -m pytest tests/unit/ tests/parity/ -q --tb=short -o "addopts="
endgroup

group "Build sdist + wheel"
rm -rf dist
python3 -m build
python3 -m twine check dist/*
endgroup

# Assert the built artifacts carry the version we think we are releasing. A
# stale build directory or an out-of-sync pyproject would otherwise ship the
# wrong version under the right tag.
compgen -G "dist/sparkless-${VERSION}-*.whl" > /dev/null \
  || { ls -la dist >&2; die "no wheel matching version ${VERSION} was built"; }
[ -f "dist/sparkless-${VERSION}.tar.gz" ] \
  || die "no sdist matching version ${VERSION} was built"

# ---------------------------------------------------------------------------
# 4. Upload. No `|| true`, no --skip-existing (see the note at the top).
# ---------------------------------------------------------------------------

group "Upload to Azure Artifacts feed '${AZURE_DEVOPS_FEED}'"
python3 -m twine upload \
  --non-interactive \
  --repository-url "${UPLOAD_URL}" \
  --username "_" \
  --password "${AZURE_DEVOPS_PAT}" \
  dist/*
endgroup

# ---------------------------------------------------------------------------
# 5. Verify the feed actually serves the version. This is the check that makes
#    "the workflow was green" mean "the package is installable".
# ---------------------------------------------------------------------------

group "Verify ${VERSION} is served by the feed"
python3 scripts/release/feed.py has --version "${VERSION}" --wait \
  || die "uploaded sparkless ${VERSION} but the feed never served it. The publish did NOT succeed -- do not trust this run. Check ${FEED_BASE}/simple/sparkless/"
endgroup

# ---------------------------------------------------------------------------
# 6. Tag and release, last, so a tag always implies a verified publish.
#    `gh release create` creates the tag as well, so this is a single operation
#    rather than a `git push` of a tag followed by a separate API call -- one
#    thing to go wrong instead of two, and no half-state where a tag exists
#    without a release.
# ---------------------------------------------------------------------------

group "Tag ${TAG} and create the GitHub release"
notes="$(python3 scripts/release/extract_changelog.py "${VERSION}")" \
  || die "could not extract the CHANGELOG section for ${VERSION}"
notes_file="$(mktemp)"
printf '%s\n' "${notes}" > "${notes_file}"

gh release create "${TAG}" \
  --target "$(git rev-parse HEAD)" \
  --title "${TAG}" \
  --notes-file "${notes_file}" \
  dist/*
rm -f "${notes_file}"
endgroup

echo "Published sparkless ${VERSION} to the ${AZURE_DEVOPS_FEED} feed and released ${TAG}."

# Deliberately NOT printing a "New tag: ..." line. changesets/action scrapes its
# publish command's stdout for that string to decide whether to create a GitHub
# release of its own -- another publish decision made by parsing free text, which
# is the failure mode this migration exists to remove. Tagging and releasing are
# owned by this script, above, where every step is fatal on failure.
