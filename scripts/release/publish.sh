#!/usr/bin/env bash
#
# Build and publish the sparkless wheel + sdist to the Azure Artifacts PyPI feed,
# then tag the commit and create the GitHub release.
#
# This is the `publish` command handed to changesets/action. The action only runs
# it when there are no pending changesets on `main` -- i.e. immediately after a
# "Version Packages" PR is merged, when `pyproject.toml` already carries the new
# version.
#
# DESIGN NOTE -- why this script is paranoid.
#
# The setup this replaced could report success while publishing nothing:
#   1. it decided whether to publish by grepping semantic-release's prose output,
#      and a miss meant "skip the publish job", not "fail";
#   2. `twine upload ... || echo "Warning: upload failed"` swallowed every upload
#      error.
# So: every command here is fatal (`set -euo pipefail`), the publish decision is
# made from a git tag rather than from parsed prose, and after uploading we ask
# the feed whether the version is actually there. If any of that does not hold,
# this script exits non-zero and the workflow goes red.

set -euo pipefail

AZURE_DEVOPS_ORG="${AZURE_DEVOPS_ORG:-solya-azure-devops}"
AZURE_DEVOPS_PROJECT="${AZURE_DEVOPS_PROJECT:-sparkless}"
AZURE_DEVOPS_FEED="${AZURE_DEVOPS_FEED:-sparkless}"

FEED_BASE="https://pkgs.dev.azure.com/${AZURE_DEVOPS_ORG}/${AZURE_DEVOPS_PROJECT}/_packaging/${AZURE_DEVOPS_FEED}/pypi"
UPLOAD_URL="${FEED_BASE}/upload/"
SIMPLE_URL="${FEED_BASE}/simple/sparkless/"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

die() {
  echo "::error::$*" >&2
  exit 1
}

group() { echo "::group::$*"; }
endgroup() { echo "::endgroup::"; }

# ---------------------------------------------------------------------------
# 1. Work out what we are releasing, and whether it is already released.
# ---------------------------------------------------------------------------

VERSION="$(python3 scripts/release/check_version_consistency.py --print)" \
  || die "version files disagree; refusing to publish (see the error above)"
TAG="v${VERSION}"

echo "Release candidate: ${TAG}"

# Ask the remote directly rather than `git fetch --tags` + `rev-parse`: a fetch
# can fail for reasons unrelated to this release (a local tag that would be
# clobbered, for instance), and the remote is the authority on what has shipped.
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
  || die "GITHUB_TOKEN is unset; cannot create the GitHub release."

# ---------------------------------------------------------------------------
# 3. Install, test, build.
# ---------------------------------------------------------------------------

group "Install build and publish tooling"
python3 -m pip install --upgrade pip
python3 -m pip install -e ".[dev]" build twine
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
if ! ls "dist/sparkless-${VERSION}-"*.whl >/dev/null 2>&1; then
  echo "dist/ contains:" >&2
  ls -la dist >&2
  die "no wheel matching version ${VERSION} was built"
fi
if ! ls "dist/sparkless-${VERSION}.tar.gz" >/dev/null 2>&1; then
  die "no sdist matching version ${VERSION} was built"
fi

# ---------------------------------------------------------------------------
# 4. Upload. No `|| true`, no warnings-as-success.
# ---------------------------------------------------------------------------

group "Upload to Azure Artifacts feed '${AZURE_DEVOPS_FEED}'"
# --skip-existing makes a re-run after a partial failure idempotent: it tolerates
# ONLY "this file is already on the feed". Every other failure (auth, network,
# rejected metadata) still exits non-zero. Step 5 then independently confirms the
# version is really there, so "skipped" can never be mistaken for "published".
python3 -m twine upload \
  --non-interactive \
  --skip-existing \
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
verified=0
for attempt in 1 2 3 4 5 6; do
  index="$(curl --silent --show-error --fail --location \
    --user "_:${AZURE_DEVOPS_PAT}" "${SIMPLE_URL}" || true)"
  if printf '%s' "${index}" | grep -q "sparkless-${VERSION}-.*\.whl"; then
    echo "Feed serves sparkless ${VERSION} (attempt ${attempt})."
    verified=1
    break
  fi
  echo "Not visible yet (attempt ${attempt}/6); the feed is eventually consistent. Waiting 10s..."
  sleep 10
done
[ "${verified}" -eq 1 ] \
  || die "uploaded sparkless ${VERSION} but the feed never served it. The publish did NOT succeed -- do not trust this run. Check ${SIMPLE_URL}"
endgroup

# ---------------------------------------------------------------------------
# 6. Tag and release. Done last so a tag always implies a verified publish.
# ---------------------------------------------------------------------------

group "Tag ${TAG} and create the GitHub release"
git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
git tag -a "${TAG}" -m "${TAG}"
git push origin "refs/tags/${TAG}"

notes="$(python3 scripts/release/extract_changelog.py "${VERSION}")" \
  || die "could not extract the CHANGELOG section for ${VERSION}"
printf '%s\n' "${notes}" > /tmp/release-notes.md

gh release create "${TAG}" \
  --title "${TAG}" \
  --notes-file /tmp/release-notes.md \
  dist/*
endgroup

echo "Published sparkless ${VERSION} to the ${AZURE_DEVOPS_FEED} feed and released ${TAG}."

# Deliberately NOT printing a "New tag: ..." line. changesets/action scrapes its
# publish command's stdout for that string to decide whether to create a GitHub
# release of its own -- another publish decision made by parsing free text, which
# is the failure mode this migration exists to remove. Tagging and releasing are
# owned by this script, above, where every step is fatal on failure.
