# Release Process

Sparkless uses [Changesets](https://github.com/changesets/changesets) for
versioning and changelogs — the same tooling as the other Solya repositories —
and a Python-specific publish step that uploads the wheel to Azure Artifacts.

## The two-step flow

```text
feature PR (carries a .changeset/*.md)
        │  merge to main
        ▼
Release workflow ──► opens/updates the "Version Packages" PR
                     (bumps the version, writes CHANGELOG.md,
                      deletes the consumed changesets)
        │  merge that PR
        ▼
Release workflow ──► builds sdist + wheel
                     uploads to the Azure Artifacts feed
                     VERIFIES the feed serves the version
                     tags vX.Y.Z + creates the GitHub release
```

Nothing publishes without a human merging the Version PR. That PR is the review
gate: it shows the exact version and the exact changelog before anything ships.

## Adding a changeset (contributors)

Any PR that changes runtime behaviour needs a changeset in the same PR:

```bash
pnpm install        # once
pnpm changeset       # interactive
```

Pick the bump and write a one-line, user-facing summary:

| Bump | When |
|------|------|
| `patch` | Bug fix, closer PySpark parity, no new API |
| `minor` | New function / new API surface, backwards compatible |
| `major` | Behaviour callers depend on changes, or API is removed |

No Node locally? The changeset is just a markdown file — write it by hand:

```markdown
---
"sparkless": patch
---

One-line summary of the user-visible change.
```

Save it as `.changeset/some-descriptive-name.md`. Docs-only, CI-only and
test-only PRs do not need one.

Commit messages no longer drive versioning. `fix:` / `feat:` prefixes are still
welcome for readability, but they have no effect on the release.

## Cutting a release (maintainers)

1. Merge the feature PRs. Each merge to `main` refreshes the **"Version
   Packages"** PR opened by the Release workflow.
2. Review that PR: it contains the new version in `package.json`,
   `pyproject.toml` and `sparkless/_version.py`, plus the `CHANGELOG.md` entry.
3. Merge it. The Release workflow then builds, uploads, verifies, tags and
   creates the GitHub release.
4. Confirm the run is green **and** that the run log ends with
   `Published sparkless X.Y.Z ...`. A green run ending in
   `Nothing further to do` means nothing was published (see below).

> **Note:** the Version PR is opened with `GITHUB_TOKEN`, so GitHub does not run
> `pull_request` workflows on it. If you want CI on it, push an empty commit to
> its branch.

### If a publish fails

Fix the cause, then re-run the Release workflow via **workflow_dispatch** on
`main`. The publish is idempotent: it exits early if the tag already exists, and
`twine --skip-existing` tolerates files already on the feed while still failing
on every other error.

## How this fails loudly

The previous semantic-release setup could report success while publishing
nothing. Two mechanisms did that, and both are gone:

1. **The publish decision was a grep over prose.** The workflow ran
   `grep -oP 'The next release version is \K[0-9.]+'` over semantic-release's
   console output; a miss set `new_release_published=false`, which *skipped* the
   publish job. A skipped job is green — so "no release was due" and "the release
   machinery broke" looked identical.

   **Now:** the decision is made from a git tag (`scripts/release/publish.sh`),
   and `scripts/release/assert_release_state.py` runs afterwards and asserts the
   run did what it was supposed to:

   | Version PR opened? | Pending changesets | Version tagged | Outcome |
   |---|---|---|---|
   | **yes** | *(irrelevant)* | *(not queried)* | green — "nothing published, by design" |
   | no | yes | *(not queried)* | **red** — the version step is broken |
   | no | no | no | **red** — the publish did not complete |
   | no | no | yes | green — "already published" |

   The Version PR is checked **first**, and nothing else is consulted when one
   exists. That ordering matters: by the time the guard runs, `changesets/action`
   has already run the version command *in the working tree*, so the changesets
   are consumed and the version files already carry the next version. A healthy
   version run therefore presents the exact fingerprint of a broken publish — no
   pending changesets, and a version with no tag. `--version-pr` is the only
   input that separates them. (Keying off tree state instead failed the job on
   the first real run, for Version PR #34.)

   Each row is pinned by a unit test in `tests/unit/test_release_guard.py`, run
   against the guard's pure `decide()` function rather than through the workflow.

2. **The version bump was two `sed` calls** over `pyproject.toml` and
   `sparkless/_version.py`. `sed` exits 0 when it matches nothing, so a rename or
   a reformat would have produced a release whose version files were never
   touched.

   **Now:** `scripts/release/sync_version.py` asserts each file contains exactly
   one version literal, reads the file back after writing, and exits non-zero if
   it does not carry the expected version. `check_version_consistency.py` re-runs
   that assertion in CI on every PR.

Two further silent failures were removed along the way:

- `twine upload ... || echo "Warning: upload failed"` swallowed every upload
  error. The upload is now fatal, and after it succeeds the script polls the feed
  and fails if the version is not actually served. A green run means the package
  is installable.
- A release with no changelog entry used to pass unnoticed — `v4.2.2` shipped
  exactly that way, bumping both version files while writing nothing to
  `CHANGELOG.md`. `extract_changelog.py` now refuses to release a version whose
  changelog section is missing.

## Files

| Path | Role |
|------|------|
| `.changeset/config.json` | Changesets config (mirrors the other Solya repos) |
| `.changeset/*.md` | Pending changesets, consumed by the Version PR |
| `package.json` | Private shim so Changesets has a package to version |
| `scripts/release/sync_version.py` | package.json version → pyproject + `_version.py` |
| `scripts/release/check_version_consistency.py` | Asserts the three agree (runs in CI) |
| `scripts/release/publish.sh` | Test, build, upload, verify, tag, release |
| `scripts/release/extract_changelog.py` | Release notes from `CHANGELOG.md` |
| `scripts/release/assert_release_state.py` | Post-run guard against silent no-ops |
| `.github/workflows/release.yml` | Wires it together |

## Configuration

### GitHub secrets

- `GITHUB_TOKEN` — auto-provided; used for the Version PR, the tag and the release
- `AZURE_DEVOPS_PAT` — Azure DevOps PAT with **Packaging read & write**

If `AZURE_DEVOPS_PAT` is missing, the publish step fails immediately with an
explicit message rather than attempting an unauthenticated upload.

### Azure Artifacts feed

- **Organization**: `solya-azure-devops`
- **Project**: `sparkless`
- **Feed**: `sparkless`
- **Upload URL**: `https://pkgs.dev.azure.com/solya-azure-devops/sparkless/_packaging/sparkless/pypi/upload/`
- **Install URL**: `https://pkgs.dev.azure.com/solya-azure-devops/sparkless/_packaging/sparkless/pypi/simple/`

## Installing from Azure Artifacts

In `pyproject.toml`:

```toml
[[tool.uv.index]]
name = "solya-sparkless"
url = "https://pkgs.dev.azure.com/solya-azure-devops/sparkless/_packaging/sparkless/pypi/simple/"
explicit = true

[tool.uv.sources]
sparkless = { index = "solya-sparkless" }
```

Authentication via environment variables:

```bash
UV_INDEX_SOLYA_SPARKLESS_USERNAME=_
UV_INDEX_SOLYA_SPARKLESS_PASSWORD=<AZURE_DEVOPS_PAT>
```

## Emergency manual publish

`.github/workflows/publish-manual.yml` republishes an existing tag
(`workflow_dispatch`, takes a tag like `v4.2.3`). Use it only to recover a failed
upload of an already-versioned release — it does not bump anything.

## Local development

```bash
# In sparkless — make changes, test
python3 -m pytest tests/unit/ tests/parity/ -q -o "addopts="

# In data-platform — test against the local checkout
pip install -e /path/to/sparkless
# or
uv add --dev --editable /path/to/sparkless
```
