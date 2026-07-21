# Changesets

This folder is managed by [Changesets](https://github.com/changesets/changesets).

Sparkless is a **Python** package published to the Azure Artifacts PyPI feed
`sparkless` (org `solya-azure-devops`). Changesets is used only for the
*versioning* half — deciding the next version, writing `CHANGELOG.md`, and
opening the "Version Packages" PR. Building and uploading the wheel is done by
`scripts/release/publish.sh`, not by `changeset publish`.

## Why there is a `package.json`

Changesets can only version a package it can see, and it only understands
`package.json`. The root `package.json` is a shim: it is `"private": true`
(so it can never be published to npm) and its `version` field is the value
Changesets bumps. `scripts/release/sync_version.py` then propagates that
version into the two files that actually matter:

- `pyproject.toml` → `[project] version`
- `sparkless/_version.py` → `__version__` fallback

The sync script **fails loudly** if either file does not end up on the new
version, and `scripts/release/check_version_consistency.py` re-asserts the
three files agree on every CI run.

## Adding a changeset

Every PR that changes runtime behaviour needs one, in the same PR:

```bash
pnpm changeset          # interactive: pick the bump, write a one-line summary
```

No Node on your machine? Write the file by hand — a changeset is just a
markdown file in this folder:

```markdown
---
"sparkless": patch
---

One-line summary of the user-visible change.
```

`patch` for bug fixes, `minor` for new API surface, `major` for anything that
breaks PySpark-compatible behaviour callers rely on.

PRs that only touch docs, CI, or tests do not need a changeset.

## Cutting a release

See [`docs/release-process.md`](../docs/release-process.md). Short version:
merging to `main` opens a "Version Packages" PR; merging **that** PR is what
publishes.
