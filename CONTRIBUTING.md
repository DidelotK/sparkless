# Contributing to Sparkless

## Setup

```bash
pip install -e ".[dev]"
pre-commit install && pre-commit install --hook-type pre-push
```

## Before you push

```bash
python3 -m ruff format .
python3 -m ruff check .
python3 -m mypy sparkless tests
python3 -m pytest tests/unit/ -q -o "addopts=" -m "not performance"
```

CI runs `mypy sparkless tests` with `disallow_untyped_decorators = true`, so
`@pytest.mark.parametrize` needs `# type: ignore[misc,untyped-decorator]`.

## Add a changeset

**Any PR that changes runtime behaviour needs a changeset in the same PR.** This
is what decides the next version number and what goes in `CHANGELOG.md`.

```bash
pnpm install        # once
pnpm changeset      # interactive: pick the bump, write a one-line summary
```

Or write the file by hand — a changeset is just markdown in `.changeset/`:

```markdown
---
"sparkless": patch
---

One-line summary of the user-visible change.
```

| Bump | When |
|------|------|
| `patch` | Bug fix, closer PySpark parity, no new API |
| `minor` | New function / new API surface, backwards compatible |
| `major` | Behaviour callers depend on changes, or API is removed |

Docs-only, CI-only and test-only PRs do not need one.

Commit messages no longer drive versioning — `fix:` / `feat:` prefixes are still
nice for readability, but a PR without a changeset ships nothing to the feed.

## How a release is cut

Merging to `main` opens a **"Version Packages"** PR that bumps the version and
writes the changelog. Merging *that* PR builds the wheel, uploads it to the
Azure Artifacts feed, verifies the feed serves it, and tags the release.

Full details, including how the workflow fails loudly when a publish does not
happen: [`docs/release-process.md`](docs/release-process.md).
