"""Tests for the Azure Artifacts feed client used by the release flow.

`scripts/release/feed.py` answers "is this version actually published?" — the
question that decides whether the publish step uploads, and whether a green
Release run means anything.

The parser is pinned against `fixtures/azure_pypi_simple_index.html`, which is a
**real** response from the `solya-azure-devops/sparkless` feed (fetched with
basic auth `_:<PAT>`, HTTP 200), with the internal GUIDs and sha256 digests
replaced by zeroes. Testing against a captured real payload is the point: the
previous version of this path assumed an index shape and a twine flag that had
never been exercised, and both assumptions were wrong.
"""

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import List

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FEED_PATH = REPO_ROOT / "scripts" / "release" / "feed.py"
FIXTURE = Path(__file__).parent / "fixtures" / "azure_pypi_simple_index.html"


def _load_feed() -> ModuleType:
    """Import the feed client by path -- scripts/ is not an importable package."""
    spec = importlib.util.spec_from_file_location("release_feed", FEED_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


feed = _load_feed()
REAL_INDEX = FIXTURE.read_text(encoding="utf-8")


class TestParseRealIndex:
    """Parsing the captured production index."""

    def test_finds_every_artifact(self) -> None:
        names = feed.parse_filenames(REAL_INDEX)
        assert len(names) == 39

    def test_every_name_looks_like_an_artifact(self) -> None:
        for name in feed.parse_filenames(REAL_INDEX):
            assert name.startswith("sparkless-")
            assert name.endswith((".whl", ".tar.gz"))

    def test_reads_the_filename_not_the_href_version_segment(self) -> None:
        """Azure normalises the version inside the download URL.

        The href for 4.2.0 contains ".../sparkless/4.2/sparkless-4.2.0-...", so
        anything reading a version out of the href path would see "4.2". The
        anchor text is the artifact filename and is authoritative.
        """
        names = feed.parse_filenames(REAL_INDEX)
        assert "sparkless-4.2.0-py3-none-any.whl" in names
        assert "sparkless-4.2" not in names

    def test_a_published_version_is_found_with_both_artifacts(self) -> None:
        found = feed.artifacts_for_version(feed.parse_filenames(REAL_INDEX), "4.2.2")
        assert sorted(found) == [
            "sparkless-4.2.2-py3-none-any.whl",
            "sparkless-4.2.2.tar.gz",
        ]

    def test_the_version_that_failed_to_publish_is_absent(self) -> None:
        """4.2.3 built and was never uploaded -- the feed must say so."""
        found = feed.artifacts_for_version(feed.parse_filenames(REAL_INDEX), "4.2.3")
        assert found == []


class TestExactVersionMatching:
    """A near-miss must never be mistaken for a published version."""

    @pytest.mark.parametrize(  # type: ignore[misc,untyped-decorator]
        ("version", "expected"),
        [
            pytest.param("4.2.1", ["sparkless-4.2.1-py3-none-any.whl"], id="exact"),
            pytest.param("4.2", [], id="prefix-of-a-real-version"),
            pytest.param("4.2.1.1", [], id="extends-a-real-version"),
            pytest.param("", [], id="empty"),
        ],
    )
    def test_matching(self, version: str, expected: List[str]) -> None:
        names = ["sparkless-4.2.1-py3-none-any.whl", "sparkless-4.2.10.tar.gz"]
        assert feed.artifacts_for_version(names, version) == expected

    def test_4_2_1_does_not_match_4_2_10(self) -> None:
        """The classic off-by-one-character release bug."""
        names = [
            "sparkless-4.2.10-py3-none-any.whl",
            "sparkless-4.2.10.tar.gz",
        ]
        assert feed.artifacts_for_version(names, "4.2.1") == []
        assert len(feed.artifacts_for_version(names, "4.2.10")) == 2

    def test_platform_wheels_are_matched(self) -> None:
        """Older releases shipped abi3 platform wheels; they still count."""
        names = feed.parse_filenames(REAL_INDEX)
        found = feed.artifacts_for_version(names, "4.5.7")
        assert len(found) == 9  # 8 platform wheels + 1 sdist
        assert sum(name.endswith(".whl") for name in found) == 8
        assert any("musllinux" in name for name in found)
        assert "sparkless-4.5.7.tar.gz" in found


class TestIndexUrl:
    def test_url_shape_matches_the_verified_endpoint(self) -> None:
        url = feed.simple_index_url("solya-azure-devops", "sparkless", "sparkless")
        assert url == (
            "https://pkgs.dev.azure.com/solya-azure-devops/sparkless"
            "/_packaging/sparkless/pypi/simple/sparkless/"
        )


class TestDegenerateInput:
    """A parse failure must look like a parse failure, not like 'not published'."""

    def test_empty_document_yields_nothing(self) -> None:
        assert feed.parse_filenames("") == []

    def test_index_for_a_package_with_no_releases(self) -> None:
        html = (
            "<html><head><title>Links for sparkless</title></head><body></body></html>"
        )
        assert feed.parse_filenames(html) == []

    def test_unrelated_anchors_are_ignored(self) -> None:
        html = '<a href="/x">back to index</a><a href="/y">sparkless-1.0.0.tar.gz</a>'
        assert feed.parse_filenames(html) == ["sparkless-1.0.0.tar.gz"]
