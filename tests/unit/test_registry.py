"""Tests for the source registry.

The real `configs/crawl/sources.yaml` is loaded here on purpose: it is a
reviewed artefact, and a typo in it would otherwise only surface mid-crawl.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from qaraqalpaqmind.crawlers.core.registry import (
    AccessMethod,
    SourceRegistry,
    SourceSpec,
    load_registry,
)


def _spec(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "news_example",
        "name": "Example",
        "kind": "news",
        "access": "crawl",
        "url": "https://example.uz/qq/",
        "legal": "government_work",
        "license": "unknown",
        "update_frequency": "daily",
        "est_size_mb": 1.0,
        "quality": 3,
    }
    return base | overrides


def test_real_registry_is_valid() -> None:
    registry = load_registry()
    assert registry.version == 1
    assert registry.sources
    assert registry.total_estimated_mb() > 0


def test_every_crawl_source_respects_robots() -> None:
    # A source that opts out of robots.txt must never land silently in a diff.
    for spec in load_registry().sources:
        if spec.access is AccessMethod.CRAWL:
            assert spec.respect_robots, f"{spec.id} disables robots.txt"


def test_enabled_sources_are_priority_ordered() -> None:
    priorities = [s.priority for s in load_registry().enabled_sources()]
    assert priorities == sorted(priorities)


def test_flores_is_registered_as_held_out() -> None:
    # Guard against someone quietly turning the benchmark into training data.
    spec = load_registry().by_id("flores_plus_kaa")
    assert "HELD OUT" in spec.notes.upper()


def test_noncommercial_sources_stay_disabled() -> None:
    for spec in load_registry().sources:
        if "NC" in spec.license.upper().split("-"):
            assert not spec.enabled, f"{spec.id} is non-commercial but enabled"


def test_robots_url_is_derived_from_the_host() -> None:
    spec = SourceSpec.model_validate(_spec(url="https://example.uz/qq/deep/path"))
    assert spec.robots_url == "https://example.uz/robots.txt"


def test_duplicate_ids_are_rejected() -> None:
    payload = {
        "version": 1,
        "contact_email": "a@b.c",
        "user_agent": "bot/1.0",
        "sources": [_spec(), _spec(name="Other")],
    }
    with pytest.raises(ValidationError, match="Duplicate source id"):
        SourceRegistry.model_validate(payload)


def test_unknown_script_is_rejected() -> None:
    with pytest.raises(ValidationError, match="Unknown scripts"):
        SourceSpec.model_validate(_spec(scripts=["arabic"]))


def test_bad_id_shape_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SourceSpec.model_validate(_spec(id="News Example"))


def test_unknown_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SourceSpec.model_validate(_spec(delay_second=2.0))


def test_by_id_raises_for_unknown_source() -> None:
    with pytest.raises(KeyError):
        load_registry().by_id("does_not_exist")
