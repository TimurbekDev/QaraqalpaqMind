"""Held-out data must never be able to reach a training set.

These are the tests that stop a benchmark from quietly becoming training data.
The failure they guard against is silent: nothing errors, scores just come out
inflated, and Phase 8 reports memorisation as capability.
"""

from __future__ import annotations

import pytest

from qaraqalpaqmind.cleaning.pipeline import available_sources, clean_source
from qaraqalpaqmind.crawlers.core.registry import load_registry


def test_flores_is_marked_held_out_in_the_registry() -> None:
    spec = load_registry().by_id("flores_plus_kaa")
    assert spec.held_out, "the translation benchmark must be flagged held_out"


def test_held_out_ids_are_exposed() -> None:
    assert "flores_plus_kaa" in load_registry().held_out_ids()


def test_cleaning_refuses_held_out_sources() -> None:
    # Cleaning writes to data/processed/, which deduplication reads in full.
    with pytest.raises(ValueError, match="held-out"):
        clean_source("flores_plus_kaa")


def test_available_sources_excludes_held_out() -> None:
    # `qm clean all` iterates this; a benchmark appearing here would be swept
    # into the corpus without anyone choosing to do so.
    assert "flores_plus_kaa" not in available_sources()


def test_available_sources_can_still_list_everything_explicitly() -> None:
    listed = available_sources(include_held_out=True)
    assert isinstance(listed, list)


def test_every_held_out_source_has_a_stated_reason() -> None:
    for spec in load_registry().sources:
        if spec.held_out:
            assert "HELD OUT" in spec.notes.upper(), spec.id
