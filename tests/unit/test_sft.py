"""Tests for SFT builders, mixture assembly and configuration."""

from __future__ import annotations

import random
from pathlib import Path

import pytest
from pydantic import ValidationError

from qaraqalpaqmind.common.config import load_config
from qaraqalpaqmind.common.io import write_jsonl
from qaraqalpaqmind.schemas import (
    InstructionRecord,
    Provenance,
    TaskType,
    TranslationRecord,
)
from qaraqalpaqmind.training.sft.builders import grammar, seeds, summarization
from qaraqalpaqmind.training.sft.config import MixtureConfig, SFTConfig
from qaraqalpaqmind.training.sft.mixture import (
    DEFAULT_MIXTURE,
    achievable_size,
    assemble,
    chain_builders,
    resolve_caps,
)

PROV = Provenance(source_id="test", license="CC0-1.0")

KAA = (
    "Qaraqalpaqstan Respublikasınıń paytaxtı Nókis qalası bolıp tabıladı hám "
    "onıń Joqarǵı Keńesi usı qalada jaylasqan."
)


# --- grammar builder ------------------------------------------------------


def test_corruption_reproduces_the_2009_apostrophe_convention() -> None:
    rng = random.Random(0)
    corrupted = grammar.corrupt("Nókis qalası ǵárezsiz", grammar.ErrorType.APOSTROPHE, rng)
    assert "No'kis" in corrupted
    assert "g'árezsiz" not in corrupted  # the á is also converted
    assert "ó" not in corrupted


def test_corruption_reproduces_the_1994_umlaut_convention() -> None:
    rng = random.Random(0)
    corrupted = grammar.corrupt("Nókis úlken ńátiyje", grammar.ErrorType.UMLAUT, rng)
    # Every acute-accented letter converts, not just the one being checked:
    # ó->ö, ú->ü, ń->ñ and á->ä.
    assert corrupted == "Nökis ülken ñätiyje"


def test_corruption_strips_diacritics() -> None:
    rng = random.Random(0)
    corrupted = grammar.corrupt("Joqarǵı Keńes", grammar.ErrorType.STRIPPED_DIACRITICS, rng)
    assert corrupted == "Joqargi Kenes"


def test_mixed_corruption_is_deterministic_for_a_seed() -> None:
    # A rebuild must reproduce the same dataset.
    first = grammar.corrupt(KAA, grammar.ErrorType.MIXED, random.Random(7))
    second = grammar.corrupt(KAA, grammar.ErrorType.MIXED, random.Random(7))
    assert first == second


def test_corruption_actually_changes_the_text() -> None:
    for error in grammar.ErrorType:
        assert grammar.corrupt(KAA, error, random.Random(3)) != KAA, error


def test_text_without_markers_is_unchanged() -> None:
    plain = "Bul plain ascii tekst"
    assert grammar.corrupt(plain, grammar.ErrorType.APOSTROPHE, random.Random(0)) == plain


# --- summarisation builder ------------------------------------------------


def test_lead_and_body_split_drops_the_title_paragraph() -> None:
    text = "Ájiniyaz\n\nÁjiniyaz shayır bolǵan.\n\nOl 1824-jılı tuwılǵan.\n\nQaytıs boldı."
    split = summarization.split_lead_and_body(text)
    assert split is not None
    lead, body = split
    assert lead.startswith("Ájiniyaz shayır")
    assert "1824" in body


def test_short_articles_are_rejected() -> None:
    assert summarization.split_lead_and_body("Bir abzac ǵana.") is None
    assert summarization.split_lead_and_body("Bir\n\nEki") is None


# --- seed loading ---------------------------------------------------------


def test_seed_files_load_and_validate() -> None:
    counts = seeds.available_tasks()
    assert counts, "no seed data found"
    # Every seed-backed task must have at least one usable example.
    for task in ("instruction", "qa", "reasoning", "math", "coding", "conversation"):
        assert counts.get(task, 0) > 0, task


def test_invalid_seed_rows_are_skipped_not_fatal(tmp_path: Path) -> None:
    # A malformed contribution must not stop the other examples loading.
    path = tmp_path / "mixed.jsonl"
    write_jsonl(
        path,
        [
            {"task": "instruction", "instruction": "Soraw?", "output": "Juwap."},
            {"task": "instruction", "instruction": "", "output": "Bos instrukciya."},
            {"task": "nonsense", "foo": "bar"},
            {"task": "qa", "question": "Qashan?", "answer": "Keshe."},
        ],
    )
    stats = seeds.SeedStats()
    records = list(seeds.load_file(path, stats))
    assert len(records) == 2
    assert stats.invalid == 2


def test_seeds_get_default_provenance(tmp_path: Path) -> None:
    path = tmp_path / "instruction.jsonl"
    write_jsonl(path, [{"task": "instruction", "instruction": "A?", "output": "B."}])
    record = next(iter(seeds.load_file(path)))
    assert record.provenance.source_id == "seed_instruction"
    assert record.provenance.human_reviewed
    assert not record.provenance.synthetic


# --- mixture assembly -----------------------------------------------------


def _instruction(index: int) -> InstructionRecord:
    return InstructionRecord(
        instruction=f"Soraw nomer {index} qanday?", output=f"Juwap nomer {index}.", provenance=PROV
    )


def _translation(index: int) -> TranslationRecord:
    return TranslationRecord(
        source_lang="kaa",
        target_lang="eng",
        source_text=f"Qaraqalpaqsha gáp nomer {index} usı jerde.",
        target_text=f"Karakalpak sentence number {index} is here.",
        provenance=PROV,
    )


def test_caps_follow_the_proportions() -> None:
    caps = resolve_caps(1000, {TaskType.TRANSLATION: 0.5, TaskType.INSTRUCTION: 0.5})
    assert caps[TaskType.TRANSLATION] == 500
    assert caps[TaskType.INSTRUCTION] == 500


def test_abundant_tasks_are_capped_not_allowed_to_dominate() -> None:
    # dilmash can supply hundreds of thousands of records; without a cap the
    # mixture becomes a translation engine that occasionally answers questions.
    records = [_translation(i) for i in range(500)] + [_instruction(i) for i in range(50)]
    train, validation, stats = assemble(
        records,
        target_size=100,
        mixture={TaskType.TRANSLATION: 0.5, TaskType.INSTRUCTION: 0.5},
        validation_split=0.0,
        check_contamination=False,
    )
    assert stats.by_task["translation"] == 50
    assert stats.by_task["instruction"] == 50
    assert stats.over_cap > 0
    assert len(train) == 100
    assert validation == []


def test_duplicates_are_removed_across_tasks() -> None:
    duplicate = _instruction(1)
    _, _, stats = assemble(
        [duplicate, duplicate, _instruction(2)],
        target_size=100,
        mixture={TaskType.INSTRUCTION: 1.0},
        validation_split=0.0,
        check_contamination=False,
    )
    assert stats.duplicates == 1
    assert stats.kept == 2


def test_validation_covers_every_task() -> None:
    # Splitting the concatenated stream would give a validation set made
    # entirely of whichever task happened to land last.
    records = [_translation(i) for i in range(50)] + [_instruction(i) for i in range(50)]
    _, validation, _ = assemble(
        records,
        target_size=100,
        mixture={TaskType.TRANSLATION: 0.5, TaskType.INSTRUCTION: 0.5},
        validation_split=0.2,
        check_contamination=False,
    )
    tasks = {record.task for record in validation}
    assert tasks == {TaskType.TRANSLATION, TaskType.INSTRUCTION}


def test_shortfall_is_reported() -> None:
    _, _, stats = assemble(
        [_instruction(i) for i in range(5)],
        target_size=100,
        mixture={TaskType.INSTRUCTION: 0.5, TaskType.MATH: 0.5},
        validation_split=0.0,
        check_contamination=False,
    )
    assert stats.shortfall["math"] == 50
    assert stats.shortfall["instruction"] == 45


def test_assembly_is_deterministic() -> None:
    records = [_instruction(i) for i in range(40)]
    kwargs = {
        "target_size": 40,
        "mixture": {TaskType.INSTRUCTION: 1.0},
        "validation_split": 0.1,
        "check_contamination": False,
    }
    first, _, _ = assemble(list(records), **kwargs)  # type: ignore[arg-type]
    second, _, _ = assemble(list(records), **kwargs)  # type: ignore[arg-type]
    assert [r.id for r in first] == [r.id for r in second]


def test_achievable_size_is_bound_by_the_scarcest_task() -> None:
    size, binding = achievable_size(
        {TaskType.TRANSLATION: 10_000, TaskType.QA: 6},
        {TaskType.TRANSLATION: 0.5, TaskType.QA: 0.5},
    )
    assert binding is TaskType.QA
    assert size == 12


def test_chain_builders_interleaves() -> None:
    # Concatenating would let translation exhaust the caps before the seed sets
    # are read at all.
    a = iter([_translation(1), _translation(2), _translation(3)])
    b = iter([_instruction(1)])
    tasks = [record.task for record in chain_builders(a, b)]
    assert tasks[:2] == [TaskType.TRANSLATION, TaskType.INSTRUCTION]


def test_default_mixture_sums_to_one() -> None:
    assert abs(sum(DEFAULT_MIXTURE.values()) - 1.0) < 1e-9


# --- configuration --------------------------------------------------------


def test_shipped_sft_config_is_valid() -> None:
    config = load_config("sft/qwen3_8b_qlora_24gb.yaml", SFTConfig)
    assert config.completion_only_loss
    assert config.cpt_adapter is not None
    # SFT should use a smaller adapter and lower rate than CPT.
    assert config.lora.r < 64
    assert config.optim.learning_rate < 1e-4


def test_shipped_mixture_config_is_valid() -> None:
    config = load_config("sft/mixture_v1.yaml", MixtureConfig)
    assert config.check_contamination
    assert abs(sum(config.proportions.values()) - 1.0) < 1e-6


def test_too_many_sft_epochs_is_rejected() -> None:
    # SFT overfits far faster than continued pretraining.
    with pytest.raises(ValidationError, match="overfit"):
        SFTConfig.model_validate({"runtime": {"num_epochs": 10}})


def test_negative_proportions_are_rejected() -> None:
    with pytest.raises(ValidationError):
        MixtureConfig(proportions={"translation": -0.5, "qa": 1.0})
