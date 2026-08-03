"""Tests for preference builders, mixture assembly and DPO configuration."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from qaraqalpaqmind.common.config import load_config
from qaraqalpaqmind.schemas import PreferenceRecord, Provenance, TaskType, parse_record
from qaraqalpaqmind.training.dpo.builders import on_policy, quality
from qaraqalpaqmind.training.dpo.config import DPOConfig, PreferenceMixtureConfig
from qaraqalpaqmind.training.dpo.mixture import (
    DEFAULT_MIXTURE,
    assemble,
    chain_builders,
    difference_ratio,
    resolve_caps,
)

PROV = Provenance(source_id="test", license="CC0-1.0")

PROSE = (
    "Qaraqalpaqstan Respublikası Ózbekstan Respublikasınıń quramındaǵı avtonomiyalı "
    "respublika bolıp tabıladı. Onıń paytaxtı Nókis qalası bolıp esaplanadı hám "
    "aymaǵı 166 mıń kvadrat kilometrden aslam maydandı iyeleydi."
)


def _pair(index: int, criterion: str = "orthography") -> PreferenceRecord:
    # The criterion is part of the content: deduplication hashes prompt plus
    # both sides, so two records differing only in their label are one record.
    return PreferenceRecord(
        prompt=f"Soraw nomer {index} {criterion} boyınsha qanday?",
        chosen=f"Durıs juwap nomer {index} usı jerde tolıq jazılǵan gáp bolıp tabıladı.",
        rejected=f"Qa'te juwap nomer {index} usi jerde toliq jazilg'an ga'p bolip tabiladi.",
        criterion=criterion,
        provenance=PROV,
    )


# --- record schema --------------------------------------------------------


def test_preference_record_round_trips() -> None:
    record = _pair(1)
    restored = parse_record(record.model_dump(mode="json"))
    assert restored.task is TaskType.PREFERENCE
    assert restored.model_dump(mode="json") == record.model_dump(mode="json")


def test_identical_sides_are_rejected() -> None:
    # A pair whose sides match teaches nothing, and DPO would still train on it.
    with pytest.raises(ValidationError, match="teaches nothing"):
        PreferenceRecord(
            prompt="Soraw?", chosen="Birdey.", rejected="Birdey.",
            criterion="orthography", provenance=PROV,
        )


def test_dpo_row_has_the_conversational_shape_trl_expects() -> None:
    row = _pair(1).to_dpo_row()
    assert set(row) == {"prompt", "chosen", "rejected"}
    assert row["prompt"][-1]["role"] == "user"
    assert row["chosen"][0]["role"] == "assistant"
    assert row["rejected"][0]["role"] == "assistant"


def test_to_messages_uses_the_chosen_side() -> None:
    # Used for contamination and length checks, which must see what would be
    # trained toward.
    record = _pair(1)
    assert record.to_messages()[-1]["content"] == record.chosen


# --- quality builder prose gate ------------------------------------------


def test_good_prose_passes_the_gate() -> None:
    assert quality.is_good_prose(PROSE)


@pytest.mark.parametrize(
    ("name", "text"),
    [
        ("arithmetic table", "\n".join(f"10 + {i} = {10 + i}" for i in range(30))),
        ("repetition", "Nókis " * 40 + "."),
        ("no sentence end", " ".join(["qaraqalpaqstan respublikası"] * 15)),
        ("too short", "Qısqa gáp bolıp tabıladı."),
    ],
)
def test_non_prose_is_rejected_as_a_chosen_side(name: str, text: str) -> None:
    # Surviving the cleaner is a lower bar than being worth imitating. An
    # arithmetic table passed cleaning and was briefly used as a `chosen` side,
    # which would have taught the model to emit multiplication tables.
    assert not quality.is_good_prose(text), name


# --- on-policy scoring ----------------------------------------------------


def test_karakalpak_scores_above_foreign_output() -> None:
    kaa, _ = on_policy.score_completion(PROSE)
    english, reason = on_policy.score_completion(
        "The Republic of Karakalpakstan is an autonomous republic within Uzbekistan "
        "and its capital is the city of Nukus, which lies on the Amu Darya."
    )
    assert kaa > english
    assert reason == "not_karakalpak"


def test_repetition_is_punished() -> None:
    # The dominant reason is the largest penalty, not whichever check ran
    # first - otherwise a repetitive answer is mislabelled "weak_karakalpak"
    # purely because the language check comes earlier.
    score, reason = on_policy.score_completion("Nókis qalası " * 30)
    assert reason == "repetition"
    assert score < 0.5


def test_too_short_scores_zero() -> None:
    score, reason = on_policy.score_completion("Awa.")
    assert score == 0.0
    assert reason == "too_short"


def test_pairs_need_a_real_score_gap() -> None:
    # DPO will learn a preference between two equally good answers if given
    # one, so near-ties must not become pairs.
    pairs = list(
        on_policy.pairs_from_generations(
            "Soraw?", [PROSE, PROSE.replace("Nókis", "Nókis qalası")], PROV
        )
    )
    assert pairs == []


def test_pairs_are_emitted_when_the_gap_is_real() -> None:
    pairs = list(
        on_policy.pairs_from_generations(
            "Soraw?",
            [PROSE, "This is an English answer to a Karakalpak question, entirely."],
            PROV,
        )
    )
    assert len(pairs) == 1
    assert pairs[0].chosen == PROSE
    assert pairs[0].meta["rejected_reason"] == "not_karakalpak"


def test_on_policy_build_accepts_an_injected_sampler() -> None:
    # The real sampler needs a GPU; the builder must remain testable without one.
    def fake(prompt: str, n: int) -> list[str]:
        return [PROSE, "An entirely English completion for this prompt, written out."]

    records = list(on_policy.build(["Soraw?"], model_path="fake/model", generate=fake))
    assert len(records) == 1
    assert records[0].provenance.synthetic
    assert records[0].provenance.generator == "fake/model"


# --- mixture --------------------------------------------------------------


def test_difference_ratio() -> None:
    assert difference_ratio("abc", "abc") == 0.0
    assert difference_ratio("abc", "xyz") == 1.0
    assert 0.0 < difference_ratio("abcdef", "abcdXf") < 1.0


def test_caps_follow_proportions() -> None:
    caps = resolve_caps(1000, {"a": 0.5, "b": 0.5})
    assert caps == {"a": 500, "b": 500}


def test_near_identical_pairs_are_dropped() -> None:
    long_text = "Bul juwap bolıp tabıladı hám tolıq jazılǵan gáp esaplanadı" * 3
    twin = PreferenceRecord(
        prompt="Soraw?",
        chosen=long_text + ".",
        rejected=long_text + "!",
        criterion="orthography",
        provenance=PROV,
    )
    _, _, stats = assemble(
        [twin, _pair(1)], target_size=10, mixture={"orthography": 1.0},
        validation_split=0.0, check_contamination=False,
    )
    assert stats.too_similar == 1
    assert stats.kept == 1


def test_criteria_are_capped_independently() -> None:
    records = [_pair(i, "orthography") for i in range(50)]
    records += [_pair(i, "language_consistency") for i in range(50)]
    _, _, stats = assemble(
        records, target_size=20,
        mixture={"orthography": 0.5, "language_consistency": 0.5},
        validation_split=0.0, check_contamination=False,
    )
    assert stats.by_criterion == {"orthography": 10, "language_consistency": 10}
    assert stats.over_cap == 80


def test_duplicates_are_removed() -> None:
    duplicate = _pair(1)
    _, _, stats = assemble(
        [duplicate, duplicate], target_size=10, mixture={"orthography": 1.0},
        validation_split=0.0, check_contamination=False,
    )
    assert stats.duplicates == 1


def test_validation_covers_every_criterion() -> None:
    records = [_pair(i, "orthography") for i in range(30)]
    records += [_pair(i, "language_consistency") for i in range(30)]
    _, validation, _ = assemble(
        records, target_size=60,
        mixture={"orthography": 0.5, "language_consistency": 0.5},
        validation_split=0.2, check_contamination=False,
    )
    assert {r.criterion for r in validation} == {"orthography", "language_consistency"}


def test_chain_builders_interleaves() -> None:
    a = iter([_pair(1, "orthography"), _pair(2, "orthography")])
    b = iter([_pair(1, "language_consistency")])
    criteria = [record.criterion for record in chain_builders(a, b)]
    assert criteria[:2] == ["orthography", "language_consistency"]


def test_default_mixture_sums_to_one() -> None:
    assert abs(sum(DEFAULT_MIXTURE.values()) - 1.0) < 1e-9


# --- configuration --------------------------------------------------------


def test_shipped_dpo_configs_are_valid() -> None:
    config = load_config("dpo/qwen3_8b_qlora_24gb.yaml", DPOConfig)
    assert config.sft_adapter is not None
    assert config.beta == 0.1
    # DPO should use a smaller adapter and much lower rate than SFT.
    assert config.lora.r < 32
    assert config.optim.learning_rate <= 1e-5

    mixture = load_config("dpo/mixture_v1.yaml", PreferenceMixtureConfig)
    assert abs(sum(mixture.proportions.values()) - 1.0) < 1e-6


def test_high_learning_rate_is_rejected() -> None:
    # DPO refines a working model; a large rate erases what SFT installed.
    with pytest.raises(ValidationError, match="too high for DPO"):
        DPOConfig.model_validate({"optim": {"learning_rate": 2e-5}})


def test_too_many_epochs_is_rejected() -> None:
    with pytest.raises(ValidationError, match="over-optimise"):
        DPOConfig.model_validate({"runtime": {"num_epochs": 5}})


def test_prompt_must_fit_inside_the_window() -> None:
    with pytest.raises(ValidationError, match="max_prompt_length"):
        DPOConfig.model_validate({"max_length": 512, "max_prompt_length": 512})
