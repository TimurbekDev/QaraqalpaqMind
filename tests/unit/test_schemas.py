"""Tests for the task dataset schemas.

Every canonical example is validated here, which is the point of keeping them
in code: a markdown example drifts silently the first time a field is renamed,
a constructed one fails the build.
"""

from __future__ import annotations

import orjson
import pytest
from pydantic import ValidationError

from qaraqalpaqmind.schemas import (
    RECORD_TYPES,
    BenchmarkRecord,
    ConversationRecord,
    GrammarRecord,
    InstructionRecord,
    Provenance,
    SummarizationRecord,
    TaskType,
    TranslationRecord,
    Turn,
    parse_record,
)
from qaraqalpaqmind.schemas.examples import ALL_EXAMPLES, EXAMPLES

PROV = Provenance(source_id="test", license="CC0-1.0")


# --- coverage -------------------------------------------------------------


def test_every_task_type_has_a_record_class() -> None:
    assert set(RECORD_TYPES) == set(TaskType)


def test_every_task_type_has_an_example() -> None:
    assert set(EXAMPLES) == set(TaskType)


@pytest.mark.parametrize("record", ALL_EXAMPLES, ids=lambda r: f"{r.task.value}-{r.id[:6]}")
def test_examples_are_valid_and_round_trip(record) -> None:  # type: ignore[no-untyped-def]
    payload = record.model_dump(mode="json")
    restored = parse_record(orjson.loads(orjson.dumps(payload)))
    assert restored.task is record.task
    assert restored.model_dump(mode="json") == payload


@pytest.mark.parametrize("record", ALL_EXAMPLES, ids=lambda r: r.task.value)
def test_examples_render_as_chat(record) -> None:  # type: ignore[no-untyped-def]
    messages = record.to_messages()
    assert messages, record.task
    assert all(set(m) == {"role", "content"} for m in messages)
    assert all(m["content"].strip() for m in messages)
    assert all(m["role"] in {"system", "user", "assistant"} for m in messages)


def test_trainable_tasks_end_on_an_assistant_turn() -> None:
    # Pretraining is raw text and has no target; everything else must have one,
    # or the loss is computed over no assistant tokens at all.
    for record in ALL_EXAMPLES:
        if record.task is TaskType.PRETRAIN:
            continue
        assert record.to_messages()[-1]["role"] == "assistant", record.task


# --- identity and provenance ---------------------------------------------


def test_ids_are_content_derived_and_stable() -> None:
    first = InstructionRecord(instruction="Salawmatsız ba?", output="Salawmatsız!", provenance=PROV)
    second = InstructionRecord(instruction="Salawmatsız ba?", output="Salawmatsız!", provenance=PROV)
    assert first.id == second.id

    different = InstructionRecord(instruction="Basqa soraw?", output="Salawmatsız!", provenance=PROV)
    assert different.id != first.id


def test_explicit_id_is_respected() -> None:
    record = InstructionRecord(id="fixed", instruction="a", output="b", provenance=PROV)
    assert record.id == "fixed"


def test_synthetic_data_must_be_labelled() -> None:
    # Training on model output without knowing it is model output is how a
    # corpus degrades across generations.
    with pytest.raises(ValidationError, match="synthetic"):
        Provenance(source_id="gen", generator="qwen3-8b")

    ok = Provenance(source_id="gen", synthetic=True, generator="qwen3-8b")
    assert ok.synthetic


# --- per-task validation --------------------------------------------------


def test_conversation_must_end_with_the_assistant() -> None:
    with pytest.raises(ValidationError, match="assistant turn"):
        ConversationRecord(
            messages=[
                Turn(role="user", content="Salawmatsız?"),
                Turn(role="assistant", content="Salawmatsız!"),
                Turn(role="user", content="Taǵı bir soraw bar."),
            ],
            provenance=PROV,
        )


def test_conversation_needs_a_user_turn() -> None:
    with pytest.raises(ValidationError, match="user turn"):
        ConversationRecord(
            messages=[
                Turn(role="system", content="Sen járdemshisen."),
                Turn(role="assistant", content="Salawmatsız!"),
            ],
            provenance=PROV,
        )


def test_translation_normalises_language_codes() -> None:
    # dilmash writes kaa_Latn, FLORES writes kaa_Latn, humans write kaa.
    record = TranslationRecord(
        source_lang="kaa_Latn",
        target_lang="uzn_Latn",
        source_text="Salawmatsız",
        target_text="Assalomu alaykum",
        provenance=PROV,
    )
    assert record.source_lang == "kaa"
    assert record.target_lang == "uzn"


def test_translation_directions_must_differ() -> None:
    with pytest.raises(ValidationError, match="must differ"):
        TranslationRecord(
            source_lang="kaa",
            target_lang="kaa_Latn",
            source_text="a",
            target_text="b",
            provenance=PROV,
        )


def test_translation_prompt_is_in_karakalpak() -> None:
    messages = EXAMPLES[TaskType.TRANSLATION].to_messages()
    # Asking in English to answer in Karakalpak makes the model translate the
    # instruction first.
    assert "awdar" in messages[1]["content"]


def test_grammar_pair_must_actually_differ() -> None:
    with pytest.raises(ValidationError, match="identical"):
        GrammarRecord(incorrect="Bul durıs.", correct="Bul durıs.", provenance=PROV)


def test_grammar_example_covers_the_orthography_error_we_actually_see() -> None:
    record = EXAMPLES[TaskType.GRAMMAR]
    assert isinstance(record, GrammarRecord)
    assert "'" in record.incorrect  # 2009 apostrophe orthography
    assert "ó" in record.correct  # 2016 standard
    assert record.error_type == "orthography"


def test_summary_must_be_shorter_than_the_document() -> None:
    with pytest.raises(ValidationError, match="shorter"):
        SummarizationRecord(document="Qısqa tekst.", summary="Bunnan da uzınıraq juwmaq.",
                            provenance=PROV)


def test_benchmark_answer_must_be_among_the_choices() -> None:
    with pytest.raises(ValidationError, match="not one of the choices"):
        BenchmarkRecord(
            subject="geography",
            question="Paytaxt qaysı qala?",
            choices=["Xiywa", "Buxara"],
            answer="Nókis",
            provenance=PROV,
        )


def test_benchmark_supports_free_form_and_multiple_choice() -> None:
    assert EXAMPLES[TaskType.BENCHMARK].is_multiple_choice  # type: ignore[attr-defined]
    free_form = BenchmarkRecord(
        subject="history", question="Qashan?", answer="XIX ásirde", provenance=PROV
    )
    assert not free_form.is_multiple_choice


def test_multiple_choice_renders_lettered_options() -> None:
    rendered = EXAMPLES[TaskType.BENCHMARK].to_messages()[1]["content"]
    assert "A) Nókis" in rendered
    assert "B) Xiywa" in rendered


# --- parsing --------------------------------------------------------------


def test_parse_record_dispatches_on_task() -> None:
    payload = EXAMPLES[TaskType.MATH].model_dump(mode="json")
    assert parse_record(payload).task is TaskType.MATH


def test_parse_record_rejects_unknown_and_missing_tasks() -> None:
    with pytest.raises(ValueError, match="no 'task' field"):
        parse_record({"text": "a"})
    with pytest.raises(ValueError, match="unknown task"):
        parse_record({"task": "sentiment", "text": "a"})


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        InstructionRecord(
            instruction="a", output="b", provenance=PROV, unexpected="boom"  # type: ignore[call-arg]
        )


def test_empty_required_text_is_rejected() -> None:
    with pytest.raises(ValidationError):
        InstructionRecord(instruction="", output="b", provenance=PROV)
