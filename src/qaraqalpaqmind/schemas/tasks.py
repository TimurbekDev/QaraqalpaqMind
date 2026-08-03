"""The eleven task record types.

Each knows how to render itself as chat messages, so the SFT trainer in Phase 6
handles one format regardless of which task a row came from. Task-specific
prompt wording lives here rather than in the trainer, which keeps the trainer
free of per-task branching.

Prompts are written in Karakalpak. A model asked in English to answer in
Karakalpak has to translate the instruction first; asking in the target
language is both more natural and one less thing to get wrong.
"""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .base import TaskRecord, TaskType, system_prompt


class PretrainRecord(TaskRecord):
    """Raw text for continued pretraining. No prompt, no target - just language."""

    task: Literal[TaskType.PRETRAIN] = TaskType.PRETRAIN
    text: str = Field(min_length=1)

    def to_messages(self) -> list[dict[str, str]]:
        # Pretraining is not chat-formatted; this exists so length statistics
        # and contamination checks work uniformly across every task type.
        return [{"role": "user", "content": self.text}]

    def training_text(self) -> str:
        return self.text


class InstructionRecord(TaskRecord):
    """Single-turn instruction following, with optional supporting input."""

    task: Literal[TaskType.INSTRUCTION] = TaskType.INSTRUCTION
    instruction: str = Field(min_length=1)
    input: str = ""
    output: str = Field(min_length=1)

    def to_messages(self) -> list[dict[str, str]]:
        user = f"{self.instruction}\n\n{self.input}".strip() if self.input else self.instruction
        return [system_prompt(), {"role": "user", "content": user},
                {"role": "assistant", "content": self.output}]


class Turn(BaseModel):
    """One message in a conversation."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ConversationRecord(TaskRecord):
    """Multi-turn dialogue."""

    task: Literal[TaskType.CONVERSATION] = TaskType.CONVERSATION
    messages: list[Turn] = Field(min_length=2)

    @model_validator(mode="after")
    def _must_end_on_the_assistant(self) -> ConversationRecord:
        # A conversation ending on a user turn has nothing to train against;
        # the loss would be computed over no assistant tokens at all.
        if self.messages[-1].role != "assistant":
            raise ValueError("a conversation must end with an assistant turn")
        if not any(m.role == "user" for m in self.messages):
            raise ValueError("a conversation needs at least one user turn")
        return self

    def to_messages(self) -> list[dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in self.messages]


class TranslationRecord(TaskRecord):
    """A translation pair in a stated direction."""

    task: Literal[TaskType.TRANSLATION] = TaskType.TRANSLATION
    source_lang: str
    target_lang: str
    source_text: str = Field(min_length=1)
    target_text: str = Field(min_length=1)

    # Karakalpak names for the languages, used in the prompt.
    LANGUAGE_NAMES: ClassVar[dict[str, str]] = {
        "kaa": "qaraqalpaq", "uzb": "ózbek", "uzn": "ózbek",
        "rus": "rus", "eng": "ingliz", "kaz": "qazaq", "tur": "túrk",
    }

    @field_validator("source_lang", "target_lang")
    @classmethod
    def _normalise_language_code(cls, value: str) -> str:
        # dilmash writes `kaa_Latn`; FLORES writes `kaa_Latn`; humans write `kaa`.
        return value.split("_")[0].lower()

    @model_validator(mode="after")
    def _directions_must_differ(self) -> TranslationRecord:
        if self.source_lang == self.target_lang:
            raise ValueError("source_lang and target_lang must differ")
        return self

    def to_messages(self) -> list[dict[str, str]]:
        target = self.LANGUAGE_NAMES.get(self.target_lang, self.target_lang)
        prompt = f"Tómendegi tekstti {target} tiline awdar:\n\n{self.source_text}"
        return [system_prompt(), {"role": "user", "content": prompt},
                {"role": "assistant", "content": self.target_text}]


class GrammarRecord(TaskRecord):
    """Correcting a grammatical or orthographic error."""

    task: Literal[TaskType.GRAMMAR] = TaskType.GRAMMAR
    incorrect: str = Field(min_length=1)
    correct: str = Field(min_length=1)
    explanation: str = ""
    error_type: str = Field(default="", description="e.g. orthography, agreement, case")

    @model_validator(mode="after")
    def _must_actually_differ(self) -> GrammarRecord:
        if self.incorrect.strip() == self.correct.strip():
            raise ValueError("incorrect and correct are identical; nothing is being taught")
        return self

    def to_messages(self) -> list[dict[str, str]]:
        prompt = f"Tómendegi gápti qátelerin dúzetip jaz:\n\n{self.incorrect}"
        answer = self.correct
        if self.explanation:
            answer = f"{self.correct}\n\nTúsindirme: {self.explanation}"
        return [system_prompt(), {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer}]


class QARecord(TaskRecord):
    """Question answering, optionally grounded in a passage."""

    task: Literal[TaskType.QA] = TaskType.QA
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    context: str = ""

    def to_messages(self) -> list[dict[str, str]]:
        prompt = (
            f"Tómendegi tekstke tiykarlanıp sorawǵa juwap ber:\n\n{self.context}\n\n"
            f"Soraw: {self.question}"
            if self.context
            else self.question
        )
        return [system_prompt(), {"role": "user", "content": prompt},
                {"role": "assistant", "content": self.answer}]


class SummarizationRecord(TaskRecord):
    """Condensing a document."""

    task: Literal[TaskType.SUMMARIZATION] = TaskType.SUMMARIZATION
    document: str = Field(min_length=1)
    summary: str = Field(min_length=1)

    @model_validator(mode="after")
    def _summary_must_be_shorter(self) -> SummarizationRecord:
        if len(self.summary) >= len(self.document):
            raise ValueError("summary is not shorter than the document")
        return self

    def to_messages(self) -> list[dict[str, str]]:
        prompt = f"Tómendegi tekstti qısqasha juwmaqlap ber:\n\n{self.document}"
        return [system_prompt(), {"role": "user", "content": prompt},
                {"role": "assistant", "content": self.summary}]


class ReasoningRecord(TaskRecord):
    """A question requiring stated intermediate steps."""

    task: Literal[TaskType.REASONING] = TaskType.REASONING
    question: str = Field(min_length=1)
    reasoning: str = Field(min_length=1)
    answer: str = Field(min_length=1)

    def to_messages(self) -> list[dict[str, str]]:
        prompt = f"{self.question}\n\nAdımlap oylap juwap ber."
        return [system_prompt(), {"role": "user", "content": prompt},
                {"role": "assistant", "content": f"{self.reasoning}\n\nJuwap: {self.answer}"}]


class CodingRecord(TaskRecord):
    """A programming task described in Karakalpak."""

    task: Literal[TaskType.CODING] = TaskType.CODING
    prompt: str = Field(min_length=1)
    code: str = Field(min_length=1)
    language: str = "python"
    explanation: str = ""
    tests: str = ""

    def to_messages(self) -> list[dict[str, str]]:
        answer = f"```{self.language}\n{self.code}\n```"
        if self.explanation:
            answer = f"{answer}\n\n{self.explanation}"
        return [system_prompt(), {"role": "user", "content": self.prompt},
                {"role": "assistant", "content": answer}]


class MathRecord(TaskRecord):
    """A mathematics problem with a worked solution."""

    task: Literal[TaskType.MATH] = TaskType.MATH
    problem: str = Field(min_length=1)
    solution: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    level: str = ""

    def to_messages(self) -> list[dict[str, str]]:
        return [system_prompt(), {"role": "user", "content": self.problem},
                {"role": "assistant", "content": f"{self.solution}\n\nJuwap: {self.answer}"}]


class BenchmarkRecord(TaskRecord):
    """An evaluation item. Never used for training.

    Multiple choice when `choices` is set, free-form otherwise.
    """

    task: Literal[TaskType.BENCHMARK] = TaskType.BENCHMARK
    subject: str = Field(description="grammar, history, geography, math, coding, ...")
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    choices: list[str] = Field(default_factory=list)
    context: str = ""

    @model_validator(mode="after")
    def _answer_must_be_available(self) -> BenchmarkRecord:
        if self.choices:
            if len(self.choices) < 2:
                raise ValueError("a multiple-choice item needs at least two choices")
            if self.answer not in self.choices:
                raise ValueError("answer is not one of the choices")
        return self

    @property
    def is_multiple_choice(self) -> bool:
        return bool(self.choices)

    def to_messages(self) -> list[dict[str, str]]:
        prompt = self.question
        if self.context:
            prompt = f"{self.context}\n\n{prompt}"
        if self.choices:
            options = "\n".join(
                f"{chr(65 + i)}) {choice}" for i, choice in enumerate(self.choices)
            )
            prompt = f"{prompt}\n\n{options}"
        return [system_prompt(), {"role": "user", "content": prompt},
                {"role": "assistant", "content": self.answer}]


class PreferenceRecord(TaskRecord):
    """A preference pair for direct preference optimisation.

    `chosen` and `rejected` are two responses to the same prompt. DPO raises the
    model's likelihood of the first relative to the second, so what matters is
    that the pair differ in exactly the dimension being taught - anything else
    they differ in is also being taught, silently.

    `criterion` names that dimension, so a mixture can be audited and a
    regression traced back to the preference that caused it.
    """

    task: Literal[TaskType.PREFERENCE] = TaskType.PREFERENCE
    prompt: str = Field(min_length=1)
    chosen: str = Field(min_length=1)
    rejected: str = Field(min_length=1)
    criterion: str = Field(
        min_length=1,
        description="What the pair teaches: orthography, language_consistency, quality, ...",
    )
    system: str = ""

    @model_validator(mode="after")
    def _sides_must_differ(self) -> PreferenceRecord:
        if self.chosen.strip() == self.rejected.strip():
            raise ValueError("chosen and rejected are identical; the pair teaches nothing")
        return self

    def to_messages(self) -> list[dict[str, str]]:
        """Prompt plus the preferred response, for length and contamination checks."""
        return [
            system_prompt(self.system or None),
            {"role": "user", "content": self.prompt},
            {"role": "assistant", "content": self.chosen},
        ]

    def to_dpo_row(self) -> dict[str, list[dict[str, str]]]:
        """The conversational format TRL's DPOTrainer expects."""
        prompt = [
            system_prompt(self.system or None),
            {"role": "user", "content": self.prompt},
        ]
        return {
            "prompt": prompt,
            "chosen": [{"role": "assistant", "content": self.chosen}],
            "rejected": [{"role": "assistant", "content": self.rejected}],
        }


RECORD_TYPES: dict[TaskType, type[TaskRecord]] = {
    TaskType.PRETRAIN: PretrainRecord,
    TaskType.INSTRUCTION: InstructionRecord,
    TaskType.CONVERSATION: ConversationRecord,
    TaskType.TRANSLATION: TranslationRecord,
    TaskType.GRAMMAR: GrammarRecord,
    TaskType.QA: QARecord,
    TaskType.SUMMARIZATION: SummarizationRecord,
    TaskType.REASONING: ReasoningRecord,
    TaskType.CODING: CodingRecord,
    TaskType.MATH: MathRecord,
    TaskType.BENCHMARK: BenchmarkRecord,
    TaskType.PREFERENCE: PreferenceRecord,
}


def parse_record(payload: dict[str, object]) -> TaskRecord:
    """Build the right record type from a raw JSONL row."""
    raw_task = payload.get("task")
    if raw_task is None:
        raise ValueError("record has no 'task' field")
    try:
        task = TaskType(str(raw_task))
    except ValueError as exc:
        known = ", ".join(t.value for t in TaskType)
        raise ValueError(f"unknown task {raw_task!r}; expected one of: {known}") from exc
    return RECORD_TYPES[task].model_validate(payload)
