"""On-policy preference pairs, sampled from the SFT model.

The three other builders make pairs from data that already exists. This one
makes them from what the model *actually does*, which is what DPO was designed
for: correcting a specific model's specific failures rather than teaching a
general preference.

For each prompt the model generates several completions; a scorer ranks them;
the best becomes `chosen` and the worst `rejected`. The scorer here is
rule-based - language consistency, orthography, length, repetition - because a
Karakalpak reward model does not exist and a judge model would be scoring a
language it barely knows.

**This module requires a trained SFT checkpoint and a GPU, and has not been
executed.** The other three builders were run against real data; this one is
written from the same spec but is unverified. Treat its first run as a debugging
session, not a data-generation run.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from ....common.logging import get_logger
from ....preprocessing.orthography import to_latin2016
from ....preprocessing.script import analyse
from ....schemas import PreferenceRecord, Provenance

logger = get_logger(__name__)

CRITERION = "on_policy"

_MIN_CHARS = 20


@dataclass(slots=True)
class OnPolicyStats:
    prompts: int = 0
    generations: int = 0
    emitted: int = 0
    no_spread: int = 0
    by_reason: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"on-policy preferences: {self.emitted:,} pairs from {self.prompts:,} prompts "
            f"({self.generations:,} generations, {self.no_spread:,} with no usable spread)"
        )


def score_completion(text: str) -> tuple[float, str]:
    """Rank a completion. Returns `(score, dominant_reason)`.

    Rule-based on purpose. A Karakalpak reward model does not exist, and an
    LLM judge would be evaluating a language it has very little of - its
    rankings would measure its own weakness rather than the candidate's.
    """
    stripped = text.strip()
    if len(stripped) < _MIN_CHARS:
        return 0.0, "too_short"

    report = analyse(stripped[:3000])

    # Collect every penalty, then report the LARGEST as the dominant reason.
    # Reporting whichever fired first would label a repetitive answer as
    # "weak_karakalpak" merely because the language check runs earlier, which
    # makes the rejection statistics misleading.
    penalties: list[tuple[float, str]] = []

    # Answering in the wrong language is the failure most worth punishing.
    if report.karakalpak_score < 0.3:
        penalties.append((0.6, "not_karakalpak"))
    elif report.karakalpak_score < 0.5:
        penalties.append((0.25, "weak_karakalpak"))

    if to_latin2016(stripped) != stripped:
        penalties.append((0.2, "orthography"))

    # Degenerate repetition, the classic small-corpus failure.
    words = stripped.split()
    if len(words) > 12:
        unique_ratio = len({w.lower() for w in words}) / len(words)
        if unique_ratio < 0.35:
            penalties.append((0.4, "repetition"))

    if not penalties:
        return 1.0, "ok"

    score = 1.0 - sum(weight for weight, _ in penalties)
    dominant = max(penalties, key=lambda item: item[0])[1]
    return max(0.0, score), dominant


def pairs_from_generations(
    prompt: str,
    completions: Sequence[str],
    provenance: Provenance,
    *,
    min_gap: float = 0.25,
    stats: OnPolicyStats | None = None,
) -> Iterator[PreferenceRecord]:
    """Turn one prompt's completions into a preference pair, if they differ enough."""
    tracker = stats if stats is not None else OnPolicyStats()
    tracker.prompts += 1
    tracker.generations += len(completions)

    scored = sorted(
        ((score_completion(c), c) for c in completions if c.strip()),
        key=lambda item: item[0][0],
        reverse=True,
    )
    if len(scored) < 2:
        tracker.no_spread += 1
        return

    (best_score, best_reason), best = scored[0]
    (worst_score, worst_reason), worst = scored[-1]

    # Without a real gap the pair teaches noise. DPO will happily learn a
    # preference between two equally good answers if given one.
    if best_score - worst_score < min_gap or best.strip() == worst.strip():
        tracker.no_spread += 1
        return

    tracker.by_reason[worst_reason] = tracker.by_reason.get(worst_reason, 0) + 1
    tracker.emitted += 1
    yield PreferenceRecord(
        prompt=prompt,
        chosen=best,
        rejected=worst,
        criterion=CRITERION,
        provenance=provenance,
        meta={
            "chosen_score": round(best_score, 3),
            "rejected_score": round(worst_score, 3),
            "rejected_reason": worst_reason,
            "chosen_reason": best_reason,
        },
    )


def build(
    prompts: Sequence[str],
    *,
    model_path: str,
    samples_per_prompt: int = 4,
    temperature: float = 0.9,
    max_new_tokens: int = 256,
    limit: int | None = None,
    generate: Any = None,
) -> Iterator[PreferenceRecord]:
    """Sample completions and pair the best against the worst.

    Args:
        generate: Optional callable `(prompt, n) -> list[str]`, injected for
            testing. When absent, a transformers pipeline is built from
            `model_path`, which requires a GPU.
    """
    stats = OnPolicyStats()
    provenance = Provenance(
        source_id="dpo_on_policy",
        license="derived from model outputs",
        # Both sides are model output. Labelling this anything else would make
        # the mixture impossible to audit.
        synthetic=True,
        generator=model_path,
        human_reviewed=False,
    )

    sampler = generate or _build_sampler(model_path, temperature, max_new_tokens)

    for prompt in prompts:
        if limit is not None and stats.emitted >= limit:
            break
        completions = sampler(prompt, samples_per_prompt)
        yield from pairs_from_generations(prompt, completions, provenance, stats=stats)

    logger.info("On-policy builder finished", extra={"summary": stats.summary()})
    if stats.by_reason:
        logger.info("Rejection reasons", extra=dict(stats.by_reason))


def _build_sampler(model_path: str, temperature: float, max_new_tokens: int) -> Any:
    """Construct a generation callable from a checkpoint. Requires a GPU."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()  # type: ignore[no-untyped-call]

    def sample(prompt: str, n: int) -> list[str]:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                max_new_tokens=max_new_tokens,
                num_return_sequences=n,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        prompt_length = inputs["input_ids"].shape[1]
        return [
            str(tokenizer.decode(output[prompt_length:], skip_special_tokens=True))
            for output in outputs
        ]

    return sample
