"""Question answering grounded in real corpus paragraphs.

Phase 6 trained on 45 hand-authored instruction examples out of 27,042 records.
The model that came out answers in Azerbaijani, because 90% of what it saw was
translation pairs and orthography drills and it has no template for "answer this
question". This builder is the fix, and its whole design follows from one
constraint: **no new Karakalpak sentence may be invented.**

So the answer is always a sentence copied verbatim out of the corpus, and the
question is assembled from a *fixed* Karakalpak string plus a noun phrase lifted
from that same sentence in the nominative. Nothing is inflected, nothing is
paraphrased, nothing is machine-translated. What could be wrong is therefore
bounded by the four question templates in `TEMPLATES` below - four strings a
Karakalpak speaker can read in a minute - rather than by ten thousand generated
rows nobody can audit.

The cost of that constraint is recall. These patterns fire on a small fraction of
sentences and deliberately refuse anything ambiguous: a sentence whose subject is
a pronoun is skipped, because `Ol 1824-jılı tuwılǵan` would produce the question
"Ol qay jılı tuwılǵan?" - grammatical, and unanswerable without the paragraph
before it.

The question is generated, so every record is `synthetic=True` with a named
generator. Review a sample before training on it:

    qm sft build
    qm sft inspect --task qa
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field

from ....common.io import read_jsonl
from ....common.logging import get_logger
from ....dedup.pipeline import output_path
from ....schemas import InstructionRecord, Provenance, QARecord

logger = get_logger(__name__)

#: Sources whose prose is expository enough for the patterns to fire. Court
#: rulings and parallel translation data are excluded: the first is boilerplate,
#: the second is sentence pairs with no paragraph context to ground an answer in.
_SOURCES = ("wiki_kaa", "news_kknews", "edu_ndpi")

#: Overviews come from encyclopedia articles only. Scraped agency sites put
#: section headings in the title field - "Basshılıq", "Baylanıs", "Xızmetler" -
#: and pairing those with a contact-details blob teaches the model to answer
#: "tell me about leadership" with a phone number.
_TITLED_SOURCES = ("wiki_kaa",)

#: The corpus is Latin 2016. Any Cyrillic character in a candidate means mixed
#: homoglyphs survived cleaning ("аgеntligi" with Cyrillic а and е), and a model
#: trained on it learns to emit letters that are invisible until they break a
#: tokenizer. Cheap to detect, so detected here rather than trusted upstream.
_CYRILLIC = re.compile(r"[Ѐ-ӿ]")

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# A context short enough to read and long enough to be a context.
_MIN_CONTEXT_CHARS = 200
_MAX_CONTEXT_CHARS = 1_500
_MIN_ANSWER_CHARS = 25
_MAX_ANSWER_CHARS = 400

#: Subjects that carry no information without the preceding sentence. A question
#: built on one of these is unanswerable even with the paragraph in front of you.
_PRONOUNS: frozenset[str] = frozenset(
    {
        "ol", "bul", "sol", "onıń", "bunıń", "sonıń", "olar", "bular", "solar",
        "olardıń", "bulardıń", "solardıń", "onda", "bunda", "sonda", "onı",
        "bunı", "sonı", "men", "sen", "siz", "biz", "ózi", "sonday", "bunday",
        "hámmesi",
        # Not pronouns, but adjectives and discourse markers that a lazy subject
        # capture leaves stranded at the front: "Házirgi qay jerde jaylasqan?"
        "házirgi", "dáslepki", "birinshi", "ekinshi", "keyin", "sonlıqtan",
        "usı", "mısalı", "rásmiy", "tiykarǵı", "ulıwma", "sonday-aq",
    }
)

#: A subject longer than this is a clause, not a noun phrase, and the question
#: built from it reads as a fragment: "Qozǵalıs reflektor jerde tek bir jóneliste
#: haqqında ne aytılǵan?" came from a six-word one.
_MAX_SUBJECT_WORDS = 4

#: Markup that means the cleaning pipeline left wikitext behind. Wikipedia
#: infoboxes survive as `{{Xalıq punkti infoqutısı | atı = ... }}`, which reads
#: as a paragraph by every length test and as noise to a reader.
_MARKUP = ("{{", "}}", "[[", "]]", "==", "|")


@dataclass(frozen=True, slots=True)
class Template:
    """One question pattern.

    `detector` must capture a `subject` group and match against a single
    sentence. `question` is a fixed Karakalpak string with `{subject}`
    substituted in the nominative - never inflected, which is what makes the
    result grammatical without a morphological analyser.
    """

    name: str
    detector: re.Pattern[str]
    question: str

    def render(self, subject: str) -> str:
        return self.question.format(subject=subject)


#: The entire audit surface of this builder. Each question form is either copied
#: from `seeds/qa.jsonl` (hand-authored) or built the same way: a nominative noun
#: phrase followed by a fixed interrogative.
TEMPLATES: tuple[Template, ...] = (
    # The em-dash definition sentence, which Karakalpak reference prose uses
    # constantly: "Aral teńizi — Oraylıq Aziyada jaylasqan kól." The dash *is*
    # the subject/predicate boundary, so the subject needs no guessing - which is
    # why this template survived review and a "where is X located" one did not.
    Template(
        name="definition",
        detector=re.compile(r"^(?P<subject>[^—]{3,60}) — .{20,}"),
        question="{subject} haqqında ne aytılǵan?",
    ),
)

# Two more templates were written and removed rather than shipped: a birth-year
# one ("X qay jılı tuwılǵan?") and a founding-year one. Both matched a year
# somewhere after the subject, and in real corpus sentences that year usually
# belongs to somebody else - "Paramount Pictures óziniń tariyxın 1912-jılı
# qurılǵan Feymos Pleyers dáwirinen baslanadı" is not a sentence about when
# Paramount was founded. Between them they produced four usable rows out of
# 37 MB of corpus, so the recall was not worth the failure mode.

#: seeds/instruction.jsonl: "Aral teńizi haqqında qısqasha maǵlıwmat ber."
_OVERVIEW_INSTRUCTION = "{title} haqqında qısqasha maǵlıwmat ber."


@dataclass(slots=True)
class GroundedStats:
    documents: int = 0
    paragraphs: int = 0
    qa_emitted: int = 0
    overview_emitted: int = 0
    skipped_pronoun: int = 0
    skipped_length: int = 0
    skipped_duplicate: int = 0
    skipped_cyrillic: int = 0
    by_template: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"grounded qa: {self.qa_emitted:,} questions and {self.overview_emitted:,} "
            f"overviews from {self.paragraphs:,} paragraphs in {self.documents:,} documents "
            f"(dropped {self.skipped_pronoun:,} pronoun subjects, "
            f"{self.skipped_length:,} on length, {self.skipped_duplicate:,} duplicates, "
            f"{self.skipped_cyrillic:,} with Cyrillic homoglyphs)"
        )


def is_usable_subject(subject: str) -> bool:
    """Reject subjects that make a question unanswerable or ungrammatical."""
    subject = subject.strip()
    words = subject.split()
    if len(subject) < 3 or not words:
        return False
    # A newline inside the subject means a section heading was glued to the
    # sentence below it: "Temir jollar\nDáslepki Xarkov temir jol uzeli".
    if "\n" in subject or _CYRILLIC.search(subject):
        return False
    # Must open like a name or a noun, not mid-clause or mid-list.
    if not subject[0].isupper():
        return False
    # A leading pronoun or ordinal means the real subject is in an earlier
    # sentence: "Onıń paytaxtı — Nókis" answers a question about *what*.
    if words[0].lower() in _PRONOUNS or subject.lower() in _PRONOUNS:
        return False
    if len(words) > _MAX_SUBJECT_WORDS:
        return False
    return not any(char.isdigit() for char in subject)


def agrees_with_title(subject: str, title: str) -> bool:
    """Is this subject the thing the document is about?

    The strongest available signal that a captured subject is a real entity and
    not a sentence fragment. In an encyclopedia article the defining sentence is
    about the title, so `Rásmiy paytaxtı — JayavardenapuraKotte` fails: the
    article is about Sri Lanka, and "what is said about its official capital" is
    a question with no subject in it.
    """
    if not title:
        return False
    one, other = subject.casefold(), title.casefold()
    return one.startswith(other) or other.startswith(one)


def match_sentence(sentence: str, title: str) -> tuple[Template, str] | None:
    """First template that fires on this sentence, with its subject."""
    for template in TEMPLATES:
        found = template.detector.match(sentence)
        if found is None:
            continue
        subject = found.group("subject").strip()
        if is_usable_subject(subject) and agrees_with_title(subject, title):
            return template, subject
    return None


def is_prose(block: str) -> bool:
    """Reject leftover markup and pipe-delimited infobox rows."""
    if any(marker in block for marker in _MARKUP) or _CYRILLIC.search(block):
        return False
    # Prose ends sentences. A block with no full stop is a heading or a caption.
    return "." in block


def _paragraphs(text: str) -> Iterator[str]:
    for block in _PARAGRAPH_SPLIT.split(text):
        block = block.strip()
        if _MIN_CONTEXT_CHARS <= len(block) <= _MAX_CONTEXT_CHARS and is_prose(block):
            yield block


def build(
    *,
    limit: int | None = None,
    dataset: str = "pretrain_v1",
    sources: tuple[str, ...] = _SOURCES,
    include_overviews: bool = True,
) -> Iterator[QARecord | InstructionRecord]:
    """Yield grounded QA pairs, and overview instructions for titled documents."""
    stats = GroundedStats()
    seen: set[tuple[str, str]] = set()

    provenance = Provenance(
        source_id="grounded_qa_from_corpus",
        license="derived from pretrain_v1; see per-source manifests",
        # The answer is corpus text; the question is generated. That makes the
        # record synthetic, and the generator has to be named for it to be
        # auditable later.
        synthetic=True,
        generator="grounded_qa_templates_v1",
        human_reviewed=False,
    )

    for row in read_jsonl(output_path(dataset)):
        if limit is not None and stats.qa_emitted + stats.overview_emitted >= limit:
            break
        if row.get("source_id") not in sources:
            continue

        stats.documents += 1
        text = str(row.get("text", ""))
        meta = row.get("meta") or {}
        title = str(meta.get("title") or "").strip()

        for paragraph in _paragraphs(text):
            stats.paragraphs += 1

            # One question per paragraph. Two would share a context and teach
            # the model that this exact passage has a fixed set of answers.
            for sentence in _SENTENCE_SPLIT.split(paragraph):
                sentence = sentence.strip()
                if not (_MIN_ANSWER_CHARS <= len(sentence) <= _MAX_ANSWER_CHARS):
                    stats.skipped_length += 1
                    continue
                # A newline inside one sentence is a hard-wrapped PDF line, and
                # those arrive hyphenated across the break ("shól-\nkemlestiriw").
                # Answering with that text teaches the model to break words.
                if "\n" in sentence:
                    stats.skipped_length += 1
                    continue

                if _CYRILLIC.search(sentence):
                    stats.skipped_cyrillic += 1
                    continue

                matched = match_sentence(sentence, title)
                if matched is None:
                    continue

                template, subject = matched
                question = template.render(subject)
                key = (question, sentence)
                if key in seen:
                    stats.skipped_duplicate += 1
                    continue
                seen.add(key)

                yield QARecord(
                    question=question,
                    answer=sentence,
                    context=paragraph,
                    provenance=provenance,
                    meta={"template": template.name, "source_id": row.get("source_id")},
                )
                stats.qa_emitted += 1
                stats.by_template[template.name] = stats.by_template.get(template.name, 0) + 1
                break

        # The overview instruction is the one form guaranteed grammatical for any
        # entity, person or place, because the title is used exactly as written.
        titled = include_overviews and row.get("source_id") in _TITLED_SOURCES
        if titled and title and is_usable_subject(title):
            lead = next(_paragraphs(text), None)
            if lead is not None and not _CYRILLIC.search(lead):
                yield InstructionRecord(
                    instruction=_OVERVIEW_INSTRUCTION.format(title=title),
                    output=lead,
                    provenance=provenance,
                    meta={"template": "overview", "source_id": row.get("source_id")},
                )
                stats.overview_emitted += 1

    logger.info(
        "Grounded QA builder finished",
        extra={"summary": stats.summary(), "by_template": stats.by_template},
    )
