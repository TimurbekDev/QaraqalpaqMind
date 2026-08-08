"""Guards on the grounded QA builder.

Every test here is a failure mode that appeared in real output during
development. The builder's contract is that no Karakalpak sentence is invented,
so what has to be tested is the *refusals*: which candidates it declines to turn
into a question, and why.
"""

from __future__ import annotations

import pytest

from qaraqalpaqmind.training.sft.builders.grounded_qa import (
    TEMPLATES,
    agrees_with_title,
    is_prose,
    is_usable_subject,
    match_sentence,
)


def test_every_template_captures_a_subject() -> None:
    # `render` substitutes {subject}; a template without that group would raise
    # at build time, on whichever corpus sentence happened to match it first.
    for template in TEMPLATES:
        assert "subject" in template.detector.groupindex
        assert "{subject}" in template.question


def test_the_defining_sentence_becomes_a_question() -> None:
    sentence = "Aral teńizi — Oraylıq Aziyada jaylasqan, bir waqıtları eń iri kóllerdiń biri."
    matched = match_sentence(sentence, title="Aral teńizi")
    assert matched is not None
    template, subject = matched
    assert subject == "Aral teńizi"
    assert template.render(subject) == "Aral teńizi haqqında ne aytılǵan?"


@pytest.mark.parametrize(
    "subject",
    [
        "Ol",  # the entity is in an earlier sentence
        "Onıń paytaxtı",  # possessive fragment, no subject in the question
        "Házirgi",  # a truncated adjective: "Házirgi qay jerde jaylasqan?"
        "Temir jollar\nDáslepki Xarkov temir jol uzeli",  # heading glued on
        "Qozǵalıs reflektor jerde tek bir jóneliste",  # a clause, not a phrase
        "agеntligi",  # Cyrillic е homoglyph
        "aral teńizi",  # mid-clause, not a name
        "1960-jılları",  # a date
    ],
)
def test_unusable_subjects_are_refused(subject: str) -> None:
    assert not is_usable_subject(subject)


def test_a_subject_the_article_is_not_about_is_refused() -> None:
    # "Rásmiy paytaxtı — JayavardenapuraKotte" is a sentence in the Sri Lanka
    # article. The question "what is said about its official capital" has no
    # subject, so title agreement is what rejects it.
    sentence = "Rásmiy paytaxtı — JayavardenapuraKotte (parlament jaylasqan)."
    assert match_sentence(sentence, title="Shri-Lanka") is None
    assert not agrees_with_title("Rásmiy paytaxtı", "Shri-Lanka")


def test_a_title_is_required() -> None:
    # Sources without titles cannot be checked, so they yield no questions
    # rather than unverified ones.
    sentence = "Stokgolm — mámlekettiń mádeniy hám ilimiy orayı."
    assert match_sentence(sentence, title="") is None
    assert match_sentence(sentence, title="Stokgolm") is not None


def test_title_agreement_allows_a_longer_subject() -> None:
    # The defining sentence often carries a parenthetical the title omits.
    assert agrees_with_title("Ájiniyaz Qosıbay ulı", "Ájiniyaz")
    assert agrees_with_title("Ájiniyaz", "Ájiniyaz Qosıbay ulı")
    assert not agrees_with_title("Nókis", "Aral teńizi")


def test_wikitext_is_not_prose() -> None:
    # Infoboxes survive cleaning and pass every length check.
    assert not is_prose("{{Xalıq punkti infoqutısı | qaraqalpaqsha atı = Tolman | túri = awıl }}")
    assert not is_prose("== Tariyxı ==")
    assert not is_prose("Bul bir sarlawha")  # no full stop
    assert is_prose("Nókis — Qaraqalpaqstan Respublikasınıń paytaxtı. Qala Ámiwdárya boyında.")
