"""Tests for tokenizer fertility analysis.

A stub tokenizer keeps the suite offline and deterministic. It splits on a
fixed rule rather than a learned vocabulary, which is enough to verify the
arithmetic and the reporting - the real numbers come from `qm tokenizer
analyze` against Qwen3.
"""

from __future__ import annotations

from qaraqalpaqmind.tokenizer.fertility import (
    FertilityReport,
    compare_languages,
    letter_split_rate,
    measure,
    relative_penalty,
)


class StubTokenizer:
    """Splits text into fixed-size chunks, emitting `isolate` letters alone.

    That mimics the behaviour actually observed on Qwen3, where `ǵ` and `ń` are
    emitted as standalone tokens because no subword in the vocabulary contains
    them. `decode` round-trips, so `letter_split_rate` can be tested too.
    """

    def __init__(self, chunk: int = 3, isolate: str = "áǵń") -> None:
        self.chunk = chunk
        self.isolate = set(isolate)
        self._last: list[str] = []

    def _pieces(self, text: str) -> list[str]:
        pieces: list[str] = []
        buffer = ""
        for char in text:
            if char in self.isolate:
                if buffer:
                    pieces.append(buffer)
                    buffer = ""
                pieces.append(char)
                continue
            buffer += char
            if len(buffer) == self.chunk:
                pieces.append(buffer)
                buffer = ""
        if buffer:
            pieces.append(buffer)
        return pieces

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        self._last = self._pieces(text)
        return list(range(len(self._last)))

    def decode(self, ids: list[int]) -> str:
        return self._last[ids[0]]


def test_stub_splits_as_intended() -> None:
    # The test double is only useful if it behaves as documented.
    assert StubTokenizer(chunk=3, isolate="")._pieces("abcdef") == ["abc", "def"]
    assert StubTokenizer(chunk=3, isolate="ǵ")._pieces("aǵb") == ["a", "ǵ", "b"]


EchoTokenizer = StubTokenizer


def test_fertility_arithmetic() -> None:
    tokenizer = StubTokenizer(chunk=3, isolate="")
    report = measure(["abcdef ghi"], tokenizer, "test", worst_n=0)

    assert report.documents == 1
    assert report.characters == 10
    assert report.words == 2
    assert report.tokens == 4  # "abc" "def" " gh" "i"
    assert report.fertility == 2.0
    assert report.chars_per_token == 2.5


def test_empty_and_blank_documents_are_skipped() -> None:
    report = measure(["", "   ", "\n"], StubTokenizer(), "test", worst_n=0)
    assert report.documents == 0
    assert report.fertility == 0.0
    assert report.chars_per_token == 0.0


def test_per_word_statistics_are_opt_in() -> None:
    # A whole-corpus count cannot afford one encode call per word.
    without = measure(["bir eki úsh"], StubTokenizer(), "t", worst_n=0)
    assert without.single_token_words == 0
    assert without.worst_words == []

    with_stats = measure(["bir eki úsh"], StubTokenizer(chunk=10), "t", worst_n=5)
    assert with_stats.single_token_words > 0


def test_worst_words_are_ranked_by_cost() -> None:
    tokenizer = StubTokenizer(chunk=2, isolate="")
    report = measure(["qısqa juwaplandırılmaǵanlıqtan"], tokenizer, "t", worst_n=3)
    costs = [cost for _, cost in report.worst_words]
    assert costs == sorted(costs, reverse=True)
    assert report.worst_words[0][0] == "juwaplandırılmaǵanlıqtan"


def test_single_token_and_fragmented_rates() -> None:
    tokenizer = StubTokenizer(chunk=100, isolate="")  # every word is one token
    report = measure(["bir eki úsh tórt"], tokenizer, "t", worst_n=5)
    assert report.single_token_rate == 1.0
    assert report.fragmented_rate == 0.0


def test_relative_penalty_uses_chars_per_token() -> None:
    # Word length differs between languages, so the comparison must not be
    # based on tokens per word.
    reports = {
        "eng": FertilityReport("eng", characters=400, words=100, tokens=100),
        "kaa": FertilityReport("kaa", characters=400, words=50, tokens=200),
    }
    penalties = relative_penalty(reports, baseline="eng")
    assert penalties["eng"] == 1.0
    assert penalties["kaa"] == 2.0


def test_relative_penalty_without_a_baseline() -> None:
    assert relative_penalty({"kaa": FertilityReport("kaa")}, baseline="eng") == {}


def test_compare_languages_labels_each_report() -> None:
    reports = compare_languages(
        {"kaa": ["bir eki úsh"], "eng": ["one two three"]}, StubTokenizer()
    )
    assert set(reports) == {"kaa", "eng"}
    assert reports["kaa"].label == "kaa"


def test_letter_split_rate_detects_isolated_letters() -> None:
    # The real finding this measures: Qwen3 emits `ǵ` as its own token 100% of
    # the time, so every word containing it pays an extra token.
    rates = letter_split_rate(["joqarǵı keńes"], EchoTokenizer(chunk=3, isolate="ǵń"), "ǵń")
    assert rates["ǵ"] == 1.0
    assert rates["ń"] == 1.0


def test_letter_split_rate_ignores_absent_letters() -> None:
    rates = letter_split_rate(["plain ascii text"], EchoTokenizer(), "áǵń")
    assert rates == {}
