"""One worked Karakalpak example per task type.

These are code, not documentation, and the test suite validates every one. An
example in a markdown file drifts from the schema the first time a field is
renamed; an example that is constructed and validated on every test run cannot.

They are also the fixtures Phase 6 builds its first SFT batches against, and
what `qm schema example <task>` prints.
"""

from __future__ import annotations

from .base import Provenance, TaskRecord, TaskType
from .tasks import (
    BenchmarkRecord,
    CodingRecord,
    ConversationRecord,
    GrammarRecord,
    InstructionRecord,
    MathRecord,
    PreferenceRecord,
    PretrainRecord,
    QARecord,
    ReasoningRecord,
    SummarizationRecord,
    TranslationRecord,
    Turn,
)

_HAND = Provenance(source_id="handwritten_examples", license="CC0-1.0", human_reviewed=True)


PRETRAIN = PretrainRecord(
    text=(
        "Qaraqalpaqstan Respublikası Ózbekstan Respublikasınıń quramındaǵı "
        "avtonomiyalı respublika bolıp tabıladı. Onıń paytaxtı — Nókis qalası."
    ),
    provenance=Provenance(source_id="wiki_kaa", license="CC-BY-SA-4.0"),
)

INSTRUCTION = InstructionRecord(
    instruction="Qaraqalpaqstannıń paytaxtı qaysı qala?",
    output="Qaraqalpaqstan Respublikasınıń paytaxtı — Nókis qalası.",
    provenance=_HAND,
)

INSTRUCTION_WITH_INPUT = InstructionRecord(
    instruction="Tómendegi gápti kóplik túrge aylandır.",
    input="Bala mektepke baradı.",
    output="Balalar mektepke baradı.",
    provenance=_HAND,
)

CONVERSATION = ConversationRecord(
    messages=[
        Turn(role="user", content="Salawmatsız ba! Maǵan járdem bere alasız ba?"),
        Turn(role="assistant", content="Salawmatsız! Álbette, qanday járdem kerek?"),
        Turn(role="user", content="Aral teńizi haqqında qısqasha aytıp beriń."),
        Turn(
            role="assistant",
            content=(
                "Aral teńizi — Oraylıq Aziyada jaylasqan, bir waqıtları dúnyadaǵı "
                "eń úlken kóllerdiń biri bolǵan. XX ásirdiń ekinshi yarımınan "
                "baslap onıń kólemi keskin azaydı."
            ),
        ),
    ],
    provenance=_HAND,
)

TRANSLATION = TranslationRecord(
    source_lang="kaa",
    target_lang="eng",
    source_text="Qaraqalpaqstan Respublikasınıń paytaxtı — Nókis qalası.",
    target_text="The capital of the Republic of Karakalpakstan is the city of Nukus.",
    provenance=Provenance(
        source_id="hf_dilmash_parallel",
        source_url="https://huggingface.co/datasets/tahrirchi/dilmash",
        license="MIT",
    ),
)

GRAMMAR = GrammarRecord(
    # A real and extremely common error: the 2009 apostrophe orthography used
    # where the current 2016 standard wants acute accents.
    incorrect="Qaraqalpaqstan Respublikasi'ning paytaxti' — No'kis qalasi.",
    correct="Qaraqalpaqstan Respublikasınıń paytaxtı — Nókis qalası.",
    explanation=(
        "Házirgi jazıw qaǵıydaları boyınsha apostrof emes, ustin belgili "
        "háripler jazıladı: o' → ó, i' → ı, n' → ń."
    ),
    error_type="orthography",
    provenance=_HAND,
)

QA = QARecord(
    context=(
        "Ájiniyaz Qosıbay ulı — XIX ásirdegi qaraqalpaq klassik ádebiyatınıń "
        "eń kórnekli wákilleriniń biri. Ol 1824-jılı tuwılǵan."
    ),
    question="Ájiniyaz qay jılı tuwılǵan?",
    answer="Ájiniyaz 1824-jılı tuwılǵan.",
    provenance=_HAND,
)

SUMMARIZATION = SummarizationRecord(
    document=(
        "Qaraqalpaqstan Respublikası Joqarǵı Keńesiniń gezekli sessiyası bolıp "
        "ótti. Sessiyada mámleketlik hám jámiyetlik ómirdiń áhmiyetli máseleleri "
        "boyınsha qararlar qabıl etildi. Deputatlar tárepinen bir neshe nızam "
        "joybarları dodalanıp, olar boyınsha tiyisli sheshimler qabıllandı. "
        "Sonday-aq, aymaqtı rawajlandırıw baǵdarlaması tastıyıqlandı."
    ),
    summary=(
        "Joqarǵı Keńes sessiyasında nızam joybarları dodalanıp, aymaqtı "
        "rawajlandırıw baǵdarlaması tastıyıqlandı."
    ),
    provenance=Provenance(source_id="gov_jokargikenes", license="unknown"),
)

REASONING = ReasoningRecord(
    question="Bir dúkanda 5 qutı bar, hár qutıda 12 alma bar. 8 alma satıldı. Neshe alma qaldı?",
    reasoning=(
        "Dáslep ulıwma alma sanın tabamız: 5 × 12 = 60 alma.\n"
        "Sonnan keyin satılǵan almalardı alıp taslaymız: 60 − 8 = 52."
    ),
    answer="52 alma",
    provenance=_HAND,
)

CODING = CodingRecord(
    prompt="Python tilinde berilgen sannıń juplıǵın tekseretuǵın funkciya jaz.",
    code=(
        "def jup_pa(san: int) -> bool:\n"
        '    """San jup bolsa True qaytaradı."""\n'
        "    return san % 2 == 0"
    ),
    language="python",
    explanation="Funkciya sannı 2 ge bólgendegi qaldıqtı tekseredi.",
    tests="assert jup_pa(4) is True\nassert jup_pa(7) is False",
    provenance=_HAND,
)

MATH = MathRecord(
    problem="Úshmúyeshliktiń tabanı 10 sm, biyikligi 6 sm. Onıń maydanın tabıń.",
    solution="Úshmúyeshlik maydanı: S = (taban × biyiklik) / 2 = (10 × 6) / 2 = 30.",
    answer="30 sm²",
    level="mektep",
    provenance=_HAND,
)

BENCHMARK_MULTIPLE_CHOICE = BenchmarkRecord(
    subject="geography",
    question="Qaraqalpaqstan Respublikasınıń paytaxtı qaysı qala?",
    choices=["Nókis", "Xiywa", "Buxara", "Samarqand"],
    answer="Nókis",
    provenance=_HAND,
)

BENCHMARK_FREE_FORM = BenchmarkRecord(
    subject="history",
    question="Ájiniyaz qaysı ásirde jasaǵan?",
    answer="XIX ásirde",
    provenance=_HAND,
)


PREFERENCE = PreferenceRecord(
    # The cleanest kind of preference pair: both sides say exactly the same
    # thing, so language is the only dimension they differ in. Anything else
    # they differed in would also be taught.
    prompt=(
        "Tómendegi mazmundı qaraqalpaq tilinde jazıp ber:\n\n"
        "The capital of the Republic of Karakalpakstan is the city of Nukus."
    ),
    chosen="Qaraqalpaqstan Respublikasınıń paytaxtı — Nókis qalası.",
    rejected="The capital of the Republic of Karakalpakstan is the city of Nukus.",
    criterion="language_consistency",
    provenance=Provenance(
        source_id="hf_dilmash_parallel",
        source_url="https://huggingface.co/datasets/tahrirchi/dilmash",
        license="MIT",
    ),
)


EXAMPLES: dict[TaskType, TaskRecord] = {
    TaskType.PRETRAIN: PRETRAIN,
    TaskType.INSTRUCTION: INSTRUCTION,
    TaskType.CONVERSATION: CONVERSATION,
    TaskType.TRANSLATION: TRANSLATION,
    TaskType.GRAMMAR: GRAMMAR,
    TaskType.QA: QA,
    TaskType.SUMMARIZATION: SUMMARIZATION,
    TaskType.REASONING: REASONING,
    TaskType.CODING: CODING,
    TaskType.MATH: MATH,
    TaskType.BENCHMARK: BENCHMARK_MULTIPLE_CHOICE,
    TaskType.PREFERENCE: PREFERENCE,
}

ALL_EXAMPLES: list[TaskRecord] = [
    PRETRAIN,
    INSTRUCTION,
    INSTRUCTION_WITH_INPUT,
    CONVERSATION,
    TRANSLATION,
    GRAMMAR,
    QA,
    SUMMARIZATION,
    REASONING,
    CODING,
    MATH,
    BENCHMARK_MULTIPLE_CHOICE,
    BENCHMARK_FREE_FORM,
    PREFERENCE,
]
