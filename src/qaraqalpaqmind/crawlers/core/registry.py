"""The source registry: a typed, machine-readable description of every place
we intend to get Karakalpak text from.

`configs/crawl/sources.yaml` is the single list of truth. Nothing crawls a
domain that is not declared there, which means legality, licence and rate
limits are reviewed in a diff before any traffic is generated.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import Field, HttpUrl, field_validator

from ...common.config import StrictModel, load_raw


class SourceKind(StrEnum):
    """What the source is, which determines the extractor used later."""

    WIKI = "wiki"
    NEWS = "news"
    GOVERNMENT = "government"
    EDUCATION = "education"
    LITERATURE = "literature"
    RELIGIOUS = "religious"
    DICTIONARY = "dictionary"
    BLOG = "blog"
    FORUM = "forum"
    SOCIAL = "social"
    DATASET = "dataset"
    PARALLEL = "parallel"


class AccessMethod(StrEnum):
    """How we obtain the bytes. Dumps and APIs are always preferred to crawling."""

    DUMP = "dump"  # official bulk export - zero crawl load, best option
    API = "api"  # documented public API
    HF = "hf"  # Hugging Face Hub dataset
    CRAWL = "crawl"  # polite HTTP crawl of HTML pages
    MANUAL = "manual"  # human download (paywalled, captcha, physical books)


class LegalStatus(StrEnum):
    """Our assessment of whether we may redistribute the derived corpus."""

    OPEN_LICENSE = "open_license"  # explicit CC/MIT/Apache/public-domain grant
    GOVERNMENT_WORK = "government_work"  # official texts, generally not copyrightable
    FAIR_USE_TRAIN = "fair_use_train"  # crawlable & robots-clean, redistribution unclear
    PERMISSION_NEEDED = "permission_needed"  # must email the owner first
    RESTRICTED = "restricted"  # do not touch


class UpdateFrequency(StrEnum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    RARELY = "rarely"
    STATIC = "static"


class SourceSpec(StrictModel):
    """One declared data source."""

    id: str = Field(pattern=r"^[a-z0-9]+(_[a-z0-9]+)*$", max_length=48)
    name: str
    kind: SourceKind
    access: AccessMethod
    url: HttpUrl
    legal: LegalStatus
    license: str = Field(description="SPDX id, 'unknown', or a short prose note.")
    update_frequency: UpdateFrequency

    # Rough expectations, refined once the source has actually been fetched.
    est_size_mb: float = Field(ge=0, description="Estimated raw Karakalpak text, in MB.")
    quality: int = Field(ge=1, le=5, description="1 = machine noise, 5 = edited prose.")
    scripts: list[str] = Field(default_factory=lambda: ["latin"])

    # Crawl behaviour. Ignored for dump/api/hf sources.
    allowed_paths: list[str] = Field(default_factory=list)
    denied_paths: list[str] = Field(default_factory=list)
    delay_seconds: float = Field(default=2.0, ge=0.0, le=60.0)
    max_pages: int | None = Field(default=None, ge=1)
    respect_robots: bool = True
    use_sitemap: bool = Field(
        default=True,
        description=(
            "Seed from robots.txt sitemaps. Set false where the sitemap indexes a "
            "different language than the one we want, or is reliably broken."
        ),
    )

    enabled: bool = True
    priority: int = Field(default=3, ge=1, le=5, description="1 = crawl first.")
    verified_on: date | None = None
    notes: str = ""

    held_out: bool = Field(
        default=False,
        description=(
            "Evaluation data. Ingested, but structurally excluded from cleaning and "
            "deduplication so it can never reach a training set. A note saying HELD "
            "OUT is not enough - `qm clean all` would sweep it into processed/ and "
            "deduplication reads everything there."
        ),
    )

    @field_validator("scripts")
    @classmethod
    def _known_scripts(cls, value: list[str]) -> list[str]:
        allowed = {"latin", "cyrillic", "mixed"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"Unknown scripts {sorted(unknown)}; allowed: {sorted(allowed)}")
        return value

    @property
    def robots_url(self) -> str:
        parsed = str(self.url).rstrip("/")
        scheme, _, rest = parsed.partition("://")
        host = rest.split("/", 1)[0]
        return f"{scheme}://{host}/robots.txt"


class SourceRegistry(StrictModel):
    """The whole `configs/crawl/sources.yaml` file."""

    version: int
    contact_email: str = Field(description="Published in the crawler User-Agent.")
    user_agent: str
    sources: list[SourceSpec]

    @field_validator("sources")
    @classmethod
    def _unique_ids(cls, value: list[SourceSpec]) -> list[SourceSpec]:
        seen: set[str] = set()
        for spec in value:
            if spec.id in seen:
                raise ValueError(f"Duplicate source id: {spec.id}")
            seen.add(spec.id)
        return value

    def enabled_sources(self) -> list[SourceSpec]:
        """Enabled sources, highest priority first."""
        return sorted((s for s in self.sources if s.enabled), key=lambda s: (s.priority, s.id))

    def by_id(self, source_id: str) -> SourceSpec:
        for spec in self.sources:
            if spec.id == source_id:
                return spec
        raise KeyError(f"Unknown source id: {source_id}")

    def total_estimated_mb(self) -> float:
        return round(sum(s.est_size_mb for s in self.sources if s.enabled), 1)

    def held_out_ids(self) -> frozenset[str]:
        """Source ids that must never reach a training set."""
        return frozenset(spec.id for spec in self.sources if spec.held_out)


def load_registry(path: str = "crawl/sources.yaml") -> SourceRegistry:
    """Load and validate the source registry."""
    return SourceRegistry.model_validate(load_raw(path))
