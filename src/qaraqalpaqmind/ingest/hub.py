"""Move built datasets between machines via the Hugging Face Hub.

`data/` is not in git, deliberately: corpora are reproducible outputs and a
repository is not a data store. But that means a fresh clone on a GPU host has
no training data, and rebuilding it there means re-crawling other people's
servers for the web portion.

So the built artefacts move through the Hub instead. Push once from the machine
that built them, pull on every GPU host. The manifest travels with the data, so
the sha256 and token count that describe a training run are never separated
from it.

Repositories default to **private**: this corpus contains crawled institutional
text whose redistribution licence is `unknown` for several sources. Publishing
it is a decision the operator makes explicitly, not a default.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..common.env import has_secret
from ..common.logging import get_logger
from ..common.paths import DATASETS_DIR, MANIFESTS_DIR, ensure_dir

logger = get_logger(__name__)

# Datasets worth moving, relative to `data/datasets/`.
TRANSFERABLE = (
    "pretrain/pretrain_v1.jsonl.zst",
    "sft/sft_v1_train.jsonl.zst",
    "sft/sft_v1_val.jsonl.zst",
    "dpo/dpo_v1_train.jsonl.zst",
    "dpo/dpo_v1_val.jsonl.zst",
    "eval/contamination_blocklist.json",
)


@dataclass(frozen=True, slots=True)
class TransferPlan:
    repo_id: str
    files: tuple[Path, ...]
    manifests: tuple[Path, ...]

    @property
    def total_bytes(self) -> int:
        return sum(p.stat().st_size for p in (*self.files, *self.manifests) if p.exists())


def _require_token() -> None:
    if not has_secret("HF_TOKEN") and not has_secret("HUGGING_FACE_HUB_TOKEN"):
        raise RuntimeError(
            "HF_TOKEN is not set. Create a token with write access at "
            "https://huggingface.co/settings/tokens and put it in .env"
        )


def plan_transfer(repo_id: str) -> TransferPlan:
    """Which local files would be pushed."""
    files = tuple(
        path
        for path in (DATASETS_DIR / name for name in TRANSFERABLE)
        if path.exists()
    )
    manifests = tuple(sorted(MANIFESTS_DIR.glob("*.json")))
    return TransferPlan(repo_id=repo_id, files=files, manifests=manifests)


def push(repo_id: str, *, private: bool = True, dry_run: bool = False) -> TransferPlan:
    """Upload built datasets and their manifests to the Hub."""
    plan = plan_transfer(repo_id)
    if not plan.files:
        raise FileNotFoundError(
            "No built datasets found. Run `qm dedup run` (and optionally `qm sft build`, "
            "`qm dpo build`) first."
        )
    if dry_run:
        return plan

    _require_token()
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)

    for path in plan.files:
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=f"datasets/{path.relative_to(DATASETS_DIR).as_posix()}",
            repo_id=repo_id,
            repo_type="dataset",
        )
        logger.info("Uploaded", extra={"file": path.name, "bytes": path.stat().st_size})

    for path in plan.manifests:
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=f"manifests/{path.name}",
            repo_id=repo_id,
            repo_type="dataset",
        )

    logger.info(
        "Push complete",
        extra={"repo": repo_id, "files": len(plan.files), "private": private},
    )
    return plan


def pull(repo_id: str) -> list[Path]:
    """Download built datasets and manifests into `data/`.

    This is what a GPU host runs after cloning: it puts the corpus in the exact
    place the training config expects, with the manifests that describe it.
    """
    _require_token()
    from huggingface_hub import snapshot_download

    logger.info("Pulling datasets", extra={"repo": repo_id})
    local = Path(
        snapshot_download(repo_id=repo_id, repo_type="dataset", allow_patterns=["*"])
    )

    written: list[Path] = []

    datasets_root = local / "datasets"
    if datasets_root.is_dir():
        for source in datasets_root.rglob("*"):
            if not source.is_file():
                continue
            target = DATASETS_DIR / source.relative_to(datasets_root)
            ensure_dir(target.parent)
            target.write_bytes(source.read_bytes())
            written.append(target)

    manifests_root = local / "manifests"
    if manifests_root.is_dir():
        ensure_dir(MANIFESTS_DIR)
        for source in manifests_root.glob("*.json"):
            target = MANIFESTS_DIR / source.name
            target.write_bytes(source.read_bytes())
            written.append(target)

    logger.info("Pull complete", extra={"repo": repo_id, "files": len(written)})
    return written
