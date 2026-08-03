"""Transformers backend: runs a model in this process.

For development. It loads a small model - `Qwen/Qwen3-0.6B` by default, which
is a few hundred megabytes and runs on CPU - so the whole API can be exercised
end to end on a laptop with the same code paths production uses.

Not for production. Generation is single-threaded and blocks a worker for the
duration; one slow request stalls everything behind it. vLLM exists for that.

Generation runs in a thread so the event loop keeps serving health checks and
other requests while a completion is in flight.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from queue import Empty, Queue
from typing import Any

from ...common.logging import get_logger
from ..schemas.chat import ChatCompletionRequest
from .base import BackendError, ModelBackend

logger = get_logger(__name__)

# Small enough to download in seconds and run on CPU, while still using the
# Qwen3 chat template and tokenizer the real model uses.
DEFAULT_DEV_MODEL = "Qwen/Qwen3-0.6B"

_QUEUE_TIMEOUT = 60.0


class TransformersBackend(ModelBackend):
    """Loads a model with transformers and streams from it."""

    def __init__(
        self,
        model_path: str = DEFAULT_DEV_MODEL,
        *,
        device: str | None = None,
        dtype: str = "auto",
        max_new_tokens: int = 512,
    ) -> None:
        self.name = model_path
        self._model_path = model_path
        self._device = device
        self._dtype = dtype
        self._max_new_tokens = max_new_tokens
        self._model: Any = None
        self._tokenizer: Any = None

    def _load(self) -> None:
        """Load lazily, so importing the app does not pull in torch."""
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise BackendError(
                "the transformers backend needs `pip install -e '.[train]'`",
                status_code=503,
            ) from exc

        logger.info("Loading model", extra={"path": self._model_path})
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        dtype = getattr(torch, self._dtype) if self._dtype != "auto" else "auto"
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_path,
            dtype=dtype,
            device_map=self._device or ("cuda" if torch.cuda.is_available() else "cpu"),
        )
        self._model.eval()
        logger.info("Model ready", extra={"path": self._model_path})

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[str]:
        await asyncio.to_thread(self._load)

        from transformers import TextIteratorStreamer

        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": request.max_tokens or self._max_new_tokens,
            "do_sample": request.temperature > 0,
            "temperature": max(request.temperature, 1e-5),
            "top_p": request.top_p,
            "pad_token_id": self._tokenizer.pad_token_id,
        }

        # Generation blocks; run it off the event loop so the server keeps
        # answering health checks while a completion is in flight.
        errors: Queue[BaseException] = Queue()

        def generate() -> None:
            try:
                import torch

                with torch.no_grad():
                    self._model.generate(**kwargs)
            except BaseException as exc:  # surfaced below, never swallowed
                errors.put(exc)

        task = asyncio.create_task(asyncio.to_thread(generate))
        try:
            for piece in streamer:
                if piece:
                    yield piece
                # Yield control so a client disconnect is noticed promptly.
                await asyncio.sleep(0)
        finally:
            await task

        try:
            raise BackendError(f"generation failed: {errors.get_nowait()}") from None
        except Empty:
            pass

    async def health(self) -> bool:
        return True

    async def aclose(self) -> None:
        self._model = None
        self._tokenizer = None
