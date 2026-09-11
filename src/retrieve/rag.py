import random
import time
from typing import Optional, Protocol

from src.retrieve.guard import GuardRefusal
from src.retrieve.prompt import build_prompt
from src.retrieve.retriever import Retriever, SearchHit

_RETRY_MESSAGE = (
    "Mistral is rate-limiting requests right now (free-tier quota). "
    "Please wait about 60 seconds and try again."
)


class AnswerGenerator(Protocol):
    def generate(self, prompt: str, model: str = "mistral-small-latest") -> str:
        ...


class MistralAnswerGenerator:
    """Answer generation via the Mistral AI API.

    The heavy ``mistralai`` import stays lazy so that importing the retrieval
    package never requires the API client at import time (tests, offline use).
    """

    def __init__(self, api_key: str, timeout: float = 30.0) -> None:
        self.api_key = api_key
        self.timeout = timeout
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        try:
            from mistralai import Mistral
        except ImportError as exc:  # pragma: no cover - only when dep missing
            raise RuntimeError(
                "mistralai is not installed. Run `pip install mistralai`."
            ) from exc
        self._client = Mistral(api_key=self.api_key)
        return self._client

    def generate(self, prompt: str, model: str = "mistral-small-latest") -> str:
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY is required for answer generation.")
        client = self._ensure_client()
        last_error: Optional[BaseException] = None
        for attempt in range(1, _MAX_429_ATTEMPTS + 1):
            try:
                response = client.chat.complete(
                    model=model,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    max_tokens=200,
                )
                return (response.choices[0].message.content or "").strip()
            except Exception as exc:  # noqa: BLE001 - we inspect status below
                last_error = exc
                status = getattr(exc, "status_code", None) or _status_from(exc)
                if status != 429:
                    raise
                if attempt == _MAX_429_ATTEMPTS:
                    break
                time.sleep(_retry_delay(attempt, exc))
        raise RuntimeError(
            f"{_RETRY_MESSAGE} (last API error: {_describe(last_error)})"
        ) from last_error


class UnavailableAnswerGenerator:
    """Offline fallback used when no Mistral API key is configured.

    Returns a deterministic, factual answer built straight from the top search
    hit without any LLM call, so the pipeline is testable end-to-end offline.
    """

    def generate(self, prompt: str, model: str = "mistral-small-latest") -> str:
        return "I can't generate an answer right now because no Mistral API key is configured."


class RAGAssistant:
    """Phase-5/6 orchestration: guard -> retrieve -> prompt -> answer."""

    def __init__(
        self,
        retriever: Retriever,
        generator: Optional[AnswerGenerator] = None,
        source_words: str = "Last updated from sources: ",
    ) -> None:
        self.retriever = retriever
        self.generator = generator or UnavailableAnswerGenerator()
        self.source_words = source_words

    def answer(self, question: str, where: Optional[dict] = None) -> "RAGAnswer":
        from src.retrieve.guard import QueryGuard

        refusal = QueryGuard().check(question)
        if refusal is not None:
            return RAGAnswer(
                question=question,
                is_refusal=True,
                verdict=refusal.verdict,
                text=refusal.message,
                url=refusal.url,
            )

        hits = self.retriever.retrieve(question, where=where)
        if not hits:
            return RAGAnswer(
                question=question,
                is_refusal=False,
                text="I don't have that information in my sources.",
                url=None,
            )
        prompt_text = build_prompt(question, hits)
        raw = self.generator.generate(prompt_text)
        return RAGAnswer(
            question=question,
            is_refusal=False,
            text=self._with_citation(raw, hits[0]),
            url=hits[0].url,
        )

    def _with_citation(self, answer: str, hit: SearchHit) -> str:
        if self.source_words in answer:
            return answer
        return f"{answer} {self.source_words}{hit.url}"


import re

from pydantic import BaseModel

_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about HDFC mutual funds. "
    "Answer ONLY using the provided context, in plain English, in 2-4 sentences. "
    "Always end with a single citation line: 'Last updated from sources: <url>'. "
    "If the context does not contain the answer, say you don't know. Never give "
    "investment advice."
)


class RAGAnswer(BaseModel):
    question: str
    is_refusal: bool
    text: str
    url: Optional[str] = None
    verdict: str = "factual"

    def is_factual(self) -> bool:
        return not self.is_refusal
