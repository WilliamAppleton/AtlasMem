"""Minimal LLM clients (stdlib only).

Anything with ``chat(messages) -> str`` works as an LLM in AtlasMem, so tests can
pass a fake and a trained curator can be served by vLLM behind OpenAICompatLLM.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Protocol

Message = dict[str, str]


_TYPOGRAPHY = str.maketrans({
    "\u00a0": " ", "\u2007": " ", "\u2009": " ", "\u200a": " ", "\u202f": " ", "\u2002": " ",
    "\u2003": " ", "\u200b": "", "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u2011": "-", "\u2212": "-", "\u2026": "...",
})


def normalize_text(text: str) -> str:
    """Map typographic spaces/quotes/dashes to ASCII so names like 'cabinet 2' stay matchable."""
    return text.translate(_TYPOGRAPHY)


class LLM(Protocol):
    def chat(self, messages: list[Message], **options: Any) -> str: ...


class LLMError(RuntimeError):
    pass


def _post_json(url: str, body: dict, headers: dict[str, str], timeout: float, retries: int) -> dict:
    data = json.dumps(body).encode()
    last: Exception | None = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:500]
            last = LLMError(f"HTTP {e.code} from {url}: {detail}")
            if e.code < 500 and e.code != 429:
                raise last from e
        except (urllib.error.URLError, TimeoutError) as e:
            last = LLMError(f"request to {url} failed: {e}")
        if attempt < retries:
            time.sleep(2**attempt)
    assert last is not None
    raise last


class OllamaLLM:
    """Ollama /api/chat client. Works with local Ollama and Ollama Cloud.

    host defaults to $OLLAMA_HOST or https://ollama.com; an $OLLAMA_API_KEY, if set,
    is sent as a bearer token.
    """

    def __init__(
        self,
        model: str,
        host: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
        think: bool | str | None = None,
        timeout: float = 300.0,
        retries: int = 2,
    ):
        self.model = model
        self.host = (host or os.environ.get("OLLAMA_HOST") or "https://ollama.com").rstrip("/")
        if not self.host.startswith("http"):
            self.host = "http://" + self.host
        self.api_key = api_key if api_key is not None else os.environ.get("OLLAMA_API_KEY")
        self.temperature = temperature
        self.think = think
        self.timeout = timeout
        self.retries = retries

    def chat(self, messages: list[Message], **options: Any) -> str:
        body: dict[str, Any] = {"model": self.model, "messages": messages, "stream": False}
        opts = dict(options)
        if self.temperature is not None:
            opts.setdefault("temperature", self.temperature)
        if opts:
            body["options"] = opts
        if self.think is not None:
            body["think"] = self.think
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        resp = _post_json(f"{self.host}/api/chat", body, headers, self.timeout, self.retries)
        try:
            return normalize_text(resp["message"]["content"]).strip()
        except (KeyError, TypeError) as e:
            raise LLMError(f"unexpected Ollama response: {str(resp)[:300]}") from e

    def __repr__(self) -> str:
        return f"OllamaLLM({self.model!r}, host={self.host!r})"


class OpenAICompatLLM:
    """OpenAI-compatible /v1/chat/completions client (vLLM, SGLang, etc.)."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: str | None = None,
        temperature: float | None = None,
        timeout: float = 300.0,
        retries: int = 2,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.temperature = temperature
        self.timeout = timeout
        self.retries = retries

    def chat(self, messages: list[Message], **options: Any) -> str:
        body: dict[str, Any] = {"model": self.model, "messages": messages, **options}
        if self.temperature is not None:
            body.setdefault("temperature", self.temperature)
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        resp = _post_json(f"{self.base_url}/chat/completions", body, headers, self.timeout, self.retries)
        try:
            return normalize_text(resp["choices"][0]["message"]["content"] or "").strip()
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"unexpected response: {str(resp)[:300]}") from e
