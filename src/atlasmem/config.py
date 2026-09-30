"""Environment-driven defaults and a one-call factory for a ready-to-use JitMemory."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .curator import Curator
from .judge import LLMJudge
from .llm import OllamaLLM
from .memory import JitMemory
from .retrieval import BM25Retriever
from .store import MemoryBank


def _env(name: str, default: str) -> str:
    return os.environ.get(name) or default


@dataclass
class Settings:
    db: str = field(default_factory=lambda: _env("ATLASMEM_DB", str(Path.home() / ".atlasmem" / "memory.db")))
    curator_model: str = field(default_factory=lambda: _env("ATLASMEM_CURATOR_MODEL", "gpt-oss:20b"))
    judge_model: str = field(default_factory=lambda: _env("ATLASMEM_JUDGE_MODEL", "gpt-oss:20b"))
    executor_model: str = field(default_factory=lambda: _env("ATLASMEM_EXECUTOR_MODEL", "gpt-oss:120b"))
    k: int = field(default_factory=lambda: int(_env("ATLASMEM_K", "3")))
    host: str | None = field(default_factory=lambda: os.environ.get("OLLAMA_HOST"))


def build_memory(settings: Settings | None = None, use_judge: bool = True) -> JitMemory:
    s = settings or Settings()
    bank = MemoryBank(s.db)
    curator = Curator(OllamaLLM(s.curator_model, host=s.host))
    judge = LLMJudge(OllamaLLM(s.judge_model, host=s.host, temperature=0.0)) if use_judge else None
    return JitMemory(bank, curator, judge, BM25Retriever(bank, k=s.k))
