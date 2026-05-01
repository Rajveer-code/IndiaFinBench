"""
rag/generator.py
----------------
LLM generation backend with Groq (primary) and Ollama (local fallback).

Backend selection: set env var RAG_LLM_BACKEND=groq|ollama, or pass
`backend` explicitly to LLMGenerator.__init__.

Prompt design decisions:
  - Explicit prohibition on general knowledge: LLMs default to mixing
    parametric memory with retrieved context, inflating faithfulness scores.
  - temperature=0.0: deterministic output is required for reproducible eval.
  - [Source N] citation format matches the source block numbering so the
    judge in evaluation.py can cross-reference claims.
  - "chunk {chunk_idx}" suffix in source header provides traceability to
    exact chunk position, not just document title.
"""

import os

from rag.models import RetrievalResult

_SYSTEM_PROMPT = (
    "You are an expert in Indian financial regulation specialising in "
    "Reserve Bank of India (RBI) and Securities and Exchange Board of India "
    "(SEBI) regulatory documents.\n\n"
    "Answer the question using ONLY the numbered source passages provided. "
    "Cite every factual claim inline as [Source N]. "
    "If the sources do not contain sufficient information to answer the "
    "question, state this explicitly — do not infer, extrapolate, or draw "
    "on general knowledge not present in the sources. "
    "Be concise and precise. Maximum 200 words unless the question requires more."
)


def _build_context_block(results: list[RetrievalResult]) -> str:
    parts = [
        f"[Source {i}] {r.chunk.title} (chunk {r.chunk.chunk_idx})\n{r.chunk.text}"
        for i, r in enumerate(results, 1)
    ]
    return "\n\n".join(parts)


class LLMGenerator:
    def __init__(
        self,
        backend:    str   = "groq",
        model:      str | None = None,
        max_tokens: int   = 512,
        temperature: float = 0.0,
    ) -> None:
        self.backend     = backend
        self.max_tokens  = max_tokens
        self.temperature = temperature

        if backend == "groq":
            from groq import Groq
            self._client = Groq(api_key=os.environ["GROQ_API_KEY"])
            self.model   = model or "llama-3.3-70b-versatile"

        elif backend == "ollama":
            import ollama as _ollama  # type: ignore[import]
            self._client = _ollama
            self.model   = model or "llama3.2:3b"

        else:
            raise ValueError(
                f"Unknown backend {backend!r}. Valid options: 'groq', 'ollama'."
            )

    def generate(self, query: str, results: list[RetrievalResult]) -> str:
        if not results:
            return (
                "No relevant passages were found in the corpus for this query. "
                "Please rephrase or ask a different question."
            )

        context_block = _build_context_block(results)
        user_msg      = (
            f"SOURCES:\n{context_block}\n\n"
            f"QUESTION:\n{query}\n\n"
            f"ANSWER:"
        )

        if self.backend == "groq":
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return resp.choices[0].message.content.strip()

        elif self.backend == "ollama":
            resp = self._client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            )
            return resp["message"]["content"].strip()

        raise RuntimeError(f"Unhandled backend: {self.backend!r}")
