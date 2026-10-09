from __future__ import annotations

import os

DEFAULT_MODEL = "openai/gpt-oss-120b"


class LLMError(RuntimeError):
    """Raised with a user-readable message when the LLM can't be reached or isn't configured."""


class GroqClient:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model or os.getenv("GROQ_MODEL", DEFAULT_MODEL)
        self._client = None

    def generate(self, prompt: str, system: str | None = None) -> str:
        if not self.api_key:
            raise LLMError("GROQ_API_KEY is not set. Add it to .env or set it as an environment variable on the host.")
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        try:
            from groq import Groq

            if self._client is None:
                self._client = Groq(api_key=self.api_key)
            resp = self._client.chat.completions.create(model=self.model, messages=messages, temperature=0.2)
        except Exception as exc:
            raise LLMError(f"Groq request failed: {exc}") from exc
        text = resp.choices[0].message.content
        if not text:
            raise LLMError("Groq returned an empty response.")
        return text
