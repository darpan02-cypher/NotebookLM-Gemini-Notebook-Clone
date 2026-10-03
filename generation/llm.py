from __future__ import annotations

import os

DEFAULT_MODEL = "gemini-2.5-flash"


class LLMError(RuntimeError):
    """Raised with a user-readable message when the LLM can't be reached or isn't configured."""


class GeminiClient:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
        self._client = None

    def generate(self, prompt: str, system: str | None = None) -> str:
        if not self.api_key:
            raise LLMError("GEMINI_API_KEY is not set. Add it as an environment variable / Space secret.")
        try:
            from google import genai
            from google.genai import types

            if self._client is None:
                self._client = genai.Client(api_key=self.api_key)
            resp = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(system_instruction=system, temperature=0.2),
            )
        except Exception as exc:
            raise LLMError(f"Gemini request failed: {exc}") from exc
        if not resp.text:
            raise LLMError("Gemini returned an empty response (possibly blocked by safety filters).")
        return resp.text
