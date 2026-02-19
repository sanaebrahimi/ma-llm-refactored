"""LLM client wrappers used by agent, coordinator, and judge roles."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

try:
    import tiktoken
except ImportError:  # pragma: no cover - optional dependency
    tiktoken = None

try:
    import llmlingua
except ImportError:  # pragma: no cover - optional dependency
    llmlingua = None

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None

try:
    from ollama import Client as OllamaNativeClient
except ImportError:  # pragma: no cover - optional dependency
    OllamaNativeClient = None


@dataclass
class RetryConfig:
    """Retry policy for network calls."""

    max_retries: int = 5
    sleep_seconds: float = 2.0


class GPTClient:
    """Minimal OpenAI chat client with persistent conversation state."""

    def __init__(
        self,
        system_template: str,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        temperature: float = 0.7,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        retry: Optional[RetryConfig] = None,
    ) -> None:
        self.system_template = system_template
        self.messages = [{"role": "system", "content": system_template}]
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.temperature = temperature
        self.model = model
        self.retry = retry or RetryConfig()
        self._client = self._build_client(api_key)

    @staticmethod
    def _build_client(api_key: Optional[str]):
        if OpenAI is None:
            return None
        return OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def get_response(self, prompt: str) -> str:
        """Call the configured OpenAI chat model for one user prompt."""

        if self._client is None:
            raise RuntimeError("OpenAI client is unavailable. Install `openai` and set OPENAI_API_KEY.")

        self.messages.append({"role": "user", "content": prompt})
        response = self._client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            temperature=self.temperature,
        )
        return response.choices[0].message.content

    def safe_api_call(self, prompt: str) -> str:
        """Retry wrapper around :meth:`get_response`."""

        last_error: Optional[Exception] = None
        for _ in range(self.retry.max_retries):
            try:
                return self.get_response(prompt)
            except Exception as exc:  # pragma: no cover - external dependency
                last_error = exc
                time.sleep(self.retry.sleep_seconds)
        raise RuntimeError(f"OpenAI call failed after retries: {last_error}")


class OllamaClient:
    """Thin wrapper for local Ollama chat completion calls."""

    def __init__(self, system_template: str, model: str, host: str = "localhost:11434") -> None:
        self.system_template = system_template
        self.model = model
        self.host = host
        self._client = self._build_client(host)

    @staticmethod
    def _build_client(host: str):
        if OllamaNativeClient is None:
            return None
        return OllamaNativeClient(host=host)

    def get_response(self, prompt: str) -> str:
        """Call local Ollama model with system+user messages."""

        if self._client is None:
            raise RuntimeError("Ollama client is unavailable. Install `ollama` Python package.")

        messages = [
            {"role": "system", "content": self.system_template},
            {"role": "user", "content": prompt},
        ]
        response = self._client.chat(self.model, messages=messages)
        return response.message.content


class JudgeClient(GPTClient):
    """Judge client that can optionally compress long prompts before scoring."""

    def __init__(self, judge_template: str, model: str = "gpt-4o-mini", **kwargs) -> None:
        super().__init__(judge_template, presence_penalty=0.0, frequency_penalty=0.0, model=model, **kwargs)
        self.system_template = judge_template
        self._compressor = self._build_compressor()

    @staticmethod
    def _build_compressor():
        if llmlingua is None:
            return None
        try:
            return llmlingua.PromptCompressor(
                model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
                use_llmlingua2=True,
            )
        except Exception:
            # On CPU-only hosts or restricted environments, loading the
            # compressor model may fail. Judge scoring can proceed without it.
            return None

    @staticmethod
    def num_tokens_from_string(text: str, encoding_name: str = "o200k_base") -> int:
        """Return token count if `tiktoken` is installed, otherwise return 0."""

        if tiktoken is None:
            return 0
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))

    def _compress_if_needed(self, prompt: str, limit: int = 8192) -> str:
        if self._compressor is None:
            return prompt
        while self.num_tokens_from_string(prompt) > limit:
            compressed = self._compressor.compress_prompt(prompt, rate=0.45, force_tokens=["\n", "?"])
            prompt = compressed["compressed_prompt"]
        return prompt

    def get_score(self, prompt: str, temperature: float = 0.7) -> str:
        """Return a judge score string for the provided prompt."""

        if self._client is None:
            raise RuntimeError("OpenAI client is unavailable. Install `openai` and set OPENAI_API_KEY.")

        prompt = self._compress_if_needed(prompt)
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_template},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content


class JudgeOllamaClient(OllamaClient):
    """Judge implementation backed by an Ollama model."""

    def get_score(self, prompt: str) -> str:
        """Return a score by querying a local Ollama model."""

        return self.get_response(prompt)
