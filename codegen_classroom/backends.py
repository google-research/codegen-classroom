# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LLM backends: Gemini (google-genai) and OpenAI-compatible servers (vLLM).

Each backend turns a (history, prompt) pair into response text and exposes the
tuple of exceptions that callers should treat as transient and retry on.
Provider SDKs are imported lazily so that only the selected backend's package
needs to be installed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
import re
from typing import Any, Protocol

GEMINI = "gemini"
OPENAI_COMPAT = "openai_compat"
# Aliases accepted in configs for the OpenAI-compatible backend.
_OPENAI_COMPAT_ALIASES = frozenset({OPENAI_COMPAT, "openai", "vllm"})

# Reasoning blocks. Some chat templates (e.g. Qwen3.5) put the opening <think>
# in the prompt, so the response holds only the reasoning and a closing tag.
_THINK_BLOCK_RE = re.compile(r"^.*?</think>\s*|<think>.*?</think>\s*", flags=re.DOTALL)


def is_openai_compat(backend: str | None) -> bool:
  """Returns True if `backend` names the OpenAI-compatible backend."""
  return (backend or "").lower() in _OPENAI_COMPAT_ALIASES


class LLMBackend(Protocol):
  """Interface for a text generation backend."""

  retryable_exceptions: tuple[type[BaseException], ...]
  # Errors that retrying cannot fix (e.g. context length exceeded). Callers
  # stop generating for this request instead of crashing the run.
  fatal_exceptions: tuple[type[BaseException], ...]

  def generate(
      self,
      history: Sequence[tuple[str, str]],
      prompt: str,
      *,
      system_instruction: str,
      temperature: float,
      max_output_tokens: int,
      json_mode: bool = False,
  ) -> str:
    """Returns the model response to `prompt` given prior (role, text) turns.

    Roles in `history` are "user" and "model".
    """
    ...


class GeminiBackend:
  """Backend for the Gemini API via google-genai."""

  def __init__(
      self,
      model_name: str,
      api_key: str | None = None,
      http_timeout_ms: int | None = None,
  ):
    from google import genai  # pylint: disable=g-import-not-at-top

    self._genai = genai
    self.model_name = model_name
    self.client = genai.Client(
        api_key=api_key or os.environ.get("GEMINI_API_KEY"),
        http_options=genai.types.HttpOptions(timeout=http_timeout_ms),
    )
    self.retryable_exceptions = (
        genai.errors.ServerError,
        genai.errors.ClientError,
        genai.errors.httpx.HTTPError,
    )
    self.fatal_exceptions = ()  # All genai API errors are retried (upstream).

  def generate(
      self,
      history: Sequence[tuple[str, str]],
      prompt: str,
      *,
      system_instruction: str,
      temperature: float,
      max_output_tokens: int,
      json_mode: bool = False,
  ) -> str:
    types = self._genai.types
    contents = [
        types.Content(role=role, parts=[types.Part.from_text(text=text)])
        for role, text in history
    ]
    contents.append(
        types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
    )
    response = self.client.models.generate_content(
        model=self.model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            response_mime_type="application/json" if json_mode else None,
        ),
    )
    return (response.text or "") if response else ""


class OpenAICompatBackend:
  """Backend for OpenAI-compatible chat servers (vLLM, SGLang, Ollama, ...)."""

  def __init__(
      self,
      model_name: str,
      base_url: str | None = None,
      api_key: str | None = None,
      http_timeout_ms: int | None = None,
      extra_body: Mapping[str, Any] | None = None,
      strip_think: bool = True,
  ):
    import openai  # pylint: disable=g-import-not-at-top

    self.model_name = model_name
    self.extra_body = dict(extra_body) if extra_body else None
    self.strip_think = strip_think
    self.client = openai.OpenAI(
        base_url=base_url,
        # vLLM ignores the key unless started with --api-key, but the client
        # requires a non-empty value.
        api_key=api_key or os.environ.get("VLLM_API_KEY") or "EMPTY",
        timeout=http_timeout_ms / 1000 if http_timeout_ms else None,
        max_retries=0,  # Retries are handled by the agents.
    )
    self.retryable_exceptions = (
        openai.APIConnectionError,  # Includes APITimeoutError.
        openai.RateLimitError,
        openai.InternalServerError,
    )
    # Remaining HTTP errors (400 context length, 401, 404, ...). Listed after
    # retryable_exceptions in except clauses, so 429/5xx are still retried.
    self.fatal_exceptions = (openai.APIStatusError,)

  def generate(
      self,
      history: Sequence[tuple[str, str]],
      prompt: str,
      *,
      system_instruction: str,
      temperature: float,
      max_output_tokens: int,
      json_mode: bool = False,
  ) -> str:
    messages = []
    if system_instruction:
      messages.append({"role": "system", "content": system_instruction})
    for role, text in history:
      messages.append(
          {"role": "assistant" if role == "model" else role, "content": text}
      )
    messages.append({"role": "user", "content": prompt})

    kwargs: dict[str, Any] = {
        "model": self.model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_output_tokens,
    }
    if json_mode:
      kwargs["response_format"] = {"type": "json_object"}
    if self.extra_body:
      kwargs["extra_body"] = self.extra_body

    response = self.client.chat.completions.create(**kwargs)
    if not response or not response.choices:
      return ""
    text = response.choices[0].message.content or ""
    if self.strip_think:
      text = _THINK_BLOCK_RE.sub("", text)
    return text


def make_backend(config: Any) -> LLMBackend:
  """Creates the backend named by `config.backend` from an AgentConfig."""
  backend = (config.backend or GEMINI).lower()
  if backend == GEMINI:
    return GeminiBackend(
        model_name=config.model_name,
        api_key=config.api_key,
        http_timeout_ms=config.http_timeout_ms,
    )
  if is_openai_compat(backend):
    return OpenAICompatBackend(
        model_name=config.model_name,
        base_url=config.base_url,
        api_key=config.api_key,
        http_timeout_ms=config.http_timeout_ms,
        extra_body=config.extra_body,
    )
  raise ValueError(
      f"Unknown backend {config.backend!r}; expected {GEMINI!r} or"
      f" {OPENAI_COMPAT!r}."
  )
