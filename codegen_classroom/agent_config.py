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

"""Global configuration defaults for models, generation, and execution."""

from __future__ import annotations

import os

import dotenv
from etils import epath


def load_env(env_path: str | epath.PathLike | None = None) -> None:
  """Loads API keys and settings from .env files into os.environ."""
  if env_path is not None:
    dotenv.load_dotenv(env_path)
  dotenv.load_dotenv(epath.Path(__file__).resolve().parent / ".env")
  dotenv.load_dotenv()


# Load .env before reading environment-driven defaults below.
load_env()

# Default LLM backend: "gemini" (Gemini API) or "openai_compat" (a local vLLM
# server or any other OpenAI-compatible endpoint).
DEFAULT_BACKEND = os.environ.get("CODEGEN_BACKEND", "gemini")

# Default base URL of the OpenAI-compatible (vLLM) server.
DEFAULT_VLLM_BASE_URL = os.environ.get(
    "VLLM_BASE_URL", "http://localhost:8000/v1"
)

# Example open-weight models to serve with vLLM. The served model name must
# match what `vllm serve` was started with (see scripts/serve_vllm.sh).
DEFAULT_VLLM_MODELS = (
    "unsloth/Qwen3.8-27B-unsloth-bnb-4bit",
    "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
)

# Default model used for standard agents and generic tasks.
# See available models at: https://ai.google.dev/gemini-api/docs/models
DEFAULT_MODEL = "gemini-3.5-flash"

# Default sampling temperature for LLM code generation.
DEFAULT_TEMPERATURE = 0.5

# Default maximum output tokens per LLM generation call.
DEFAULT_MAX_OUTPUT_TOKENS = 32768

# Default maximum number of retries when LLM code generation fails.
DEFAULT_MAX_RETRIES = 5

# Default sleep duration in seconds between LLM retry attempts.
DEFAULT_RETRY_SLEEP_SECONDS = 30

# Default wall-clock timeout in seconds for executing student-generated code.
DEFAULT_EXECUTION_TIMEOUT_SECONDS = 30

# Default HTTP request timeout in milliseconds for GenAI API calls.
DEFAULT_HTTP_TIMEOUT_MS = 300_000

# Default Gemma open-weight models available for student agents.
DEFAULT_GEMMA_MODELS = (
    "gemma-4-31b-it",
    "gemma-4-26b-a4b-it",
)

# Default Gemini models available for student and analysis agents.
DEFAULT_GEMINI_MODELS = (
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.1-pro-preview",
)

# Floating latest-model endpoints (not recommended for reproducible runs).
# See available models at: https://ai.google.dev/gemini-api/docs/models
DEFAULT_LATEST_MODELS = (
    "gemini-flash-lite-latest",
    "gemini-flash-latest",
    "gemini-pro-latest",
)

# Combined tuple of default models used when spawning a classroom of students.
DEFAULT_STUDENT_MODELS = DEFAULT_GEMMA_MODELS + DEFAULT_GEMINI_MODELS

