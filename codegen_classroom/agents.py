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

"""LLM agents and their configs: Student and Solution Analysis."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, MutableSequence, Sequence
import dataclasses
import enum
import functools
import json
import logging
import os
import time
from typing import Any

from codegen_classroom import agent_config
from codegen_classroom import backends
from codegen_classroom import data_utils
import numpy as np

logger = logging.getLogger(__name__)


class MutationStrategy(enum.Enum):
  """Enum for different mutation strategies with associated prompt templates."""

  def __init__(self, value: str, prompt: str):
    self._value_ = value
    self.prompt = prompt

  # SELF_BEST_CODE is a fallback and used if no mutators are available.
  SELF_BEST_CODE = (
      "self_best_code",
      "Analyze your own best solution and try to improve it.",
  )
  BEST_PEER_REPLACEMENT = (
      "best_peer_replacement",
      "",  # No prompt needed, as this is a direct solution replacement.
  )
  BEST_PEER_CROSSOVER = (
      "best_peer_crossover",
      (
          "{peer_examples}\n"
          "Analyze the better solution and incorporate its techniques and ideas"
          " to create an improved version of your own solution."
      ),
  )
  CONTRASTIVE_PEER_CROSSOVER = (
      "contrastive_peer_crossover",
      (
          "Improve your solution using a contrastive approach.\n\n1. "
          " **Inspirational Solution**: Incorporate ideas from the"
          " following:{positive_examples}\n\n2.  **Solution to Avoid**: Do NOT"
          " use patterns from the following:{negative_examples}\n\nAnalyze the"
          " better solution and incorporate its techniques while strictly"
          " avoiding the mistakes in the worse solution."
      ),
  )


class GenerationError(Exception):
  """Exceptions for errors during LLM code generation (often service issues)."""


@dataclasses.dataclass
class AgentConfig:
  """Configuration for an LLM agent (Gemini API or OpenAI-compatible server)."""

  name: str
  api_key: str | None = dataclasses.field(default=None, repr=False)
  # "gemini" or "openai_compat" (aliases: "openai", "vllm").
  backend: str = agent_config.DEFAULT_BACKEND
  # Server URL for OpenAI-compatible backends, e.g. http://localhost:8000/v1.
  base_url: str | None = None
  # Extra request fields for OpenAI-compatible backends (e.g. top_p,
  # chat_template_kwargs).
  extra_body: Mapping[str, Any] | None = None
  system_instruction: str = ""
  model_name: str = agent_config.DEFAULT_MODEL
  temperature: float = agent_config.DEFAULT_TEMPERATURE
  max_output_tokens: int = agent_config.DEFAULT_MAX_OUTPUT_TOKENS
  max_code_generation_retries: int = agent_config.DEFAULT_MAX_RETRIES
  retry_sleep_seconds: int = agent_config.DEFAULT_RETRY_SLEEP_SECONDS
  response_mime_type: str | None = None
  http_timeout_ms: int = agent_config.DEFAULT_HTTP_TIMEOUT_MS


@dataclasses.dataclass
class StudentCycleResult:
  """Holds all metadata and results from a single student learning cycle."""

  cycle: int  # Cycle number.
  code: str = ""  # Extracted Python code.
  code_hash: str = ""  # Hash of the extracted Python code.
  plan: str = ""  # Text before and after the code block.
  code_length: int = 0  # Length of the *extracted* Python code.
  traceback: str = ""  # Traceback if any error occurred.
  execution_feedback: str = ""  # Optional feedback from student code.
  retry_count: int = 0  # Number of retries within this cycle's execution phase.
  exclude_reason: str = ""  # Reason for exclusion. Empty means do not exclude.
  early_stop: bool = False  # Flag indicating if early stopping for this cycle.
  mutation_strategy: str | None = None  # The mutation strategy used, if any.
  overall_score: float | None = None  # The final score for this cycle.
  metrics: MutableMapping[str, float] = dataclasses.field(default_factory=dict)

  def flatten(self, prefix: str = "metrics_") -> Mapping[str, Any]:
    """Flattens the result, including all metrics calculated in this cycle."""
    flat_dict = {
        field.name: getattr(self, field.name)
        for field in dataclasses.fields(self)
        if field.name != "metrics"
    }
    # Add all metric keys from the result's metrics dictionary
    for metric_name, metric_value in self.metrics.items():
      full_key = f"{prefix}{metric_name}"
      flat_dict[full_key] = metric_value
    return flat_dict


class LLMAgent:
  """Base class for the Student and Solution Analysis LLM agents."""

  def __init__(self, config: AgentConfig, run_base_dir: str):
    self.config = config
    self.name = self.config.name
    self.run_base_dir = run_base_dir
    if backends.is_openai_compat(self.config.backend):
      self.config.base_url = (
          self.config.base_url or agent_config.DEFAULT_VLLM_BASE_URL
      )
    self.backend = backends.make_backend(self.config)
    self.system_instruction = self.config.system_instruction
    self.model_name = self.config.model_name
    self.temperature = self.config.temperature
    self.max_output_tokens = self.config.max_output_tokens
    self.history: MutableSequence[tuple[str, str]] = []

    # Save agent config
    agent_config_dir = os.path.join(self.run_base_dir, "agent_configs")
    data_utils.create_dir(agent_config_dir)
    config_path = os.path.join(agent_config_dir, f"{self.name}_config.json")
    data_utils.save_dict_to_json(
        data_utils.config_to_serializable_dict(self.config),
        config_path,
    )

  def reset_history(self):
    """Resets conversation history, preserving system instructions."""
    self.history = []

  def generate(self, prompt: str) -> str:
    """Generates text from the prompt using the model.

    Maintains multi-turn `self.history` for within-cycle interactions (e.g.
    error-fix retries) until `reset_history()` is called between cycles. Turns
    are recorded only after receiving a non-empty response so failed or empty
    calls never corrupt history.

    Args:
        prompt: The prompt to send to the model for text generation.

    Returns:
        The generated response text, or an empty string if generation failed.
    """
    response_text = self.backend.generate(
        self.history,
        prompt,
        system_instruction=self.system_instruction,
        temperature=self.temperature,
        max_output_tokens=self.max_output_tokens,
        json_mode=self.config.response_mime_type == "application/json",
    )

    if response_text.strip():
      self.history.append(("user", prompt))
      self.history.append(("model", response_text))
    return response_text


# Backwards-compatible alias for the upstream class name.
GeminiAgent = LLMAgent


@functools.total_ordering
class StudentAgent(LLMAgent):
  """Student agent that generates solutions and learns from feedback."""

  def __init__(
      self,
      config: AgentConfig,
      run_base_dir: str,
      seed_solution: str | None = None,
  ):
    """Initializes the StudentAgent.

    Args:
        config: The agent configuration.
        run_base_dir: The base directory for the current run.
        seed_solution: Initial model response (solution) to start with.
    """
    print(
        f"Initializing Student Agent ({config.name}) using"
        f" {config.backend}:{config.model_name}..."
    )
    super().__init__(config, run_base_dir)
    self.current_solution: str = seed_solution or ""  # Full model output.
    self.student_dir: str = os.path.join(self.run_base_dir, "students")
    data_utils.create_dir(self.student_dir)
    self.student_result_path: str = os.path.join(
        self.student_dir, f"{self.name}.csv"
    )
    self.cycle_results: MutableSequence[StudentCycleResult] = []
    self.mutation_count: int = 0
    self.generate_code_count: int = 0
    self.best_cycle: int = -1  # Index of the best performing cycle.
    self.best_score: float = -float("inf")  # Higher is better, start low.
    self.best_solution: str = ""  # Stores the best full model response.

  def __lt__(self, other: "StudentAgent") -> bool:
    return self.get_best_score() < other.get_best_score()

  def __eq__(self, other: "StudentAgent") -> bool:  # pyrefly: ignore[bad-override]
    return self.get_best_score() == other.get_best_score()

  def __gt__(self, other: "StudentAgent") -> bool:
    return self.get_best_score() > other.get_best_score()

  def __add__(self, other: "StudentAgent") -> str:
    """Creates a prompt snippet for a better performing peer."""
    if not isinstance(other, StudentAgent):
      return NotImplemented
    # Assumes other > self (better) based on usage in mutate.
    positive_example_prompt = (
        f" {other.name} produced a better solution"
        f" with a score of {other.get_best_score():.4f}.\n"
        "Here is their solution:\n---"
        f"\n{other.best_solution}\n---"
    )
    return positive_example_prompt

  def __sub__(self, other: "StudentAgent") -> str:
    """Creates a prompt snippet for a worse performing peer."""
    if not isinstance(other, StudentAgent):
      return NotImplemented
    # Assumes other < self (worse) based on usage in mutate.
    negative_example_prompt = (
        f" {other.name} produced a worse solution"
        f" with a score of {other.get_best_score():.4f}.\n"
        "Here is their solution:\n---"
        f"\n{other.best_solution}\n---"
    )
    return negative_example_prompt

  def __repr__(self) -> str:
    """Returns a string representation of the student's state."""
    if self.best_cycle == -1:
      return (
          f"StudentAgent(name='{self.name}', "
          f"Learning Cycles = {len(self.cycle_results)}, Best Cycle = N/A)"
      )
    best_result = self.cycle_results[self.best_cycle]
    return (
        f"StudentAgent(name='{self.name}', "
        f"Learning Cycles={len(self.cycle_results)}\n"
        f"Best Cycle = {self.best_cycle}, Score = {self.get_best_score():.4f})"
        f"\nBest Plan:\n{best_result.plan}"
        f"\nBest Code:\n{best_result.code}"
    )

  def generate_solution(self, prompt: str) -> str:
    """Generates a solution using the language model with retry logic.

    The generated solution (full response) is also stored in
    `self.current_solution`.

    Handles ServerError and ClientError by waiting and retrying up to
    self.config.max_code_generation_retries times.

    Args:
        prompt: The prompt to send to the model for solution generation.

    Returns:
        The generated solution as a string.

    Raises:
        GenerationError: If the model generation fails after all retries.
    """
    self.generate_code_count += 1
    last_exception = None
    for attempt in range(self.config.max_code_generation_retries):
      try:
        solution = super().generate(prompt)
        self.current_solution = solution
        return solution
      except self.backend.retryable_exceptions as e:
        last_exception = e
        error_type = type(e).__name__
        if attempt < self.config.max_code_generation_retries - 1:
          # Only sleep if this is not the last attempt
          logger.warning(
              "[%s] %s on generate attempt %d/%d: %s. Retrying in %d seconds.",
              self.name,
              error_type,
              attempt + 1,
              self.config.max_code_generation_retries,
              e,
              self.config.retry_sleep_seconds,
          )
          time.sleep(self.config.retry_sleep_seconds)
        else:
          logger.error(
              "[%s] %s on final generate attempt %d: %s",
              self.name,
              error_type,
              attempt + 1,
              e,
              exc_info=True,
          )
      except self.backend.fatal_exceptions as e:
        logger.error("[%s] Non-retryable LLM API error: %s", self.name, e)
        raise GenerationError(f"Non-retryable LLM API error: {e}") from e
    raise GenerationError(
        "Failed to generate solution after"
        f" {self.config.max_code_generation_retries} attempts due to API"
        " issues."
    ) from last_exception

  def add_cycle_result(self, result: StudentCycleResult):
    """Adds a learning cycle result to the history."""
    # logger.debug("[%s] Adding result for cycle %d", self.name, result.cycle)
    self.cycle_results.append(result)

  def update_best_solution(
      self,
      current_score: float,
      current_cycle: int,
      current_solution: str,
  ) -> bool:
    """Updates the best solution if the current score is better.

    Higher overall_score is always better.

    Args:
        current_score: The score achieved in the current cycle.
        current_cycle: The current cycle number.
        current_solution: The full model response for the current cycle.

    Returns:
        True if the current score is better, False otherwise.
    """
    if current_score is None or np.isnan(current_score):
      return False
    if current_score > self.best_score:  # Higher is always better
      logger.info(
          "[%s] New best score: %.4f -> %.4f at cycle %d",
          self.name,
          self.best_score,
          current_score,
          current_cycle,
      )
      self.best_score = current_score
      self.best_cycle = current_cycle
      self.best_solution = current_solution
      return True

    return False

  def get_best_score(self) -> float:
    """Helper to get the best score (higher is better)."""
    if self.best_cycle == -1 or not self.cycle_results:
      return -float("inf")
    # best_cycle should be a valid index if not -1
    score = self.cycle_results[self.best_cycle].overall_score
    return score if score is not None and np.isfinite(score) else -float("inf")

  def mutate(
      self,
      prompt: str,
      mutators: Sequence["StudentAgent"],
      strategy: MutationStrategy = MutationStrategy.SELF_BEST_CODE,
  ):
    """Generates a new solution by learning from peer best solutions.

    Args:
        prompt: The base prompt for generation.
        mutators: A sequence of student agents whose solutions can be used for
          mutation.
        strategy: The mutation strategy to apply.
    """
    self.mutation_count += 1
    valid_mutators = [m for m in mutators if m.best_cycle != -1]
    peer_mutators = [m for m in valid_mutators if m is not self]
    current_strategy = strategy

    # Determine the effective strategy
    if not valid_mutators:
      logger.warning(
          "[%s] No valid mutators, falling back to SELF_BEST_CODE from %s.",
          self.name,
          strategy.name,
      )
      current_strategy = MutationStrategy.SELF_BEST_CODE
    elif (
        self.best_cycle == -1
        and strategy.name != MutationStrategy.BEST_PEER_REPLACEMENT.name
    ):
      logger.warning(
          "[%s] No best solution yet, falling back to SELF_BEST_CODE from %s.",
          self.name,
          strategy.name,
      )
      current_strategy = MutationStrategy.SELF_BEST_CODE
    elif not peer_mutators and strategy.name in [
        MutationStrategy.BEST_PEER_CROSSOVER.name,
        MutationStrategy.CONTRASTIVE_PEER_CROSSOVER.name,
    ]:
      logger.warning(
          "[%s] No peers for %s, falling back to SELF_BEST_CODE.",
          self.name,
          strategy.name,
      )
      current_strategy = MutationStrategy.SELF_BEST_CODE

    strategy_name = current_strategy.name
    if strategy_name == MutationStrategy.BEST_PEER_REPLACEMENT.name:
      best_mutator = max(valid_mutators)
      if best_mutator > self or self.best_cycle == -1:
        logger.info(
            "[%s] Mutating with BEST_PEER_REPLACEMENT from %s (Score: %.4f)",
            self.name,
            best_mutator.name,
            best_mutator.get_best_score(),
        )
        self.current_solution = best_mutator.best_solution
      else:
        logger.info(
            "[%s] No better peer for replacement, keeping own (Score: %.4f).",
            self.name,
            self.get_best_score(),
        )
      return

    # Prepare context for prompt templates
    context = {}
    if strategy_name == MutationStrategy.BEST_PEER_CROSSOVER.name:
      best_peer = max(peer_mutators)
      if best_peer > self:
        context["peer_examples"] = self + best_peer
      else:
        logger.warning(
            "[%s] Best peer (%s) not better for BEST_PEER_CROSSOVER, falling"
            " back to SELF_BEST_CODE.",
            self.name,
            best_peer.name,
        )
        current_strategy = MutationStrategy.SELF_BEST_CODE

    elif strategy_name == MutationStrategy.CONTRASTIVE_PEER_CROSSOVER.name:
      better = sorted([m for m in peer_mutators if m > self], reverse=True)
      worse = sorted([m for m in peer_mutators if m < self])
      context["positive_examples"] = (
          "\n\n".join([self + m for m in better]) or "\nNone"
      )
      context["negative_examples"] = (
          "\n\n".join([self - m for m in worse]) or "\nNone"
      )

    # Base prompt
    if self.best_cycle != -1:
      base_prompt = (
          "Your previous best solution achieved an overall score of"
          f" {self.get_best_score():.4f}.\nHere was your best"
          f" solution:\n{self.best_solution}\n\n"
      )
    else:
      base_prompt = (
          "This is your first attempt to mutate. Here is your current"
          f" solution:\n{self.current_solution}\n\n"
      )

    # Generate and format strategy prompt
    try:
      strategy_prompt = current_strategy.prompt.format(**context)
    except KeyError as e:
      logger.error(
          "[%s] Missing context for strategy %s with context: %s: %s",
          self.name,
          current_strategy.name,
          context,
          e,
          exc_info=True,
      )
      raise ValueError(
          f"Missing context for strategy {current_strategy.name}: {e}"
      ) from e

    final_mutation_prompt = base_prompt + strategy_prompt
    logger.debug(
        "[%s] Generating mutated solution with strategy %s",
        self.name,
        current_strategy.name,
    )
    self.generate_solution(prompt + "\n" + final_mutation_prompt)


class SolutionAnalysisAgent(LLMAgent):
  """Analyzes the best student model solution using an LLM agent.

  Extracts metadata (tags, description, insights, etc.), writes a JSON
  summary to storage (`best_analysis.json`), and returns it as a dictionary.
  """

  def __init__(
      self,
      config: AgentConfig,
      run_base_dir: str | None = None,
      extra_json_fields: Mapping[str, str] | None = None,
      json_fields: Mapping[str, str] | None = None,
      run_cns_base_dir: str | None = None,
  ):
    actual_run_dir = run_base_dir or run_cns_base_dir or "."
    if json_fields is None:
      json_fields = {
          "model_tags": (
              "Comma-separated string of tags describing the modeling"
              " architectures."
          ),
          "model_description": "1-2 sentences summarizing the modeling method.",
          "model_insights": (
              "1-2 sentences of notable insights or feature analysis."
          ),
      }
    else:
      json_fields = dict(json_fields)

    if extra_json_fields:
      json_fields.update(extra_json_fields)

    fields_str = "\n".join(f'  - "{k}": {v}' for k, v in json_fields.items())

    if not config.system_instruction:
      config.system_instruction = (
          "You are an expert machine learning algorithmic analyst.\n"
          "Extract the following information from the provided machine learning"
          " solution and return it strictly as a JSON object:\n"
          f"{fields_str}"
      )

    if not config.response_mime_type:
      config.response_mime_type = "application/json"

    super().__init__(config, actual_run_dir)

  def analyze(
      self,
      student_name: str,
      best_code: str,
      best_plan: str,
      best_test_metrics: Mapping[str, Any],
      best_val_metrics: Mapping[str, Any],
      student_instructions: str,
      metrics_description: str,
      task_description_for_prompt: str = "the task",
      extra_prompt_instructions: str = "",
      max_retries: int | None = None,
  ) -> dict[str, Any]:
    """Analyzes the best student model solution.

    Args:
        student_name: The name of the winning student.
        best_code: Python code of the winning solution.
        best_plan: Plan/reasoning of the winning solution.
        best_test_metrics: Evaluated metrics dict on the unseen test set.
        best_val_metrics: Validation/learning metrics dict (OOF baseline).
        student_instructions: Main student system instructions.
        metrics_description: Main metrics description.
        task_description_for_prompt: Descriptive name of the task (e.g. "the
          activity classification task" or "predicting house prices").
        extra_prompt_instructions: Any extra instruction text to append to the
          prompt.
        max_retries: Optional override for the number of retry attempts.

    Returns:
        A dict containing the extracted solution metadata keys.

    Raises:
        ValueError: If the model's response is not a valid JSON document.
    """

    analysis_prompt = f"""
Please analyze the following machine learning solution.
The {student_name} model was selected based on validation performance and achieved these scores on the unseen Test Set:
{best_test_metrics}

Validation Metrics:
{best_val_metrics}

Plan:
{best_plan}

Code:
```python
{best_code}
```

Task Description:
{student_instructions}

Metrics Description:
{metrics_description}

Please provide a deep dive into how this specific student approached {task_description_for_prompt}.
{extra_prompt_instructions}
"""

    self.reset_history()
    current_prompt = analysis_prompt
    max_retries = (
        max_retries
        if max_retries is not None
        else self.config.max_code_generation_retries
    )
    last_exception = None

    for attempt in range(max_retries):
      try:
        res_json = super().generate(current_prompt).strip()
        res_dict = json.loads(res_json)

        # Save the analysis metadata to file for persistence
        analysis_path = os.path.join(self.run_base_dir, "best_analysis.json")
        data_utils.save_dict_to_json(res_dict, analysis_path)
        return res_dict

      except ValueError as e:
        last_exception = e
        logger.warning(
            "[%s] JSON parse error: %s. Conversational retry %d/%d.",
            self.name,
            e,
            attempt + 1,
            max_retries,
        )
        current_prompt = (
            f"Failed to parse your response as JSON due to error: {e}.\nPlease"
            " regenerate the response as a valid JSON object matching the"
            " requested schema."
        )

      except self.backend.retryable_exceptions as e:
        last_exception = e
        error_type = type(e).__name__
        if attempt < max_retries - 1:
          logger.warning(
              "[%s] %s on attempt %d/%d. Retrying in %d seconds.",
              self.name,
              error_type,
              attempt + 1,
              max_retries,
              self.config.retry_sleep_seconds,
          )
          time.sleep(self.config.retry_sleep_seconds)
        else:
          logger.error(
              "[%s] %s on final generate attempt %d: %s",
              self.name,
              error_type,
              attempt + 1,
              e,
              exc_info=True,
          )
      except self.backend.fatal_exceptions as e:
        raise ValueError(f"Non-retryable LLM API error: {e}") from e

    raise ValueError(
        "Failed to obtain and parse valid JSON from model."
    ) from last_exception
