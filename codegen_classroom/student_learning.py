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

"""Learning cycle and final evaluation logic for student code generation."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, MutableSequence, Sequence
from concurrent import futures
import hashlib
import logging
import os
import random
import re
import time
import traceback
from typing import Any

from codegen_classroom import agent_config
from codegen_classroom import agents
from codegen_classroom import code_execution
from codegen_classroom import data_utils
from codegen_classroom import environments
from codegen_classroom import scoring
import numpy as np
import pandas as pd

Classroom = MutableSequence[agents.StudentAgent]
ClassroomResults = MutableSequence[agents.StudentCycleResult]

logger = logging.getLogger(__name__)
MAX_FEEDBACK_LEN = 1_000_000


def _validate_dataframe_inputs(**kwargs):
  """Raises TypeError if any input is not a pandas DataFrame."""

  for name, value in kwargs.items():
    if value is None:
      continue
    if not isinstance(value, pd.DataFrame):
      raise TypeError(
          f"Input '{name}' must be a pandas DataFrame, but got {type(value)}."
          " If you are passing a single column, use df[[column_name]] to keep"
          " it as a DataFrame."
      )


def _execute_with_cv_splits(
    extracted_code: str,
    target_fn_name: str,
    x_train: pd.DataFrame,
    y_train: pd.DataFrame | None,
    x_val: pd.DataFrame | None = None,
    initial_namespace: Mapping[str, Any] | None = None,
    validator: scoring.ValidatorFunction = scoring.execution_validator,
    cv_splits: Sequence[tuple[np.ndarray, np.ndarray]] | None = None,
    cv_sample_feedback_count: int = 1,
    auxiliary_train_data: Mapping[str, pd.DataFrame] | None = None,
    timeout_seconds: float | None = None,
    filename_prefix: str = "<code_execution>",
) -> tuple[pd.DataFrame, str, float]:
  """Executes student code across CV folds and aggregates results.

  Args:
      extracted_code: The Python code extracted from the student's solution.
      target_fn_name: Name of the function within the extracted code to execute.
      x_train: DataFrame containing training features.
      y_train: DataFrame containing training labels.
      x_val: DataFrame containing validation features (used for prediction).
      initial_namespace: A mapping of variables to pre-populate the execution
        namespace with (e.g., {"pd": pd, "np": np}).
      validator: A function to validate the output of the executed code.
      cv_splits: A sequence of tuples, where each tuple contains two numpy
        (train_indices, validation_indices) for a single CV fold.
      cv_sample_feedback_count: The number of random CV folds from which to
        collect and return execution feedback.
      auxiliary_train_data: Optional mapping of auxiliary training DataFrames.
      timeout_seconds: Optional timeout in seconds for each code execution.
      filename_prefix: Prefix for the temporary file created for code execution.

  Returns:
    A tuple containing:
      - y_pred_df: A pandas DataFrame with aggregated out-of-fold predictions
        across all CV splits.
      - execution_feedback: A string containing concatenated execution feedback
        from a sample of the CV folds.
      - exec_time_total: The total execution time across all CV folds.
  """
  if x_val is None:
    # If no validation set is provided, we assume we want OOF predictions
    # on the training set.
    x_val = x_train

  if cv_splits is None:
    # If no splits are provided, we just run once on the whole training set
    # to predict on the validation set.
    cv_splits = [(np.arange(len(x_train)), np.arange(len(x_val)))]

  y_pred_dfs = []
  execution_feedbacks = []
  exec_time_total = 0.0

  # Randomly select a subset of folds from which to collect execution feedback.
  feedback_folds = set(
      random.sample(
          range(len(cv_splits)), min(cv_sample_feedback_count, len(cv_splits))
      )
  )

  for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
    x_train_fold = x_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx] if y_train is not None else None
    x_val_fold = x_val.iloc[val_idx]

    aux_train_fold = None
    if auxiliary_train_data is not None:
      aux_train_fold = {
          k: v.iloc[train_idx] for k, v in auxiliary_train_data.items()
      }

    fold_fn_args = [x_train_fold, y_train_fold]
    if aux_train_fold is not None:
      fold_fn_args.extend([aux_train_fold[k] for k in sorted(aux_train_fold)])
    fold_fn_args.append(x_val_fold)

    exec_output, fold_exec_time = code_execution.execute_code(
        extracted_code,
        target_fn_name,
        fn_args=fold_fn_args,
        initial_namespace=initial_namespace,
        local_filename=f"{filename_prefix}_fold_{fold_idx}>",
        timeout_seconds=timeout_seconds,
    )
    exec_time_total += fold_exec_time

    validator(fold_fn_args, {}, exec_output)
    y_pred_fold, exec_feedback = exec_output
    y_pred_dfs.append(y_pred_fold)
    if fold_idx in feedback_folds:
      note = (
          f"\n[Note: Executed on {len(cv_splits)} folds. Feedback from fold"
          f" {fold_idx + 1} shown.]"
      )
      execution_feedbacks.append(exec_feedback + note)

  y_pred_combined = pd.concat(y_pred_dfs)
  y_pred_df = (
      y_pred_combined.groupby(y_pred_combined.index)
      .mean()
      .reindex(x_val.index)
      .dropna(how="all")
  )
  execution_feedback = "\n\n---\n\n".join(execution_feedbacks)

  return y_pred_df, execution_feedback, exec_time_total


def _align_to_predictions(
    y_pred_df: pd.DataFrame,
    y_true: pd.DataFrame | None,
    auxiliary: Mapping[str, pd.DataFrame] | None,
) -> tuple[pd.DataFrame | None, Mapping[str, pd.DataFrame] | None]:
  """Restricts ground truth to the rows actually predicted out-of-fold.

  Identity no-op when the prediction index already covers y_true exactly.
  In time-series forecasting tasks like `gift_eval`, validation splits may
  contain the full series (context history plus forecast horizon) while the
  student's `y_pred` only returns rows indexed over the forecast horizon.
  `_align_to_predictions` slices `y_true` (and row-aligned `auxiliary` data)
  to `y_pred.index` so metrics are evaluated precisely on the predicted
  subset without index mismatch errors.

  Args:
    y_pred_df: Concatenated out-of-fold prediction DataFrame.
    y_true: Ground truth target DataFrame, or None.
    auxiliary: Optional mapping of auxiliary ground truth DataFrames.

  Returns:
    A tuple of (aligned_y_true, aligned_auxiliary).
  """
  if y_true is None or y_pred_df.index.equals(y_true.index):
    return y_true, auxiliary
  logger.warning(
      "CV folds predicted %d of %d ground-truth rows; scoring on the "
      "predicted subset.",
      len(y_pred_df.index),
      len(y_true.index),
  )
  index = y_pred_df.index
  aligned_aux = (
      {k: v.loc[index] for k, v in auxiliary.items()}
      if auxiliary
      else auxiliary
  )
  return y_true.loc[index], aligned_aux


def _retry_with_traceback(
    student: agents.StudentAgent,
    attempt: int,
    max_retries: int,
    last_error_traceback: str,
    result: agents.StudentCycleResult,
) -> tuple[bool, str]:
  """Handles errors within a run_learning_cycle attempt."""

  if attempt < max_retries:

    error_prompt = (
        "The solution you provided (or the code within it) caused an error. "
        "Here is the traceback:\n"
        f"{last_error_traceback}\n\n"
        "Please revise the solution to address the error."
    )
    try:
      student.current_solution = student.generate_solution(error_prompt)
      return True, ""  # Continue to next attempt
    except agents.GenerationError as e:
      logger.warning(
          "[%s] Solution generation for fix failed: %s",
          student.name,
          e,
      )
      result.exclude_reason = "Solution generation for fix failed in retry."
      return False, traceback.format_exc()  # Stop retrying
  else:
    return False, last_error_traceback  # Stop retrying


def _refine_solution(
    student: agents.StudentAgent,
    result: agents.StudentCycleResult,
    classroom: Classroom,
    score_feedback_str: str,
    student_refine_instruction_options: Sequence[str],
    max_mutation_count: int,
    mutation_strategy_options: Sequence[agents.MutationStrategy],
    early_stopping_patience: int,
    current_cycle: int,
    classroom_feedback: str | None = None,
    render_feedback_as_md: bool = False,
):
  """Prepares the agent's solution for the next cycle (refine or mutate)."""
  if result.exclude_reason:
    # logger.debug("[%s] Skipping refine, student excluded.", student.name)
    return

  # Determine if early stopping criteria met for *this* cycle
  result.early_stop = (
      current_cycle - student.best_cycle >= early_stopping_patience
      if student.best_cycle != -1
      else False
  )

  refine_instruction = random.choice(student_refine_instruction_options)
  prompt_parts = [
      refine_instruction,
      f"\n\n**Performance Feedback:**\n{score_feedback_str}",
  ]

  if result.execution_feedback:
    if render_feedback_as_md:
      prompt_parts.append(
          "\n\n**Your Code's Execution Output Feedback:**\n"
          f"\n{result.execution_feedback}\n"
      )
    else:
      prompt_parts.append(
          "\n\n**Your Code's Execution Output:**\n"
          f"```\n{result.execution_feedback}\n```\n"
      )

  if classroom_feedback is not None:  # render as markdown
    prompt_parts.append(
        f"\n\n**Current Classroom State**:\n{classroom_feedback}"
    )

  refine_prompt = "\n".join(prompt_parts)
  logger.debug("[%s] Refine prompt:\n%s", student.name, refine_prompt)

  try:
    if not result.early_stop:  # If not meeting early stop criteria, refine.
      logger.info(
          "[%s] Refining solution for cycle %d.",
          student.name,
          current_cycle + 1,
      )
      student.generate_solution(
          refine_prompt
          + "\n\n**Your solution from this cycle:**"
          f"\n{student.current_solution}"
      )
    elif student.mutation_count < max_mutation_count:
      # Early stopping patience met, try mutating.
      selected_strategy = random.choice(mutation_strategy_options)
      result.mutation_strategy = selected_strategy.name
      logger.info(
          "[%s] Early stop, Mutating with %s (Mutation %d/%d) for cycle %d",
          student.name,
          selected_strategy.name,
          student.mutation_count + 1,
          max_mutation_count,
          current_cycle + 1,
      )
      student.mutate(
          prompt=refine_prompt,
          mutators=classroom,
          strategy=selected_strategy,
      )
    else:  # Early stop and max mutations.
      logger.info(
          "[%s] Early stopping. Max mutations reached. Excluding from cycle %d",
          student.name,
          current_cycle + 1,
      )
      result.exclude_reason = "Max mutations reached."

  except agents.GenerationError as e:
    logger.error(
        "[%s] Solution generation failed in refine/mutate: %s",
        student.name,
        e,
        exc_info=True,
    )
    result.traceback += f"\nRefine Error: {traceback.format_exc()}"
    result.exclude_reason = "Solution generation failed in refine/mutate."


def run_learning_cycle(
    student: agents.StudentAgent,
    x_train: pd.DataFrame,
    y_train: pd.DataFrame | None,
    x_val: pd.DataFrame | None,
    y_val: pd.DataFrame | None,
    current_cycle: int,
    classroom: Classroom,
    student_initial_prompt: str,
    student_function_name: str,
    metric_weights: Mapping[str, float],
    early_stopping_patience: int,
    student_refine_instruction_options: Sequence[str],
    max_mutation_count: int,
    mutation_strategy_options: Sequence[agents.MutationStrategy],
    code_execution_max_retries: int,
    code_execution_timeout_seconds: float | None = None,
    auxiliary_train_data: Mapping[str, pd.DataFrame] | None = None,
    auxiliary_val_data: Mapping[str, pd.DataFrame] | None = None,
    validator: scoring.ValidatorFunction = scoring.execution_validator,
    classroom_feedback: str | None = None,
    cv_splits: Sequence[tuple[np.ndarray, np.ndarray]] | None = None,
    cv_sample_feedback_count: int = 1,
    metric_binary_threshold: float | None = None,
    verbose_logging: bool = False,
    render_feedback_as_md: bool = False,
    env: environments.TaskEnvironment | None = None,
) -> agents.StudentCycleResult:
  """Runs a single learning cycle for a given student.

  This function orchestrates the process of code extraction, execution,
  validation, and scoring for one student in a single cycle. It handles retries
  for execution errors and manages the student's state for the next cycle.

  Args:
      student: The StudentAgent instance for this cycle.
      x_train: The training features.
      y_train: The training labels for the primary regression task.
      x_val: The validation features.
      y_val: The validation labels for the primary regression task.
      current_cycle: The current learning cycle number.
      classroom: A sequence of all students in the learning process.
      student_initial_prompt: The initial prompt for the student if no solution
        exists.
      student_function_name: The name of the function to execute from the
        student's code.
      metric_weights: Weights for calculating the overall score from metrics.
      early_stopping_patience: Number of cycles without improvement to trigger
        early stop.
      student_refine_instruction_options: Sequence of prompts for
        self-refinement.
      max_mutation_count: Maximum number of times a student can mutate.
      mutation_strategy_options: Available mutation strategies to choose from.
      code_execution_max_retries: Max retries for code execution on error.
      code_execution_timeout_seconds: Optional timeout in seconds for code
        execution.
      auxiliary_train_data: Optional mapping of auxiliary training DataFrames.
      auxiliary_val_data: Optional mapping of auxiliary validation DataFrames.
      validator: Function to validate the output of the executed code.
      classroom_feedback: Optional string describing the state of the classroom.
      cv_splits: Custom CV splits (sequence of train_idx, val_idx tuples).
      cv_sample_feedback_count: Number of random folds to provide feedback from.
      metric_binary_threshold: Optional threshold to use for converting
        regression predictions to binary labels for classification metrics.
      verbose_logging: If True, log full code and other verbose details.
      render_feedback_as_md: If True, render execution feedback as markdown.
      env: Optional TaskEnvironment that executes the code instead of the
        DataFrame contract. When set, x/y data, auxiliary data, cv_splits and
        validator are ignored.

  Returns:
      A StudentCycleResult object containing the results of the cycle.
  """
  result = agents.StudentCycleResult(cycle=current_cycle)
  t_cycle = time.time()
  if student.current_solution and current_cycle == 0:
    logger.info(
        "[%s] Starting with seed solution generation for cycle %d",
        student.name,
        current_cycle,
    )
  elif not student.current_solution and current_cycle == 0:
    try:
      logger.info(
          "[%s] Generating initial solution for cycle %d",
          student.name,
          current_cycle,
      )
      student.current_solution = student.generate_solution(
          student_initial_prompt
      )
      if verbose_logging:
        logger.debug(
            "[%s] Initial solution:\n%s", student.name, student.current_solution
        )
    except agents.GenerationError as e:
      result.exclude_reason = "Initial solution generation failed."
      result.traceback = (
          f"Initial solution generation failed: {traceback.format_exc()}"
      )
      logger.warning(
          "[%s] Excluding due to initial generation failure: %s",
          student.name,
          e,
          # exc_info=True,
      )
      student.add_cycle_result(result)
      return result

  initial_namespace = {"pd": pd, "np": np}
  last_error_traceback = ""
  pre_computed_metrics = {}
  score_feedback_str = "Execution/Scoring failed."

  # Retry loop for code extraction, execution, validation, and scoring.
  for attempt in range(code_execution_max_retries + 1):
    result.retry_count = attempt
    last_error_traceback = ""
    exec_time = 0.0
    current_attempt_metrics = dict(pre_computed_metrics)
    try:
      # Extract `python` Code from the student's current solution.
      if verbose_logging:
        logger.debug(
            "[%s] Current Solution:\n%s",
            student.name,
            student.current_solution,
        )
      extracted_code, extracted_plan = code_execution.extract_python_code(
          student.current_solution, target_fn_name=student_function_name
      )
      result.code = extracted_code
      result.code_hash = hashlib.md5(extracted_code.encode("utf-8")).hexdigest()
      result.code_length = len(extracted_code)
      result.plan = extracted_plan
      logger.debug(
          "[%s] Cycle %d Attempt %d, Executing code with hash: %s",
          student.name,
          current_cycle,
          attempt,
          result.code_hash[:8],
      )

      # Execute the extracted code over splits or single set
      if env is not None:
        execution = env.execute(
            extracted_code,
            phase=environments.LEARNING,
            label=f"{student.name}_cycle_{current_cycle}_attempt_{attempt}",
            timeout_seconds=code_execution_timeout_seconds,
        )
        y_pred_df = execution.y_pred
        execution_feedback = execution.execution_feedback
        exec_time = execution.exec_time
        scoring_y_val = execution.y_true
        scoring_aux_val = execution.auxiliary

      elif cv_splits is not None:
        logger.debug(
            "[%s] Executing code with %d CV splits (OOF)...",
            student.name,
            len(cv_splits),
        )
        y_pred_df, execution_feedback, exec_time = _execute_with_cv_splits(
            extracted_code=extracted_code,
            target_fn_name=student_function_name,
            x_train=x_train,
            y_train=y_train,
            x_val=None,  # Use OOF
            cv_splits=cv_splits,
            initial_namespace=initial_namespace,
            validator=validator,
            auxiliary_train_data=auxiliary_train_data,
            timeout_seconds=code_execution_timeout_seconds,
            filename_prefix=(
                f"<{student.name}_cycle_{current_cycle}_attempt_{attempt}"
            ),
            cv_sample_feedback_count=cv_sample_feedback_count,
        )
        # Score against training labels for CV/OOF
        scoring_y_val, scoring_aux_val = _align_to_predictions(
            y_pred_df,
            y_train,
            (
                auxiliary_val_data
                if auxiliary_val_data is not None
                else auxiliary_train_data
            ),
        )

      else:
        logger.debug("[%s] Executing code with validation set...", student.name)
        fn_args = [x_train, y_train]
        if auxiliary_train_data is not None:
          fn_args.extend(
              [auxiliary_train_data[k] for k in sorted(auxiliary_train_data)]
          )
        fn_args.append(x_val)

        exec_output, exec_time = code_execution.execute_code(
            extracted_code,
            student_function_name,
            fn_args=fn_args,
            initial_namespace=initial_namespace,
            local_filename=(
                f"<{student.name}_cycle_{current_cycle}_attempt_{attempt}>"
            ),
            timeout_seconds=code_execution_timeout_seconds,
        )
        validator(fn_args, {}, exec_output)
        y_pred_df, execution_feedback = exec_output
        scoring_y_val = y_val if y_val is not None else y_train
        scoring_aux_val = auxiliary_val_data or auxiliary_train_data

      current_attempt_metrics["exec_time"] = exec_time
      result.execution_feedback = execution_feedback[:MAX_FEEDBACK_LEN]

      # Score the predictions against the validation labels.
      overall_score, metrics_dict, score_feedback_str, score_components = (
          scoring.score_predictions(
              scoring_y_val,  # pyrefly: ignore[bad-argument-type]
              y_pred_df,
              current_attempt_metrics,
              metric_weights,
              auxiliary_ground_truths=scoring_aux_val,
              binary_threshold=metric_binary_threshold,
          )
      )
      logger.debug(
          "[%s] Score Components: %s",
          student.name,
          ", ".join([f"{k}={v:.4f}" for k, v in score_components.items()]),
      )
      result.overall_score = overall_score
      result.metrics = metrics_dict
      pre_computed_metrics = metrics_dict
      break  # Successful attempt, exit retry loop.

    except (
        code_execution.ExtractionError,
        code_execution.ExecutionError,
        code_execution.ExecutionTimeoutError,
        scoring.ValidatorError,
        scoring.ScoreError,
    ) as e:
      error_type = type(e).__name__
      last_error_traceback = traceback.format_exc()
      logger.debug(
          "[%s] %s in attempt %d of %d: %s\n%s",
          student.name,
          error_type,
          attempt + 1,
          code_execution_max_retries + 1,
          e,
          last_error_traceback,
      )
    except Exception as e:  # pylint: disable=broad-except
      last_error_traceback = traceback.format_exc()
      logger.error("[%s] Unexpected Error: %s", student.name, e, exc_info=True)

    # Error Handling & Retry Logic
    if last_error_traceback:
      should_continue, tb = _retry_with_traceback(
          student,
          attempt,
          code_execution_max_retries,
          last_error_traceback,
          result,
      )
      if not should_continue:
        last_error_traceback = tb
        break
    if exec_time > 0:
      pre_computed_metrics["exec_time"] = exec_time

    if attempt >= code_execution_max_retries:
      logger.warning("[%s] Max retries reached for this cycle.", student.name)
      break
    if result.exclude_reason:
      logger.warning("[%s] Student excluded, stopping attempts.", student.name)
      break
  # End of Retry Loop

  if last_error_traceback or result.overall_score is None:
    if not result.exclude_reason:
      result.exclude_reason = "Scoring failed to complete."
    result.traceback = (
        result.traceback or last_error_traceback or result.exclude_reason
    )
    result.overall_score = -float("inf")
    result.metrics = pre_computed_metrics
    logger.warning(
        "[%s] Excluding Cycle %d: Failed after %d attempts.",
        student.name,
        current_cycle,
        result.retry_count,
    )
  else:  # Success
    student.update_best_solution(
        result.overall_score, current_cycle, student.current_solution
    )
    # Print one line summary of the final eval results
    print(
        f"[{student.name}]  Score: {result.overall_score:.4f}, Metrics:"
        f" {', '.join([f'{k}: {v:.4f}' for k, v in result.metrics.items()])}"
        f" (Cycle: {current_cycle}, Retries: {result.retry_count},"
        f" Code Hash: {result.code_hash[:8]})"
    )
  student.add_cycle_result(result)

  _refine_solution(
      student,
      result,
      classroom,
      score_feedback_str,
      student_refine_instruction_options,
      max_mutation_count,
      mutation_strategy_options,
      early_stopping_patience,
      current_cycle,
      classroom_feedback,
      render_feedback_as_md=render_feedback_as_md,
  )
  logger.info(
      "[%s] Cycle %d finished in %.2fs",
      student.name,
      current_cycle,
      time.time() - t_cycle,
  )
  return result


def run_parallel_classroom_learning_cycle(
    current_classroom: Classroom,
    x_train: pd.DataFrame,
    y_train: pd.DataFrame | None,
    x_val: pd.DataFrame | None,
    y_val: pd.DataFrame | None,
    current_cycle: int,
    student_initial_prompt: str,
    student_function_name: str,
    metric_weights: Mapping[str, float],
    early_stopping_patience: int,
    student_refine_instruction_options: Sequence[str],
    max_mutation_count: int,
    mutation_strategy_options: Sequence[agents.MutationStrategy],
    code_execution_max_retries: int,
    code_execution_timeout_seconds: float | None = None,
    auxiliary_train_data: Mapping[str, pd.DataFrame] | None = None,
    auxiliary_val_data: Mapping[str, pd.DataFrame] | None = None,
    validator: scoring.ValidatorFunction = scoring.prediction_validator,
    classroom_feedback: str | None = None,
    num_workers: int | None = None,
    cv_splits: Sequence[tuple[np.ndarray, np.ndarray]] | None = None,
    cv_sample_feedback_count: int = 1,
    metric_binary_threshold: float | None = None,
    verbose_logging: bool = False,
    render_feedback_as_md: bool = False,
    env: environments.TaskEnvironment | None = None,
) -> ClassroomResults:
  """Runs the learning cycle for all students in the classroom in parallel.

  Args:
      current_classroom: Sequence of active StudentAgent instances.
      x_train: Training features.
      y_train: Training labels.
      x_val: Validation features.
      y_val: Validation labels.
      current_cycle: The current learning cycle number.
      student_initial_prompt: Prompt for the first cycle.
      student_function_name: Name of the function to execute.
      metric_weights: Weights for scoring metrics.
      early_stopping_patience: Patience for early stopping.
      student_refine_instruction_options: Options for refinement prompts.
      max_mutation_count: Maximum number of mutations.
      mutation_strategy_options: Available mutation strategies.
      code_execution_max_retries: Max retries for code execution.
      code_execution_timeout_seconds: Timeout in seconds for code execution.
      auxiliary_train_data: Optional mapping of additional ground truth
        DataFrames needed for specific metrics. Keys should match what specific
        metrics expect (e.g., 'positive_label', 'negative_label').
      auxiliary_val_data: Optional mapping of additional ground truth DataFrames
        needed for specific metrics. Keys should match what specific metrics
        expect (e.g., 'positive_label', 'negative_label').
      validator: Function to validate execution output.
      classroom_feedback: Feedback string about the classroom state.
      num_workers: Number of parallel workers. Defaults to # of students.
      cv_splits: Custom CV splits (sequence of train_idx, val_idx tuples).
      cv_sample_feedback_count: Number of random folds to provide feedback from.
      metric_binary_threshold: Optional threshold to use for converting
        regression predictions to binary labels for classification metrics.
      verbose_logging: If True, log full code and other verbose details.
      render_feedback_as_md: If True, render execution feedback as markdown.
      env: Optional TaskEnvironment that executes student code (see
        `run_learning_cycle`). DataFrame inputs may be None when set.

  Returns:
      A sequence of StudentCycleResult objects, one for each student.
  """
  if not current_classroom:
    return []
  if env is None:
    dataframe_inputs = {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
    }
    if auxiliary_train_data is not None:
      dataframe_inputs.update(auxiliary_train_data)
    if auxiliary_val_data is not None:
      dataframe_inputs.update(auxiliary_val_data)
    _validate_dataframe_inputs(**dataframe_inputs)
  parallel_run_args = [
      {
          "student": student,
          "x_train": x_train,
          "y_train": y_train,
          "x_val": x_val,
          "y_val": y_val,
          "current_cycle": current_cycle,
          "classroom": current_classroom,
          "student_initial_prompt": student_initial_prompt,
          "student_function_name": student_function_name,
          "code_execution_max_retries": code_execution_max_retries,
          "code_execution_timeout_seconds": code_execution_timeout_seconds,
          "metric_weights": metric_weights,
          "early_stopping_patience": early_stopping_patience,
          "student_refine_instruction_options": (
              student_refine_instruction_options
          ),
          "max_mutation_count": max_mutation_count,
          "mutation_strategy_options": mutation_strategy_options,
          "auxiliary_train_data": auxiliary_train_data,
          "auxiliary_val_data": auxiliary_val_data,
          "validator": validator,
          "classroom_feedback": classroom_feedback,
          "cv_splits": cv_splits,
          "cv_sample_feedback_count": cv_sample_feedback_count,
          "metric_binary_threshold": metric_binary_threshold,
          "verbose_logging": verbose_logging,
          "render_feedback_as_md": render_feedback_as_md,
          "env": env,
      }
      for student in current_classroom
  ]

  if num_workers is None:
    num_workers = len(current_classroom)

  t_cycle = time.time()
  with futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
    tasks = [
        executor.submit(run_learning_cycle, **args)  # pyrefly: ignore[bad-argument-type]
        for args in parallel_run_args
    ]
    students_cycle_results: ClassroomResults = [t.result() for t in tasks]
  logger.info(
      "Finished learning cycle %s in %.2fs.",
      current_cycle,
      time.time() - t_cycle,
  )

  return students_cycle_results


def process_cycle_results(
    current_classroom: Classroom,
    students_cycle_results: ClassroomResults,
    results_history: MutableMapping[str, ClassroomResults],
    current_cycle: int,
) -> Classroom:
  """Processes and saves results for each student, returns the next classroom.

  Args:
      current_classroom: The sequence of students that participated in cycle.
      students_cycle_results: The results from run_all_students_learning_cycle.
      results_history: Mapping to store results history for each student.
      current_cycle: The current learning cycle number.

  Returns:
      A sequence of StudentAgent instances for the next cycle.
  """
  next_classroom: Classroom = []
  for student, cycle_result in zip(current_classroom, students_cycle_results):
    results_history[student.name].append(cycle_result)  # Add to history first
    if cycle_result.exclude_reason:
      logger.warning(
          "[%s] Excluded in cycle %s: %s",
          student.name,
          current_cycle,
          cycle_result.exclude_reason,
      )
      continue

    student_dir = os.path.dirname(student.student_result_path)
    if not os.path.exists(student_dir):
      data_utils.create_dir(student_dir)
    data_utils.save_dataframe_to_csv(
        pd.DataFrame([cycle_result.flatten()]),
        student.student_result_path,
        mode="a",
    )
    next_classroom.append(student)

  logger.info(
      "%d students remain active, %d students were excluded this cycle.",
      len(next_classroom),
      len(current_classroom) - len(next_classroom),
  )

  return next_classroom


def run_final_evaluation(
    student_name: str,
    x_learning: pd.DataFrame,
    y_learning: pd.DataFrame,
    x_test_final: pd.DataFrame | None,
    y_test_final: pd.DataFrame | None,
    strategies: Sequence[str],
    students: Mapping[str, agents.StudentAgent],
    student_function_name: str,
    metric_weights: Mapping[str, float],
    metric_binary_threshold: float | None = None,
    code_execution_timeout_seconds: float | None = None,
    enforce_timeout: bool = False,
    auxiliary_learning_data: Mapping[str, pd.DataFrame] | None = None,
    auxiliary_test_data: Mapping[str, pd.DataFrame] | None = None,
    validator: scoring.ValidatorFunction = scoring.execution_validator,
    cv_splits: Sequence[tuple[np.ndarray, np.ndarray]] | None = None,
    env: environments.TaskEnvironment | None = None,
) -> tuple[Mapping[str, Any], Mapping[str, pd.DataFrame]]:
  """Runs the final evaluation for a single student on the fixed test set.

  This function evaluates a student's code (either their first, last, or best
  solution) against a final, unseen test set, using the new scoring system.
  It also returns the predictions for later saving.

  Args:
      student_name: The name of the student (agent).
      x_learning: The training features used during learning (for fitting
        model).
      y_learning: The training labels used during learning (for fitting model).
      x_test_final: The final test features for prediction.
      y_test_final: The final test labels for evaluation.
      strategies: A sequence of strings indicating which code to use. Options
        include "first", "last", and "best".
      students: A mapping of all students in the learning process.
      student_function_name: The name of the function to execute from the
        student's code.
      metric_weights: Weights for calculating the overall score from metrics.
      metric_binary_threshold: Optional threshold to use for converting
        regression predictions to binary labels for classification metrics.
      code_execution_timeout_seconds: Optional timeout for code execution.
      enforce_timeout: Whether to enforce execution timeout during final
        evaluation. Defaults to False (timeout disabled).
      auxiliary_learning_data: Optional mapping of additional ground truth
        DataFrames needed for specific metrics. Keys should match what specific
        metrics expect (e.g., 'positive_label', 'negative_label').
      auxiliary_test_data: Optional mapping of additional ground truth
        DataFrames needed for specific metrics. Keys should match what specific
        metrics expect (e.g., 'positive_label', 'negative_label').
      validator: Function to validate the output of the executed code.
      cv_splits: Optional sequence of tuples of indices (train_idx, val_idx) to
        evaluate OOF predictions in CV fashion.
      env: Optional TaskEnvironment that executes the code on its held-out
        split. When set, x/y data, auxiliary data, cv_splits and validator are
        ignored, and the environment's own final timeout applies unless
        enforce_timeout is set.

  Returns:
    A tuple containing:
      - evaluation_results: A mapping containing the final evaluation results
        for the student for each specified strategy.
      - predictions_to_save: A mapping where keys are strategic names
        (e.g., "student_name_last_y_pred", "student_name_best_y_pred") and
        values
        are the corresponding pandas DataFrames of predictions.
  """

  if x_test_final is None or y_test_final is None:
    x_test_final = x_learning
    y_test_final = y_learning

  student = students[student_name]
  effective_timeout = (
      code_execution_timeout_seconds if enforce_timeout else None
  )
  final_eval_results: MutableMapping[str, Any] = {"student_name": student_name}
  predictions_to_save: MutableMapping[str, pd.DataFrame] = {}
  initial_namespace = {"pd": pd, "np": np}
  logger.info(
      "[%s] Starting Final Evaluation for strategies: %s",
      student_name,
      strategies,
  )

  for strategy in set(strategies):
    # Initialize variables for each strategy run
    eval_metrics: MutableMapping[str, float] = {}
    traceback_str = ""
    pre_computed_metrics: MutableMapping[str, float] = {}
    y_pred_df = None
    eval_overall_score = -float("inf")
    cycle_info = ""
    logger.info(
        "[%s] Final evaluation using '%s' strategy...", student.name, strategy
    )
    try:
      if not student.cycle_results:
        raise scoring.EvaluatorError(
            f"No cycle results found for {student.name}"
        )

      if strategy == "first":
        target_cycle = 0
      elif strategy == "last":
        target_cycle = len(student.cycle_results) - 1
      elif strategy == "best":
        if student.best_cycle == -1:
          raise scoring.EvaluatorError(
              f"No best solution found for {student.name}"
          )
        target_cycle = student.best_cycle
      elif strategy.startswith("cycle_"):
        try:
          target_cycle = int(strategy.split("_")[1])
          if not (0 <= target_cycle < len(student.cycle_results)):
            raise IndexError(
                f"Cycle index {target_cycle} out of range for student"
                f" {student.name} with {len(student.cycle_results)} cycles."
            )
        except (ValueError, IndexError) as exc:
          raise scoring.EvaluatorError(
              f"Invalid or out-of-range cycle index in strategy '{strategy}'"
          ) from exc
      else:
        logger.warning(
            "[%s] Unknown evaluation strategy '%s'. Skipping.",
            student.name,
            strategy,
        )
        final_eval_results[strategy] = {"error": "Unknown strategy"}
        continue

      extracted_code = student.cycle_results[target_cycle].code
      cycle_info = f"Cycle: {target_cycle}"
      if not extracted_code:
        raise scoring.EvaluatorError(
            f"Extracted code is empty for strategy '{strategy}'"
        )

      # Execute the extracted code.
      if env is not None:
        execution = env.execute(
            extracted_code,
            phase=environments.FINAL,
            label=f"{student_name}_{strategy}_final_eval",
            timeout_seconds=effective_timeout,
        )
        y_pred_df = execution.y_pred
        exec_time = execution.exec_time
        y_test_scored = execution.y_true
        aux_test_scored = execution.auxiliary

      elif cv_splits is not None:
        y_pred_df, _, exec_time = _execute_with_cv_splits(
            extracted_code=extracted_code,
            target_fn_name=student_function_name,
            x_train=x_learning,
            y_train=y_learning,
            x_val=x_test_final,
            cv_splits=cv_splits,
            initial_namespace=initial_namespace,
            validator=validator,
            auxiliary_train_data=auxiliary_learning_data,
            timeout_seconds=effective_timeout,
            filename_prefix=f"<{student_name}_{strategy}_final_eval",
        )
        y_test_scored, aux_test_scored = _align_to_predictions(
            y_pred_df, y_test_final, auxiliary_test_data
        )

      else:
        # The function should be trained on the full learning set.
        fn_args = [x_learning, y_learning]
        if auxiliary_learning_data is not None:
          fn_args.extend([
              auxiliary_learning_data[k]
              for k in sorted(auxiliary_learning_data)
          ])
        fn_args.append(x_test_final)
        exec_output, exec_time = code_execution.execute_code(
            extracted_code=extracted_code,
            target_fn_name=student_function_name,
            fn_args=fn_args,
            initial_namespace=initial_namespace,
            local_filename=f"<{student_name}_{strategy}_final_eval>",
            timeout_seconds=effective_timeout,
        )
        # Check output has valid predictions and feedback, then unpack
        validator(fn_args, {}, exec_output)
        y_pred_df, _ = exec_output  # execution_feedback is ignored
        y_test_scored = y_test_final
        aux_test_scored = auxiliary_test_data

      pre_computed_metrics["exec_time"] = exec_time
      logger.info(
          "[%s] Final eval execution time: %.2fs", student.name, exec_time
      )

      # Score predictions
      eval_overall_score, eval_metrics, _, _ = scoring.score_predictions(
          y_test_scored if y_test_scored is not None else y_test_final,
          y_pred_df,
          pre_computed_metrics,
          metric_weights,
          auxiliary_ground_truths=aux_test_scored,
          binary_threshold=metric_binary_threshold,
      )

      # Print one line summary of the final eval results
      print(
          f"[{student.name}] '{strategy}':"
          f" Score: {eval_overall_score:.4f}, , Metrics:"
          f" {', '.join([f'{k}: {v:.4f}' for k, v in eval_metrics.items()])}"
          f" ({cycle_info})"
      )

    except (
        code_execution.ExtractionError,
        code_execution.ExecutionError,
        code_execution.ExecutionTimeoutError,
        scoring.ValidatorError,
        scoring.ScoreError,
        scoring.EvaluatorError,
    ) as e:
      traceback_str = traceback.format_exc()
      logger.warning(
          "[%s] Final evaluation failed for '%s': %s\n%s",
          student.name,
          strategy,
          e,
          traceback_str,
      )
      eval_metrics.update(pre_computed_metrics)
    except Exception as e:  # pylint: disable=broad-except
      traceback_str = traceback.format_exc()
      logger.error(
          "[%s] UNEXPECTED ERROR in final eval for '%s': %s",
          student.name,
          strategy,
          e,
          exc_info=True,
      )
      eval_metrics.update(pre_computed_metrics)
      eval_overall_score = -float("inf")

    # Store results for the current strategy
    strategy_results = {
        "overall_score": eval_overall_score,
        "traceback": traceback_str,
    }
    # Prefix metrics with "metrics_"
    for metric_name, metric_value in eval_metrics.items():
      strategy_results[f"metrics_{metric_name}"] = metric_value

    if strategy == "best" and student.best_cycle != -1:
      strategy_results["best_cycle"] = student.best_cycle
    strategy_results["cycle_info"] = cycle_info

    final_eval_results[strategy] = strategy_results

    # Store predictions if successful
    if y_pred_df is not None:
      filename = f"{student_name}_{strategy}_y_predictions"
      predictions_to_save[filename] = y_pred_df

  return final_eval_results, predictions_to_save


def run_parallel_classroom_final_evaluation(
    students: Mapping[str, agents.StudentAgent],
    x_learning: pd.DataFrame,
    y_learning: pd.DataFrame,
    x_test_final: pd.DataFrame | None,
    y_test_final: pd.DataFrame | None,
    student_function_name: str,
    metric_weights: Mapping[str, float],
    metric_binary_threshold: float | None = None,
    code_execution_timeout_seconds: float | None = None,
    enforce_timeout: bool = False,
    strategies: Sequence[str] | None = None,
    auxiliary_learning_data: Mapping[str, pd.DataFrame] | None = None,
    auxiliary_test_data: Mapping[str, pd.DataFrame] | None = None,
    validator: scoring.ValidatorFunction = scoring.prediction_validator,
    num_workers: int | None = None,
    final_results_path: str | None = None,
    predictions_dir: str | None = None,
    cv_splits: Sequence[tuple[np.ndarray, np.ndarray]] | None = None,
    env: environments.TaskEnvironment | None = None,
) -> tuple[Sequence[Mapping[str, Any]], Mapping[str, pd.DataFrame]]:
  """Runs final evaluation for all students in parallel and saves results.

  Args:
      students: Mapping of all student agents.
      x_learning: Training features for fitting.
      y_learning: Training labels for fitting.
      x_test_final: Test features DataFrame. If None, uses x_learning for OOF.
      y_test_final: Test labels DataFrame. If None, uses y_learning for OOF.
      student_function_name: Name of the function to execute.
      metric_weights: Weights for scoring metrics.
      metric_binary_threshold: Optional threshold to use for converting
        regression predictions to binary labels for classification metrics.
      code_execution_timeout_seconds: Optional timeout for code execution.
      enforce_timeout: Whether to enforce execution timeout during final
        evaluation. Defaults to False (timeout disabled).
      strategies: Evaluation strategies (e.g., ["last", "best"]).
      auxiliary_learning_data: Optional mapping of additional ground truth
        DataFrames needed for specific metrics. Keys should match what specific
        metrics expect (e.g., 'positive_label', 'negative_label').
      auxiliary_test_data: Optional mapping of additional ground truth
        DataFrames needed for specific metrics. Keys should match what specific
        metrics expect (e.g., 'positive_label', 'negative_label').
      validator: Function to validate execution output.
      num_workers: Number of parallel workers. Defaults to # of students.
      final_results_path: Full path to save the final evaluation results JSON.
        If None, results are not saved.
      predictions_dir: Full directory path to save prediction CSV files. If
        None, predictions are not saved.
      cv_splits: Optional sequence of tuples of indices (train_idx, val_idx) to
        evaluate OOF predictions in CV fashion.
      env: Optional TaskEnvironment that executes student code (see
        `run_final_evaluation`). DataFrame inputs may be None when set.

  Returns:
      A tuple containing:
      - evaluation_results: A sequence of final evaluation results.
      - all_predictions: A mapping of all predictions.
  """

  # If using cross validation and test set not provided, use learning set.
  if x_test_final is None or y_test_final is None:
    x_test_final = x_learning
    y_test_final = y_learning

  if strategies is None:
    strategies = ["best"]
  if env is None:
    dataframe_inputs = {
        "x_learning": x_learning,
        "y_learning": y_learning,
        "x_test_final": x_test_final,
        "y_test_final": y_test_final,
    }
    if auxiliary_learning_data is not None:
      dataframe_inputs.update(auxiliary_learning_data)
    if auxiliary_test_data is not None:
      dataframe_inputs.update(auxiliary_test_data)
    _validate_dataframe_inputs(**dataframe_inputs)

  logger.info(
      "Starting Final Evaluation for %d students (Strategies: %s)",
      len(students),
      strategies,
  )
  final_eval_args = []
  for name in students:
    final_eval_args.append({
        "student_name": name,
        "x_learning": x_learning,
        "y_learning": y_learning,
        "x_test_final": x_test_final,
        "y_test_final": y_test_final,
        "strategies": strategies,
        "students": students,
        "student_function_name": student_function_name,
        "code_execution_timeout_seconds": code_execution_timeout_seconds,
        "enforce_timeout": enforce_timeout,
        "metric_weights": metric_weights,
        "auxiliary_learning_data": auxiliary_learning_data,
        "auxiliary_test_data": auxiliary_test_data,
        "validator": validator,
        "cv_splits": cv_splits,
        "metric_binary_threshold": metric_binary_threshold,
        "env": env,
    })

  num_workers = num_workers if num_workers is not None else len(students)
  with futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
    tasks = [
        executor.submit(run_final_evaluation, **args)
        for args in final_eval_args
    ]
    parallel_results = [t.result() for t in tasks]

  # Unpack results and predictions in one go
  final_eval_results = []
  all_predictions = {}
  for res, preds_dict in parallel_results:
    final_eval_results.append(res)
    all_predictions.update(preds_dict or {})

  # Save final evaluation results JSON if path is provided
  if final_results_path:
    data_utils.save_dict_to_json(final_eval_results, final_results_path)
    logger.info("Saved final evaluation results to: %s", final_results_path)

  # Save predictions if directory is provided
  if predictions_dir and all_predictions:
    data_utils.create_dir(predictions_dir)
    logger.info(
        "Saving %d Predictions to CSV in %s",
        len(all_predictions),
        predictions_dir,
    )

    for pred_key, y_pred_df in all_predictions.items():
      if y_pred_df is None or y_pred_df.empty:
        logger.warning("Skipping empty/None prediction for '%s'", pred_key)
        continue
      if env is None:
        y_pred_df = y_pred_df.set_index(x_test_final.index)
        y_pred_df.columns = [y_test_final.columns[0]]
      pred_path = os.path.join(predictions_dir, f"{pred_key}.csv")
      data_utils.save_dataframe_to_csv(y_pred_df, pred_path, index=True)
      logger.debug("Saved prediction for '%s' to: %s", pred_key, pred_path)
  elif predictions_dir:
    logger.info(
        "Predictions directory provided, but no predictions were generated."
    )

  return final_eval_results, all_predictions


def get_best_solution_from_eval_results(
    eval_results: Sequence[Mapping[str, Any]],
    metric_key: str = "overall_score",
    solution_strategy: str = "best",
    higher_is_better: bool = True,
    min_cycle: int = 0,
    max_exec_time: float | None = None,
) -> tuple[str | None, Mapping[str, Any] | None]:
  """Finds the best student solution from a sequence of evaluation results.

  Args:
      eval_results: Sequence of evaluation result mappings.
      metric_key: Key within the solution_strategy dict for comparison.
      solution_strategy: Which evaluation strategy ('best' or 'last').
      higher_is_better: True if higher metric values are better.
      min_cycle: Minimum 'best_cycle' to consider (for 'best' strategy).
      max_exec_time: Maximum 'metrics_exec_time' allowed.

  Returns:
    (best_student_name, best_solution_data), or (None, None).
  """
  best_entry = None
  extreme_value = -float("inf") if higher_is_better else float("inf")

  for item in eval_results:
    student_name = item.get("student_name", "")
    eval_data = item.get(solution_strategy, {})
    if eval_data is None:
      logger.debug(
          "[%s] Skipping: Solution strategy '%s' not found.",
          student_name,
          solution_strategy,
      )
      continue

    current_metric = eval_data.get(metric_key)
    if current_metric is None:
      logger.debug(
          "[%s] Skipping: Metric '%s' not found in '%s'.",
          student_name,
          metric_key,
          solution_strategy,
      )
      continue

    exec_time = eval_data.get("metrics_exec_time", float("inf"))
    if max_exec_time is not None and exec_time > max_exec_time:
      logger.debug(
          "[%s] Skipping: Execution time %.2f exceeds max %.2f.",
          student_name,
          exec_time,
          max_exec_time,
      )
      continue

    best_cycle = eval_data.get("best_cycle", -1)
    if solution_strategy == "best" and best_cycle < min_cycle:
      logger.debug(
          "[%s] Skipping: Best cycle %d is less than min cycle %d.",
          student_name,
          best_cycle,
          min_cycle,
      )
      continue

    is_better = (higher_is_better and current_metric > extreme_value) or (
        not higher_is_better and current_metric < extreme_value
    )

    if is_better:
      extreme_value = current_metric
      best_entry = item

  if not best_entry:
    logger.warning("No best solution found matching criteria.")
    return None, None

  best_student_name = best_entry.get("student_name", "")
  best_solution_data = best_entry[solution_strategy]
  logger.info(
      "Best solution from %s (%s) with %s: %.3f",
      best_student_name,
      solution_strategy,
      metric_key,
      best_solution_data.get(metric_key, float("nan")),
  )
  return best_student_name, best_solution_data


def get_classroom_student_configs(
    system_instruction: str,
    api_key: str | None = None,
    model_names: Sequence[str | tuple[str, str]] | None = None,
    temperature: float = 1.0,
    max_output_tokens: int = 32768,
    max_code_generation_retries: int | None = None,
    retry_sleep_seconds: int | None = None,
    backend: str | None = None,
    base_url: str | None = None,
    extra_body: Mapping[str, Any] | None = None,
) -> Sequence[agents.AgentConfig]:
  """Returns a sequence of student configurations for a classroom.

  Args:
      system_instruction: The base system instruction to use for each agent.
      api_key: The API key for the generative model service.
      model_names: Optional sequence of model names to use. Entries may also be
        `(backend, model_name)` tuples to mix backends in one classroom (e.g.
        `[("vllm", "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"), ("gemini",
        "gemini-3.5-flash")]`).
      temperature: The default temperature for the models.
      max_output_tokens: The default maximum output tokens for the models.
      max_code_generation_retries: Optional maximum number of retries for code
        generation.
      retry_sleep_seconds: Optional sleep time in seconds between retries.
      backend: Backend for plain-string `model_names` entries ("gemini" or
        "openai_compat"/"vllm"). Defaults to `agent_config.DEFAULT_BACKEND`.
      base_url: Server URL for OpenAI-compatible backends.
      extra_body: Extra request fields for OpenAI-compatible backends.

  Returns:
      A sequence of agents.AgentConfig objects.
  """
  if model_names is None:
    model_names = agent_config.DEFAULT_STUDENT_MODELS
  default_backend = backend or agent_config.DEFAULT_BACKEND

  student_cfgs = []
  for i, entry in enumerate(model_names):
    if isinstance(entry, str):
      student_backend, model_name = default_backend, entry
    else:
      student_backend, model_name = entry
    # Student names are used in file paths, so keep them filesystem-safe.
    clean_model_name = re.sub(r"[^A-Za-z0-9_.]", "_", model_name)
    student_name = f"student_{i+1}_{clean_model_name}"
    kwargs = {
        "name": student_name,
        "api_key": api_key,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "system_instruction": system_instruction,
        "model_name": model_name,
        "backend": student_backend,
        "base_url": base_url,
        "extra_body": extra_body,
    }
    if max_code_generation_retries is not None:
      kwargs["max_code_generation_retries"] = max_code_generation_retries
    if retry_sleep_seconds is not None:
      kwargs["retry_sleep_seconds"] = retry_sleep_seconds
    student_cfgs.append(agents.AgentConfig(**kwargs))
  return student_cfgs
