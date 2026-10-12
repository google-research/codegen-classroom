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

"""Scoring functions, configurations, and type definitions for code-generating agents."""

from __future__ import annotations

import collections
from collections.abc import Callable, Mapping, MutableMapping, MutableSequence, Sequence
import logging
from typing import Any

from codegen_classroom import metrics as metric_lib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ValidatorError(Exception):
  """Custom exception for validation failures. Validators should raise this."""


class ScoreError(Exception):
  """Custom exception for scoring failures."""


class EvaluatorError(Exception):
  """Custom exception for errors during code final evaluation."""


ValidatorFunction = Callable[
    [
        Sequence[Any],  # fn_args: Positional args passed to function.
        Mapping[str, Any],  # fn_kwargs: Keyword args passed to function.
        Any,  # result: The raw output of the function.
    ],
    None,  # Validators raise ValidatorError or return None.
]


def execution_validator(
    fn_args: Sequence[Any],
    fn_kwargs: Mapping[str, Any],
    result: Any | None,
) -> None:
  """Validates the output of a solution function (pd.DataFrame, str)."""
  del fn_args, fn_kwargs  # Unused in this example validator.
  if not isinstance(result, tuple):
    raise ValidatorError(
        'Function must return a tuple of (pd.DataFrame, str), but got'
        f' {type(result)}.'
    )

  if len(result) != 2:
    raise ValidatorError(
        'Returned tuple must have exactly 2 elements (pd.DataFrame, str), but'
        f' got {len(result)} elements.'
    )

  y_pred_df, execution_feedback = result

  if not isinstance(y_pred_df, pd.DataFrame):
    raise ValidatorError(
        'The first element of the return tuple must be a pandas DataFrame, but'
        f' got {type(y_pred_df)}.'
    )

  if not isinstance(execution_feedback, str):
    raise ValidatorError(
        'The second element of the return tuple must be a string, but got'
        f' {type(execution_feedback)}.'
    )


def prediction_validator(
    fn_args: Sequence[Any],
    fn_kwargs: Mapping[str, Any],
    result: Any,
    expected_columns: int = 1,
    max_feedback_len: int = 1_000_000,
    expected_dtype: Any | None = None,
) -> None:
  """Validates the output of a solution function.

  Ensures the output is (pd.DataFrame, str) and checks DataFrame contents.

  Args:
      fn_args: Positional arguments passed to the original function. The
        validation data (X_val) is expected at the last index.
      fn_kwargs: Keyword arguments passed to the original function (unused).
      result: The return value of the student's function.
      expected_columns: The expected number of columns in the prediction
        DataFrame.
      max_feedback_len: Maximum allowed length for the execution_feedback
        string.
      expected_dtype: If provided, validates that the prediction DataFrame
        columns match this dtype ('int' for integers).

  Raises:
      ValidatorError: If the code_output fails any validation check.
  """

  execution_validator(fn_args, fn_kwargs, result)
  y_pred_df, execution_feedback = result

  if y_pred_df.empty:
    raise ValidatorError('Predictions DataFrame is empty.')

  if len(fn_args) < 3:
    raise ValidatorError(f'Expected at least 3 fn_args, got {len(fn_args)}')
  x_val = fn_args[-1]  # X_val is expected to be the last argument
  if not isinstance(x_val, pd.DataFrame):
    logger.error('X_val in validator is not a DataFrame, type: %s', type(x_val))
    raise ValidatorError(
        f'Internal Error: X_val is not a DataFrame, got {type(x_val)}'
    )

  if y_pred_df.shape[0] != x_val.shape[0]:
    raise ValidatorError(
        'Predictions DataFrame has an incorrect number of rows. '
        f'Expected {x_val.shape[0]}, but got {y_pred_df.shape[0]}.'
    )

  if not y_pred_df.index.equals(x_val.index):
    raise ValidatorError(
        'Predictions DataFrame index does not match X_val index.'
    )

  if y_pred_df.shape[1] != expected_columns:
    raise ValidatorError(
        f'Predictions DataFrame must have {expected_columns} column(s). '
        f'Got {y_pred_df.shape[1]} columns: {y_pred_df.columns.tolist()}.'
    )

  if y_pred_df.isnull().values.any():
    raise ValidatorError('Predictions DataFrame contains NaN values.')

  # Example custom dtype validation for int predictions (i.e. classification).
  if expected_dtype == 'int':
    if not all(np.issubdtype(dtype, np.integer) for dtype in y_pred_df.dtypes):
      raise ValidatorError(
          'Predictions DataFrame must contain only integers for classification'
          ' tasks.'
      )
  # Validate that all columns in the prediction DataFrame are numeric.
  elif not all(np.issubdtype(dtype, np.number) for dtype in y_pred_df.dtypes):
    raise ValidatorError(
        'All columns in Predictions DataFrame must be numeric.'
    )

  # Validate execution_feedback length
  if len(execution_feedback) > max_feedback_len:
    logger.warning(
        'Execution feedback length (%d) exceeds maximum allowed (%d). It will'
        ' be truncated by the caller.',
        len(execution_feedback),
        max_feedback_len,
    )


def classification_prediction_validator(
    fn_args: Sequence[Any],
    fn_kwargs: Mapping[str, Any],
    result: Any,
    expected_columns: int = 1,
    max_feedback_len: int = 1_000_000,
) -> None:
  """Validates classification predictions, ensuring integer output."""
  prediction_validator(
      fn_args,
      fn_kwargs,
      result,
      expected_columns,
      max_feedback_len,
      expected_dtype='int',
  )


def get_metrics_description(metric_weights: Mapping[str, float]) -> str:
  """Generates a string describing the active metrics and their weights."""
  lines = []
  for name, weight in sorted(metric_weights.items()):
    if weight > 0:
      if name not in metric_lib.ALL_METRICS_CONFIG:
        raise ScoreError(
            f"Unsupported metric '{name}' in metric_weights. Registered"
            f' metrics: {sorted(metric_lib.ALL_METRICS_CONFIG)}'
        )
      metric = metric_lib.ALL_METRICS_CONFIG[name]
      lines.append(
          f'-   **{name}:** {metric.description}, {metric.goal_string}'
          f' (Weight: {weight:.2f})'
      )
  return '\n'.join(lines)


def get_score_feedback_string(
    score: float,
    metrics: Mapping[str, float],
    metric_configs: Mapping[str, metric_lib.Metric],
) -> str:
  """Creates a formatted feedback string for the LLM."""
  lines = [
      f'Overall Score: {score:.4f} (Higher is Better)',
      'Included Metrics:',
  ]
  for name in sorted(metric_configs.keys()):
    value = metrics.get(name)
    metric = metric_configs[name]
    if value is not None:
      val_str = f'{value:.4f}'
      lines.append(
          f'-   **{metric.description}:** {val_str} ({metric.goal_string})'
      )
  return '\n'.join(lines)


def get_classroom_feedback_string(
    cycle_num: int,
    current_classroom: Sequence[Any],
    classroom_results_history: Mapping[str, Sequence[Any]],
    metric_keys: Sequence[str],
) -> str:
  """Generates concise markdown feedback string about the classroom state."""
  if not classroom_results_history and cycle_num == 0:
    return ''

  num_active_current = len(current_classroom)
  total_students = len(classroom_results_history)

  all_metric_values: MutableMapping[str, MutableSequence[float]] = (
      collections.defaultdict(list)
  )
  num_successful_results = 0

  for student_results in classroom_results_history.values():
    for result in student_results:
      if result.exclude_reason:
        continue
      num_successful_results += 1
      for key in metric_keys:
        metric_val = (
            result.overall_score
            if key == 'overall_score'
            else result.metrics.get(key)
        )
        if metric_val is not None and not np.isnan(metric_val):
          all_metric_values[key].append(metric_val)

  summary = (
      f'Completed **{cycle_num} cycles** with {num_active_current} of '
      f'{total_students} students currently active.\n'
  )

  if num_successful_results == 0:
    return summary + 'No successful results yet.\n'

  stat_lines = []
  for key in sorted(all_metric_values.keys()):
    if key not in metric_keys:
      continue
    data = all_metric_values[key]

    stats_str = (
        f'Avg={np.nanmean(data):.3f}, Min={np.nanmin(data):.3f}, '
        f'Max={np.nanmax(data):.3f}'
    )
    stat_lines.append(f'-   **{key}:** {stats_str}')

  if not stat_lines:
    return summary

  stats_header = (
      f'Stats from **{num_successful_results}** successful results so far:\n'
  )
  return summary + stats_header + '\n'.join(stat_lines) + '\n'


def score_predictions(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    pre_computed_metrics: Mapping[str, float],
    metric_weights: Mapping[str, float],
    auxiliary_ground_truths: Mapping[str, pd.DataFrame] | None = None,
    binary_threshold: float | None = None,
) -> tuple[float, MutableMapping[str, float], str, MutableMapping[str, float]]:
  """Calculates metrics, overall score, and feedback string based on weights.

  Assumptions:
  - y_pred is a DataFrame from code_output validated by prediction_validator.
    This implies correct shape, index, numeric types, and no NaNs.
  - y_true is a DataFrame.

  Args:
      y_true: Ground truth values for the primary regression task.
      y_pred: Predicted values, as a pandas DataFrame.
      pre_computed_metrics: Mapping of metrics already computed (e.g., time).
      metric_weights: Mapping metric names to their weights.
      auxiliary_ground_truths: Optional mapping of additional ground truth
        DataFrames needed for specific metrics. Keys should match what specific
        metrics expect (e.g., 'positive_label', 'negative_label').
      binary_threshold: Optional float threshold used to binarize predictions
        and ground truths for binary classification metrics (e.g., precision,
        recall, AUROC).

  Returns:
      Tuple of (overall_score, metrics_dict, feedback_string, score_components).

  Raises:
      ScoreError: If inputs are not DataFrames, contain NaNs, or if indices
        or shapes don't match, or if a metric calculation/normalization fails.
  """
  unknown = sorted(
      name
      for name, weight in metric_weights.items()
      if weight > 0 and name not in metric_lib.ALL_METRICS_CONFIG
  )
  if unknown:
    raise ScoreError(
        f'Unknown metric name(s) in metric_weights: {unknown}. Registered'
        f' metrics: {sorted(metric_lib.ALL_METRICS_CONFIG)}'
    )

  if not isinstance(y_true, pd.DataFrame):
    raise ScoreError('y_true must be a pandas DataFrame.')
  # y_pred type is guaranteed by execution_validator via prediction_validator

  if not y_true.index.equals(y_pred.index):
    raise ScoreError('Indices of y_true and y_pred do not match.')

  if y_true.shape != y_pred.shape:
    raise ScoreError(
        f'Shapes of y_true {y_true.shape} and y_pred'
        f' {y_pred.shape} do not match.'
    )

  if y_true.isnull().values.any():
    raise ScoreError('y_true contains NaN values.')

  if auxiliary_ground_truths:
    for aux_name, aux_df in auxiliary_ground_truths.items():
      if not isinstance(aux_df, pd.DataFrame):
        raise ScoreError(
            f"Auxiliary ground truth '{aux_name}' must be a DataFrame."
        )
      if not aux_df.index.equals(y_pred.index):
        raise ScoreError(
            f"Indices of auxiliary ground truth '{aux_name}' and y_pred do"
            ' not match.'
        )
      if aux_df.isnull().values.any():
        raise ScoreError(
            f"Auxiliary ground truth '{aux_name}' contains NaN values."
        )

  # Assuming y_pred has no NaNs due to prior validation.

  y_pred_np = y_pred.values.flatten()
  y_true_np = y_true.values.flatten()

  # Flatten auxiliary ground truths if present
  aux_np = {}
  if auxiliary_ground_truths:
    for key, df in auxiliary_ground_truths.items():
      aux_np[key] = df.values.flatten()

  active_metrics: MutableMapping[str, metric_lib.Metric] = {
      name: metric_lib.ALL_METRICS_CONFIG[name]
      for name, weight in metric_weights.items()
      if weight > 0 and name in metric_lib.ALL_METRICS_CONFIG
  }
  metrics_results: MutableMapping[str, float] = dict(pre_computed_metrics)

  # Calculate metric values
  for name, metric in active_metrics.items():
    if name not in metrics_results and metric.func:
      try:
        if name in metric_lib.BINARY_CLASSIFICATION_METRICS:
          # If binary_threshold provided, derive binary labels from regression.
          y_true_bin = None
          threshold = binary_threshold

          if threshold is not None:
            y_true_bin = (y_true_np >= threshold).astype(int)
          elif 'binary_ground_truth' in aux_np:
            y_true_bin = aux_np['binary_ground_truth']
            threshold = aux_np.get('binary_threshold', np.array([0.0]))[0]

          if y_true_bin is not None:
            # AUROC/AUPRC have continuous scores; others use binary labels.
            y_pred_bin = (
                y_pred_np
                if name in ('auroc', 'auprc')
                else (y_pred_np >= threshold).astype(int)
            )
            args = (y_true_bin, y_pred_bin)
          else:
            args = (y_true_np, y_pred_np)

        else:
          args = (y_true_np, y_pred_np)

        kwargs = {}
        if name == 'worst_group_accuracy':
          # Attempt to get 'subgroup_labels' from auxiliary ground truths
          if 'subgroup_labels' in aux_np:
            kwargs['subgroup_labels'] = aux_np['subgroup_labels']
        elif name == 'mase':
          if metric_lib.MASE_SCALE_AUX_KEY not in aux_np:
            raise ScoreError(
                "Metric 'mase' requires the auxiliary ground truth"
                f" '{metric_lib.MASE_SCALE_AUX_KEY}' (per-row in-sample"
                ' seasonal-naive MAE of the training split).'
            )
          kwargs['seasonal_scale'] = aux_np[metric_lib.MASE_SCALE_AUX_KEY]

        metric_value = metric.compute(*args, **kwargs)

      except Exception as e:
        logger.error(
            "Error calculating metric '%s': %s", name, e, exc_info=True
        )
        raise ScoreError(f"Error calculating metric '{name}'") from e

      metrics_results[name] = float(metric_value)

  # Compute overall score
  overall_score = 0.0
  total_weight = 0.0
  score_components: MutableMapping[str, float] = {}
  for name, metric in active_metrics.items():
    value = metrics_results.get(name)
    if value is not None:
      weight = metric_weights[name]
      if not np.isfinite(value):
        # Single choke point for computed AND pre-computed values. NaN and -inf
        # would otherwise normalize to a PERFECT 1.0 via inverse_normalize, and
        # +inf to an unbounded component via clip_negative.
        logger.warning(
            "Metric '%s' is non-finite (%s); scoring it as the worst possible"
            ' contribution (0.0).',
            name,
            value,
        )
        component_score = 0.0
      else:
        try:
          component_score = metric.normalize_score(value)
        except Exception as e:
          logger.error(
              "Error or non-finite result during normalization of metric '%s'"
              ' with value %s: %s',
              name,
              value,
              e,
              exc_info=True,
          )
          raise ScoreError(
              f"Error during normalization of metric '{name}' with value"
              f' {value}'
          ) from e

      weighted_component = weight * component_score
      score_components[name] = weighted_component
      overall_score += weighted_component
      total_weight += weight

  score = (overall_score / total_weight) if total_weight > 0 else 0.0
  feedback = get_score_feedback_string(score, metrics_results, active_metrics)
  return score, metrics_results, feedback, score_components
