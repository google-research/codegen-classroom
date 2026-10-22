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

"""Pure metric functions for common regression and classification tasks."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import dataclasses

import numpy as np
import scipy.stats
import sklearn.metrics

# Standard sklearn metrics to be used directly
mean_absolute_error = sklearn.metrics.mean_absolute_error
mean_squared_error = sklearn.metrics.mean_squared_error
r2_score = sklearn.metrics.r2_score
accuracy_score = sklearn.metrics.accuracy_score
balanced_accuracy_score = sklearn.metrics.balanced_accuracy_score
cohen_kappa_score = sklearn.metrics.cohen_kappa_score


def _correlation_compute(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    func: Callable[[np.ndarray, np.ndarray], tuple[float, float]],
) -> float:
  """Helper for calculating correlations, handling zero variance."""
  if y_true.size <= 1 or y_pred.size <= 1:
    return 0.0
  if np.var(y_true) < 1e-9 or np.var(y_pred) < 1e-9:
    return 0.0
  # Errors from func (e.g., scipy.stats) will propagate up.
  corr, _ = func(y_true, y_pred)
  return float(corr) if np.isfinite(corr) else 0.0


def pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
  """Calculates Pearson correlation."""
  return _correlation_compute(y_true, y_pred, scipy.stats.pearsonr)


def kendall_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
  """Calculates Kendall's tau correlation."""
  return _correlation_compute(y_true, y_pred, scipy.stats.kendalltau)


def spearman_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
  """Calculates Spearman rank correlation."""
  return _correlation_compute(y_true, y_pred, scipy.stats.spearmanr)


def f1_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
  """Calculates F1 score for a binary classification task."""
  return float(
      sklearn.metrics.f1_score(
          y_true, y_pred, average='binary', zero_division=0.0
      )
  )


def precision_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
  """Calculates precision for a binary classification task."""
  # Precision is the precision of positive class (Positive Predictive Value).
  return float(
      sklearn.metrics.precision_score(
          y_true, y_pred, average='binary', zero_division=0.0
      )
  )


def recall_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
  """Calculates recall for a binary classification task."""
  # Recall is the recall of positive class (True Positive Rate).
  return float(
      sklearn.metrics.recall_score(
          y_true, y_pred, average='binary', zero_division=0.0
      )
  )


def specificity_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
  """Calculates specificity for a binary classification task."""
  # Specificity is the recall of negative class (True Negative Rate).
  return float(
      sklearn.metrics.recall_score(
          y_true, y_pred, pos_label=0, average='binary', zero_division=0.0
      )
  )


def f1_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
  """Calculates F1 score with macro averaging."""
  return float(
      sklearn.metrics.f1_score(
          y_true, y_pred, average='macro', zero_division=0.0
      )
  )


def precision_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
  """Calculates precision with macro averaging."""
  return float(
      sklearn.metrics.precision_score(
          y_true, y_pred, average='macro', zero_division=0.0
      )
  )


def recall_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
  """Calculates recall with macro averaging."""
  return float(
      sklearn.metrics.recall_score(
          y_true, y_pred, average='macro', zero_division=0.0
      )
  )


def f1_weighted(y_true: np.ndarray, y_pred: np.ndarray) -> float:
  """Calculates F1 score with weighted averaging."""
  return float(
      sklearn.metrics.f1_score(
          y_true, y_pred, average='weighted', zero_division=0.0
      )
  )


def precision_weighted(y_true: np.ndarray, y_pred: np.ndarray) -> float:
  """Calculates precision with weighted averaging."""
  return float(
      sklearn.metrics.precision_score(
          y_true, y_pred, average='weighted', zero_division=0.0
      )
  )


def recall_weighted(y_true: np.ndarray, y_pred: np.ndarray) -> float:
  """Calculates recall with weighted averaging."""
  return float(
      sklearn.metrics.recall_score(
          y_true, y_pred, average='weighted', zero_division=0.0
      )
  )


def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
  """Calculates AUROC (Area Under the Receiver Operating Characteristic curve)."""
  return float(sklearn.metrics.roc_auc_score(y_true, y_score))


def auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
  """Calculates AUPRC (Average Precision Score)."""
  return float(sklearn.metrics.average_precision_score(y_true, y_score))


def worst_group_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subgroup_labels: np.ndarray | None = None,
) -> float:
  """Calculates worst-group accuracy over the provided subgroup labels."""
  if subgroup_labels is None:
    return float(sklearn.metrics.accuracy_score(y_true, y_pred))

  groups = np.unique(subgroup_labels)
  if len(groups) == 0:
    return float(sklearn.metrics.accuracy_score(y_true, y_pred))

  accuracies = []
  for g in groups:
    mask = subgroup_labels == g
    if np.any(mask):
      acc = sklearn.metrics.accuracy_score(y_true[mask], y_pred[mask])
      accuracies.append(acc)

  if not accuracies:
    return 0.0
  return float(np.min(accuracies))


def mean_absolute_percentage_error(
    y_true: np.ndarray, y_pred: np.ndarray
) -> float:
  """Calculates MAPE, handling cases where y_true contains zeros."""
  y_true, y_pred = np.array(y_true), np.array(y_pred)
  # Create a mask to avoid division by zero
  mask = y_true != 0
  if not np.all(mask):
    # Calculate error only for non-zero true values to prevent errors
    if not np.any(mask):
      return 0.0 if np.allclose(y_pred, 0) else 100.0
    return float(
        np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    )
  return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
  """Calculates Root Mean Squared Error."""
  return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def wasserstein_distance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
  """Calculates the Wasserstein distance between the two distributions."""
  return scipy.stats.wasserstein_distance(y_true, y_pred)


MASE_SCALE_AUX_KEY = 'mase_scale'


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    seasonal_scale: np.ndarray | float | None = None,
) -> float:
  """Mean Absolute Scaled Error (Hyndman & Koehler, 2006).

  MASE = mean(|y - y_hat| / q), where q is the in-sample MAE of the seasonal
  naive forecast on the TRAINING split, supplied per row. Because the scorer
  flattens predictions to 1-D, the result is the point-weighted mean of the
  per-series MASE values (equal to the unweighted mean when every series
  contributes the same number of evaluation points).

  Degenerate denominators are not clamped: q == 0 (constant training series)
  yields 0.0 when the forecast is exact and +inf otherwise, which scoring maps
  to the worst contribution.

  Args:
    y_true: Ground-truth values.
    y_pred: Predicted values, same shape as y_true.
    seasonal_scale: Per-row (or scalar) q. REQUIRED.

  Returns:
    The MASE value, possibly +inf; NaN for empty input.

  Raises:
    ValueError: If seasonal_scale is missing or has an incompatible size.
  """
  if seasonal_scale is None:
    raise ValueError(
        'mase requires `seasonal_scale`, the in-sample seasonal-naive MAE of'
        f' the training split. Supply it as auxiliary ground truth'
        f" '{MASE_SCALE_AUX_KEY}'."
    )
  abs_err = np.abs(
      np.asarray(y_true, dtype=float).reshape(-1)
      - np.asarray(y_pred, dtype=float).reshape(-1)
  )
  scale = np.asarray(seasonal_scale, dtype=float).reshape(-1)
  if scale.size == 1:
    scale = np.full(abs_err.shape, scale.item())
  elif scale.size != abs_err.size:
    raise ValueError(
        f'seasonal_scale has {scale.size} entries but {abs_err.size} were'
        ' expected (one per predicted value, or a single scalar).'
    )
  if abs_err.size == 0:
    return float('nan')
  with np.errstate(divide='ignore', invalid='ignore'):
    ratio = np.where(
        scale > 0.0,
        abs_err / scale,
        np.where(abs_err == 0.0, 0.0, np.inf),
    )
  return float(np.mean(ratio))


# Normalization Functions
# Used to scale any metric so 0 is worst ~1 is best. For weighted overall_score
# Shift correlation from [-1, 1] to [0, 1]
correlation_normalize = lambda x: (x + 1.0) / 2.0
# Larger errors result in scores closer to 0, max score of 1.
inverse_normalize = lambda x: 1.0 / (1.0 + max(0.0, x))
# Clip negative values to 0 (e.g., for r2).
clip_negative = lambda x: max(0.0, x)


@dataclasses.dataclass(frozen=True)
class Metric:
  """Configuration and computation for a single metric."""

  name: str
  description: str
  higher_is_better: bool
  normalize: Callable[[float], float] | None
  func: Callable[..., float] | None = None

  @property
  def goal_string(self) -> str:
    """Returns a string indicating if higher or lower values are better."""
    return 'Higher is better' if self.higher_is_better else 'Lower is better'

  def compute(self, *args, **kwargs) -> float:
    """Computes the raw metric value."""
    if self.func is None:
      raise TypeError(
          f"Metric '{self.name}' is pre-computed and has no computation"
          ' function.'
      )
    return self.func(*args, **kwargs)

  def normalize_score(self, raw_value: float) -> float:
    """Normalizes the raw metric value to a score between 0 and 1."""
    if self.normalize is None:
      return float(raw_value)
    score = self.normalize(raw_value)
    return float(score)


# Metrics for classification tasks where `binary_threshold` can be used.
BINARY_CLASSIFICATION_METRICS: Sequence[str] = (
    'accuracy',
    'worst_group_accuracy',
    'balanced_accuracy',
    'specificity_binary',
    'recall_binary',
    'precision_binary',
    'f1_binary',
    'auroc',
    'auprc',
)

# All metrics supported by the scoring library.
ALL_METRICS_CONFIG: dict[str, Metric] = {
    'worst_group_accuracy': Metric(
        name='worst_group_accuracy',
        func=worst_group_accuracy,
        description='Worst-group Accuracy',
        higher_is_better=True,
        normalize=None,
    ),
    'accuracy': Metric(
        name='accuracy',
        func=accuracy_score,
        description='Accuracy',
        higher_is_better=True,
        normalize=None,  # Range [0, 1]. Non-negative by definition.
    ),
    'cohen_kappa': Metric(
        name='cohen_kappa',
        func=cohen_kappa_score,
        description="Cohen's Kappa",
        higher_is_better=True,
        normalize=clip_negative,  # Range [-1, 1]; below chance clips to 0.
    ),
    'balanced_accuracy': Metric(
        name='balanced_accuracy',
        func=balanced_accuracy_score,
        description='Balanced Accuracy',
        higher_is_better=True,
        normalize=None,  # Range [0, 1]. Non-negative by definition.
    ),
    'f1_binary': Metric(
        name='f1_binary',
        func=f1_binary,
        description='F1 Score (Binary)',
        higher_is_better=True,
        normalize=None,  # Range [0, 1]. Non-negative by definition.
    ),
    'precision_binary': Metric(
        name='precision_binary',
        func=precision_binary,
        description='Precision (Binary)',
        higher_is_better=True,
        normalize=None,  # Range [0, 1]. Non-negative by definition.
    ),
    'recall_binary': Metric(
        name='recall_binary',
        func=recall_binary,
        description='Recall (Binary)',
        higher_is_better=True,
        normalize=None,  # Range [0, 1]. Non-negative by definition.
    ),
    'f1_macro': Metric(
        name='f1_macro',
        func=f1_macro,
        description='F1 Score (Macro)',
        higher_is_better=True,
        normalize=None,  # Range [0, 1]. Non-negative by definition.
    ),
    'precision_macro': Metric(
        name='precision_macro',
        func=precision_macro,
        description='Precision (Macro)',
        higher_is_better=True,
        normalize=None,  # Range [0, 1]. Non-negative by definition.
    ),
    'recall_macro': Metric(
        name='recall_macro',
        func=recall_macro,
        description='Recall (Macro)',
        higher_is_better=True,
        normalize=None,  # Range [0, 1]. Non-negative by definition.
    ),
    'f1_weighted': Metric(
        name='f1_weighted',
        func=f1_weighted,
        description='F1 Score (Weighted)',
        higher_is_better=True,
        normalize=None,  # Range [0, 1]. Non-negative by definition.
    ),
    'precision_weighted': Metric(
        name='precision_weighted',
        func=precision_weighted,
        description='Precision (Weighted)',
        higher_is_better=True,
        normalize=None,  # Range [0, 1]. Non-negative by definition.
    ),
    'recall_weighted': Metric(
        name='recall_weighted',
        func=recall_weighted,
        description='Recall (Weighted)',
        higher_is_better=True,
        normalize=None,  # Range [0, 1]. Non-negative by definition.
    ),
    'mae': Metric(
        name='mae',
        func=mean_absolute_error,
        description='MAE (Mean Absolute Error)',
        higher_is_better=False,
        # Sum of absolute differences is non-negative. inverse_normalize
        # maps to [0, 1] and applies clip_negative internally.
        normalize=inverse_normalize,
    ),
    'mse': Metric(
        name='mse',
        func=mean_squared_error,
        description='MSE (Mean Squared Error)',
        higher_is_better=False,
        # Sum of squared differences is non-negative. inverse_normalize
        # maps to [0, 1] and applies clip_negative internally.
        normalize=inverse_normalize,
    ),
    'rmse': Metric(
        name='rmse',
        func=root_mean_squared_error,
        description='RMSE (Root Mean Squared Error)',
        higher_is_better=False,
        # Square root of non-negative MSE is non-negative. inverse_normalize
        # maps to [0, 1] and applies clip_negative internally.
        normalize=inverse_normalize,
    ),
    'mape': Metric(
        name='mape',
        func=mean_absolute_percentage_error,
        description='MAPE (Mean Absolute Percentage Error)',
        higher_is_better=False,
        # Percentage error from absolute differences is non-negative.
        # inverse_normalize maps to [0, 1] and applies clip_negative internally.
        normalize=inverse_normalize,
    ),
    'mase': Metric(
        name='mase',
        func=mase,
        description=(
            'MASE (Mean Absolute Scaled Error vs. the in-sample seasonal naive'
            ' forecast)'
        ),
        higher_is_better=False,
        # Non-negative ratio. inverse_normalize maps to (0, 1]; MASE ~ 1 for a
        # seasonal-naive forecast, so it lands in the responsive band.
        normalize=inverse_normalize,
    ),
    'r2': Metric(
        name='r2',
        func=r2_score,
        description='R2 (R-squared)',
        higher_is_better=True,
        normalize=clip_negative,
    ),
    'pearson': Metric(
        name='pearson',
        func=pearson_correlation,
        description='Pearson Correlation',
        higher_is_better=True,
        normalize=correlation_normalize,
    ),
    'spearman': Metric(
        name='spearman',
        func=spearman_correlation,
        description='Spearman Correlation',
        higher_is_better=True,
        normalize=correlation_normalize,
    ),
    'kendall': Metric(
        name='kendall',
        func=kendall_correlation,
        description="Kendall's Tau Correlation",
        higher_is_better=True,
        # Correlations are [-1, 1]. correlation_normalize maps to [0, 1].
        normalize=correlation_normalize,
    ),
    'wasserstein_distance': Metric(
        name='wasserstein_distance',
        func=wasserstein_distance,
        description='Wasserstein Distance',
        higher_is_better=False,
        # Distance metrics are non-negative. inverse_normalize maps to [0, 1]
        # and applies clip_negative internally.
        normalize=inverse_normalize,
    ),
    'exec_time': Metric(
        name='exec_time',
        func=None,  # Pre-computed
        description='Time taken to execute generated code.',
        higher_is_better=False,
        normalize=inverse_normalize,
    ),
    'overall_score': Metric(
        name='overall_score',
        func=None,  # Computed externally
        description='Overall score is a combination of other metrics.',
        higher_is_better=True,  # Always True for this convention.
        normalize=clip_negative,
    ),
    'specificity_binary': Metric(
        name='specificity_binary',
        func=specificity_binary,
        description='Specificity (Binary)',
        higher_is_better=True,
        normalize=None,  # Range [0, 1]. Non-negative by definition.
    ),
    'auroc': Metric(
        name='auroc',
        func=auroc,
        description='AUROC (Area Under ROC Curve)',
        higher_is_better=True,
        normalize=None,  # Range [0, 1]. Non-negative by definition.
    ),
    'auprc': Metric(
        name='auprc',
        func=auprc,
        description='AUPRC (Area Under Precision-Recall Curve)',
        higher_is_better=True,
        normalize=None,  # Range [0, 1]. Non-negative by definition.
    ),
}


def register_metric(metric: Metric) -> None:
  """Registers a metric in ALL_METRICS_CONFIG."""
  ALL_METRICS_CONFIG[metric.name] = metric
