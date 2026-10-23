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

"""Shared task configuration and student prompts for the template task.

Both `run_template.py` and `task_example.ipynb` import constants and prompt
builders from this module so configuration and instructions stay in sync.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import importlib
import re

from codegen_classroom import scoring
from codegen_classroom.tasks.template import data_loader

# ==============================================================================
# === 1. TASK CONFIGURATION (CUSTOMIZE: Update for your benchmark task) ===
# ==============================================================================

DEFAULT_STUDENT_MODELS: Sequence[str] = (
    "gemini-3.5-flash",
    "gemini-3.1-pro-preview",
    "gemini-3.5-flash-lite",
    "gemma-4-26b-a4b-it",
    "gemma-4-31b-it",
    "gemini-4-argon",
)

TASK_NAME = "template"
TASK_TITLE = "Multi-Regime Thermal Load Regression (Template Task)"
STUDENT_FUNCTION_NAME = "solution"
VAL_SEED = 42

RUN_DEFAULTS = {
    "num_students": 3,
    "num_cycles": 8,
    "timeout_s": 60,
    "final_eval_timeout_s": 180,
    "patience": 2,
    "max_mutations": 2,
    "max_output_tokens": 32768,
}

# CUSTOMIZE: Define metric weights matching `metrics.ALL_METRICS_CONFIG`.
DEFAULT_METRIC_WEIGHTS: Mapping[str, float] = {
    "r2": 0.6,
    "mae": 0.4,
    "exec_time": 0.01,
}
METRIC_WEIGHTS: Mapping[str, float] = DEFAULT_METRIC_WEIGHTS

CANDIDATE_MODULES: tuple[str, ...] = (
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "lightgbm",
)

FORBIDDEN_MODULES: tuple[str, ...] = (
    "xgboost",
    "torch",
    "tensorflow",
    "jax",
)

SECTION_TITLES: tuple[str, ...] = (
    "1. Objective & What Is Rewarded",
    "2. Data Contract",
    "3. Evaluation & Budget",
    "4. Environment",
    "5. Principles & Levers",
    "6. Pitfalls",
    "7. Iteration Discipline",
    "8. Output Contract Checklist",
)

# ==============================================================================
# === 2. STUDENT PROMPTS & INSTRUCTIONS ===
# ==============================================================================

# CUSTOMIZE: Edit the 8-section Markdown template below for your task domain.
STUDENT_INSTRUCTIONS_TEMPLATE = """# Task: {task_title}

## 1. Objective & What Is Rewarded

Predict `{target_column}` across multi-regime thermal and workload telemetry observations.

Your goal is to build an accurate, well-calibrated regression pipeline. Overall score is a weighted combination of Coefficient of Determination (R², weight {r2_w:.2f}), Mean Absolute Error (MAE, weight {mae_w:.2f}), and execution time (weight {exec_time_w:.2f}).

Metrics summary:
{metrics_desc}

## 2. Data Contract

- **Inputs:**
  * `X_train` (pd.DataFrame): Training feature matrix ({train_rows:,} rows).
  * `y_train` (pd.DataFrame): Training target labels with single column `'{target_column}'`.
  * `X_val` (pd.DataFrame): Validation feature matrix ({val_rows:,} rows).
- **Features ({num_features} columns):** {features_list_str}
  * Includes continuous telemetry measurements, discrete state indicators, and partially missing sensor readings.
- **Target:** `{target_column}` (continuous float).
- **Output:** Return `(y_pred_df, feedback)`:
  * `y_pred_df` (pd.DataFrame): Index matching `X_val.index`, single float column named `'predictions'`.
  * `feedback` (str): 3 to 6 concise lines logging what changed, internal CV R² and MAE, and diagnostic observations.

## 3. Evaluation & Budget

- **Validation Protocol:** Hold-out validation during cycles; use internal K-fold cross-validation on `X_train` to guide feature selection and model blending.
- **Per-Cycle Wall-Clock Budget:** **{timeout_s} s** on {train_rows:,} training rows.
- **Final Evaluation Budget:** **{final_eval_timeout_s} s** on {final_eval_rows:,} combined learning rows.
- **Hard Resource Caps:** Single-threaded execution preferred (`n_jobs=1`); predictions must be finite 1D floats in column `'predictions'`.

## 4. Environment

- **Importable Libraries:** {imp_str}.
- **Forbidden / Unavailable:** {forb_str}.
- **Network & Pretrained Weights:** No network access during execution.
- **Single Code Block:** Provide one complete Python block defining `solution(X_train, y_train, X_val)`.

## 5. Principles & Levers

- **Regime-Conditioned Behavior:** Consider whether discrete operating states shift the baseline response or alter interactions among continuous variables, and how different model families represent state-dependent relationships.
- **Non-Linear & Periodic Structure:** Inspect predictor distributions for skewness, cyclical patterns, or saturation effects, and consider appropriate transformations or basis expansions.
- **Missingness & Signal Selection:** Handle missing entries cleanly without data leakage, and evaluate whether all input columns carry genuine out-of-fold signal versus noise.
- **Complementary Model Families:** Consider the inductive biases of piecewise-constant tree ensembles versus smooth function approximators, and how blending structurally distinct models can improve generalization.

## 6. Pitfalls

- **Index Alignment:** Returning a DataFrame whose index differs from `X_val.index` fails validation immediately.
- **Missing Value Handling:** Many linear and neural estimators fail on raw `NaN` entries unless imputed first.
- **Data Leakage:** Fit imputers, scalers, and feature selectors strictly on training folds before transforming validation data.
- **Unbounded Grid Searches:** Exhaustive hyperparameter grids risk exceeding the per-cycle timeout.

## 7. Iteration Discipline

- **Cycle 0 Baseline:** Start with a fast, verified baseline model to establish a clean benchmark score.
- **One Lever per Cycle:** Test one well-defined hypothesis per cycle (feature representation, model family, or ensembling) and retain changes only when internal CV improves.
- **Use the Feedback Log:** Record out-of-fold R² and MAE deltas in `feedback` so subsequent cycles build on verified gains.

## 8. Output Contract Checklist

- [ ] Signature is `solution(X_train, y_train, X_val)` returning `(y_pred_df, feedback)`.
- [ ] `y_pred_df` is a `pd.DataFrame` with index identical to `X_val.index` and one float column `'predictions'`.
- [ ] `y_pred_df` has no `NaN`, `None`, or infinite values, and variance > 0.
- [ ] Execution completes well within {timeout_s} s (and scales to {final_eval_timeout_s} s on {final_eval_rows:,} rows).
- [ ] `feedback` logs the hypothesis tested, internal CV metrics, and runtime.
"""

STUDENT_INITIAL_PROMPT = (
    "Start with a concise 1-2 sentence plan summarizing your approach, then"
    " provide the Cycle 0 `solution(X_train, y_train, X_val)`: build a clean,"
    " fast baseline model with cross-validation logging in `feedback` and"
    " return `(y_pred_df, feedback)`."
)

STUDENT_REFINE_INSTRUCTION_OPTIONS: list[str] = [
    (
        "Regime & Interaction Exploration: Investigate how continuous"
        " predictors interact with one another and whether discrete operating"
        " states alter those relationships. Validate engineered features using"
        " out-of-fold cross-validation."
    ),
    (
        "Distributional & Periodic Transformations: Examine predictor"
        " distributions for skewness, cyclical behavior, or non-linear"
        " saturation, and test appropriate mathematical transformations."
    ),
    (
        "Complementary Model Families: Compare tree-based ensembles against"
        " smooth function approximators (such as regularized linear/kernel"
        " models or neural networks) on properly imputed and scaled features."
    ),
    (
        "Feature Relevance & Out-of-Fold Blending: Audit feature importance to"
        " prune uninformative variables, and explore combining predictions"
        " from structurally diverse models using out-of-fold validation."
    ),
]

_FORBIDDEN_RECIPE_PATTERNS = (
    re.compile(r"\b(lr|learning_rate)\s*=\s*[0-9.e-]+", re.IGNORECASE),
    re.compile(
        r"\b(max_iter|n_estimators|max_depth)\s*=\s*[0-9]+", re.IGNORECASE
    ),
    re.compile(r"\b(C|alpha|l2_regularization)\s*=\s*[0-9.]+", re.IGNORECASE),
    re.compile(r"\b0\.\d+\s*\*\s*\w*(?:_pred|pred)\b"),
)


def assert_no_recipes(text: str) -> None:
  """Verifies that prompts provide principles rather than hyperparameter recipes."""
  for pattern in _FORBIDDEN_RECIPE_PATTERNS:
    match = pattern.search(text)
    if match:
      raise AssertionError(
          "Forbidden recipe or hyperparameter assignment found:"
          f" {match.group(0)!r}"
      )


def probe_available_modules(
    candidates: Sequence[str] = CANDIDATE_MODULES,
) -> tuple[str, ...]:
  """Probes which candidate libraries are importable in the runtime."""
  available = []
  for module_name in candidates:
    try:
      importlib.import_module(module_name)
      available.append(module_name)
    except (ImportError, ModuleNotFoundError):
      continue
  return tuple(available)


def build_student_instructions(
    *,
    feature_names: Sequence[str] = data_loader.FEATURE_COLUMNS,
    target_column: str = data_loader.TARGET_COLUMN,
    train_rows: int = 960,
    val_rows: int = 240,
    final_eval_rows: int = 1200,
    timeout_s: int = 60,
    final_eval_timeout_s: int = 180,
    metric_weights: Mapping[str, float] | None = None,
    available_modules: Sequence[str] | None = None,
) -> str:
  """Formats `STUDENT_INSTRUCTIONS_TEMPLATE` with dataset and runtime parameters."""
  if metric_weights is None:
    metric_weights = DEFAULT_METRIC_WEIGHTS
  if available_modules is None:
    available_modules = probe_available_modules()

  forbidden = list(FORBIDDEN_MODULES)
  if "lightgbm" not in available_modules:
    forbidden.append("lightgbm")

  return STUDENT_INSTRUCTIONS_TEMPLATE.format(
      task_title=TASK_TITLE,
      target_column=target_column,
      r2_w=metric_weights.get("r2", 0.6),
      mae_w=metric_weights.get("mae", 0.4),
      exec_time_w=metric_weights.get("exec_time", 0.01),
      metrics_desc=scoring.get_metrics_description(metric_weights).strip(),
      train_rows=train_rows,
      val_rows=val_rows,
      final_eval_rows=final_eval_rows,
      num_features=len(feature_names),
      features_list_str=", ".join(f"`{f}`" for f in feature_names),
      timeout_s=timeout_s,
      final_eval_timeout_s=final_eval_timeout_s,
      imp_str=", ".join(f"`{m}`" for m in available_modules),
      forb_str=", ".join(f"`{m}`" for m in forbidden),
  )


STUDENT_INSTRUCTIONS: str = build_student_instructions()
