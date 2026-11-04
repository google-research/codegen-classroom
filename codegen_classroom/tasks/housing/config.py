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

"""Shared task configuration and student prompts for California Housing regression.

Both `run_housing.py` and `housing.ipynb` import constants and prompt
builders from this module so configuration and instructions stay in sync.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import importlib
import re

from codegen_classroom import scoring

# ==============================================================================
# === 1. TASK CONFIGURATION ===
# ==============================================================================

DEFAULT_STUDENT_MODELS: Sequence[str] = (
    "gemini-3.5-flash",
    "gemini-3.1-pro-preview",
    "gemini-3.5-flash-lite",
    "gemma-4-26b-a4b-it",
    "gemma-4-31b-it",
    "gemini-4-argon",
)

TASK_NAME = "housing"
TASK_TITLE = "California Housing Price Regression"
STUDENT_FUNCTION_NAME = "solution"
VAL_SEED = 42

RUN_DEFAULTS = {
    "num_students": 6,
    "num_cycles": 12,
    "early_stopping_patience": 6,
    "max_mutation_count": 6,
    "timeout_s": 60,
    "final_eval_timeout_s": 180,
    "max_output_tokens": 32768,
}

DEFAULT_METRIC_WEIGHTS: Mapping[str, float] = {
    "mape": 0.5,
    "r2": 1.0,
    "pearson": 0.5,
    "exec_time": 0.01,
}

DEFAULT_FEATURE_NAMES: tuple[str, ...] = (
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
    "ocean_proximity_INLAND",
    "ocean_proximity_ISLAND",
    "ocean_proximity_NEAR_BAY",
    "ocean_proximity_NEAR_OCEAN",
)

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
    "1. Objective",
    "2. Data Contract",
    "3. Evaluation",
    "4. Environment",
    "5. Principles",
    "6. Pitfalls",
    "7. Iteration Discipline",
    "8. Output Contract",
)

# ==============================================================================
# === 2. STUDENT PROMPTS ===
# ==============================================================================

STUDENT_INSTRUCTIONS_TEMPLATE = """# Task: {task_title}

## 1. Objective

Predict `{target_column}` for 1990 California census blocks.

Score combines weighted R² (primary, weight {r2_w:.2f}), MAPE ({mape_w:.2f}), Pearson correlation ({pearson_w:.2f}), and a tiny execution-time penalty term.

You need **both** good ranking/variance explanation (R², Pearson) and low *relative* percentage error (MAPE). These pull in slightly different directions on a right-skewed, top-coded target ($500,000 ceiling).

Metrics summary:
{metrics_desc}

## 2. Data Contract

- **Inputs:**
  * `X_train` (pd.DataFrame): Training feature matrix ({train_rows:,} rows).
  * `y_train` (pd.DataFrame): Training target labels with single column `'{target_column}'`.
  * `X_val` (pd.DataFrame): Validation feature matrix ({val_rows:,} rows).
- **Features ({num_features} columns):**
  {features_list_str}
  (Note: categorical ocean proximity is pre-encoded as binary indicator columns; all features are numeric float/int/bool).
- **Target:** `{target_column}` (numeric dollar value, right-skewed, capped at $500,000).
- **Output:** A tuple `(y_pred_df, feedback)`:
  * `y_pred_df` (pd.DataFrame): Must have identical index to `X_val.index` and a single column named `'predictions'` of float dtype.
  * `feedback` (str): Mandatory experiment log.

**Required Feedback Format:**
Your function returns `(y_pred_df, feedback)`. The `feedback` string is your experiment log and will be provided back to you in the next cycle's Performance Feedback. It MUST contain 3 to 6 concise lines covering:
- What changed vs. previous cycle (model family, preprocessing pipeline, and lever applied)
- Internal CV or holdout validation R², MAPE, and Pearson correlation
- Wall-clock time used and fitting iterations/epochs completed vs. planned
- Model complexity summary (e.g., estimator family, feature count, or regularization)
Do NOT leave feedback empty; without your own diagnostic notes, subsequent cycles have only scalar metric values to guide refinement.

## 3. Evaluation

- **Validation Protocol:** Random train/validation split (no group structure).
- **Per-Cycle Wall-Clock Budget:** **{timeout_s} s** (hard execution timeout) on {train_rows:,} training rows.
- **Final Evaluation Budget:** **{final_eval_timeout_s} s** when retrained on the combined learning set ({final_eval_rows:,} rows).
- **Budgeting Strategy:** Time your first iteration or epoch, then plan iterations by a deadline with at least a 15% safety margin. Do not train by fixed high iteration counts that risk timing out. A timeout scores 0 for that cycle.
- **Hard Resource Caps:**
  * Single core CPU execution (set n_jobs=1 or threads=1)
  * Peak memory must not exceed host sandbox limits

## 4. Environment

- **Importable Libraries:** {imp_str}.
- **Forbidden Libraries:** {forb_str} (do NOT import or use; will raise ModuleNotFoundError or cause validator failure).
- **Network Access:** No internet access; no downloading pretrained model weights or external files during execution.
- **Single Code Block:** Provide exactly one complete python code block containing all imports, helpers, and the `solution` function.
- Always fill missing values on a copy: `df['col'] = df['col'].fillna(val)` instead of `inplace=True`.
- Ensure thread/worker parameters are restricted (e.g. `n_jobs=1`) to avoid process stalls.

## 5. Principles

- **Candidate Models:**
  * Tree-Based Models: HistGradientBoostingRegressor, LightGBM (LGBMRegressor), ExtraTreesRegressor, and RandomForestRegressor are well-suited for tabular census data with complex non-linear splits.
  * Linear and Kernel Models: Regularized linear models (Ridge, Lasso, ElasticNet) and Kernel Ridge with spatial basis expansions provide fast, interpretable baselines with low variance.
  * Neural Networks: Multi-Layer Perceptrons (MLPRegressor) with normalized features can capture smooth spatial manifolds when combined with non-linear activations.
  * **Model Architecture:** Fit a well-tuned model for your final predictions rather than blending or stacking multiple models, model families, or multi-loss heads. Invest your iteration budget in feature representation, spatial context, target/loss alignment, and regularization.
- **Spatial Feature Engineering:** House values vary continuously with geographic coordinates. Decision trees split on axis-aligned cuts; providing derived spatial representations—such as distances or bearings to major population centers or coastal boundaries, rotated coordinate projections, unsupervised spatial cluster structure, or smooth basis expansions over latitude and longitude—helps models capture regional sub-markets.
- **Local Spatial Context:** Housing prices exhibit strong local spatial autocorrelation. Summarizing local neighborhood statistics (such as localized target or socio-economic distributions computed strictly out-of-fold on training data to prevent self-leakage) provides fine-grained neighborhood context beyond global coordinates.
- **Density Ratios:** Raw census counts (total rooms, bedrooms, population, households) carry stronger density and living-condition signals when expressed as normalized per-household, per-room, or per-capita relationships, with robust handling of extreme ratio outliers.
- **Target Transformations:** The target is right-skewed and top-coded at $500,000. Evaluating target transformations (e.g. logarithmic or power scaling), sample weighting, or loss objectives alongside lightweight post-processing calibration and domain-boundary clipping—balances variance explanation (R²) and relative percentage error (MAPE) without multi-model ensembling.
- **Peer Collaboration:** When learning from peer solutions, synthesize complementary feature discoveries (e.g. combining spatial representations with socioeconomic ratios) into a model rather than copying scripts wholesale or stacking peers' models together. Avoid ensemble bloat and validation overfitting.
- **Metric Trade-offs:**
  * R² (weight 1.0) measures overall explained variance and penalizes large dollar residuals.
  * MAPE (weight 0.5) measures relative percentage error, heavily penalizing misses on affordable/lower-income census blocks.
  * Pearson correlation (weight 0.5) evaluates monotonic ordering across the price spectrum.
  * If R² improves but MAPE degrades, investigate whether extreme predictions skew relative errors on lower-value blocks, and adjust your target transform, sample weights, or post-prediction calibration.

## 6. Pitfalls

- **Multi-Model Ensembles & Blending:** Do not build multi-model ensembles, multi-head blends, or stacked estimators; keep your pipeline focused so each cycle tests a clear feature, representation, or calibration hypothesis and generalizes cleanly without overfitting.
- **Column Name Matching:** Use exact column names from `X_train.columns`. Do not append datatype strings or assume extra columns exist.
- **Exhaustive Grid Searches:** The per-cycle budget is tight; do not run extensive GridSearchCV or heavy random sweeps that cause timeouts.
- **Prediction Range:** Predictions must be in a DataFrame with column `'predictions'`. Never return all NaNs, constant values, negative prices, or unclipped extreme values.
- **Modifying Inputs In-Place:** Never mutate `X_train` or `X_val` in-place; work on explicit copies.
- **Multiprocessing Overheads:** Multi-threaded fitting (`n_jobs=-1`) can suffer severe contention in containerized evaluation environments; prefer `n_jobs=1`.

## 7. Iteration Discipline

- **Cycle 0 Baseline:** Build the simplest correct model baseline that returns a valid prediction dataframe in under half the time budget. Prioritize a working end-to-end pipeline before tuning.
- **One Change at a Time:** Once you have a working baseline, change exactly ONE conceptual component per cycle (e.g. one feature family, one regularization sweep, or one loss formulation).
- **Keep-Best-So-Far:** Track your best score. If a modification degrades performance or adds excessive latency without accuracy gain, revert to your best-so-far solution on the following cycle.
- **Use the Feedback String:** Record your validation findings in the returned feedback string. Use it as an experiment log across cycles.
- **Focus on the Primary Lever:** Spend compute on the lever with the highest expected gain for the primary objective. Avoid architectural rewrites when the headroom is small.
- **Pre-Return Check:** Verify your output dataframe against the checklist in Section 8 before returning.
- **Concise Model Summary:** Start your response with a concise 1-line model/technique summary rather than conversational filler.

## 8. Output Contract

- [ ] Function signature is exactly `solution(X_train, y_train, X_val)` and returns a tuple `(y_pred_df, feedback)`.
- [ ] `y_pred_df` is a pandas DataFrame with index identical to `X_val.index`.
- [ ] `y_pred_df` has exactly one column named `'predictions'` with dtype `float`.
- [ ] `y_pred_df` contains no `NaN`, `None`, or infinite values.
- [ ] Predictions are non-trivial (not a constant value or single class).
- [ ] Total execution completes comfortably within the {timeout_s} s per-cycle limit and scales to {final_eval_timeout_s} s on final evaluation.
- [ ] Predictions are strictly finite floats and within realistic housing price bounds ($10,000 to $500,000+).
- [ ] Column names match `X_train.columns` exactly without syntax errors or missing keys.
- [ ] Feedback string contains the required items (change, validation score, elapsed wall-clock, size).
"""

STUDENT_INITIAL_PROMPT = (
    "Provide the first `solution`: a fast, clean baseline"
    " regression model on the raw columns with a correct output contract"
    " `(y_pred_df, feedback)` and the required feedback items; nothing more."
)

STUDENT_REFINE_INSTRUCTION_OPTIONS: list[str] = [
    (
        "Spatial Representations: Decision trees split on"
        " axis-aligned coordinates. Explore continuous spatial representations"
        " such as distances or projections relative to major economic hubs and"
        " coastline geometry, rotated coordinate axes, unsupervised coordinate"
        " clustering, or smooth spatial basis functions, recording R² and MAPE"
        " in feedback."
    ),
    (
        "Density Ratios: Derive normalized"
        " socioeconomic and housing density relationships (such as"
        " per-household or per-room ratios and income-to-housing interactions)"
        " with outlier handling to capture localized living conditions."
    ),
    (
        "Model Exploration: Compare distinct model"
        " families—such as HistGradientBoostingRegressor, LGBMRegressor,"
        " ExtraTreesRegressor, Multi-Layer Perceptrons (MLPRegressor), or"
        " regularized polynomial linear models—to determine which standalone"
        " estimator generalizes best across California census tracts without"
        " multi-model blending."
    ),
    (
        "Target Transformations: California housing values are"
        " right-skewed and top-coded at $500,000. Explore log-space or"
        " power-space target transforms, sample weighting, or domain-boundary"
        " clipping to stabilize residual variance"
        " and balance R² vs MAPE."
    ),
    (
        "Neighborhood Aggregations: Compute local spatial"
        " neighborhood summaries (e.g. k-nearest neighbors' socio-economic or"
        " target statistics computed strictly out-of-fold on training data to"
        " prevent self-leakage) to capture fine-grained geographic context."
    ),
    (
        "Non-Linear Basis Expansions: Test spline or polynomial basis"
        " expansions on continuous predictors like median_income,"
        " housing_median_age, or geographic coordinates to model smooth"
        " non-linear valuation gradients."
    ),
    (
        "Residual Error Analysis: Use the feedback string to"
        " inspect census tracts with the largest prediction residuals."
        " Identify whether errors concentrate in extreme income brackets,"
        " top-coded tracts, rural coastal areas, or high-density urban"
        " clusters, and propose corrective features for your model."
    ),
    (
        "Feature Selection & Regularization: Prune redundant or noisy"
        " interaction terms, tune tree depth, leaf regularization, and"
        " learning schedule, and verify that validation improvements come from"
        " a parsimonious model that resists overfitting."
    ),
    (
        "Loss Alignment & Post-Processing Calibration: Compare"
        " squared-error, absolute-error, and Huber or power-scaled target"
        " formulations alongside lightweight post-processing calibration on"
        " out-of-fold predictions to jointly balance R² and MAPE within a"
        " model."
    ),
    (
        "Peer Crossover: Review peer feedback notes in"
        " Performance Feedback. Integrate strong feature discoveries (e.g."
        " spatial context or ratio engineering) into your model"
        " architecture rather than ensembling or blending multiple models."
    ),
]

_FORBIDDEN_RECIPE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(lr|learning_rate)\s*=\s*[0-9.e-]+", re.IGNORECASE),
    re.compile(
        r"\b(max_iter|n_estimators|max_depth)\s*=\s*[0-9]+", re.IGNORECASE
    ),
    re.compile(r"\b(C|alpha|l2_regularization)\s*=\s*[0-9.]+", re.IGNORECASE),
    re.compile(r"\b0\.\d+\s*\*\s*\w*(?:_pred|pred)\b"),
    re.compile(r"\bweights\s*=\s*\[[0-9.,\s]+\]", re.IGNORECASE),
    re.compile(r"\bthreshold\s*=\s*0\.[0-9]+", re.IGNORECASE),
)


def assert_no_recipes(prompt: str) -> None:
  """Verifies that prompts provide principles rather than hyperparameter recipes."""
  violations = []
  for pat in _FORBIDDEN_RECIPE_PATTERNS:
    m = pat.search(prompt)
    if m:
      violations.append(f"Pattern '{pat.pattern}' matched: '{m.group(0)}'")
  if violations:
    raise AssertionError(
        "Prompt contains concrete recipe / hyperparameter leaks:\n"
        + "\n".join(f"  - {v}" for v in violations)
    )


def probe_available_modules(
    candidates: Sequence[str] = CANDIDATE_MODULES,
) -> tuple[str, ...]:
  """Probes which candidate libraries are genuinely importable in this environment."""
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
    feature_names: Sequence[str] = DEFAULT_FEATURE_NAMES,
    target_column: str = "median_house_value",
    train_rows: int = 13076,
    val_rows: int = 3270,
    final_eval_rows: int = 16346,
    timeout_s: int = RUN_DEFAULTS["timeout_s"],
    final_eval_timeout_s: int = RUN_DEFAULTS["final_eval_timeout_s"],
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
      r2_w=metric_weights.get("r2", 1.0),
      mape_w=metric_weights.get("mape", 0.5),
      pearson_w=metric_weights.get("pearson", 0.5),
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
