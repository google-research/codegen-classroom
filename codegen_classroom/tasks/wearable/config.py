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

"""Shared task configuration and student prompts for Wearable Activity Classification.

Both `run_wearable.py` and `wearable.ipynb` import constants and prompt
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

TASK_NAME = "wearable"
TASK_TITLE = "Wearable Sensor Physical Activity Classification"
STUDENT_FUNCTION_NAME = "solution"
VAL_SEED = 42

RUN_DEFAULTS = {
    "num_students": 6,
    "num_cycles": 15,
    "early_stopping_patience": 3,
    "max_mutation_count": 8,
    "timeout_s": 60,
    "final_eval_timeout_s": 180,
    "max_output_tokens": 32768,
}

DEFAULT_METRIC_WEIGHTS: Mapping[str, float] = {
    "accuracy": 1.0,
    "f1_macro": 0.5,
    "exec_time": 0.01,
}

DEFAULT_FEATURE_NAMES: tuple[str, ...] = (
    "age",
    "height",
    "weight",
    "steps",
    "heart_rate",
    "calories",
    "distance",
    "entropy_heart",
    "entropy_steps",
    "resting_heart",
    "corr_heart_steps",
    "norm_heart",
    "intensity_karvonen",
    "sd_norm_heart",
    "steps_times_distance",
    "gender",
    "device_encoded",
)

CANDIDATE_MODULES: tuple[str, ...] = (
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
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

Classify physical activity type into 6 discrete intensity classes from one-row-per-sample wearable sensor summaries.

Score = accuracy ({accuracy_w:.2f}) + F1-macro ({f1_w:.2f}) - execution time penalty ({exec_w:.2f}).

**Objective Priority:** F1-macro rewards performance across all 6 classes equally, including low-frequency minority activities. Most importantly, **evaluation users are completely unseen during training**. What is rewarded is physiological generalization across different individuals, not memorizing specific participant characteristics.

Metrics summary:
{metrics_desc}

## 2. Data Contract

- **Inputs:**
  * `X_train` (pd.DataFrame): Training features ({train_rows:,} samples from {train_users} participants).
  * `y_train` (pd.DataFrame): Target class labels with single column `'activity_encoded'`.
  * `X_val` (pd.DataFrame): Validation features ({val_rows:,} samples from {val_users} unseen participants).
- **Features ({num_features} columns):**
| Feature | Description |
| :--- | :--- |
| `age` | Participant age in years |
| `gender` | Biological sex (0 = female, 1 = male) |
| `height` | Body height in centimeters |
| `weight` | Body weight in kilograms |
| `device_encoded` | Wearable device brand indicator (0 = Apple Watch, 1 = Fitbit) |
| `steps` | Total steps recorded in observation window |
| `heart_rate` | Mean heart rate (beats per minute) |
| `calories` | Estimated calories burned in window |
| `distance` | Estimated distance traversed |
| `resting_heart` | Estimated baseline resting heart rate |
| `norm_heart` | Heart rate normalized relative to resting rate |
| `intensity_karvonen` | Cardiovascular reserve intensity using Karvonen formula |
| `sd_norm_heart` | Standard deviation of normalized heart rate (variability) |
| `entropy_heart` | Approximate sample entropy of heart rate series |
| `entropy_steps` | Approximate sample entropy of step dynamics |
| `corr_heart_steps` | Correlation between heart rate and step rate |
| `steps_times_distance` | Interaction term between step count and distance |

- **Feature Grouping:**
  * **Demographics:** `age`, `gender`, `height`, `weight`
  * **Device Context:** `device_encoded` (0: Apple Watch, 1: Fitbit)
  * **Biometrics & Motion:** `steps`, `heart_rate`, `calories`, `distance`, `resting_heart`, `norm_heart`, `intensity_karvonen`, `sd_norm_heart`, `entropy_heart`, `entropy_steps`, `corr_heart_steps`, `steps_times_distance`

- **Target Classes (6 classes, integer IDs 0 to 5 from alphabetical `LabelEncoder`):**
| Class ID | Activity Label | Description |
| :---: | :--- | :--- |
| **0** | `Lying (resting / sedentary)` | Class index in `predictions` |
| **1** | `Running at 3 METs (light running / jog)` | Class index in `predictions` |
| **2** | `Running at 5 METs (moderate running)` | Class index in `predictions` |
| **3** | `Running at 7 METs (vigorous running)` | Class index in `predictions` |
| **4** | `Self-paced walking` | Class index in `predictions` |
| **5** | `Sitting (sedentary)` | Class index in `predictions` |

- **Output:** A tuple `(y_pred_df, feedback)`:
  * `y_pred_df` (pd.DataFrame): Must have identical index to `X_val.index` and a single column named `'predictions'` containing integer class IDs (0..5).
  * `feedback` (str): Mandatory experiment log.

**Required Feedback Format:**
Your function returns `(y_pred_df, feedback)`. The `feedback` string is your experiment log and will be provided back to you in the next cycle's Performance Feedback. It MUST contain 3 to 6 concise lines covering:
- What changed vs. previous cycle (model architecture, preprocessing, class balancing, or feature lever applied)
- Internal validation F1-macro and overall accuracy
- Per-class recall / error analysis across the 6 classes (especially worst-confused activities)
- Wall-clock execution time used and iterations/models completed
Do NOT leave feedback empty; without your own diagnostic notes, subsequent cycles have only scalar metric values to guide refinement.

## 3. Evaluation

- **Validation Protocol:** Evaluation splits hold out unseen participants: validation and test observations belong to individuals not present in `X_train`, so naive random row splits overestimate cross-person generalization. Design your internal validation strategy to respect participant separation and estimate true cross-subject performance.
- **Inductive Inference:** Each row in `X_val` represents an independent sensor observation that must be predicted without referencing other rows in `X_val`. All feature scalers, imputers, and encoders must be fit on `X_train` and applied via `.transform(X_val)`. Never group `X_val` by participant or compute per-participant statistics across multiple rows of `X_val`.
- **Per-Cycle Wall-Clock Budget:** **{timeout_s} s** (hard execution timeout) on {train_rows:,} training rows.
- **Final Evaluation Budget:** **{final_eval_timeout_s} s** when retraining on the combined learning set ({final_eval_rows:,} rows across {final_eval_users} participants).
- **Budgeting Strategy:** Time your first iteration or epoch, then plan iterations by a deadline with at least a 15% safety margin. Do not train by fixed high iteration counts that risk timing out. A timeout scores 0 for that cycle.
- **Hard Resource Caps:**
  * Single core CPU execution (set n_jobs=1 or threads=1)
  * Predictions must be integers in range [0, 5]

## 4. Environment

- **Importable Libraries:** {imp_str}.
- **Forbidden Libraries:** {forb_str} (do NOT import or use; will raise ModuleNotFoundError or cause validator failure).
- **Network Access:** No internet access; no downloading pretrained model weights or external files during execution.
- **Single Code Block:** Provide exactly one complete python code block containing all imports, helpers, and the `solution` function.
- scikit-learn and NumPy are fully supported. Avoid deprecated or removed scikit-learn keyword arguments.
- Do NOT attempt to implement neural networks or gradient boosting from scratch in pure NumPy/JAX; standard library implementations are 100x faster and far more accurate.
- Restrict parallel threads (`n_jobs=1`) to avoid execution timeout in containerized evaluation environments.

## 5. Principles

- **Candidate Models:**
  * Gradient-boosted decision trees, random forests, and extra-trees handle non-linear physiological interactions and mixed-scale features naturally.
  * Regularized linear and kernel classifiers on standardized features provide smooth, low-variance decision boundaries across subjects.
  * Multi-Layer Perceptrons (MLPClassifier) with scaled inputs capture non-linear sensor manifolds without memorizing demographic IDs.
  * **Model Architecture:** Prefer a well-tuned model or parsimonious pipeline over heavy multi-model ensembles that consume the runtime budget for marginal gain.
- **Baseline & Anthropometric Context:** Raw heart rate and energy readings vary with an individual's resting physiology, age, and body dimensions; contextualizing active signals relative to individual baseline characteristics helps generalize across unseen subjects without memorizing participant identities.
- **Movement & Exertion Coupling:** Physical activities differ in how mechanical motion relates to cardiorespiratory response and signal variability. Combine accelerometer and metabolic features to capture physically grounded activity patterns.
- **Cross-Device Differences:** Apple Watch and Fitbit (`device_encoded`) have distinct sensor sensitivities and bias profiles. Modeling device-aware interactions or brand-specific scaling can harmonize cross-brand measurements.
- **Activity Hierarchy:** Classes span sedentary postures (Lying `0`, Sitting `5`), self-paced walking (`4`), and ascending running intensities (`1`, `2`, `3` corresponding to 3, 5, 7 METs). Inspect confusion patterns between adjacent running tiers and sedentary states.
- **Validation & Metric Balance:** Design internal validation with unseen-participant generalization in mind, and balance overall `Accuracy` (weight {accuracy_w:.2f}) with `Macro-F1` (weight {f1_w:.2f}) across rarer or harder classes without over-predicting minority activities.

## 6. Pitfalls

- **Transductive Grouping:** Never group `X_val` by participant or compute per-participant `mean`, `std`, `min`, `max`, or `rank` across rows of `X_val`. Because each lab participant completed all 6 activities in sequence, aggregating across a subject's test rows leaks their full session range and breaks real-time single-window inference. Every row of `X_val` must be transformed independently using only its own single-row values and statistics fit on `X_train`.
- **Temporal Fallacy:** Each sample in this benchmark is an independent summary snapshot. There is NO continuous time axis or timestamp column; do NOT write dead code attempting rolling window statistics or datetime decomposition.
- **From-Scratch Models:** Do not attempt from-scratch NumPy or JAX neural networks; standard library implementations are 100x faster and far more accurate.
- **Deprecated Arguments:** Do not pass removed keyword arguments (such as `multi_class` to `LogisticRegression` on modern scikit-learn; raises TypeError).
- **Random Split Leakage:** Evaluating on random row splits rather than participant-separated splits inflates validation metrics and masks cross-person failure modes.
- **Probability Outputs:** Output must contain integer class predictions (0..5), not probabilities, logits, or continuous floats.

## 7. Iteration Discipline

- **Cycle 0 Baseline:** Build the simplest correct baseline that returns a valid prediction dataframe in under half the time budget. Prioritize a working end-to-end pipeline before tuning.
- **One Change at a Time:** Once you have a working baseline, change exactly ONE conceptual component per cycle (e.g. one feature family, one regularization sweep, or one loss formulation).
- **Keep-Best-So-Far:** Track your best score. If a modification degrades performance or adds excessive latency without accuracy gain, revert to your best-so-far solution on the following cycle.
- **Use the Feedback String:** Record your validation findings in the returned feedback string. Use it as an experiment log across cycles.
- **Focus on the Primary Lever:** Spend compute on the lever with the highest expected gain for the primary objective. Avoid architectural rewrites when the headroom is small.
- **Pre-Return Check:** Verify your output dataframe against the checklist in Section 8 before returning.
- **Concise Model Summary:** Start your response with a concise 1-line model/technique summary rather than conversational filler.

## 8. Output Contract

- [ ] Function signature is exactly `solution(X_train, y_train, X_val)` and returns a tuple `(y_pred_df, feedback)`.
- [ ] `y_pred_df` is a pandas DataFrame with index identical to `X_val.index`.
- [ ] `y_pred_df` has exactly one column named `'predictions'` with dtype `int (0 to 5)`.
- [ ] `y_pred_df` contains no `NaN`, `None`, or infinite values.
- [ ] Predictions are non-trivial (not a constant value or single class).
- [ ] Total execution completes comfortably within the {timeout_s} s per-cycle limit and scales to {final_eval_timeout_s} s on final evaluation.
- [ ] Output predictions are strictly integers within {{0, 1, 2, 3, 4, 5}}.
- [ ] All 6 classes are represented in predictions (not collapsing minority classes).
- [ ] Each row of `X_val` is predicted independently without cross-row participant `groupby` or test-batch rank/min/max aggregation.
- [ ] Validation diagnostics and per-class recall are logged in the feedback string.
- [ ] Feedback string contains the required items (change, validation score, per-class recall, elapsed wall-clock).
"""

STUDENT_INITIAL_PROMPT = (
    "Provide the first `solution`: a fast, clean baseline classification model"
    " on the raw columns with a correct output contract `(y_pred_df, feedback)`"
    " and the required feedback items; nothing more."
)

STUDENT_REFINE_INSTRUCTION_OPTIONS: list[str] = [
    (
        "Individual Baseline Context: Active sensor signals vary significantly"
        " across individuals due to differing resting physiology and body"
        " dimensions. Formulate normalized indicators that relate dynamic"
        " exertion signals to baseline physical characteristics, enabling the"
        " model to generalize across unseen subjects without memorizing"
        " participant identities."
    ),
    (
        "Motion & Exertion Coupling: Physical activities differ in how"
        " mechanical motion coordinates with cardiorespiratory response and"
        " signal variability. Combine motion, metabolic, and heart rate"
        " features to capture physically grounded activity patterns and"
        " exertion states."
    ),
    (
        "Cross-Device Harmonization: Different recording devices exhibit"
        " distinct sensor sensitivities, bias profiles, and scaling"
        " characteristics. Explore device-aware interactions or calibration"
        " strategies (fit strictly on training data) to harmonize cross-device"
        " measurements."
    ),
    (
        "Algorithmic Diversity: Compare distinct model families (e.g."
        " gradient-boosted trees, random forests, extra trees, multi-layer"
        " perceptrons, or regularized linear classifiers) under"
        " participant-aware cross-validation, logging comparative"
        " generalization performance in the feedback string."
    ),
    (
        "Intensity & Posture Disambiguation: Analyze confusion patterns"
        " between adjacent exercise intensities and sedentary postures."
        " Explore monotonic exertion representations or ordinal separation"
        " techniques to resolve ambiguities between closely related activity"
        " classes."
    ),
    (
        "Class Balance & Decision Calibration: Investigate loss weighting,"
        " class balancing, probability calibration, or tuned decision"
        " thresholds to jointly optimize overall accuracy and macro-F1 without"
        " over-predicting minority activity classes."
    ),
    (
        "Participant-Aware Validation: Structure internal validation to hold"
        " out unseen participants across folds. Log per-fold and overall"
        " macro-F1 diagnostics in the feedback string to evaluate true"
        " cross-individual generalization."
    ),
    (
        "Regularization & Capacity Control: Tune model complexity, tree depth,"
        " leaf constraints, or regularization penalties under"
        " participant-grouped validation so the classifier avoids memorizing"
        " participant-specific idiosyncratic artifacts."
    ),
    (
        "Feature Representation & Selection: Audit feature importances and"
        " validation gains on participant-grouped splits to prune noisy or"
        " redundant interaction terms while preserving informative"
        " physiological signals."
    ),
    (
        "Selective Peer Synthesis: Review peer feedback notes in Performance"
        " Feedback. Integrate promising feature representations, normalization"
        " principles, or calibration strategies with your own best model"
        " pipeline, rather than adopting monolithic peer implementations."
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
    train_rows: int = 3760,
    val_rows: int = 1260,
    final_eval_rows: int = 5020,
    train_users: int = 29,
    val_users: int = 10,
    final_eval_users: int = 39,
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

  return STUDENT_INSTRUCTIONS_TEMPLATE.format(
      task_title=TASK_TITLE,
      accuracy_w=metric_weights.get("accuracy", 1.0),
      f1_w=metric_weights.get("f1_macro", 0.5),
      exec_w=metric_weights.get("exec_time", 0.01),
      metrics_desc=scoring.get_metrics_description(metric_weights).strip(),
      num_features=len(feature_names),
      train_rows=train_rows,
      val_rows=val_rows,
      final_eval_rows=final_eval_rows,
      train_users=train_users,
      val_users=val_users,
      final_eval_users=final_eval_users,
      timeout_s=timeout_s,
      final_eval_timeout_s=final_eval_timeout_s,
      imp_str=", ".join(f"`{m}`" for m in available_modules),
      forb_str=", ".join(f"`{m}`" for m in forbidden),
  )


STUDENT_INSTRUCTIONS: str = build_student_instructions()
