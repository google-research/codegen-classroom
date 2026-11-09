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

"""Standalone script to run the Wearable Activity Classification task.

Demonstrates:
  - Participant-grouped multi-class tabular classification (6 classes)
  - Prompt generation and student instruction rendering
  - RUN CONSTANTS with empirical rationale
  - Top solution archive export
"""

from __future__ import annotations

import collections
import functools
import json
import os
import tempfile
import time
from typing import MutableMapping

from absl import app
from absl import flags
from codegen_classroom import agent_config
from codegen_classroom import agents
from codegen_classroom import data_utils
from codegen_classroom import plotting
from codegen_classroom import reporting
from codegen_classroom import scoring
from codegen_classroom import student_learning
from codegen_classroom.tasks.wearable import config
from codegen_classroom.tasks.wearable import data_loader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==============================================================================
# === 1. TASK CONFIGURATION ===
# ==============================================================================


FLAGS = flags.FLAGS

flags.DEFINE_integer(
    "num_cycles",
    config.RUN_DEFAULTS["num_cycles"],
    "Number of learning cycles.",
)
flags.DEFINE_integer(
    "num_students",
    config.RUN_DEFAULTS["num_students"],
    "Number of student agents to run.",
)
flags.DEFINE_integer(
    "timeout_s",
    config.RUN_DEFAULTS["timeout_s"],
    "Per-cycle execution timeout in seconds.",
)
flags.DEFINE_integer(
    "final_eval_timeout_s",
    config.RUN_DEFAULTS["final_eval_timeout_s"],
    "Final evaluation execution timeout in seconds.",
)
flags.DEFINE_integer(
    "max_output_tokens",
    config.RUN_DEFAULTS["max_output_tokens"],
    "Maximum output tokens for student LLM responses.",
)
flags.DEFINE_string("experiment_name", "wearable_headless", "Experiment name.")
flags.DEFINE_list(
    "model_names",
    list(config.DEFAULT_STUDENT_MODELS),
    "Model names to use.",
)
flags.DEFINE_integer(
    "early_stopping_patience",
    config.RUN_DEFAULTS["early_stopping_patience"],
    "Early stopping patience in cycles without improvement.",
)
flags.DEFINE_integer(
    "max_mutation_count",
    config.RUN_DEFAULTS["max_mutation_count"],
    "Max mutations before student exclusion.",
)
flags.DEFINE_string(
    "run_dir",
    None,
    "Explicit directory for this run's outputs. If not set, generated"
    " automatically under experiments/<experiment_name>/.",
)
flags.DEFINE_string(
    "report_only",
    None,
    "Regenerate report.md for an existing run directory (pass path or 'latest')"
    " and exit without running any learning cycles.",
)
flags.DEFINE_bool(
    "mock_students",
    False,
    "Use mock student producing baseline model for hermetic unit testing.",
)
flags.DEFINE_integer(
    "export_top_n",
    3,
    "Number of top solutions to export to solutions/.",
)
flags.DEFINE_string(
    "seed_from",
    None,
    "Optional seed solution ('best', solution ID/path, or run_id).",
)
flags.DEFINE_enum(
    "action",
    "run",
    ["run", "diagnostics"],
    "Action to perform: 'run' (default) or 'diagnostics'.",
)

agent_config.load_env()
reporting.suppress_common_warnings()

# ==============================================================================
# === EXECUTION HELPERS ===
# ==============================================================================


def _wearable_baseline_predictions(
    x_learning: pd.DataFrame,
    y_learning: pd.DataFrame,
    x_test: pd.DataFrame,
) -> np.ndarray:
  """Computes Cycle 0 baseline (balanced HistGradientBoostingClassifier) predictions."""
  from sklearn.ensemble import HistGradientBoostingClassifier  # pylint: disable=g-import-not-at-top

  model = HistGradientBoostingClassifier(random_state=42)
  model.fit(x_learning, y_learning.iloc[:, 0])
  return np.asarray(model.predict(x_test), dtype=int).ravel()


def _generate_wearable_dataset_overview(
    x_all: pd.DataFrame,
    y_all: pd.DataFrame,
    plots_dir: str,
) -> str:
  """Generates a 4-panel dataset & physiological sensor overview figure."""
  os.makedirs(plots_dir, exist_ok=True)
  overview_abs = os.path.join(plots_dir, "dataset_overview.png")
  class_order = [0, 5, 4, 1, 2, 3]
  class_labels = ["Lying", "Sitting", "Walk", "Run 3M", "Run 5M", "Run 7M"]
  y_arr = np.asarray(y_all.iloc[:, 0], dtype=int).ravel()
  hr_arr = pd.to_numeric(x_all["heart_rate"], errors="coerce").to_numpy()
  cal_arr = pd.to_numeric(x_all["calories"], errors="coerce").to_numpy()
  dev_arr = np.asarray(x_all["device_encoded"], dtype=int).ravel()
  gen_arr = np.asarray(x_all["gender"], dtype=int).ravel()

  fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(19.2, 4.2), dpi=150)

  # Panel 1: Heart Rate Boxplot by Activity (Intensity Order)
  hr_by_act = [hr_arr[y_arr == c] for c in class_order]
  bp = ax1.boxplot(
      hr_by_act,
      tick_labels=class_labels,
      patch_artist=True,
      showfliers=False,
      widths=0.55,
      medianprops={"color": "#111827", "linewidth": 1.4},
  )
  palette = ["#93c5fd", "#60a5fa", "#34d399", "#fbbf24", "#f97316", "#ef4444"]
  for patch, col in zip(bp["boxes"], palette):
    patch.set_facecolor(col)
    patch.set_alpha(0.80)
    patch.set_edgecolor("#334155")
  plotting.style_axes(
      ax1,
      "Heart Rate Distribution by Activity",
      "Activity (Sedentary → Vigorous)",
      "Heart Rate (BPM)",
  )
  ax1.tick_params(axis="x", rotation=18.0)

  # Panel 2: Calories Burned by Activity & Brand (Apple Watch vs. Fitbit)
  aw_means = [
      float(np.nanmean(cal_arr[(y_arr == c) & (dev_arr == 0)]))
      for c in class_order
  ]
  fb_means = [
      float(np.nanmean(cal_arr[(y_arr == c) & (dev_arr == 1)]))
      for c in class_order
  ]
  plotting.plot_grouped_bar_panel(
      ax2,
      class_labels,
      aw_means,
      fb_means,
      label_first="Apple Watch (n=3,656)",
      label_best="Fitbit (n=2,608)",
      title="Mean Calories by Activity & Device",
      xlabel="Activity (Sedentary → Vigorous)",
      ylabel="Calories Burned (kcal)",
      fmt="{:.1f}",
      color_first="#64748b",
      color_best="#0d9488",
      rotation=18.0,
  )

  # Panel 3: Heart Rate by Activity & Gender Split
  fem_means = [
      float(np.nanmean(hr_arr[(y_arr == c) & (gen_arr == 0)]))
      for c in class_order
  ]
  male_means = [
      float(np.nanmean(hr_arr[(y_arr == c) & (gen_arr == 1)]))
      for c in class_order
  ]
  plotting.plot_grouped_bar_panel(
      ax3,
      class_labels,
      fem_means,
      male_means,
      label_first="Gender = 0 (Female)",
      label_best="Gender = 1 (Male)",
      title="Mean Heart Rate by Activity & Gender",
      xlabel="Activity (Sedentary → Vigorous)",
      ylabel="Mean Heart Rate (BPM)",
      fmt="{:.0f}",
      color_first="#8b5cf6",
      color_best="#f59e0b",
      rotation=18.0,
  )

  # Panel 4: Physiological & Biomechanical Feature Correlation Matrix
  corr_cols = [
      ("steps", "Steps"),
      ("heart_rate", "Heart Rate"),
      ("calories", "Calories"),
      ("distance", "Distance"),
      ("resting_heart", "Rest HR"),
      ("intensity_karvonen", "Karvonen"),
      ("sd_norm_heart", "SD Norm HR"),
      ("age", "Age"),
      ("weight", "Weight"),
  ]
  avail = [(c, lbl) for c, lbl in corr_cols if c in x_all.columns]
  corr_df = x_all[[c for c, _ in avail]].apply(pd.to_numeric, errors="coerce")
  corr_df["Activity"] = y_arr
  c_labels = [lbl for _, lbl in avail] + ["Activity"]
  c_mat = corr_df.corr().to_numpy()
  im = ax4.imshow(c_mat, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
  cbar = fig.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
  cbar.ax.tick_params(labelsize=7.5)
  ax4.set_xticks(np.arange(len(c_labels)))
  ax4.set_yticks(np.arange(len(c_labels)))
  ax4.set_xticklabels(c_labels, rotation=38, ha="right", fontsize=7.5)
  ax4.set_yticklabels(c_labels, fontsize=7.5)
  for r_idx in range(len(c_labels)):
    for c_idx in range(len(c_labels)):
      val = c_mat[r_idx, c_idx]
      ax4.text(
          c_idx,
          r_idx,
          f"{val:.2f}",
          ha="center",
          va="center",
          fontsize=5.8,
          color="white" if abs(val) > 0.55 else "#111827",
      )
  ax4.set_title(
      "Sensor & Biomechanical Correlation Matrix",
      fontsize=10.5,
      fontweight="bold",
      pad=8,
  )

  fig.tight_layout(pad=1.2)
  fig.savefig(overview_abs, dpi=150, bbox_inches="tight")
  plt.close(fig)
  return "plots/dataset_overview.png"


def _generate_wearable_diagnostics(
    task_dir: str,
    md_files: list[str],
) -> None:
  """Generates 4-panel wearable dataset overview and evaluation plots and embeds them in md_files."""
  if not md_files:
    return
  data_dir = os.path.join(task_dir, "data")
  if not os.path.isfile(os.path.join(data_dir, "aw_fb_data.csv")):
    return
  try:
    x, y, groups = data_loader.load_preprocessed_wearable_data(
        data_dir=data_dir, y_column="activity_encoded"
    )
    (
        _,
        _,
        x_test,
        _,
        _,
        y_test,
        _,
        _,
        _,
        x_learning,
        y_learning,
        _,
    ) = data_loader.split_wearable_data(
        x, y, groups, random_state=config.VAL_SEED
    )
  except Exception:  # pylint: disable=broad-exception-caught
    return

  plots_dir = os.path.join(os.path.dirname(md_files[0]), "plots")
  os.makedirs(plots_dir, exist_ok=True)
  overview_rel = "plots/dataset_overview.png"
  if not os.path.isfile(os.path.join(plots_dir, "dataset_overview.png")):
    try:
      overview_rel = _generate_wearable_dataset_overview(x, y, plots_dir)
    except Exception:  # pylint: disable=broad-exception-caught
      pass

  from sklearn import metrics as sk_metrics  # pylint: disable=g-import-not-at-top

  class_order = [0, 5, 4, 1, 2, 3]
  class_labels = ["Lying", "Sitting", "Walk", "Run 3M", "Run 5M", "Run 7M"]
  y_true = np.asarray(y_test.iloc[:, 0], dtype=int).ravel()

  for md_path in md_files:
    try:
      preds = reporting.resolve_solution_predictions(
          md_path,
          x_learning,
          y_learning,
          x_test,
          baseline_fn=_wearable_baseline_predictions,
          timeout_s=180.0,
      )
      if preds is None:
        continue
      y_best, y_first = preds
      y_best = np.asarray(np.rint(y_best), dtype=int).ravel()
      y_first = np.asarray(np.rint(y_first), dtype=int).ravel()
      if len(y_best) != len(y_true) or len(y_first) != len(y_true):
        continue

      acc_first = float(sk_metrics.accuracy_score(y_true, y_first))
      acc_best = float(sk_metrics.accuracy_score(y_true, y_best))
      f1_first = float(sk_metrics.f1_score(y_true, y_first, average="macro"))
      f1_best = float(sk_metrics.f1_score(y_true, y_best, average="macro"))

      sol_id = os.path.splitext(os.path.basename(md_path))[0]
      plot_rel = f"plots/{sol_id}_diagnostics.png"
      plot_abs = os.path.join(os.path.dirname(md_path), plot_rel)

      fig, (ax1, ax2, ax3, ax4) = plt.subplots(
          1, 4, figsize=(19.2, 4.2), dpi=150
      )

      # Panel 1: Row-Normalized Confusion Matrix (Best Solution)
      plotting.plot_confusion_matrix_panel(
          ax1,
          fig,
          y_true,
          y_best,
          class_order,
          class_labels,
          title="Best Solution Confusion Matrix (Row %)",
          xlabel="Predicted Activity (Intensity Order)",
          ylabel="True Activity (Intensity Order)",
      )

      # Panel 2: Per-Activity F1 Score (Cycle 0 vs. Best)
      f1_per_first = sk_metrics.f1_score(
          y_true, y_first, labels=class_order, average=None, zero_division=0
      )
      f1_per_best = sk_metrics.f1_score(
          y_true, y_best, labels=class_order, average=None, zero_division=0
      )
      plotting.plot_grouped_bar_panel(
          ax2,
          class_labels,
          list(f1_per_first),
          list(f1_per_best),
          label_first=f"Cycle 0 (Macro={f1_first:.3f})",
          label_best=f"Best (Macro={f1_best:.3f})",
          title="Per-Activity Macro-F1 (Cycle 0 vs. Best)",
          xlabel="Activity Class (Sedentary → Vigorous)",
          ylabel="F1 Score (0.0 – 1.0)",
          fmt="{:.2f}",
          color_best="#8b5cf6",
          rotation=18.0,
      )

      # Panel 3: Precision vs. Recall Tradeoff (Best Solution)
      prec_best = sk_metrics.precision_score(
          y_true, y_best, labels=class_order, average=None, zero_division=0
      )
      rec_best = sk_metrics.recall_score(
          y_true, y_best, labels=class_order, average=None, zero_division=0
      )
      plotting.plot_grouped_bar_panel(
          ax3,
          class_labels,
          list(prec_best),
          list(rec_best),
          label_first=f"Precision (Macro={float(np.mean(prec_best)):.3f})",
          label_best=f"Recall (Macro={float(np.mean(rec_best)):.3f})",
          title="Per-Activity Precision vs. Recall (Best)",
          xlabel="Activity Class (Sedentary → Vigorous)",
          ylabel="Score (0.0 – 1.0)",
          fmt="{:.2f}",
          color_first="#0284c7",
          color_best="#f97316",
          rotation=18.0,
      )

      # Panel 4: Cross-Device & Physiological Regime Accuracy
      dev_col = (
          np.asarray(x_test["device_encoded"], dtype=int).ravel()
          if "device_encoded" in x_test.columns
          else np.zeros_like(y_true)
      )
      cohort_specs = [
          ("Apple Watch", dev_col == 0),
          ("Fitbit", dev_col == 1),
          ("Sedentary", np.isin(y_true, [0, 5])),
          ("Walking", y_true == 4),
          ("Running (3-7M)", np.isin(y_true, [1, 2, 3])),
      ]
      c_names = [c[0] for c in cohort_specs]
      c_acc_first = [
          float(np.mean(y_first[m] == y_true[m])) * 100.0 if np.any(m) else 0.0
          for _, m in cohort_specs
      ]
      c_acc_best = [
          float(np.mean(y_best[m] == y_true[m])) * 100.0 if np.any(m) else 0.0
          for _, m in cohort_specs
      ]
      plotting.plot_grouped_bar_panel(
          ax4,
          c_names,
          c_acc_first,
          c_acc_best,
          label_first=f"Cycle 0 (Acc={acc_first * 100:.1f}%)",
          label_best=f"Best (Acc={acc_best * 100:.1f}%)",
          title="Accuracy by Device & Intensity Tier",
          xlabel="Hardware Brand / Physiological Regime",
          ylabel="Test Accuracy (%)",
          fmt="{:.1f}%",
          color_best="#2ca02c",
          rotation=18.0,
      )

      fig.tight_layout(pad=1.2)
      fig.savefig(plot_abs, dpi=150, bbox_inches="tight")
      plt.close(fig)

      aw_m = dev_col == 0
      fb_m = dev_col == 1
      sed_m = np.isin(y_true, [0, 5])
      run_m = np.isin(y_true, [1, 2, 3])
      aw_f, aw_b = (
          float(np.mean(y_first[aw_m] == y_true[aw_m])) * 100,
          float(np.mean(y_best[aw_m] == y_true[aw_m])) * 100,
      )
      fb_f, fb_b = (
          float(np.mean(y_first[fb_m] == y_true[fb_m])) * 100,
          float(np.mean(y_best[fb_m] == y_true[fb_m])) * 100,
      )
      sed_f, sed_b = (
          float(np.mean(y_first[sed_m] == y_true[sed_m])) * 100,
          float(np.mean(y_best[sed_m] == y_true[sed_m])) * 100,
      )
      run_f, run_b = (
          float(np.mean(y_first[run_m] == y_true[run_m])) * 100,
          float(np.mean(y_best[run_m] == y_true[run_m])) * 100,
      )

      summary_md = (
          "| Diagnostic Metric (Unseen Test Participants) | Cycle 0 Baseline |"
          " Best Solution | Improvement |\n| :--- | :---: | :---: | :---: |\n|"
          f" **Overall Test Accuracy** | {acc_first * 100:.2f}% |"
          f" **{acc_best * 100:.2f}%** | {(acc_best - acc_first) * 100:+.2f}%"
          f" |\n| **Test Macro-F1 (6 Classes)** | {f1_first:.4f} |"
          f" **{f1_best:.4f}** | {f1_best - f1_first:+.4f} |\n| **Apple Watch"
          f" Accuracy (`device=0`)** | {aw_f:.1f}% | **{aw_b:.1f}%** |"
          f" {aw_b - aw_f:+.1f}% |\n| **Fitbit Accuracy (`device=1`)** |"
          f" {fb_f:.1f}% | **{fb_b:.1f}%** | {fb_b - fb_f:+.1f}% |\n|"
          f" **Sedentary Recall (`Lying` / `Sitting`)** | {sed_f:.1f}% |"
          f" **{sed_b:.1f}%** | {sed_b - sed_f:+.1f}% |\n| **Running Tiers"
          f" Recall (`3 / 5 / 7 METs`)** | {run_f:.1f}% | **{run_b:.1f}%** |"
          f" {run_b - run_f:+.1f}% |"
      )
      reporting.embed_diagnostic_section(
          md_path, plot_rel, summary_md, overview_plot_rel_path=overview_rel
      )
    except Exception:  # pylint: disable=broad-exception-caught
      pass


def export_top_solutions(
    task_dir: str,
    top_n: int = 3,
    write: bool = True,
) -> list[str]:
  """Exports top solution reports from leaderboard.tsv to solutions/."""
  return reporting.export_top_solutions(
      task_dir,
      sort_by=["test_f1_macro", "test_accuracy"],
      ascending=[False, False],
      top_n=top_n,
      write=write,
      metric_weights=config.DEFAULT_METRIC_WEIGHTS,
      experiments_root=data_utils.experiments_root(config.TASK_NAME),
      diagnose=functools.partial(_generate_wearable_diagnostics, task_dir),
  )


def resolve_task_dir(base_dir: str | None = None) -> str:
  """Resolves the live task directory."""
  return data_utils.resolve_task_dir(__file__, base_dir)


def _report_only() -> None:
  """Rebuilds report.md for an existing run without re-executing code."""
  run_dir = data_utils.resolve_run_dir(
      FLAGS.report_only,
      os.path.join(
          data_utils.experiments_root(config.TASK_NAME), FLAGS.experiment_name
      ),
  )
  reporting.write_run_report(
      run_dir,
      metric_weights=config.DEFAULT_METRIC_WEIGHTS,
  )


class _MockStudent(agents.StudentAgent):
  """Deterministic mock student for fast offline unit testing."""

  def __init__(
      self,
      config: agents.AgentConfig,  # pylint: disable=redefined-outer-name
      run_base_dir: str = "",
      seed_solution: str | None = None,
  ):
    config.api_key = config.api_key or "mock_key"
    super().__init__(
        config=config,
        run_base_dir=run_base_dir or tempfile.gettempdir(),
        seed_solution=seed_solution,
    )

  def generate_solution(self, prompt: str) -> str:
    del prompt
    self.generate_code_count += 1
    self.current_solution = (
        "```python\n"
        "import pandas as pd\n"
        "from sklearn.ensemble import HistGradientBoostingClassifier\n"
        "def solution(X_train, y_train, X_val):\n"
        "    model = HistGradientBoostingClassifier(max_iter=20,"
        " random_state=42)\n"
        "    model.fit(X_train, y_train.iloc[:, 0])\n"
        "    preds = model.predict(X_val)\n"
        "    return pd.DataFrame({'predictions': preds},"
        " index=X_val.index), 'Mock HGBC baseline'\n"
        "```\n"
    )
    return self.current_solution


def main(argv: list[str]) -> None:
  del argv

  if FLAGS.report_only:
    _report_only()
    return

  if FLAGS.action == "diagnostics":
    task_dir = resolve_task_dir()
    solutions_dir = os.path.join(task_dir, "solutions")
    md_files = [
        os.path.join(solutions_dir, f)
        for f in sorted(os.listdir(solutions_dir))
        if f.endswith(".md")
    ]
    _generate_wearable_diagnostics(task_dir, md_files)
    return

  # 1. Load Data
  print("Loading data...")
  task_dir = resolve_task_dir()
  data_dir = os.path.join(task_dir, "data")
  data_loader.ensure_data(data_dir)

  try:
    x, y, groups = data_loader.load_preprocessed_wearable_data(
        data_dir=data_dir, y_column="activity_encoded"
    )
  except FileNotFoundError as e:
    print(f"Error: {e}")
    return

  # 2. Split Data
  (
      x_train,
      x_val,
      x_test_final,
      y_train,
      y_val,
      y_test_final,
      groups_train,
      groups_val,
      _,
      x_learning,
      y_learning,
      groups_learning,
  ) = data_loader.split_wearable_data(
      x, y, groups, random_state=config.VAL_SEED
  )

  # 3. Define Task
  metric_weights = dict(config.DEFAULT_METRIC_WEIGHTS)

  student_instructions = config.build_student_instructions(
      train_rows=len(x_train),
      val_rows=len(x_val),
      final_eval_rows=len(x_learning),
      train_users=groups_train.nunique(),
      val_users=groups_val.nunique(),
      final_eval_users=groups_learning.nunique(),
      timeout_s=FLAGS.timeout_s,
      final_eval_timeout_s=FLAGS.final_eval_timeout_s,
      metric_weights=metric_weights,
  )

  # 4. Config
  runs_base_path = os.path.join(
      data_utils.experiments_root(config.TASK_NAME), FLAGS.experiment_name
  )
  os.makedirs(runs_base_path, exist_ok=True)

  if FLAGS.run_dir:
    run_dir = os.path.abspath(FLAGS.run_dir)
    run_id = os.path.basename(run_dir)
  else:
    run_id = "headless_" + time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(runs_base_path, run_id)
  os.makedirs(run_dir, exist_ok=True)

  num_learning_cycles = FLAGS.num_cycles
  code_execution_max_retries = 3
  code_execution_max_duration = FLAGS.timeout_s

  early_stopping_patience = FLAGS.early_stopping_patience
  max_mutation_count = FLAGS.max_mutation_count

  mutation_strategy_options = [
      agents.MutationStrategy.SELF_BEST_CODE,
      agents.MutationStrategy.BEST_PEER_CROSSOVER,
      agents.MutationStrategy.CONTRASTIVE_PEER_CROSSOVER,
  ]

  # Setup logging
  logger = reporting.setup_logger(
      "codegen_classroom", log_file=os.path.join(run_dir, "run.log")
  )
  logger.info("Starting wearable classification headless run: %s", run_id)

  if len(FLAGS.model_names) < FLAGS.num_students:
    models = [
        FLAGS.model_names[i % len(FLAGS.model_names)]
        for i in range(FLAGS.num_students)
    ]
  else:
    models = FLAGS.model_names[: FLAGS.num_students]
  student_cfgs = student_learning.get_classroom_student_configs(
      system_instruction=student_instructions,
      model_names=models,
      temperature=0.85,
      max_output_tokens=FLAGS.max_output_tokens,
  )

  agent_cls = _MockStudent if FLAGS.mock_students else agents.StudentAgent
  seed_md = data_utils.load_seed_solution(FLAGS.seed_from, task_dir=task_dir)
  students = {
      cfg.name: agent_cls(
          config=cfg,
          run_base_dir=run_dir,
          seed_solution=seed_md,
      )
      for cfg in student_cfgs
  }

  current_classroom = list(students.values())
  results_history: MutableMapping[str, student_learning.ClassroomResults] = (
      collections.defaultdict(list)
  )
  active_metrics = ["overall_score"] + list(metric_weights.keys())

  validator = functools.partial(
      scoring.classification_prediction_validator, expected_columns=1
  )

  # Run learning cycles
  for current_cycle in range(num_learning_cycles):
    if not current_classroom:
      break
    logger.info("=== Starting Cycle %d ===", current_cycle)

    classroom_feedback = scoring.get_classroom_feedback_string(
        cycle_num=current_cycle,
        current_classroom=current_classroom,
        classroom_results_history=results_history,
        metric_keys=active_metrics,
    )

    classroom_results = student_learning.run_parallel_classroom_learning_cycle(
        current_classroom=current_classroom,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        current_cycle=current_cycle,
        student_initial_prompt=config.STUDENT_INITIAL_PROMPT,
        student_function_name=config.STUDENT_FUNCTION_NAME,
        code_execution_max_retries=code_execution_max_retries,
        code_execution_timeout_seconds=code_execution_max_duration,
        metric_weights=metric_weights,
        early_stopping_patience=early_stopping_patience,
        student_refine_instruction_options=config.STUDENT_REFINE_INSTRUCTION_OPTIONS,
        max_mutation_count=max_mutation_count,
        mutation_strategy_options=mutation_strategy_options,
        validator=validator,
        classroom_feedback=classroom_feedback,
        num_workers=len(current_classroom),
    )

    current_classroom = student_learning.process_cycle_results(
        current_classroom, classroom_results, results_history, current_cycle
    )

    for student in students.values():
      student.reset_history()

    if current_classroom and all(
        s.mutation_count >= max_mutation_count for s in current_classroom
    ):
      logger.warning(
          "All active students reached max mutations (%s). Stopping after"
          " cycle %s",
          max_mutation_count,
          current_cycle,
      )
      break

  # Final Evaluation on held-out test set
  logger.info("Evaluating on held-out test set...")
  test_results_path = os.path.join(run_dir, "final_eval_results.json")
  test_prediction_dir = os.path.join(run_dir, "predictions")

  student_learning.run_parallel_classroom_final_evaluation(
      students=students,
      x_learning=x_learning,
      y_learning=y_learning,
      x_test_final=x_test_final,
      y_test_final=y_test_final,
      student_function_name=config.STUDENT_FUNCTION_NAME,
      metric_weights=metric_weights,
      strategies=["first", "best", "last"],
      validator=validator,
      code_execution_timeout_seconds=FLAGS.final_eval_timeout_s,
      enforce_timeout=True,
      num_workers=len(students),
      final_results_path=test_results_path,
      predictions_dir=test_prediction_dir,
  )

  # Append results to leaderboard
  row_data = None
  if os.path.isfile(test_results_path):
    with open(test_results_path, "r", encoding="utf-8") as f:
      eval_json = json.load(f)
    best_student_obj = max(
        students.values(),
        key=lambda s: float(s.best_score)
        if s.best_score is not None
        else -float("inf"),
    )
    best_eval_entry = next(
        (
            e
            for e in eval_json
            if e.get("student_name") == best_student_obj.name
        ),
        {},
    )
    best_strategy_eval = best_eval_entry.get("best", {})
    b_cycle = best_student_obj.best_cycle
    best_res = (
        best_student_obj.cycle_results[b_cycle]
        if 0 <= b_cycle < len(best_student_obj.cycle_results)
        else None
    )
    best_val_metrics = best_res.metrics if best_res else {}
    best_plan = best_res.plan if best_res else ""
    meta = {}
    if not FLAGS.mock_students and best_student_obj.best_solution:
      try:
        analyzer = agents.SolutionAnalysisAgent(
            config=agents.AgentConfig(name="solution_metadata"),
            run_base_dir=run_dir,
        )
        meta = (
            analyzer.analyze(
                student_name=best_student_obj.name,
                best_code=best_student_obj.best_solution,
                best_plan=best_plan,
                best_test_metrics=best_strategy_eval,
                best_val_metrics=best_val_metrics,
                student_instructions=student_instructions,
                metrics_description=scoring.get_metrics_description(
                    metric_weights
                ),
                task_description_for_prompt="wearable activity classification",
            )
            or {}
        )
      except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning("SolutionAnalysisAgent failed: %s", e)

    row_data = {
        "timestamp": pd.Timestamp.now().strftime("%m/%d/%Y"),
        "test_accuracy": best_strategy_eval.get("metrics_accuracy"),
        "test_f1_macro": best_strategy_eval.get("metrics_f1_macro"),
        "test_exec_time_s": best_strategy_eval.get("metrics_exec_time"),
        "val_accuracy": best_val_metrics.get("accuracy"),
        "val_f1_macro": best_val_metrics.get("f1_macro"),
        "train_cycles": FLAGS.num_cycles,
        "train_best_cycle": best_student_obj.best_cycle,
        "train_val_seed": config.VAL_SEED,
        "train_num_students": len(students),
        "codegen_version": 2,
        "solution_type": "best_validation_solution",
        "total_tokens": None,
        "est_cost_usd": None,
        "experiment_name": FLAGS.experiment_name or "wearable_headless",
        "student": best_student_obj.name,
        "run_id": run_id,
        "run_dir": run_dir,
        "user": os.environ.get("USER", ""),
        "tags": "mock" if FLAGS.mock_students else meta.get("model_tags", ""),
        "description": meta.get("model_description", ""),
        "insights": meta.get("model_insights", ""),
        "test_precision_macro": best_strategy_eval.get(
            "metrics_precision_macro"
        ),
        "test_recall_macro": best_strategy_eval.get("metrics_recall_macro"),
        "solution_dir": "",
    }

  reporting.write_run_report(
      run_dir,
      students=list(students.values()),
      metric_weights=config.DEFAULT_METRIC_WEIGHTS,
  )
  logger.info("Report generated in %s/report.md", run_dir)

  if row_data is not None and not FLAGS.mock_students:
    lb_path = os.path.join(task_dir, "leaderboard.tsv")
    data_utils.submit_results_to_local_leaderboard(
        lb_path, pd.DataFrame([row_data])
    )
    logger.info("Results appended to leaderboard: %s", lb_path)

  # Export top solutions
  if not FLAGS.mock_students:
    try:
      export_top_solutions(task_dir, top_n=FLAGS.export_top_n)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logger.warning("Solution archive export failed: %s", e)


if __name__ == "__main__":
  app.run(main)
