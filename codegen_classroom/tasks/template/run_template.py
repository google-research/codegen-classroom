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

"""Canonical runner script for the template benchmark task.

Copy `tasks/template/` to `tasks/<your_task>/` and update the sections marked
`# CUSTOMIZE:` to configure your dataset, evaluation metrics, and
domain guidance.
"""

from __future__ import annotations

import collections
import functools
import json
import os
import tempfile
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
from codegen_classroom.tasks.template import config
from codegen_classroom.tasks.template import data_loader
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import impute
from sklearn import linear_model
from sklearn import metrics as sk_metrics
from sklearn import model_selection

# ==============================================================================
# === 1. TASK CONFIGURATION & FLAGS ===
# ==============================================================================


FLAGS = flags.FLAGS

flags.DEFINE_integer(
    "num_cycles",
    config.RUN_DEFAULTS["num_cycles"],
    "Number of learning cycles.",
)
flags.DEFINE_string(
    "experiment_name", "template_benchmark", "Experiment group name."
)
flags.DEFINE_list(
    "model_names",
    list(config.DEFAULT_STUDENT_MODELS),
    "Student LLM model names to use.",
)
flags.DEFINE_integer(
    "num_students",
    config.RUN_DEFAULTS["num_students"],
    "Number of students in the classroom.",
)
flags.DEFINE_integer(
    "code_execution_max_duration",
    config.RUN_DEFAULTS["timeout_s"],
    "Per-solution execution timeout in seconds.",
)
flags.DEFINE_integer(
    "final_eval_timeout",
    config.RUN_DEFAULTS["final_eval_timeout_s"],
    "Final evaluation timeout in seconds.",
)
flags.DEFINE_integer(
    "early_stopping_patience",
    config.RUN_DEFAULTS["patience"],
    "Cycles without improvement before student mutation.",
)
flags.DEFINE_integer(
    "max_mutation_count",
    config.RUN_DEFAULTS["max_mutations"],
    "Maximum mutation attempts before student exclusion.",
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
    "Use a deterministic mock student for hermetic testing without API calls.",
)
flags.DEFINE_integer(
    "max_output_tokens",
    config.RUN_DEFAULTS["max_output_tokens"],
    "Maximum output tokens for student LLM completions.",
)
flags.DEFINE_integer(
    "export_top_n",
    3,
    "Number of top solutions to export to solutions/.",
)
flags.DEFINE_string(
    "seed_from",
    None,
    "Optional seed solution ('best', solution ID, or path to report .md).",
)

agent_config.load_env()


# ==============================================================================
# === 2. DIAGNOSTICS & TOP-SOLUTION ARCHIVE ===
# ==============================================================================


def _generate_template_diagnostics(
    task_dir: str,
    md_files: list[str],
) -> None:
  """Generates multi-panel regime and residual diagnostic plots for solutions."""
  if not md_files:
    return
  x, y = data_loader.load_data(os.path.join(task_dir, "data"))
  x_learning, x_test, y_learning, y_test = model_selection.train_test_split(
      x, y, test_size=0.2, random_state=config.VAL_SEED
  )
  y_true = y_test.iloc[:, 0].to_numpy(dtype=float).ravel()
  regimes = x_test["operating_regime"].to_numpy(dtype=int).ravel()
  plots_dir = os.path.join(task_dir, "solutions", "plots")
  os.makedirs(plots_dir, exist_ok=True)

  def _baseline_fn(
      x_tr: pd.DataFrame, y_tr: pd.DataFrame, x_te: pd.DataFrame
  ) -> np.ndarray:
    imp = impute.SimpleImputer(strategy="median")
    m = linear_model.Ridge(alpha=1.0)
    m.fit(imp.fit_transform(x_tr), y_tr.iloc[:, 0])
    return m.predict(imp.transform(x_te))

  matplotlib.use("Agg")
  for md_path in md_files:
    sol_id = os.path.splitext(os.path.basename(md_path))[0]
    plot_path = os.path.join(plots_dir, f"{sol_id}_diagnostics.png")
    preds = reporting.resolve_solution_predictions(
        md_path,
        x_learning,
        y_learning,
        x_test,
        baseline_fn=_baseline_fn,
    )
    if preds is None:
      continue
    y_best, y_first = preds
    r2_0 = float(sk_metrics.r2_score(y_true, y_first))
    r2_b = float(sk_metrics.r2_score(y_true, y_best))
    mae_0 = float(sk_metrics.mean_absolute_error(y_true, y_first))
    mae_b = float(sk_metrics.mean_absolute_error(y_true, y_best))

    with plt.style.context("default"):
      fig, axes = plt.subplots(1, 4, figsize=(19.2, 4.2))
      ax0, ax1, ax2, ax3 = axes

      # Panel 1: Predicted vs. Observed by Operating Regime
      ax0.scatter(
          y_true,
          y_first,
          color="#7f7f7f",
          alpha=0.25,
          s=18,
          label=f"Cycle 0 Baseline (R²={r2_0:.3f})",
      )
      reg_colors = {
          0: ("#1f77b4", "Regime 0 (Base)"),
          1: ("#ff7f0e", "Regime 1 (Mod)"),
          2: ("#2ca02c", "Regime 2 (Surge)"),
      }
      for r_id, (col, r_label) in reg_colors.items():
        mask = regimes == r_id
        if np.any(mask):
          ax0.scatter(
              y_true[mask],
              y_best[mask],
              color=col,
              alpha=0.75,
              s=22,
              label=f"Best: {r_label}",
          )
      lims = [
          min(float(y_true.min()), float(y_best.min())) - 1.0,
          max(float(y_true.max()), float(y_best.max())) + 1.0,
      ]
      ax0.plot(lims, lims, "k--", lw=1.2, alpha=0.75, label="Ideal (y = x)")
      ax0.set_xlim(lims)
      ax0.set_ylim(lims)
      plotting.style_axes(
          ax0,
          f"Parity by Regime (R²: {r2_0:.3f} → {r2_b:.3f})",
          "Observed Thermal Load",
          "Predicted Thermal Load",
      )
      ax0.legend(loc="upper left", fontsize=8, framealpha=0.9)

      # Panel 2: Per-Regime MAE Reduction (Baseline vs. Best)
      labels = [
          "Regime 0\n(Base)",
          "Regime 1\n(Moderate)",
          "Regime 2\n(Surge)",
          "Overall\nTest",
      ]
      m0_vals = [
          float(
              sk_metrics.mean_absolute_error(
                  y_true[regimes == r_id], y_first[regimes == r_id]
              )
          )
          for r_id in (0, 1, 2)
      ] + [mae_0]
      mb_vals = [
          float(
              sk_metrics.mean_absolute_error(
                  y_true[regimes == r_id], y_best[regimes == r_id]
              )
          )
          for r_id in (0, 1, 2)
      ] + [mae_b]
      plotting.plot_grouped_bar_panel(
          ax1,
          labels,
          m0_vals,
          mb_vals,
          title=f"Per-Regime MAE (Overall: {mae_0:.2f} → {mae_b:.2f})",
          xlabel="Operating Regime Stratum",
          ylabel="Mean Absolute Error (MAE)",
          color_best="#ff7f0e",
      )

      # Panel 3: Residual Error Distribution
      res_0, res_b = y_first - y_true, y_best - y_true
      bound = max(float(np.percentile(np.abs(res_0), 98)), 3.0)
      bins = np.linspace(-bound, bound, 36)
      ax2.hist(
          res_0,
          bins=bins,
          density=True,
          color="#1f77b4",
          alpha=0.35,
          edgecolor="#1f77b4",
          label=f"Cycle 0 (SD={np.std(res_0):.2f})",
      )
      ax2.hist(
          res_b,
          bins=bins,
          density=True,
          color="#ff7f0e",
          alpha=0.65,
          edgecolor="#d62728",
          label=f"Best (SD={np.std(res_b):.2f})",
      )
      ax2.axvline(0.0, color="k", linestyle="--", lw=1.2, alpha=0.8)
      plotting.style_axes(
          ax2,
          "Residual Error Distribution (y_pred - y_true)",
          "Prediction Residual",
          "Empirical Density",
      )
      ax2.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

      # Panel 4: Bland-Altman Agreement (Pred - True vs. Mean)
      plotting.plot_bland_altman_panel(
          ax3,
          y_true,
          y_best,
          y_first,
          title="Bland–Altman Agreement (Pred − True)",
          xlabel="Mean Thermal Load",
          ylabel="Difference: Pred − True",
      )

      plt.tight_layout(w_pad=1.6)
      plt.savefig(plot_path, bbox_inches="tight", dpi=150)
      plt.close(fig)

    r0_pct = 100.0 * (m0_vals[0] - mb_vals[0]) / max(m0_vals[0], 1e-6)
    r1_pct = 100.0 * (m0_vals[1] - mb_vals[1]) / max(m0_vals[1], 1e-6)
    r2_pct = 100.0 * (m0_vals[2] - mb_vals[2]) / max(m0_vals[2], 1e-6)
    tot_pct = 100.0 * (mae_0 - mae_b) / max(mae_0, 1e-6)
    summary_md = (
        "| Stratum | Cycle 0 Baseline MAE | Best Solution MAE | Relative Error"
        " Reduction |\n| --- | --- | --- | --- |\n"
        f"| Regime 0 (Base) | {m0_vals[0]:.3f} | {mb_vals[0]:.3f} |"
        f" {r0_pct:.1f}% |\n"
        f"| Regime 1 (Moderate) | {m0_vals[1]:.3f} | {mb_vals[1]:.3f} |"
        f" {r1_pct:.1f}% |\n"
        f"| Regime 2 (Surge) | {m0_vals[2]:.3f} | {mb_vals[2]:.3f} |"
        f" {r2_pct:.1f}% |\n"
        f"| **Overall Held-Out Test** | **{mae_0:.3f}** (R²={r2_0:.3f}) |"
        f" **{mae_b:.3f}** (R²={r2_b:.3f}) | **{tot_pct:.1f}%** |"
    )
    reporting.embed_diagnostic_section(
        md_path, f"plots/{sol_id}_diagnostics.png", summary_md
    )


def export_top_solutions(
    task_dir: str,
    top_n: int = 3,
    write: bool = True,
) -> list[str]:
  """Exports top solution reports from leaderboard.tsv to solutions/."""
  return reporting.export_top_solutions(
      task_dir,
      sort_by=["test_r2", "test_mae"],
      ascending=[False, True],
      top_n=top_n,
      write=write,
      metric_weights=config.DEFAULT_METRIC_WEIGHTS,
      experiments_root=data_utils.experiments_root(config.TASK_NAME),
      diagnose=functools.partial(_generate_template_diagnostics, task_dir),
  )


def resolve_task_dir(base_dir: str | None = None) -> str:
  """Resolves the live task directory."""
  return data_utils.resolve_task_dir(__file__, base_dir)


# ==============================================================================
# === 3. MAIN CLASSROOM LOOP ===
# ==============================================================================


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
        "from sklearn.ensemble import HistGradientBoostingRegressor\n"
        "def solution(X_train, y_train, X_val):\n"
        "    model = HistGradientBoostingRegressor(max_iter=50,"
        " random_state=42)\n"
        "    model.fit(X_train, y_train.iloc[:, 0])\n"
        "    preds = model.predict(X_val)\n"
        "    return pd.DataFrame({'predictions': preds}, index=X_val.index),"
        " 'Mock HGBR baseline'\n"
        "```\n"
    )
    return self.current_solution


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


def main(argv: list[str]) -> None:
  del argv

  if FLAGS.report_only:
    _report_only()
    return

  task_dir = resolve_task_dir()
  data_path = os.path.join(task_dir, "data")
  x, y = data_loader.load_data(data_path)

  x_learning, x_test, y_learning, y_test = model_selection.train_test_split(
      x, y, test_size=0.2, random_state=config.VAL_SEED
  )
  x_train, x_val, y_train, y_val = model_selection.train_test_split(
      x_learning, y_learning, test_size=0.2, random_state=config.VAL_SEED
  )

  runs_base_path = os.path.join(
      data_utils.experiments_root(config.TASK_NAME), FLAGS.experiment_name
  )
  os.makedirs(runs_base_path, exist_ok=True)
  run_id = data_utils.generate_run_id()
  run_dir = os.path.join(runs_base_path, run_id)
  data_utils.create_dir(run_dir)

  logger = reporting.setup_logger(
      "codegen_classroom", log_file=os.path.join(run_dir, "run.log")
  )
  metric_weights = dict(config.DEFAULT_METRIC_WEIGHTS)
  validator = functools.partial(
      scoring.prediction_validator, expected_columns=1
  )

  student_instructions = config.build_student_instructions(
      feature_names=list(x.columns),
      target_column=y.columns[0],
      train_rows=len(x_train),
      val_rows=len(x_val),
      final_eval_rows=len(x_learning),
      timeout_s=FLAGS.code_execution_max_duration,
      final_eval_timeout_s=FLAGS.final_eval_timeout,
      metric_weights=metric_weights,
  )

  models = [
      FLAGS.model_names[i % len(FLAGS.model_names)]
      for i in range(FLAGS.num_students)
  ]
  student_cfgs = student_learning.get_classroom_student_configs(
      system_instruction=student_instructions,
      model_names=models,
      max_output_tokens=FLAGS.max_output_tokens,
  )
  agent_cls = _MockStudent if FLAGS.mock_students else agents.StudentAgent
  seed_md = data_utils.load_seed_solution(FLAGS.seed_from, task_dir=task_dir)
  students = {
      cfg.name: agent_cls(
          config=cfg, run_base_dir=run_dir, seed_solution=seed_md
      )
      for cfg in student_cfgs
  }

  current_classroom = list(students.values())
  results_history: MutableMapping[str, student_learning.ClassroomResults] = (
      collections.defaultdict(list)
  )
  active_metrics = ["overall_score"] + list(metric_weights.keys())

  for current_cycle in range(FLAGS.num_cycles):
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
        code_execution_max_retries=2,
        code_execution_timeout_seconds=FLAGS.code_execution_max_duration,
        metric_weights=metric_weights,
        early_stopping_patience=FLAGS.early_stopping_patience,
        student_refine_instruction_options=list(
            config.STUDENT_REFINE_INSTRUCTION_OPTIONS
        ),
        max_mutation_count=FLAGS.max_mutation_count,
        mutation_strategy_options=[
            agents.MutationStrategy.SELF_BEST_CODE,
            agents.MutationStrategy.BEST_PEER_CROSSOVER,
            agents.MutationStrategy.CONTRASTIVE_PEER_CROSSOVER,
        ],
        validator=validator,
        classroom_feedback=classroom_feedback,
        num_workers=len(current_classroom),
    )
    current_classroom = student_learning.process_cycle_results(
        current_classroom, classroom_results, results_history, current_cycle
    )
    for student in students.values():
      student.reset_history()

  test_results_path = os.path.join(run_dir, "final_eval_results.json")
  test_prediction_dir = os.path.join(run_dir, "predictions")
  student_learning.run_parallel_classroom_final_evaluation(
      students=students,
      x_learning=x_learning,
      y_learning=y_learning,
      x_test_final=x_test,
      y_test_final=y_test,
      student_function_name=config.STUDENT_FUNCTION_NAME,
      metric_weights=metric_weights,
      code_execution_timeout_seconds=FLAGS.final_eval_timeout,
      enforce_timeout=True,
      strategies=["first", "best", "last"],
      validator=validator,
      num_workers=len(students),
      final_results_path=test_results_path,
      predictions_dir=test_prediction_dir,
  )

  row_data = None
  if os.path.isfile(test_results_path):
    with open(test_results_path, "r", encoding="utf-8") as f:
      eval_json = json.load(f)
    best_student = max(
        students.values(),
        key=lambda s: float(s.best_score)
        if s.best_score is not None
        else -float("inf"),
    )
    best_eval_entry = next(
        (e for e in eval_json if e.get("student_name") == best_student.name),
        {},
    )
    best_eval = best_eval_entry.get("best", {})
    b_cycle = best_student.best_cycle
    val_metrics = (
        best_student.cycle_results[b_cycle].metrics
        if 0 <= b_cycle < len(best_student.cycle_results)
        else {}
    )
    best_plan = (
        best_student.cycle_results[b_cycle].plan
        if 0 <= b_cycle < len(best_student.cycle_results)
        else ""
    )
    meta = {}
    if not FLAGS.mock_students and best_student.best_solution:
      try:
        analyzer = agents.SolutionAnalysisAgent(
            config=agents.AgentConfig(name="solution_metadata"),
            run_base_dir=run_dir,
        )
        meta = (
            analyzer.analyze(
                student_name=best_student.name,
                best_code=best_student.best_solution,
                best_plan=best_plan,
                best_test_metrics=best_eval,
                best_val_metrics=val_metrics,
                student_instructions=student_instructions,
                metrics_description=scoring.get_metrics_description(
                    metric_weights
                ),
                task_description_for_prompt=config.TASK_TITLE,
            )
            or {}
        )
      except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning("SolutionAnalysisAgent failed: %s", e)

    row_data = {
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "test_r2": best_eval.get("metrics_r2"),
        "test_mae": best_eval.get("metrics_mae"),
        "test_exec_time_s": best_eval.get("metrics_exec_time"),
        "val_r2": val_metrics.get("r2"),
        "val_mae": val_metrics.get("mae"),
        "val_exec_time_s": val_metrics.get("exec_time"),
        "overall_score": best_student.best_score,
        "train_cycles": FLAGS.num_cycles,
        "train_best_cycle": b_cycle,
        "train_num_students": len(students),
        "total_tokens": None,
        "est_cost_usd": None,
        "experiment_name": FLAGS.experiment_name,
        "student": best_student.name,
        "tags": "mock" if FLAGS.mock_students else meta.get("model_tags", ""),
        "run_id": run_id,
        "run_dir": run_dir,
        "description": meta.get("model_description", ""),
        "insights": meta.get("model_insights", ""),
    }

  reporting.write_run_report(
      run_dir,
      students=list(students.values()),
      metric_weights=metric_weights,
  )

  if row_data is not None and not FLAGS.mock_students:
    lb_path = os.path.join(task_dir, "leaderboard.tsv")
    data_utils.submit_results_to_local_leaderboard(
        lb_path, pd.DataFrame([row_data])
    )

  if not FLAGS.mock_students:
    try:
      export_top_solutions(task_dir, top_n=FLAGS.export_top_n)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logger.warning("Solution archive export failed: %s", e)


if __name__ == "__main__":
  app.run(main)
