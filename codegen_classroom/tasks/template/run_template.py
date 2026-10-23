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
import re
import tempfile
from typing import MutableMapping

from absl import app
from absl import flags
from codegen_classroom import agent_config
from codegen_classroom import agents
from codegen_classroom import data_utils
from codegen_classroom import reporting
from codegen_classroom import scoring
from codegen_classroom import student_learning
from codegen_classroom.tasks.template import config
from codegen_classroom.tasks.template import data_loader
import pandas as pd
from sklearn import model_selection

# ==============================================================================
# === 1. TASK CONFIGURATION & PROMPTS (Imported from config.py) ===
# ==============================================================================

TASK_NAME = config.TASK_NAME
TASK_TITLE = config.TASK_TITLE
DEFAULT_STUDENT_MODELS = config.DEFAULT_STUDENT_MODELS
RUN_DEFAULTS = config.RUN_DEFAULTS
DEFAULT_METRIC_WEIGHTS = config.DEFAULT_METRIC_WEIGHTS
CANDIDATE_MODULES = config.CANDIDATE_MODULES
FORBIDDEN_MODULES = config.FORBIDDEN_MODULES
SECTION_TITLES = config.SECTION_TITLES
STUDENT_INITIAL_PROMPT = config.STUDENT_INITIAL_PROMPT
STUDENT_REFINE_INSTRUCTION_OPTIONS = config.STUDENT_REFINE_INSTRUCTION_OPTIONS

assert_no_recipes = config.assert_no_recipes
probe_available_modules = config.probe_available_modules
build_student_instructions = config.build_student_instructions


FLAGS = flags.FLAGS

flags.DEFINE_integer(
    "num_cycles", RUN_DEFAULTS["num_cycles"], "Number of learning cycles."
)
flags.DEFINE_string(
    "experiment_name", "template_benchmark", "Experiment group name."
)
flags.DEFINE_list(
    "model_names",
    list(DEFAULT_STUDENT_MODELS),
    "Student LLM model names to use.",
)
flags.DEFINE_integer(
    "num_students",
    RUN_DEFAULTS["num_students"],
    "Number of students in the classroom.",
)
flags.DEFINE_integer(
    "code_execution_max_duration",
    RUN_DEFAULTS["timeout_s"],
    "Per-solution execution timeout in seconds.",
)
flags.DEFINE_integer(
    "final_eval_timeout",
    RUN_DEFAULTS["final_eval_timeout_s"],
    "Final evaluation timeout in seconds.",
)
flags.DEFINE_integer(
    "early_stopping_patience",
    RUN_DEFAULTS["patience"],
    "Cycles without improvement before student mutation.",
)
flags.DEFINE_integer(
    "max_mutation_count",
    RUN_DEFAULTS["max_mutations"],
    "Maximum mutation attempts before student exclusion.",
)
flags.DEFINE_string(
    "report_only",
    None,
    "Regenerate report.md for an existing run directory (or 'latest') and"
    " exit.",
)
flags.DEFINE_integer(
    "precision",
    3,
    "Decimal precision for float values in tables and report.md.",
)
flags.DEFINE_bool(
    "mock_students",
    False,
    "Use a deterministic mock student for hermetic testing without API calls.",
)
flags.DEFINE_string(
    "task_dir",
    None,
    "Explicit path to the task directory (overrides automatic resolution).",
)
flags.DEFINE_integer(
    "max_output_tokens",
    RUN_DEFAULTS["max_output_tokens"],
    "Maximum output tokens for student LLM completions.",
)
flags.DEFINE_integer(
    "export_top_n",
    3,
    "Number of top solutions to export to solutions/.",
)

agent_config.load_env()


# ==============================================================================
# === 3. PATH RESOLUTION & TOP-SOLUTION ARCHIVE ===
# ==============================================================================


def resolve_task_dir(base_dir: str | None = None) -> str:
  """Resolves the live task directory."""
  if FLAGS.is_parsed() and FLAGS.task_dir:
    return os.path.abspath(FLAGS.task_dir)
  if base_dir is not None:
    return os.path.abspath(base_dir)
  return data_utils.resolve_workspace_dir(
      os.path.dirname(os.path.realpath(__file__))
  )


def get_experiment_runs_base_path() -> str:
  """Returns the directory for experiment outputs (`CODEGEN_EXPERIMENTS_DIR`)."""
  runs_base = os.environ.get(
      "CODEGEN_EXPERIMENTS_DIR",
      os.path.join(tempfile.gettempdir(), "codegen_classroom", TASK_NAME),
  )
  return os.path.join(runs_base, FLAGS.experiment_name)


def export_top_solutions(
    task_dir: str,
    top_n: int = 3,
    write: bool = True,
) -> list[str]:
  """Exports top `top_n` non-mock leaderboard runs to `solutions/<id>.md`."""
  lb_path = os.path.join(task_dir, "leaderboard.tsv")
  if not os.path.isfile(lb_path):
    return []
  try:
    df = pd.read_csv(lb_path, sep="\t")
  except Exception:  # pylint: disable=broad-exception-caught
    return []
  if df.empty or "test_r2" not in df.columns:
    return []

  if "tags" in df.columns:
    df = df[df["tags"].fillna("").astype(str).str.strip().str.lower() != "mock"]
  df = df[pd.to_numeric(df["test_r2"], errors="coerce").notna()].copy()
  if df.empty:
    return []

  df = df.sort_values(by=["test_r2", "test_mae"], ascending=[False, True])
  solutions_dir = os.path.join(task_dir, "solutions")
  os.makedirs(solutions_dir, exist_ok=True)

  experiments_root = os.environ.get(
      "CODEGEN_EXPERIMENTS_DIR",
      os.path.join(tempfile.gettempdir(), "codegen_classroom", TASK_NAME),
  )

  exported: list[str] = []
  seen_runs: set[str] = set()
  for c in df.to_dict("records"):
    if len(exported) >= top_n:
      break
    raw_rd = str(c.get("run_dir", "") or "").strip()
    if raw_rd in seen_runs:
      continue
    seen_runs.add(raw_rd)

    ts = str(c.get("timestamp", ""))
    dt = pd.to_datetime(ts, errors="coerce")
    date_str = "00000000" if pd.isna(dt) else dt.strftime("%Y%m%d")
    exp_slug = re.sub(
        r"[^a-z0-9]+", "_", str(c.get("experiment_name", "")).lower()
    ).strip("_")
    exp_slug = re.sub(r"_(deep_)?crossover", "", exp_slug)
    exp_slug = re.sub(r"(deep_)?crossover_?", "", exp_slug).strip("_")
    student_slug = re.sub(
        r"[^a-z0-9]+", "_", str(c.get("student", "")).lower()
    ).strip("_")
    raw_cycle = c.get("train_best_cycle")
    cycle_str = (
        f"c{int(float(raw_cycle))}"
        if raw_cycle is not None and pd.notna(raw_cycle)
        else "c0"
    )
    r2_val = float(c.get("test_r2", 0.0))
    r2_str = f"r2_{r2_val:.3f}".replace(".", "p")
    suffix = f"_{date_str}" if date_str != "00000000" else ""
    sol_id = f"{exp_slug}_{student_slug}_{cycle_str}_{r2_str}{suffix}"
    exported.append(sol_id)

    if not write:
      continue

    rd = None
    if raw_rd:
      for cand in (
          raw_rd,
          os.path.expanduser(raw_rd),
          os.path.join(experiments_root, raw_rd),
          os.path.join(task_dir, raw_rd),
      ):
        if os.path.isdir(cand):
          rd = os.path.abspath(cand)
          break
    if not rd:
      continue

    out_file = os.path.join(solutions_dir, f"{sol_id}.md")
    report_sections = reporting.build_run_report(
        rd,
        metric_weights=DEFAULT_METRIC_WEIGHTS,
        solution_id=sol_id,
        metadata=c,
    )
    reporting.write_report(out_file, report_sections, precision=FLAGS.precision)

  if write and exported:
    keep_mds = {f"{sid}.md" for sid in exported}
    keep_pngs = {f"{sid}.png" for sid in exported}
    for fname in os.listdir(solutions_dir):
      if fname.endswith(".md") and fname not in keep_mds:
        os.remove(os.path.join(solutions_dir, fname))
    plots_dir = os.path.join(solutions_dir, "plots")
    if os.path.isdir(plots_dir):
      for fname in os.listdir(plots_dir):
        if fname.endswith(".png") and fname not in keep_pngs:
          os.remove(os.path.join(plots_dir, fname))

  return exported


# ==============================================================================
# === 4. MAIN CLASSROOM LOOP ===
# ==============================================================================


class _MockStudent(agents.StudentAgent):
  """Deterministic mock student for fast offline unit testing."""

  def generate_code(self, prompt: str) -> str:
    del prompt
    self.generate_code_count += 1
    return (
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


def main(argv: list[str]) -> None:
  for var in (
      "OMP_NUM_THREADS",
      "MKL_NUM_THREADS",
      "OPENBLAS_NUM_THREADS",
      "VECLIB_MAXIMUM_THREADS",
      "NUMEXPR_NUM_THREADS",
  ):
    os.environ[var] = "4"

  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  task_dir = resolve_task_dir()
  runs_base_path = get_experiment_runs_base_path()

  if FLAGS.report_only:
    target_dir = FLAGS.report_only
    if target_dir == "latest":
      subdirs = sorted([
          os.path.join(runs_base_path, d)
          for d in os.listdir(runs_base_path)
          if os.path.isdir(os.path.join(runs_base_path, d))
      ])
      if not subdirs:
        return
      target_dir = subdirs[-1]
    sections = reporting.build_run_report(
        target_dir, metric_weights=DEFAULT_METRIC_WEIGHTS
    )
    reporting.write_report(target_dir, sections, precision=FLAGS.precision)
    export_top_solutions(task_dir, top_n=FLAGS.export_top_n, write=True)
    return

  data_path = os.path.join(task_dir, "data")
  x, y = data_loader.load_data(data_path)

  x_learning, x_test, y_learning, y_test = model_selection.train_test_split(
      x, y, test_size=0.2, random_state=42
  )
  x_train, x_val, y_train, y_val = model_selection.train_test_split(
      x_learning, y_learning, test_size=0.2, random_state=42
  )

  run_id = data_utils.generate_run_id()
  run_dir = os.path.join(runs_base_path, run_id)
  data_utils.create_dir(run_dir)

  logger = reporting.setup_logger(
      "codegen_classroom", log_file=os.path.join(run_dir, "run.log")
  )
  metric_weights = dict(DEFAULT_METRIC_WEIGHTS)
  validator = functools.partial(
      scoring.prediction_validator, expected_columns=1
  )

  student_instructions = build_student_instructions(
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
  students = {
      cfg.name: agent_cls(config=cfg, run_base_dir=run_dir)
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
        student_initial_prompt=STUDENT_INITIAL_PROMPT,
        student_function_name=config.STUDENT_FUNCTION_NAME,
        code_execution_max_retries=2,
        code_execution_timeout_seconds=FLAGS.code_execution_max_duration,
        metric_weights=metric_weights,
        early_stopping_patience=FLAGS.early_stopping_patience,
        student_refine_instruction_options=list(
            STUDENT_REFINE_INSTRUCTION_OPTIONS
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
      student_function_name="solution",
      metric_weights=metric_weights,
      code_execution_timeout_seconds=FLAGS.final_eval_timeout,
      enforce_timeout=True,
      strategies=["first", "best", "last"],
      validator=validator,
      num_workers=len(students),
      final_results_path=test_results_path,
      predictions_dir=test_prediction_dir,
  )

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
    if not FLAGS.mock_students and best_student.best_solution:
      try:
        analyzer = agents.SolutionAnalysisAgent(
            config=agents.AgentConfig(name="solution_metadata"),
            run_base_dir=run_dir,
        )
        analyzer.analyze(
            student_name=best_student.name,
            best_code=best_student.best_solution,
            best_plan=best_plan,
            best_test_metrics=best_eval,
            best_val_metrics=val_metrics,
            student_instructions=student_instructions,
            metrics_description=scoring.get_metrics_description(metric_weights),
            task_description_for_prompt=config.TASK_TITLE,
        )
      except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning("SolutionAnalysisAgent failed: %s", e)

    sections = reporting.build_run_report(
        run_dir,
        students=list(students.values()),
        metric_weights=metric_weights,
    )
    reporting.write_report(run_dir, sections, precision=FLAGS.precision)

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
        "experiment_name": (
            re.sub(r"(deep_)?crossover_?", "", FLAGS.experiment_name).strip("_")
        ),
        "student": best_student.name,
        "tags": "mock" if FLAGS.mock_students else "",
        "run_id": run_id,
        "run_dir": run_dir,
    }
    lb_path = os.path.join(task_dir, "leaderboard.tsv")
    if not FLAGS.mock_students:
      data_utils.submit_results_to_local_leaderboard(
          lb_path, pd.DataFrame([row_data])
      )
      export_top_solutions(task_dir, top_n=FLAGS.export_top_n, write=True)
  else:
    sections = reporting.build_run_report(
        run_dir,
        students=list(students.values()),
        metric_weights=metric_weights,
    )
    reporting.write_report(run_dir, sections, precision=FLAGS.precision)


if __name__ == "__main__":
  app.run(main)
