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

"""Reporting module for codegen classroom."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import difflib
import json
import logging
import os
import re
import sys
from typing import Any
import warnings

from codegen_classroom import data_utils
from codegen_classroom import plotting
from codegen_classroom import scoring
from etils import epath
import numpy as np
import pandas as pd


class _CodegenStreamHandler(logging.StreamHandler):
  """Console handler, tagged by type so setup_logger can find and reuse it."""


class _CodegenFileHandler(logging.StreamHandler):
  """File handler over an epath-opened stream, tagged with its destination."""

  def __init__(self, stream, log_file: str):
    super().__init__(stream)
    self.log_file = log_file


def setup_logger(
    name: str, level: int = logging.INFO, log_file: str | None = None
) -> logging.Logger:
  """Sets up an idempotent logger with a StreamHandler and optional file handler.

  Reconfigures handlers in place instead of blindly replacing them (fixes Bug
  A1, where calling setup_logger twice in a notebook silently destroyed the
  file handler mid-run).

  Args:
    name: Name of the logger instance.
    level: Logging level for the stream handler.
    log_file: Optional path to a log file for DEBUG output.

  Returns:
    Configured logging.Logger instance.
  """
  logger = logging.getLogger(name)
  logger.setLevel(logging.DEBUG)
  logger.propagate = False

  formatter = logging.Formatter(
      "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
  )

  stream_handler = next(
      (h for h in logger.handlers if isinstance(h, _CodegenStreamHandler)),
      None,
  )
  if stream_handler is None:
    stream_handler = _CodegenStreamHandler()
    logger.addHandler(stream_handler)
  stream_handler.setLevel(level)
  stream_handler.setFormatter(formatter)

  if log_file:
    existing = next(
        (h for h in logger.handlers if isinstance(h, _CodegenFileHandler)),
        None,
    )
    if existing is not None and existing.log_file != log_file:
      logger.removeHandler(existing)
      existing.close()
      existing = None

    if existing is None:
      epath.Path(log_file).parent.mkdir(parents=True, exist_ok=True)
      existing = _CodegenFileHandler(epath.Path(log_file).open("a"), log_file)
      logger.addHandler(existing)

    existing.setLevel(logging.DEBUG)
    existing.setFormatter(formatter)

  return logger


def suppress_common_warnings() -> None:
  """Suppresses common noisy warnings related to pandas updates and dtype issues."""
  warnings.filterwarnings(
      "ignore",
      message=".*The behavior will change in pandas 3.0.*",
      category=FutureWarning,
  )
  warnings.filterwarnings(
      "ignore",
      message=re.escape("Setting an item of incompatible dtype is deprecated")
      + r".*",
      category=FutureWarning,
  )
  warnings.filterwarnings(
      "ignore",
      message=re.escape(
          "Downcasting object dtype arrays on .fillna, .ffill, .bfill is"
          " deprecated"
      )
      + r".*",
      category=FutureWarning,
  )
  warnings.filterwarnings(
      "ignore",
      message=re.escape(
          "Series.__getitem__ treating keys as positions is deprecated"
      )
      + r".*",
      category=FutureWarning,
  )
  for msg in (
      "invalid value encountered in log1p",
      "invalid value encountered in sqrt",
      "divide by zero encountered in divide",
  ):
    warnings.filterwarnings("ignore", message=msg, category=RuntimeWarning)
  warnings.filterwarnings(
      "ignore", category=FutureWarning, message=".*multi_class.*"
  )
  warnings.filterwarnings(
      "ignore", category=FutureWarning, message=".*liblinear.*"
  )
  warnings.filterwarnings(
      "ignore",
      message="Skipping features without any observed values",
      category=UserWarning,
  )


def is_colab() -> bool:
  """Returns True if the code is running in a Google Colab notebook."""
  return "google.colab" in sys.modules


def is_notebook() -> bool:
  """Returns True if running in a Jupyter-like notebook environment (including Colab)."""
  if is_colab():
    return True
  try:
    from IPython import get_ipython  # pylint: disable=g-import-not-at-top

    shell_obj = get_ipython()
    if not shell_obj:
      return False
    shell = shell_obj.__class__.__name__
    if shell == "ZMQInteractiveShell":
      return True
    if shell == "TerminalInteractiveShell":
      return False
  except (NameError, AttributeError, ImportError):
    pass
  return False


def render_markdown(text: str) -> None:
  """Renders Markdown text in notebooks, or prints plain text in terminals."""
  if is_notebook():
    from IPython import display as ipython_display  # pylint: disable=g-import-not-at-top

    ipython_display.display(ipython_display.Markdown(text))
  else:
    print(text)


def render_code(code: str, language: str = "python") -> None:
  """Renders a fenced code block."""
  fenced = f"```{language}\n{code}\n```"
  if is_notebook():
    from IPython import display as ipython_display  # pylint: disable=g-import-not-at-top

    ipython_display.display(ipython_display.Markdown(fenced))
  else:
    print(fenced)


def colorized_html_diff(text1: str, text2: str, dark_mode: bool = False) -> str:
  """For Colab: generates a colorized HTML diff between two code texts."""
  diff = difflib.ndiff(
      text1.splitlines(keepends=True), text2.splitlines(keepends=True)
  )
  result = []
  add_color = "#223922" if dark_mode else "#d4fcdc"
  del_color = "#451e1e" if dark_mode else "#ffe0e0"

  for line in diff:
    sign = line[0]
    content = line[2:]
    escaped = (
        content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )
    if sign == " ":
      result.append(f" {escaped}")
    elif sign == "+":
      result.append(
          f"<span style='background-color: {add_color};'>+ {escaped}</span>"
      )
    elif sign == "-":
      result.append(
          f"<span style='background-color: {del_color};'>- {escaped}</span>"
      )
  return "".join(result)


def render_code_diff(old: str, new: str, dark_mode: bool = False) -> None:
  """Renders a code diff (colored HTML in notebooks, unified diff text in CLI)."""
  if is_notebook():
    from IPython import display as ipython_display  # pylint: disable=g-import-not-at-top

    code_diff_html = colorized_html_diff(old, new, dark_mode=dark_mode)
    ipython_display.display(
        ipython_display.HTML(
            f"<pre style='font-family: monospace;'>{code_diff_html}</pre>"
        )
    )
  else:
    diff = difflib.unified_diff(
        old.splitlines(keepends=False),
        new.splitlines(keepends=False),
        lineterm="",
    )
    print("\n".join(diff))


def render_json(data: dict[str, Any]) -> None:
  """Renders JSON data."""
  json_str = json.dumps(data, indent=2)
  if is_notebook():
    from IPython import display as ipython_display  # pylint: disable=g-import-not-at-top

    ipython_display.display(
        ipython_display.Markdown(f"```json\n{json_str}\n```")
    )
  else:
    print(f"```json\n{json_str}\n```")


def render_dataframe(df: pd.DataFrame, precision: int | None = 3) -> None:
  """Renders a pandas DataFrame."""
  df_display = df.round(precision) if precision is not None else df
  if is_notebook():
    from IPython import display as ipython_display  # pylint: disable=g-import-not-at-top

    ipython_display.display(df_display)
  else:
    print(df_display.to_string())


def render_leaderboard(path: str | epath.Path) -> None:
  """Reads a leaderboard TSV file via epath and renders it."""
  try:
    df = data_utils.load_leaderboard(path)
    render_dataframe(df)
  except OSError as e:
    render_markdown(f"**Error reading leaderboard from {path}:** {e}")
  except pd.errors.EmptyDataError:
    render_markdown(f"**Leaderboard file {path} is empty.**")


@dataclasses.dataclass(frozen=True)
class ReportSection:
  title: str
  kind: str
  payload: Any
  level: int = 2


def _format_cell_value(val: Any, precision: int = 3) -> str:
  """Formats a table or key-value cell, rounding floats to `precision` places."""
  if val is None or pd.isna(val):
    return ""
  if type(val) in (int, np.integer, bool):
    return str(val)
  try:
    f = float(val)
    if not np.isfinite(f):
      return ""
    return f"{f:.{precision}f}"
  except (ValueError, TypeError):
    return str(val)


def _is_empty_cell(val: Any) -> bool:
  if val is None or pd.isna(val):
    return True
  s = str(val).strip().lower()
  return s in ("", "—", "-", "none", "nan", "null")


def build_run_report(
    run_dir: str | epath.Path,
    students: list[Any] | None = None,
    metric_weights: Mapping[str, float] | None = None,
    top_n: int = 2,
    worst_n: int = 1,
    include_plots: bool = True,
    solution_id: str | None = None,
) -> list[ReportSection]:
  """Builds a complete run report reading only from the run directory."""
  run_dir_path = epath.Path(run_dir)
  students_dir = run_dir_path / "students"
  final_eval_path = run_dir_path / "final_eval_results.json"
  effective_solution_id = solution_id or run_dir_path.name

  student_results = (
      data_utils.load_student_results(students_dir)
      if students_dir.exists()
      else {}
  )

  def _get_cycle_num(rec: Mapping[str, Any] | None) -> int | None:
    if not rec:
      return None
    try:
      c = int(float(rec.get("cycle", -1)))
      return c if c >= 0 else None
    except (ValueError, TypeError):
      return None

  # Compute stats
  total_cycles = 0
  total_code_len = 0
  total_generations = 0
  valid_students = 0

  for _, recs in student_results.items():
    if any(
        r.get("overall_score") is not None
        and not pd.isna(r.get("overall_score"))
        for r in recs
    ):
      valid_students += 1
    for r in recs:
      c_num = _get_cycle_num(r)
      if c_num is not None:
        total_cycles = max(total_cycles, c_num + 1)
      total_code_len += int(r.get("code_length", 0))
      total_generations += 1 + int(r.get("retry_count", 0))

  if students:
    total_gen = sum(getattr(s, "generate_code_count", 0) for s in students)
    if total_gen > 0:
      total_generations = total_gen

  # Load final eval results
  eval_map: dict[str, Any] = {}
  if final_eval_path.exists():
    raw_eval = data_utils.load_json_from_file(str(final_eval_path))
    for item in (raw_eval if isinstance(raw_eval, Sequence) else [raw_eval]):
      if isinstance(item, Mapping) and "student_name" in item:
        eval_map[str(item["student_name"])] = item

  all_students = list(
      dict.fromkeys(
          [getattr(s, "name", str(s)) for s in students or []]
          + list(eval_map.keys())
          + list(student_results.keys())
      )
  )

  overview_payload = {
      "Run Path": str(run_dir_path),
      "Name": run_dir_path.name,
      "Total Cycles": total_cycles,
      "Valid Students": valid_students,
      "Total Generations": total_generations,
      "Total Code Length Processed": total_code_len,
  }
  if metric_weights:
    overview_payload["_metric_weights"] = dict(metric_weights)

  sections: list[ReportSection] = [
      ReportSection("Overview", "keyvalue", overview_payload, level=1)
  ]

  if metric_weights:
    metrics_desc = scoring.get_metrics_description(metric_weights)
    if metrics_desc:
      sections.append(
          ReportSection("Objectives", "markdown", metrics_desc, level=2)
      )

  # Student Leaderboard
  students_info = []
  for s_name in all_students:
    recs = student_results.get(s_name, [])
    valid_recs = []
    for r in recs:
      score_val = r.get("overall_score")
      if score_val is not None and not pd.isna(score_val):
        try:
          f = float(score_val)
          if np.isfinite(f):
            valid_recs.append((f, r))
        except (ValueError, TypeError):
          pass

    best_val_row = (
        max(valid_recs, key=lambda pair: pair[0])[1] if valid_recs else None
    )
    val_score = float(best_val_row["overall_score"]) if best_val_row else None
    best_cycle = _get_cycle_num(best_val_row)

    eval_best = (eval_map.get(s_name) or {}).get("best") or {}
    test_score = eval_best.get(
        "overall_score", eval_best.get("test_overall_score")
    )
    if test_score is not None and not pd.isna(test_score):
      try:
        f_test = float(test_score)
        test_score = f_test if np.isfinite(f_test) else None
      except (ValueError, TypeError):
        test_score = None
    else:
      test_score = None

    raw_metrics = {**(best_val_row or {}), **eval_best}
    metrics = {}
    for k, v in raw_metrics.items():
      if v is None or pd.isna(v):
        continue
      clean_k = k.removeprefix("metrics_")
      if k.startswith("metrics_") or (
          metric_weights and k in metric_weights and k != "overall_score"
      ):
        metrics[clean_k] = v

    students_info.append({
        "name": s_name,
        "best_cycle": best_cycle,
        "val_score": val_score,
        "test_score": test_score,
        "metrics": metrics,
        "best_val_row": best_val_row,
        "eval_best": eval_best,
    })

  if metric_weights:
    candidate_metric_cols = [
        m for m in metric_weights.keys() if m != "overall_score"
    ]
  else:
    collected_keys = set()
    for s in students_info:
      collected_keys.update(s["metrics"])
    candidate_metric_cols = sorted(collected_keys)

  # Only keep metric columns that have at least one valid non-empty value
  metric_cols = [
      m
      for m in candidate_metric_cols
      if any(not _is_empty_cell(s["metrics"].get(m)) for s in students_info)
  ]

  students_info.sort(
      key=lambda s: (
          s["test_score"] if s["test_score"] is not None else -float("inf"),
          s["val_score"] if s["val_score"] is not None else -float("inf"),
      ),
      reverse=True,
  )

  has_best_cycle = any(s["best_cycle"] is not None for s in students_info)
  has_val_score = any(s["val_score"] is not None for s in students_info)
  has_test_score = any(s["test_score"] is not None for s in students_info)

  leaderboard_rows = []
  for rank, s in enumerate(students_info, 1):
    row: dict[str, Any] = {
        "Rank": rank,
        "Student": s["name"],
    }
    if has_best_cycle:
      row["Best Cycle"] = (
          s["best_cycle"] if s["best_cycle"] is not None else "—"
      )
    if has_val_score:
      row["Val Score"] = (
          _format_cell_value(s["val_score"])
          if s["val_score"] is not None
          else "—"
      )
    if has_test_score:
      row["Test Score"] = (
          _format_cell_value(s["test_score"])
          if s["test_score"] is not None
          else "—"
      )
    for m in metric_cols:
      m_val = s["metrics"].get(m)
      row[m] = _format_cell_value(m_val) if not _is_empty_cell(m_val) else "—"
    leaderboard_rows.append(row)

  if leaderboard_rows:
    sections.append(
        ReportSection(
            "Student Leaderboard",
            "dataframe",
            pd.DataFrame(leaderboard_rows),
            level=2,
        )
    )

  if include_plots:
    sections.append(
        ReportSection(
            "Learning Trajectory",
            "markdown",
            f"![Classroom Progress](plots/{effective_solution_id}.png)\n",
            level=2,
        )
    )

  # Pivoted progression table
  progression_students = [
      s
      for s in all_students
      if any(
          r.get("overall_score") is not None
          and not pd.isna(r.get("overall_score"))
          for r in student_results.get(s, [])
      )
  ]
  if total_cycles > 0 and progression_students:
    progression_rows = []
    student_running_best = {s: -float("inf") for s in progression_students}
    classroom_running_best = -float("inf")

    for cycle_idx in range(total_cycles):
      row: dict[str, Any] = {"Cycle": cycle_idx}
      for s in progression_students:
        c_rec = next(
            (
                r
                for r in student_results.get(s, [])
                if _get_cycle_num(r) == cycle_idx
            ),
            None,
        )
        if not c_rec:
          row[s] = "—"
        else:
          score_val = c_rec.get("overall_score")
          if score_val is not None and not pd.isna(score_val):
            try:
              score_f = float(score_val)
              if np.isfinite(score_f):
                is_class_best = score_f > classroom_running_best
                is_self_best = score_f > student_running_best[s]
                tag = (
                    " *(CLASSROOM_BEST)*"
                    if is_class_best
                    else (" *(SELF_BEST)*" if is_self_best else "")
                )
                if is_class_best:
                  classroom_running_best = score_f
                if is_self_best:
                  student_running_best[s] = score_f
                row[s] = f"{score_f:.3f}{tag}"
              else:
                row[s] = "—"
            except (ValueError, TypeError):
              row[s] = "—"
          else:
            row[s] = "—"
      progression_rows.append(row)

    has_numeric = any(
        any(v != "—" for k, v in r.items() if k != "Cycle")
        for r in progression_rows
    )
    if has_numeric:
      sections.append(
          ReportSection(
              "Cycle Score Progression",
              "dataframe",
              pd.DataFrame(progression_rows),
              level=3,
          )
      )

  # Candidate solutions (Top N and Lowest-Scoring)
  valid_candidates = []
  for s_info in students_info:
    s_name, best_c = s_info["name"], s_info["best_cycle"]
    if best_c is None or str(best_c) in ("N/A", "None", "", "-1"):
      continue

    b_row = s_info["best_val_row"] or {}
    plan_text = (
        b_row.get("plan")
        or b_row.get("student_plan")
        or b_row.get("description")
    )
    code_text = b_row.get("code")
    exec_feedback = (
        b_row.get("execution_feedback")
        or b_row.get("stdout")
        or b_row.get("output")
    )

    if (
        not code_text or not plan_text or not exec_feedback
    ) and students_dir.exists():
      csv_p = students_dir / f"{s_name}.csv"
      if csv_p.exists():
        try:
          df_hist = pd.read_csv(csv_p)
          m = (
              df_hist[df_hist["cycle"] == best_c]
              if "cycle" in df_hist.columns
              else pd.DataFrame()
          )
          if not m.empty:
            row0 = m.iloc[0]
            plan_text = (
                plan_text
                or row0.get("plan")
                or row0.get("student_plan")
                or row0.get("description")
            )
            code_text = code_text or row0.get("code")
            exec_feedback = (
                exec_feedback
                or row0.get("execution_feedback")
                or row0.get("stdout")
                or row0.get("output")
            )
        except Exception:  # pylint: disable=broad-exception-caught
          pass

    eval_best = s_info.get("eval_best") or {}
    plan_text = (
        plan_text
        or eval_best.get("plan")
        or eval_best.get("student_plan")
        or eval_best.get("description")
    )
    code_text = code_text or eval_best.get("code")
    exec_feedback = (
        exec_feedback
        or eval_best.get("execution_feedback")
        or eval_best.get("stdout")
        or eval_best.get("output")
    )

    if not code_text or not str(code_text).strip():
      continue

    valid_candidates.append({
        "name": s_name,
        "best_cycle": best_c,
        "val_score": s_info["val_score"],
        "test_score": s_info["test_score"],
        "plan": str(plan_text) if plan_text and pd.notna(plan_text) else None,
        "code": str(code_text).strip(),
        "feedback": (
            str(exec_feedback)
            if exec_feedback and pd.notna(exec_feedback)
            else None
        ),
    })

  valid_candidates.sort(
      key=lambda c: (
          1
          if c["test_score"] is not None and np.isfinite(c["test_score"])
          else 0,
          (
              c["test_score"]
              if c["test_score"] is not None and np.isfinite(c["test_score"])
              else -float("inf")
          ),
          (
              1
              if c["val_score"] is not None and np.isfinite(c["val_score"])
              else 0
          ),
          (
              c["val_score"]
              if c["val_score"] is not None and np.isfinite(c["val_score"])
              else -float("inf")
          ),
      ),
      reverse=True,
  )

  def _add_cand_sections(cand: dict[str, Any], title: str) -> None:
    v_s, t_s = cand["val_score"], cand["test_score"]
    sections.append(
        ReportSection(
            title=title,
            kind="keyvalue",
            payload={
                "Student": cand["name"],
                "Best Cycle": cand["best_cycle"],
                "Validation Score": (
                    _format_cell_value(v_s) if v_s is not None else "—"
                ),
                "Test Score": (
                    _format_cell_value(t_s) if t_s is not None else "—"
                ),
            },
            level=2,
        )
    )
    if cand["plan"]:
      sections.append(
          ReportSection(
              f"Plan ({cand['name']})", "markdown", cand["plan"], level=3
          )
      )
    sections.append(
        ReportSection(f"Code ({cand['name']})", "code", cand["code"], level=3)
    )
    if cand["feedback"]:
      sections.append(
          ReportSection(
              f"Execution Output ({cand['name']})",
              "text",
              cand["feedback"],
              level=3,
          )
      )

  top_candidates = valid_candidates[:top_n]
  remaining_candidates = valid_candidates[top_n:]

  for rank_idx, cand in enumerate(top_candidates, 1):
    _add_cand_sections(
        cand,
        f"Top Solution {rank_idx}: {cand['name']} (Cycle {cand['best_cycle']})",
    )

  if worst_n > 0 and remaining_candidates:
    worst_candidates = remaining_candidates[-worst_n:]
    for w_idx, cand in enumerate(worst_candidates, 1):
      if len(worst_candidates) == 1:
        title = (
            f"Lowest-Scoring Solution: {cand['name']} (Cycle"
            f" {cand['best_cycle']})"
        )
      else:
        title = (
            f"Lowest-Scoring Solution {w_idx}: {cand['name']} (Cycle"
            f" {cand['best_cycle']})"
        )
      _add_cand_sections(cand, title)

  # Solution analysis
  best_analysis_path = run_dir_path / "best_analysis.json"
  if best_analysis_path.exists():
    try:
      analysis = data_utils.load_json_from_file(str(best_analysis_path))
      if hasattr(analysis, "items"):
        sections.append(
            ReportSection(
                title="Analysis",
                kind="markdown",
                payload="\n\n".join(
                    f"### {k}\n{v}" for k, v in analysis.items()
                ),
                level=2,
            )
        )
    except Exception:  # pylint: disable=broad-exception-caught
      pass

  return sections


def build_student_report(
    student_csv: str | epath.Path, cycle: int | None = None
) -> list[ReportSection]:
  """Builds a student progress report from their CSV."""
  student_name = os.path.splitext(os.path.basename(str(student_csv)))[0]
  try:
    student_results_full = data_utils.read_dict_from_csv(str(student_csv))
  except OSError as e:
    return [
        ReportSection(
            "Error", "markdown", f"**Error reading {student_csv}:** {e}"
        )
    ]

  if not student_results_full:
    return [
        ReportSection(
            "No Data", "markdown", f"**No data found for {student_name}.**"
        )
    ]

  if cycle is not None:
    student_results = [
        c for c in student_results_full if int(c["cycle"]) == cycle
    ]
    if not student_results:
      return [
          ReportSection(
              "No Data",
              "markdown",
              f"**No data found for `{student_name}` cycle `{cycle}`.**",
          )
      ]
  else:
    student_results = student_results_full

  sections = []
  if cycle is not None:
    sections.append(
        ReportSection(
            f"Report: {student_name} (Cycle {cycle})",
            "markdown",
            "",
            level=1,
        )
    )
  else:
    sections.append(
        ReportSection(f"Report: {student_name}", "markdown", "", level=1)
    )

  for _, cycle_result in enumerate(student_results):
    current_cycle = int(cycle_result["cycle"])
    previous_cycle_result = None
    if current_cycle > 0:
      prev_cycle_num = current_cycle - 1
      previous_cycle_results = [
          c for c in student_results_full if int(c["cycle"]) == prev_cycle_num
      ]
      if previous_cycle_results:
        previous_cycle_result = previous_cycle_results[0]

    if cycle is None:
      sections.append(
          ReportSection(f"Cycle {current_cycle}", "markdown", "", level=2)
      )

    plan = str(cycle_result.get("plan", "No plan text."))
    code = str(cycle_result.get("code", ""))

    sections.append(ReportSection("Plan", "markdown", plan, level=3))
    sections.append(
        ReportSection(
            "Code",
            "code",
            code or "No code extracted for this cycle.",
            level=3,
        )
    )

    if "code" in cycle_result:
      previous_code = (
          str(previous_cycle_result["code"])
          if previous_cycle_result and "code" in previous_cycle_result
          else ""
      )
      sections.append(
          ReportSection(
              "Code Diff",
              "code_diff",
              (previous_code, str(cycle_result["code"])),
              level=3,
          )
      )

    if "code_length" in cycle_result:
      try:
        code_length = int(cycle_result["code_length"])
        previous_code_length = 0
        if previous_cycle_result and "code_length" in previous_cycle_result:
          try:
            previous_code_length = int(previous_cycle_result["code_length"])
          except (ValueError, TypeError):
            previous_code_length = 0
        delta_len = code_length - previous_code_length
        len_change = f"+{delta_len}" if delta_len > 0 else str(delta_len)

        kv_payload = {
            "Previous": str(previous_code_length),
            "Current": f"{code_length} ({len_change})",
        }
        sections.append(
            ReportSection("Code Length", "keyvalue", kv_payload, level=3)
        )
      except (ValueError, TypeError):
        pass

    exec_times = {}
    for key, value in cycle_result.items():
      if "time" in key.lower():
        try:
          exec_times[key.replace("_", " ").upper()] = f"{float(value):.4f} s"
        except (ValueError, TypeError):
          exec_times[key.replace("_", " ").upper()] = f"{value} (Non-numeric)"
    if exec_times:
      sections.append(
          ReportSection("Execution Time", "keyvalue", exec_times, level=3)
      )

    scores = {}
    for key, value in cycle_result.items():
      if "score" in key.lower() or key.startswith("metrics_"):
        try:
          scores[key.replace("_", " ").upper()] = f"{float(value):.4f}"
        except (ValueError, TypeError):
          scores[key.replace("_", " ").upper()] = f"{value} (Non-numeric)"
    if scores:
      sections.append(ReportSection("Scores", "keyvalue", scores, level=3))

  return sections


def render_report(
    sections: list[ReportSection],
    dark_mode: bool = False,
    precision: int = 3,
) -> None:
  """Renders a report to the notebook or terminal."""
  for sec in sections:
    if sec.title:
      heading = "#" * sec.level
      render_markdown(f"{heading} {sec.title}")
      if sec.kind != "plot":
        pass
    if sec.kind == "markdown":
      if sec.payload:
        render_markdown(str(sec.payload))
    elif sec.kind == "code":
      render_code(str(sec.payload))
    elif sec.kind == "text":
      render_code(str(sec.payload), language="text")
    elif sec.kind == "code_diff":
      old, new = sec.payload
      render_code_diff(old, new, dark_mode=dark_mode)
    elif sec.kind == "dataframe":
      render_dataframe(sec.payload, precision=precision)
    elif sec.kind == "plot":
      # In notebook, plot live. Headless, note it skip.
      if is_notebook():
        if sec.payload["type"] == "progress":
          plotting.plot_progress(
              sec.payload["data"],
              metric_names=["overall_score"],
              backend="plotly",
              dark_mode=dark_mode,
              show=True,
          )
        elif sec.payload["type"] == "final":
          plotting.plot_final_test(
              metrics=["overall_score"],
              final_results_path=sec.payload["data"],
              backend="plotly",
              dark_mode=dark_mode,
              show=True,
          )
      else:
        print("(Plot skipped in terminal display; noted as saved)")
    elif sec.kind == "keyvalue":
      items = {
          k: v for k, v in sec.payload.items() if not str(k).startswith("_")
      }
      if is_notebook():
        bullets = "\n".join([
            f"* **{k}:** {_format_cell_value(v, precision=precision)}"
            for k, v in items.items()
        ])
        render_markdown(bullets)
      else:
        for k, v in items.items():
          print(f"{k}: {_format_cell_value(v, precision=precision)}")


def write_report(
    run_dir: str | epath.Path,
    sections: list[ReportSection],
    precision: int = 3,
    metric_weights: Mapping[str, float] | None = None,
    save_plot: bool = True,
) -> None:
  """Writes a report to .md file."""
  target_path = epath.Path(run_dir)
  if target_path.suffix == ".md":
    md_file = target_path
    md_dir = target_path.parent
    solution_id = target_path.stem
  else:
    md_dir = target_path
    md_file = target_path / "report.md"
    solution_id = target_path.name
  plot_path = md_dir / "plots" / f"{solution_id}.png"

  md_dir.mkdir(parents=True, exist_ok=True)

  if save_plot:
    if metric_weights is None:
      for sec in sections:
        if sec.kind == "keyvalue" and hasattr(sec.payload, "get"):
          metric_weights = sec.payload.get("_metric_weights")
          if metric_weights:
            break

    source_dir = None
    for sec in sections:
      if sec.kind == "keyvalue" and hasattr(sec.payload, "get"):
        rp = sec.payload.get("Run Path")
        if rp and epath.Path(rp).exists():
          source_dir = epath.Path(rp)
          break
    if source_dir is None:
      if (md_dir / "students").exists():
        source_dir = md_dir
      elif target_path.is_dir() and (target_path / "students").exists():
        source_dir = target_path

    if source_dir:
      try:
        plotting.plot_progress_static(
            source_dir,
            metric_weights=metric_weights,
            save_path=plot_path,
            show=False,
        )
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.warning("Failed to save progress plot: %s", e)

  lines = []

  for sec in sections:
    if sec.kind == "plot":
      continue

    if (
        sec.kind == "markdown"
        and "![" in str(sec.payload)
        and "plots/" in str(sec.payload)
    ):
      if not plot_path.exists():
        continue
      if sec.title:
        lines.append(f"{'#' * sec.level} {sec.title}")
        lines.append("")
      p_str = re.sub(
          r"!\[(.*?)\]\(plots/.*?\)",
          rf"![\1](plots/{solution_id}.png)",
          str(sec.payload),
      )
      lines.append(p_str)
      lines.append("")
      continue

    if sec.title:
      lines.append(f"{'#' * sec.level} {sec.title}")
      lines.append("")

    if sec.kind == "markdown":
      if sec.payload:
        lines.append(str(sec.payload))
        lines.append("")
    elif sec.kind in ("code", "text"):
      is_text = (
          sec.kind == "text"
          or "output" in sec.title.lower()
          or "feedback" in sec.title.lower()
      )
      lang = "text" if is_text else "python"
      lines.append(f"```{lang}\n{sec.payload}\n```")
      lines.append("")
    elif sec.kind == "code_diff":
      old, new = sec.payload
      diff = difflib.unified_diff(
          old.splitlines(keepends=False),
          new.splitlines(keepends=False),
          lineterm="",
      )
      diff_text = "\n".join(diff)
      lines.append(f"```diff\n{diff_text}\n```")
      lines.append("")
    elif sec.kind == "dataframe":
      if hasattr(sec.payload, "columns") and not sec.payload.empty:
        cols = [str(c) for c in sec.payload.columns]
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join("---" for _ in cols) + " |"
        rows = [
            "| "
            + " | ".join(
                _format_cell_value(v, precision=precision) for v in row
            )
            + " |"
            for row in sec.payload.itertuples(index=False)
        ]
        lines.append("\n".join([header, sep, *rows]))
        lines.append("")
    elif sec.kind == "keyvalue":
      for k, v in sec.payload.items():
        if str(k).startswith("_"):
          continue
        val_str = _format_cell_value(v, precision=precision)
        lines.append(f"* **{k}**: {val_str}")
      lines.append("")

  output_lines = [
      "<!-- disableFinding(LINE_OVER_80) -->",
      "<!-- disableFinding(WHITESPACE_LINES) -->",
      "<!-- disableFinding(WHITESPACE_TRAILING) -->",
      "<!-- disableFinding(LIST_NO_LINE) -->",
      "",
  ]
  output_lines.extend(lines)
  clean_text = "\n".join(line.rstrip() for line in output_lines).strip() + "\n"
  md_file.write_text(clean_text)
