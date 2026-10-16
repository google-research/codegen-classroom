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

"""Task readiness verifier for Codegen Classroom benchmark tasks."""

from __future__ import annotations

import argparse
import ast
from collections.abc import Sequence
import dataclasses
import json
import os
import pathlib
import re
import sys

REQUIRED_TASK_FILES = (
    "README.md",
    "config.py",
    "data_loader.py",
)
CONFIG_FILENAME = "config.py"
LEADERBOARD_FILENAME = "leaderboard.tsv"
SOLUTIONS_DIRNAME = "solutions"
NOTEBOOK_SUFFIX = ".ipynb"
TEST_FILE_SUFFIX = "_test.py"

IGNORED_WALK_DIRS = (
    ".ipynb_checkpoints",
    "__pycache__",
    "artifacts",
    "cache",
    "data",
    "experiments",
    "models",
)
BUDGET_EXCLUDED_DIRS = frozenset({"solutions", "leaderboards", "data"})
MAX_TASK_FILES = 25
MIN_NBFORMAT_VERSION = 4

REQUIRED_SOLUTION_SECTIONS = (
    "# Overview",
    "## Student Leaderboard",
    "## Learning Trajectory",
)
CHECK_COLUMNS = ("Files", "Runner", "Notebook", "Solutions")


@dataclasses.dataclass
class TaskReport:
  """Readiness audit result for a single benchmark task."""

  task: str
  errors: list[str] = dataclasses.field(default_factory=list)
  warnings: list[str] = dataclasses.field(default_factory=list)
  checks: dict[str, str] = dataclasses.field(default_factory=dict)

  @property
  def passed(self) -> bool:
    return not self.errors


def _collect_task_files(task_dir: pathlib.Path) -> list[pathlib.Path]:
  """Walks a task directory, skipping ignored cache and artifact folders."""
  all_files: list[pathlib.Path] = []
  for root, dirs, files in os.walk(task_dir):
    dirs[:] = [d for d in dirs if d not in IGNORED_WALK_DIRS]
    root_path = pathlib.Path(root)
    for fname in files:
      if fname.endswith(".log"):
        continue
      all_files.append(root_path / fname)
  return all_files


def _check_config(cfg: pathlib.Path) -> list[str]:
  """Verifies config.py syntax, metric weights, and student instructions."""
  try:
    source = cfg.read_text(encoding="utf-8")
    ast.parse(source, filename=str(cfg))
  except SyntaxError as e:
    return [f"{CONFIG_FILENAME}: SyntaxError: {e}"]

  low = source.lower()
  errors = []
  if "metric_weights" not in low:
    errors.append(f"{CONFIG_FILENAME}: missing metric_weights configuration")
  if "student_instruction" not in low and "task_description" not in low:
    errors.append(f"{CONFIG_FILENAME}: missing student instructions definition")
  return errors


def _check_runner(runner: pathlib.Path) -> list[str]:
  """Verifies runner syntax, main(), codegen_classroom import, and metrics."""
  try:
    source = runner.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(runner))
  except SyntaxError as e:
    return [f"{runner.name}: SyntaxError: {e}"]

  errors = []
  has_main = any(
      isinstance(n, ast.FunctionDef) and n.name == "main"
      for n in ast.walk(tree)
  )
  if not has_main:
    errors.append(f"{runner.name}: missing 'main()' function")

  has_import = any(
      (
          isinstance(n, ast.Import)
          and any("codegen_classroom" in a.name for a in n.names)
      )
      or (
          isinstance(n, ast.ImportFrom)
          and n.module
          and "codegen_classroom" in n.module
      )
      for n in ast.walk(tree)
  )
  if not has_import:
    errors.append(f"{runner.name}: does not import 'codegen_classroom'")

  if "metric_weights" not in source.lower():
    errors.append(f"{runner.name}: missing metric_weights configuration")
  return errors


def _check_notebook(nb: pathlib.Path, base_dir: pathlib.Path) -> list[str]:
  """Verifies notebook JSON, nbformat >= 4, cell IDs, cleared outputs, and syntax."""
  rel = nb.relative_to(base_dir)
  try:
    data = json.loads(nb.read_text(encoding="utf-8"))
    if data.get("nbformat", 0) < MIN_NBFORMAT_VERSION:
      return [f"{rel}: nbformat < {MIN_NBFORMAT_VERSION}"]
    for idx, cell in enumerate(data.get("cells", []), 1):
      cell_id = cell.get("id") or cell.get("metadata", {}).get("id")
      if not cell_id or not isinstance(cell_id, str):
        return [f"{rel}:cell_{idx}: missing required string field 'id'"]
      if not isinstance(cell.get("source"), list):
        return [f"{rel}:cell_{idx}: cell 'source' must be a list of strings"]
      if cell.get("cell_type") != "code":
        continue
      if cell.get("outputs") or cell.get("execution_count") is not None:
        return [f"{rel}:cell_{idx}: outputs not cleared"]
      lines_out = []
      for line in "".join(cell.get("source", [])).splitlines():
        stripped = line.strip()
        is_magic = stripped.startswith("!") or bool(
            re.match(r"^%+[a-zA-Z]", stripped)
        )
        lines_out.append(f"# {line}" if is_magic else line)
      ast.parse("\n".join(lines_out), filename=f"{rel}:cell_{idx}")
  except (json.JSONDecodeError, SyntaxError, ValueError, OSError) as e:
    return [f"{rel}: {e}"]
  return []


def _check_leaderboard(lb: pathlib.Path, base_dir: pathlib.Path) -> list[str]:
  """Verifies that a leaderboard TSV has a valid header and no mock rows."""
  rel = lb.relative_to(base_dir)
  lines = [
      line
      for line in lb.read_text(encoding="utf-8").splitlines()
      if line.strip() and not line.strip().startswith("#")
  ]
  if not lines or "\t" not in lines[0]:
    return [f"{rel}: empty or non-tab-separated header"]

  errors = []
  for idx, line in enumerate(lines[1:], 2):
    if re.search(r"\bmock\b", line, re.I):
      errors.append(f"{rel}:{idx}: contains mock row")
  return errors


def _check_solution_md(
    sol: pathlib.Path, sol_dir: pathlib.Path, base_dir: pathlib.Path
) -> tuple[list[str], list[str]]:
  """Verifies required sections, table values, and plots in a solution report."""
  rel = sol.relative_to(base_dir)
  text = sol.read_text(encoding="utf-8", errors="replace")
  errors = []
  warnings = []

  for sec in REQUIRED_SOLUTION_SECTIONS:
    if sec == "# Overview":
      if "# Overview" not in text and "# Objectives" not in text:
        errors.append(f"{rel}: missing '{sec}'")
    elif sec not in text:
      errors.append(f"{rel}: missing '{sec}'")

  if not re.search(r"##\s+Top\s+Solution", text, re.I):
    warnings.append(f"{rel}: missing '## Top Solutions'")
  if re.search(r"\|\s*(nan|none)\s*\|", text, re.I):
    errors.append(f"{rel}: contains NaN or None table cells")

  for img in re.findall(r"!\[.*?\]\((plots/[^\)]+)\)", text):
    if not (sol.parent / img).exists() and not (sol_dir / img).exists():
      errors.append(f"{rel}: referenced plot '{img}' not found")

  stem = sol.stem
  if re.match(r"^\d", stem):
    warnings.append(
        f"{rel}: solution filename starts with a digit ('{sol.name}');"
        " place date at the end (<slug>_YYYYMMDD.md) to comply with Blaze"
        " package rules"
    )
  if "." in stem:
    warnings.append(
        f"{rel}: solution filename contains dots ('{sol.name}'); use"
        " underscores or 'p'"
    )
  if re.search(r"(?:^|_)crossover", stem, re.I):
    warnings.append(
        f"{rel}: solution filename contains 'crossover' ('{sol.name}');"
        " simplify naming"
    )

  return errors, warnings


def verify_task(
    task_dir: pathlib.Path,
) -> TaskReport:
  """Verifies that a benchmark task has all required components and contracts."""
  d = task_dir.resolve()
  rep = TaskReport(task=d.name)
  all_files = _collect_task_files(d)

  # 1. Required Files & Directory Budget
  for req_file in REQUIRED_TASK_FILES:
    if not (d / req_file).is_file():
      rep.errors.append(f"Missing {req_file}")

  runner_name = f"run_{d.name}.py"
  runners = [f for f in all_files if f.parent == d and f.name == runner_name]
  if not runners:
    rep.errors.append(f"Missing runner script {runner_name}")

  tests = [
      f
      for f in all_files
      if f.parent == d and f.name.endswith(TEST_FILE_SUFFIX)
  ]
  if not tests:
    rep.errors.append(f"Missing unit test run_{d.name}{TEST_FILE_SUFFIX}")

  leaderboards = [f for f in all_files if f.name == LEADERBOARD_FILENAME]
  if not leaderboards:
    rep.errors.append(f"Missing {LEADERBOARD_FILENAME}")

  notebooks = [f for f in all_files if f.suffix == NOTEBOOK_SUFFIX]
  if not notebooks:
    rep.warnings.append(f"Missing companion notebook (*{NOTEBOOK_SUFFIX})")

  budget_files = [
      f
      for f in all_files
      if not BUDGET_EXCLUDED_DIRS.intersection(f.relative_to(d).parts)
  ]
  if len(budget_files) > MAX_TASK_FILES:
    rep.errors.append(
        f"File budget exceeded: {len(budget_files)} files outside"
        f" {SOLUTIONS_DIRNAME}/ (max {MAX_TASK_FILES})"
    )
  rep.checks["Files"] = (
      "ERR" if rep.errors else ("WARN" if not notebooks else "OK")
  )

  # 2. Runner, Config & Metrics Contract
  runner_errors = []
  if (d / CONFIG_FILENAME).is_file():
    runner_errors.extend(_check_config(d / CONFIG_FILENAME))
  for runner in runners:
    runner_errors.extend(_check_runner(runner))
  rep.errors.extend(runner_errors)
  rep.checks["Runner"] = "ERR" if runner_errors else "OK"

  # 3. Companion Notebook
  nb_errors = []
  for nb in notebooks:
    nb_errors.extend(_check_notebook(nb, d))
  rep.errors.extend(nb_errors)
  rep.checks["Notebook"] = (
      "ERR" if nb_errors else ("WARN" if not notebooks else "OK")
  )

  # 4. Leaderboard & Solutions
  sol_errors = []
  for lb in leaderboards:
    sol_errors.extend(_check_leaderboard(lb, d))

  sol_dir = d / SOLUTIONS_DIRNAME
  if sol_dir.is_dir():
    sol_files = [
        f
        for f in all_files
        if f.suffix == ".md" and SOLUTIONS_DIRNAME in f.relative_to(d).parts
    ]
    for sol in sol_files:
      errs, warns = _check_solution_md(sol, sol_dir, d)
      sol_errors.extend(errs)
      rep.warnings.extend(warns)

  rep.errors.extend(sol_errors)
  rep.checks["Solutions"] = "ERR" if sol_errors else "OK"

  return rep


def format_table(reports: Sequence[TaskReport]) -> str:
  """Outputs concise status lines for audited tasks."""
  lines = []
  for r in reports:
    status = "PASS" if r.passed else "FAIL"
    lines.append(f"{status}: {r.task}")
  return "\n".join(lines)


def format_report(reports: Sequence[TaskReport]) -> str:
  """Formats simple status lines along with any per-task errors or warnings."""
  out = []
  for r in reports:
    status = "PASS" if r.passed else "FAIL"
    out.append(f"{status}: {r.task}")
    for e in r.errors:
      out.append(f"  ERROR: {e}")
    for w in r.warnings:
      out.append(f"  WARN:  {w}")
  return "\n".join(out) + "\n"


def find_tasks_root(start_dir: pathlib.Path | None = None) -> pathlib.Path:
  """Locates the benchmark tasks root directory."""
  if start_dir and start_dir.is_dir():
    return start_dir.resolve()
  curr = pathlib.Path(__file__).resolve()
  candidates = [curr.parent / "tasks", curr.parent]
  ws = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
  if ws:
    candidates.insert(0, pathlib.Path(ws) / "tasks")
    candidates.insert(0, pathlib.Path(ws) / "codegen_classroom/tasks")
  for cand in candidates:
    if cand and (cand / "template").is_dir():
      return cand.resolve()
  return (curr.parent / "tasks").resolve()


def main(argv: Sequence[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description="Verify benchmark tasks.")
  parser.add_argument("--task", type=str, help="Specific task name.")
  parser.add_argument("--task_dir", type=str, help="Path to task dir.")
  parser.add_argument("--all", action="store_true", help="Audit all tasks.")
  parser.add_argument(
      "--strict", action="store_true", help="Treat warnings as errors."
  )
  parser.add_argument(
      "--json", action="store_true", help="Output JSON results."
  )
  parser.add_argument(
      "--log_file",
      type=str,
      default=None,
      help="Optional path to write the combined verification .log output.",
  )
  args = parser.parse_args(argv)

  tasks_root = find_tasks_root(
      pathlib.Path(args.task_dir) if args.task_dir else None
  )
  if args.task_dir:
    task_dirs = [pathlib.Path(args.task_dir).resolve()]
  elif args.task:
    task_dirs = [tasks_root / args.task]
  else:
    task_dirs = sorted([
        d
        for d in tasks_root.iterdir()
        if d.is_dir() and not d.name.startswith((".", "_"))
    ])

  reports = [verify_task(td) for td in task_dirs if td.is_dir()]
  report_text = (
      json.dumps([dataclasses.asdict(r) for r in reports], indent=2) + "\n"
      if args.json
      else "\n" + format_report(reports)
  )
  print(report_text, end="")

  if args.log_file:
    log_path = pathlib.Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(report_text.lstrip("\n"), encoding="utf-8")

  if all(r.passed for r in reports) and not (
      args.strict and any(r.warnings for r in reports)
  ):
    return 0
  return 1


if __name__ == "__main__":
  sys.exit(main())
