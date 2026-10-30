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

"""Data utils for working with DataFrames and other data."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import datetime
import json
import logging
import os
import re
import tempfile
from typing import Any

from etils import epath
from google import genai
import pandas as pd

logger = logging.getLogger(__name__)

JsonData = Sequence[Mapping[str, Any]] | Mapping[str, Any]


def generate_run_id() -> str:
  """Generates a unique run identifier."""
  return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def create_dir(path: str) -> None:
  """Creates a directory if it doesn't exist."""
  dir_path = epath.Path(path)
  if not dir_path.exists():
    dir_path.mkdir(parents=True, exist_ok=True)


def save_dict_to_json(data: JsonData, filepath: str) -> None:
  """Saves a Python mapping or sequence of mappings to a JSON file."""
  create_dir(os.path.dirname(filepath))
  path = epath.Path(filepath)
  path.write_text(json.dumps(data, indent=4))


def load_json_from_file(filepath: str) -> JsonData:
  """Loads a JSON file. Assumes file exists and is valid JSON."""
  path = epath.Path(filepath)
  return json.loads(path.read_text())


def save_dataframe_to_csv(
    df: pd.DataFrame,
    filepath: str,
    mode: str = "w",
    header: bool = True,
    index: bool = False,
) -> None:
  """Saves or appends a pandas DataFrame to a CSV file.

  Args:
      df: The pandas DataFrame to save.
      filepath: The path to the output CSV file.
      mode: File opening mode, either "w" (write, overwrite) or "a" (append).
      header: Whether to write the DataFrame column names as a header.
      index: Whether to write DataFrame index as a column(s).
  """
  create_dir(os.path.dirname(filepath))
  path = epath.Path(filepath)
  if mode == "a" and path.exists():
    header = False
  with path.open(mode) as f:
    df.to_csv(f, index=index, header=header)


def read_dict_from_csv(filepath: str) -> Sequence[Mapping[str, Any]]:
  """Reads a CSV file into a sequence of mappings.

  Args:
      filepath: The path to the input CSV file.

  Returns:
    A sequence of mappings, where each mapping represents a row.
  """
  path = epath.Path(filepath)
  try:
    with path.open("r") as f:
      df = pd.read_csv(f, on_bad_lines="warn")
    return df.to_dict(orient="records")
  except pd.errors.ParserError as e:
    print(f"Error parsing CSV file: {filepath} - {e}")
    return []


def load_student_results(
    run_dir: epath.PathLike | str,
    students: list[str] | None = None,
) -> dict[str, Sequence[Mapping[str, Any]]]:
  """Loads and returns progress data for specified students from a directory.

  Args:
    run_dir: Run directory containing a 'students' subfolder, or the students
      directory itself containing student CSV files.
    students: Optional list of student names to filter by. If None, all students
      found in the directory are loaded.

  Returns:
    A mapping from student name to a sequence of progress data records (rows).
  """
  path = epath.Path(run_dir)
  if (path / "students").is_dir():
    path = path / "students"

  if not path.exists():
    return {}

  try:
    csv_files = sorted(path.glob("*.csv"))
  except OSError as e:
    logger.warning("Error accessing directory %s: %s", run_dir, e)
    return {}

  all_students = {p.stem: str(p) for p in csv_files}
  target_students = (
      students if students is not None else list(all_students.keys())
  )

  results = {}
  for student_name in target_students:
    if student_name in all_students:
      csv_path = all_students[student_name]
      try:
        results[student_name] = read_dict_from_csv(csv_path)
      except OSError:
        pass

  return results


def config_to_serializable_dict(cfg: Any) -> Mapping[str, Any]:
  """Converts a dataclass config object to a JSON serializable dictionary."""
  if not dataclasses.is_dataclass(cfg):
    raise TypeError("Input must be a dataclass instance.")

  def _dataclass_factory(data):
    """Custom dict factory for dataclasses.asdict to handle ModelEntry."""

    def convert_value(obj):
      if hasattr(genai, "types") and isinstance(obj, genai.types.Model):
        return str(obj)
      return obj

    return {k: convert_value(v) for k, v in data}

  return dataclasses.asdict(cfg, dict_factory=_dataclass_factory)


def sanitize_path(path: epath.PathLike | str | None) -> str:
  """Sanitizes a filesystem path into `<parent>/<basename>`."""
  if not path or (isinstance(path, float) and pd.isna(path)):
    return ""
  norm = os.path.normpath(str(path).strip())
  base = os.path.basename(norm)
  parent = os.path.basename(os.path.dirname(norm))
  return f"{parent}/{base}" if parent else base


def round_leaderboard_metrics(
    df: pd.DataFrame, precision: int = 3
) -> pd.DataFrame:
  """Rounds float metric columns in a leaderboard DataFrame to precision decimal places."""
  return df.round(precision)


def submit_results_to_local_leaderboard(
    leaderboard_path: str,
    results_df_row: pd.DataFrame,
    precision: int = 3,
):
  """Appends results to a local TSV leaderboard file."""
  if not leaderboard_path.endswith(".tsv"):
    logger.warning("Leaderboard path does not end with .tsv.")

  results_df_row = results_df_row.copy()
  for col in ("run_dir", "solution_dir"):
    if col in results_df_row.columns:
      results_df_row[col] = results_df_row[col].apply(
          lambda p: sanitize_path(p) if pd.notna(p) and p else p
      )

  path = epath.Path(leaderboard_path)

  if path.exists():
    with path.open("r") as f:
      leaderboard_df = pd.read_csv(f, sep="\t")
    if leaderboard_df.empty:
      leaderboard_df = results_df_row
    else:
      # Align columns by name, filling missing fields in older rows with NaN
      # so historical runs are preserved when new analysis fields are added.
      leaderboard_df = pd.concat(
          [leaderboard_df, results_df_row], ignore_index=True
      )
  else:
    leaderboard_df = results_df_row

  leaderboard_df = round_leaderboard_metrics(
      leaderboard_df, precision=precision
  )

  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w") as f:
    leaderboard_df.to_csv(f, sep="\t", index=False)


def load_leaderboard(path: epath.PathLike | str) -> pd.DataFrame:
  """Loads a leaderboard TSV file into a pandas DataFrame.

  Args:
    path: Path to the leaderboard TSV file.

  Returns:
    A DataFrame containing the leaderboard contents.

  Raises:
    OSError: If the leaderboard file does not exist or cannot be read.
    pd.errors.EmptyDataError: If the leaderboard file is empty.
  """
  with epath.Path(path).open("r") as f:
    return pd.read_csv(f, sep="\t")


def resolve_workspace_dir(path: str) -> str:
  """Resolves a directory path, safely breaking out of Bazel runfiles if needed."""
  workspace_dir = os.environ.get("BUILD_WORKSPACE_DIRECTORY")

  if workspace_dir:
    for marker in (".runfiles/", "runfiles/"):
      if marker in path:
        suffix = path.split(marker, 1)[-1]
        parts = suffix.split("/", 1)
        if len(parts) > 1:
          return os.path.join(workspace_dir, parts[1])

  return path


def load_seed_solution(seed_from: str | None, task_dir: str) -> str | None:
  """Returns the top solution code block from a `solutions/*.md` report.

  Args:
    seed_from: None to skip seeding, "best" for the highest-scoring report in
      `<task_dir>/solutions/`, a report name (stem) in that directory, or a path
      to a report file.
    task_dir: Task directory containing the `solutions/` folder.

  Returns:
    The seeded plan and code block, or None if `seed_from` is unset.

  Raises:
    FileNotFoundError: If no matching solution report exists.
    ValueError: If the report has no `## Top Solution 1` python code block.
  """
  if not seed_from:
    return None
  sol_dir = epath.Path(task_dir) / "solutions"
  if seed_from == "best":
    score_re = re.compile(r"\*\*Best Test Score\*\*:\s*([-+]?\d*\.?\d+)")
    scored = [
        (float(m.group(1)), p)
        for p in sol_dir.glob("*.md")
        if (m := score_re.search(p.read_text()))
    ]
    if not scored:
      raise FileNotFoundError(f"No scored solutions in {sol_dir}")
    path = max(scored, key=lambda t: t[0])[1]
  else:
    path = next(
        (
            p
            for p in (epath.Path(seed_from), sol_dir / f"{seed_from}.md")
            if p.is_file()
        ),
        None,
    )
    if path is None:
      raise FileNotFoundError(f"Seed solution {seed_from!r} not found")
  m = re.search(
      r"## Top Solution 1:.*?(```python\b.*?```)", path.read_text(), flags=re.S
  )
  if not m:
    raise ValueError(f"No Top Solution 1 python code block in {path}")
  return f"Plan: Seeded from {path.name}.\n\n{m.group(1)}"


def resolve_task_dir(module_file: str, override: str | None = None) -> str:
  """Resolves the live task directory, respecting an override if set.

  Args:
    module_file: Path to the task module's __file__.
    override: Optional explicit task directory path.

  Returns:
    The resolved absolute task directory path.
  """
  if override:
    return os.path.abspath(override)
  return resolve_workspace_dir(os.path.dirname(os.path.realpath(module_file)))


def experiments_root(task_name: str) -> str:
  """Returns `$CODEGEN_EXPERIMENTS_DIR` or a temp dir for the task's runs."""
  return os.environ.get("CODEGEN_EXPERIMENTS_DIR") or os.path.join(
      tempfile.gettempdir(), "codegen_classroom", task_name
  )


def resolve_run_dir(report_only: str, runs_base_dir: str) -> str:
  """Returns `report_only` as a run dir, or the newest run if it is 'latest'."""
  run_dir = report_only
  if report_only == "latest" and os.path.isdir(runs_base_dir):
    runs = [e.path for e in os.scandir(runs_base_dir) if e.is_dir()]
    run_dir = max(runs, key=os.path.getmtime, default="")
  if not os.path.isdir(run_dir):
    raise FileNotFoundError(f"No run directory for {report_only!r}.")
  return run_dir
