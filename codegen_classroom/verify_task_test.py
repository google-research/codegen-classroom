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

"""Unit tests for the Codegen Classroom Task Verifier."""

import os
import pathlib
import tempfile
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
from codegen_classroom import verify_task


class VerifyTaskTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.temp_dir = tempfile.TemporaryDirectory()
    self.d = pathlib.Path(self.temp_dir.name)
    (self.d / "README.md").write_text("# Sample Task\nOverview.\n")
    (self.d / "config.py").write_text(
        "DEFAULT_METRIC_WEIGHTS = {'r2': 1.0}\n"
        "def build_student_instructions(): return 'Prompt'\n"
    )
    (self.d / "data_loader.py").write_text("def load_data(): pass\n")
    (self.d / f"run_{self.d.name}.py").write_text(
        "from codegen_classroom import scoring\n"
        "METRIC_WEIGHTS = {'r2': 1.0}\n"
        "def main(): pass\n"
    )
    (self.d / f"run_{self.d.name}_test.py").write_text("pass\n")
    (self.d / f"{self.d.name}.ipynb").write_text('{"nbformat": 4, "cells": []}')
    (self.d / "leaderboard.tsv").write_text(
        "model\tscore\trun_dir\nm1\t0.9\ttask/run1\n"
    )

  def tearDown(self):
    self.temp_dir.cleanup()
    super().tearDown()

  def test_valid_task_passes_all_checks(self):
    rep = verify_task.verify_task(self.d)
    self.assertTrue(rep.passed, f"Errors: {rep.errors}")
    self.assertEqual(
        list(rep.checks.keys()), ["Files", "Runner", "Notebook", "Solutions"]
    )

  @parameterized.named_parameters(
      (
          "missing_readme",
          lambda d: (d / "README.md").unlink(),
          "Files",
          "Missing README.md",
      ),
      (
          "missing_config",
          lambda d: (d / "config.py").unlink(),
          "Files",
          "Missing config.py",
      ),
      (
          "missing_data_loader",
          lambda d: (d / "data_loader.py").unlink(),
          "Files",
          "Missing data_loader.py",
      ),
      (
          "missing_runner",
          lambda d: [
              f.unlink()
              for f in d.glob("run_*.py")
              if not f.name.endswith("_test.py")
          ],
          "Files",
          "Missing runner",
      ),
      (
          "missing_test",
          lambda d: [f.unlink() for f in d.glob("*_test.py")],
          "Files",
          "Missing unit test",
      ),
      (
          "missing_leaderboard",
          lambda d: (d / "leaderboard.tsv").unlink(),
          "Files",
          "leaderboard.tsv",
      ),
      (
          "file_budget",
          lambda d: [(d / f"f_{i}.txt").write_text("x") for i in range(26)],
          "Files",
          "budget",
      ),
      (
          "config_missing_metrics",
          lambda d: (d / "config.py").write_text(
              "STUDENT_INSTRUCTIONS = 'Follow the data contract.'\n"
          ),
          "Runner",
          "config.py: missing metric_weights",
      ),
      (
          "config_missing_instructions",
          lambda d: (d / "config.py").write_text(
              "DEFAULT_METRIC_WEIGHTS = {'r2': 1.0}\n"
          ),
          "Runner",
          "student instructions",
      ),
      (
          "config_syntax_error",
          lambda d: (d / "config.py").write_text("def broken(:\n"),
          "Runner",
          "config.py: SyntaxError",
      ),
      (
          "runner_missing_main",
          lambda d: (d / f"run_{d.name}.py").write_text(
              "import codegen_classroom\nMETRIC_WEIGHTS = {'r2': 1.0}\n"
          ),
          "Runner",
          "main",
      ),
      (
          "runner_missing_import",
          lambda d: (d / f"run_{d.name}.py").write_text(
              "METRIC_WEIGHTS = {'r2': 1.0}\ndef main(): pass\n"
          ),
          "Runner",
          "codegen_classroom",
      ),
      (
          "runner_missing_metrics",
          lambda d: (d / f"run_{d.name}.py").write_text(
              "import codegen_classroom\ndef main(): pass\n"
          ),
          "Runner",
          "metric_weights",
      ),
      (
          "runner_syntax_error",
          lambda d: (d / f"run_{d.name}.py").write_text("def main(:\n"),
          "Runner",
          "SyntaxError",
      ),
      (
          "notebook_bad_format",
          lambda d: (d / f"{d.name}.ipynb").write_text('{"nbformat": 3}'),
          "Notebook",
          "nbformat < 4",
      ),
      (
          "notebook_invalid_json",
          lambda d: (d / f"{d.name}.ipynb").write_text("{not valid json"),
          "Notebook",
          ".ipynb",
      ),
      (
          "notebook_missing_cell_id",
          lambda d: (d / f"{d.name}.ipynb").write_text(
              '{"nbformat": 4, "cells": [{"cell_type": "markdown", '
              '"source": ["# Title\\n"]}]}'
          ),
          "Notebook",
          "missing required string field 'id'",
      ),
      (
          "notebook_cell_syntax_error",
          lambda d: (d / f"{d.name}.ipynb").write_text(
              '{"nbformat": 4, "cells": [{"id": "c1", "cell_type": "code", '
              '"source": ["def bad(:\\n"]}]}'
          ),
          "Notebook",
          "cell_1",
      ),
      (
          "notebook_outputs_not_cleared",
          lambda d: (d / f"{d.name}.ipynb").write_text(
              '{"nbformat": 4, "cells": [{"id": "c1", "cell_type": "code", '
              '"execution_count": 1, "outputs": [], "source": ["x = 1\\n"]}]}'
          ),
          "Notebook",
          "outputs not cleared",
      ),
      (
          "leaderboard_empty",
          lambda d: (d / "leaderboard.tsv").write_text("# comment only\n\n"),
          "Solutions",
          "empty or non-tab-separated header",
      ),
      (
          "leaderboard_not_tab_separated",
          lambda d: (d / "leaderboard.tsv").write_text("model,score\nm1,0.9\n"),
          "Solutions",
          "empty or non-tab-separated header",
      ),
      (
          "mock_leaderboard",
          lambda d: (d / "leaderboard.tsv").write_text(
              "model\tscore\nmock\t0.5\n"
          ),
          "Solutions",
          "mock row",
      ),
      (
          "solution_missing_overview",
          lambda d: _add_raw_sol(
              d,
              "## Student Leaderboard\n## Learning Trajectory\n## Top"
              " Solutions\n| a | b |\n| - | - |\n| 1 | 2 |\n",
          ),
          "Solutions",
          "missing '# Overview'",
      ),
      (
          "solution_missing_leaderboard_with_objectives",
          lambda d: _add_raw_sol(
              d,
              "# Objectives\n## Learning Trajectory\n## Top Solutions\n| a |"
              " b |\n| - | - |\n| 1 | 2 |\n",
          ),
          "Solutions",
          "missing '## Student Leaderboard'",
      ),
      (
          "solution_missing_learning_trajectory",
          lambda d: _add_raw_sol(
              d,
              "# Overview\n## Student Leaderboard\n## Top Solutions\n| a |"
              " b |\n| - | - |\n| 1 | 2 |\n",
          ),
          "Solutions",
          "missing '## Learning Trajectory'",
      ),
      (
          "solution_nan_table_cell",
          lambda d: _add_sol(d, "| metric | NaN |\n"),
          "Solutions",
          "contains NaN or None table cells",
      ),
      (
          "solution_missing_referenced_plot",
          lambda d: _add_sol(d, "![traj](plots/missing_plot.png)\n"),
          "Solutions",
          "referenced plot 'plots/missing_plot.png' not found",
      ),
  )
  def test_audit_failures(self, mutate_fn, check_key, expected_substring):
    mutate_fn(self.d)
    rep = verify_task.verify_task(self.d)
    self.assertFalse(rep.passed)
    self.assertEqual(rep.checks.get(check_key), "ERR")
    self.assertTrue(
        any(expected_substring in err for err in rep.errors),
        f"Expected '{expected_substring}' in {rep.errors}",
    )

  def test_config_accepts_task_description_alias(self):
    (self.d / "config.py").write_text(
        "METRIC_WEIGHTS = {'r2': 1.0}\n"
        "def build_task_description(): return 'Task instructions'\n"
    )
    rep = verify_task.verify_task(self.d)
    self.assertTrue(rep.passed, f"Errors: {rep.errors}")

  def test_notebook_ignores_shell_and_magic_commands(self):
    nb_content = (
        '{"nbformat": 4, "cells": ['
        '{"id": "c1", "cell_type": "markdown", "source": ["# Header"]}, '
        '{"id": "c2", "cell_type": "code", "source": ['
        '"!pip install -q scikit-learn\\n", '
        '"%matplotlib inline\\n", '
        '"%%time\\n", '
        '"x = 1 + 1\\n"'
        "]}]}"
    )
    (self.d / f"{self.d.name}.ipynb").write_text(nb_content)
    rep = verify_task.verify_task(self.d)
    self.assertTrue(rep.passed, f"Errors: {rep.errors}")
    self.assertEqual(rep.checks["Notebook"], "OK")

  def test_missing_notebook_produces_warning_not_error(self):
    (self.d / f"{self.d.name}.ipynb").unlink()
    rep = verify_task.verify_task(self.d)
    self.assertTrue(rep.passed)
    self.assertEqual(rep.checks["Files"], "WARN")
    self.assertEqual(rep.checks["Notebook"], "WARN")
    self.assertIn("Missing companion notebook (*.ipynb)", rep.warnings)

  def test_solution_missing_top_solutions_produces_warning(self):
    _add_raw_sol(
        self.d,
        "# Overview\n## Student Leaderboard\n## Learning Trajectory\n",
    )
    rep = verify_task.verify_task(self.d)
    self.assertTrue(rep.passed, f"Errors: {rep.errors}")
    self.assertTrue(
        any("missing '## Top Solutions'" in w for w in rep.warnings),
        f"Expected Top Solutions warning in {rep.warnings}",
    )

  def test_solution_with_existing_plot_passes(self):
    plots_dir = self.d / "solutions" / "plots"
    plots_dir.mkdir(parents=True)
    (plots_dir / "traj.png").write_bytes(b"PNG")
    _add_sol(self.d, "![traj](plots/traj.png)\n")
    rep = verify_task.verify_task(self.d)
    self.assertTrue(rep.passed, f"Errors: {rep.errors}")

  def test_table_and_root_resolution(self):
    rep_pass = verify_task.verify_task(self.d)
    rep_fail = verify_task.TaskReport(
        task="broken_task",
        errors=["Missing README.md"],
        checks={
            "Files": "ERR",
            "Runner": "OK",
            "Notebook": "OK",
            "Solutions": "OK",
        },
    )
    tbl = verify_task.format_table([rep_pass, rep_fail])
    self.assertIn(f"PASS: {self.d.name}", tbl)
    self.assertIn("FAIL: broken_task", tbl)

    # Explicit start_dir
    self.assertEqual(verify_task.find_tasks_root(self.d), self.d.resolve())
    # Default discovery locates tasks root directory
    default_root = verify_task.find_tasks_root()
    self.assertEqual(default_root.name, "tasks")

  def test_main_cli_modes(self):
    # 1. Valid task directory passes
    rc = verify_task.main(["--task_dir", str(self.d)])
    self.assertEqual(rc, 0)

    # 2. JSON output mode and --log_file overwrite
    custom_log = self.d / "artifacts" / "audit.log"
    rc_json = verify_task.main(
        ["--task_dir", str(self.d), "--json", "--log_file", str(custom_log)]
    )
    self.assertEqual(rc_json, 0)
    self.assertTrue(custom_log.is_file())

    # 3. Warning passes normally (0) but fails under --strict (1)
    (self.d / f"{self.d.name}.ipynb").unlink()
    self.assertEqual(verify_task.main(["--task_dir", str(self.d)]), 0)
    self.assertEqual(
        verify_task.main(["--task_dir", str(self.d), "--strict"]), 1
    )

    # 4. --task and --all via BUILD_WORKSPACE_DIRECTORY
    with tempfile.TemporaryDirectory() as ws_tmp:
      ws_path = pathlib.Path(ws_tmp)
      tmpl_dir = ws_path / "codegen_classroom" / "tasks" / "template"
      tmpl_dir.mkdir(parents=True)
      for f in self.d.iterdir():
        if f.is_file():
          target_name = (
              f.name.replace(self.d.name, "template")
              if self.d.name in f.name
              else f.name
          )
          (tmpl_dir / target_name).write_text(f.read_text())
      with mock.patch.dict(
          os.environ, {"BUILD_WORKSPACE_DIRECTORY": str(ws_path)}
      ):
        self.assertEqual(
            verify_task.find_tasks_root(),
            (ws_path / "codegen_classroom" / "tasks").resolve(),
        )
        self.assertEqual(verify_task.main(["--task", "template"]), 0)
        self.assertEqual(verify_task.main(["--all"]), 0)

    # 5. Error returns 1
    (self.d / "README.md").unlink()
    self.assertEqual(verify_task.main(["--task_dir", str(self.d)]), 1)


def _add_raw_sol(d: pathlib.Path, content: str):
  s = d / "solutions"
  s.mkdir(exist_ok=True)
  (s / "sol.md").write_text(content)


def _add_sol(d: pathlib.Path, extra: str):
  s = d / "solutions"
  s.mkdir(exist_ok=True)
  (s / "sol.md").write_text(
      "# Overview\n## Student Leaderboard\n## Learning Trajectory\n## Top"
      f" Solutions\n| a | b |\n| - | - |\n| 1 | 2 |\n{extra}"
  )


if __name__ == "__main__":
  absltest.main()
