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

"""End-to-end integration test for the core Codegen Classroom workflow."""

import os
from absl.testing import absltest
from codegen_classroom import agents
from codegen_classroom import data_utils
from codegen_classroom import plotting
from codegen_classroom import reporting
from codegen_classroom import student_learning
from etils import epath
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_MOCK_SOLUTION = """
import pandas as pd
from sklearn.linear_model import Ridge

def solution(x_train, y_train, x_val):
    model = Ridge().fit(x_train, y_train)
    return pd.DataFrame(model.predict(x_val), index=x_val.index, columns=y_train.columns), "Trained Ridge"
"""


class MockStudentAgent(agents.StudentAgent):
  """Deterministic mock student agent returning valid executable python solutions."""

  def generate_solution(self, prompt: str) -> str:
    del prompt
    self.generate_code_count += 1
    self.current_solution = f"```python\n{_MOCK_SOLUTION}\n```"
    return self.current_solution

  def generate(self, prompt: str) -> str:
    del prompt
    return f"```python\n{_MOCK_SOLUTION}\n```"


class ClassroomIntegrationTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.tmpdir = self.create_tempdir().full_path
    self.run_dir = os.path.join(self.tmpdir, "run_001")
    os.makedirs(self.run_dir, exist_ok=True)

    # Synthetic tabular dataset
    self.x_train = pd.DataFrame(
        {"f1": [1.0, 2.0, 3.0, 4.0], "f2": [10.0, 20.0, 30.0, 40.0]}
    )
    self.y_train = pd.DataFrame({"target": [2.0, 4.0, 6.0, 8.0]})
    self.x_val = pd.DataFrame({"f1": [5.0, 6.0], "f2": [50.0, 60.0]})
    self.y_val = pd.DataFrame({"target": [10.0, 12.0]})
    self.x_test = pd.DataFrame({"f1": [7.0, 8.0], "f2": [70.0, 80.0]})
    self.y_test = pd.DataFrame({"target": [14.0, 16.0]})

  def test_end_to_end_classroom_lifecycle(self):
    # 1. Config creation
    configs = student_learning.get_classroom_student_configs(
        system_instruction="You are an expert ML engineer.",
        model_names=["gemini-2.5-flash", "gemini-1.5-pro"],
        api_key="mock_test_key",
    )
    self.assertLen(configs, 2)

    # 2. Agent instantiation
    classroom = [
        MockStudentAgent(cfg, run_base_dir=self.run_dir) for cfg in configs
    ]

    # 3. Parallel learning cycle execution
    cycle_results = student_learning.run_parallel_classroom_learning_cycle(
        current_classroom=classroom,
        x_train=self.x_train,
        y_train=self.y_train,
        x_val=self.x_val,
        y_val=self.y_val,
        current_cycle=0,
        student_initial_prompt="Predict target using linear regression.",
        student_function_name="solution",
        metric_weights={"r2": 1.0, "mae": 1.0},
        early_stopping_patience=2,
        student_refine_instruction_options=["Tune regularization."],
        max_mutation_count=2,
        mutation_strategy_options=[agents.MutationStrategy.SELF_BEST_CODE],
        code_execution_max_retries=1,
    )
    self.assertLen(cycle_results, 2)
    for res in cycle_results:
      self.assertIsNotNone(res)
      self.assertGreater(res.overall_score, 0.9)

    # 4. Final evaluation on held-out test split
    eval_results, _ = student_learning.run_parallel_classroom_final_evaluation(
        students={s.name: s for s in classroom},
        x_learning=self.x_train,
        y_learning=self.y_train,
        x_test_final=self.x_test,
        y_test_final=self.y_test,
        student_function_name="solution",
        metric_weights={"r2": 1.0, "mae": 1.0},
        strategies=["best", "last"],
    )
    self.assertLen(eval_results, 2)

    # 5. Reporting and artifact persistence
    os.makedirs(os.path.join(self.run_dir, "students"), exist_ok=True)
    report_sections = reporting.build_run_report(
        self.run_dir,
        students=classroom,
        metric_weights={"r2": 1.0, "mae": 1.0},
        include_plots=True,
    )
    self.assertNotEmpty(report_sections)
    self.assertEqual(report_sections[-1].title, "Student Instructions")
    reporting.write_report(
        epath.Path(self.run_dir), report_sections, precision=3
    )
    report_md = os.path.join(self.run_dir, "report.md")
    self.assertTrue(os.path.exists(report_md))
    with open(report_md, "r", encoding="utf-8") as f:
      report_text = f.read()
    self.assertIn("## Student Instructions", report_text)

    # 6. Local leaderboard submission and loading
    leaderboard_tsv = os.path.join(self.tmpdir, "leaderboard.tsv")
    row = pd.DataFrame([{
        "run_id": "run_001",
        "student_name": classroom[0].name,
        "score": 0.99,
    }])
    data_utils.submit_results_to_local_leaderboard(leaderboard_tsv, row)
    self.assertTrue(os.path.exists(leaderboard_tsv))

    df_lb = data_utils.load_leaderboard(leaderboard_tsv)
    self.assertLen(df_lb, 1)
    self.assertEqual(df_lb.iloc[0]["run_id"], "run_001")

    # 7. Diagnostic prediction resolution, embedding, and panel helpers
    sol_md = os.path.join(self.tmpdir, "sol.md")
    epath.Path(sol_md).write_text(
        "## Top Solution 1: s1\n\n### Code\n\n"
        f"```python\n{_MOCK_SOLUTION}\n```\n"
    )
    preds = reporting.resolve_solution_predictions(
        sol_md,
        self.x_train,
        self.y_train,
        self.x_test,
        baseline_fn=lambda xl, yl, xt: np.zeros(len(xt)),
    )
    self.assertIsNotNone(preds)
    y_best, y_first = preds
    self.assertLen(y_best, len(self.x_test))
    self.assertLen(y_first, len(self.x_test))

    reporting.embed_diagnostic_section(
        sol_md, "plots/diag.png", "* **Test MAE**: 0.10"
    )
    self.assertIn(
        "### Diagnostic & Impact Analysis", epath.Path(sol_md).read_text()
    )

    fig, axes = plt.subplots(1, 3)
    stats = plotting.plot_bland_altman_panel(
        axes[0], self.y_test.to_numpy().ravel(), y_best, y_first
    )
    self.assertIn("bias_best", stats)
    plotting.plot_grouped_bar_panel(axes[1], ["A", "B"], [0.5, 0.6], [0.8, 0.9])
    cm = plotting.plot_confusion_matrix_panel(
        axes[2], fig, np.array([0, 1, 0, 1]), np.array([0, 1, 1, 1]), [0, 1]
    )
    self.assertEqual(cm.shape, (2, 2))
    plt.close(fig)

  def test_write_run_report(self):
    run_dir = os.path.join(self.tmpdir, "run_report_dir")
    students_dir = os.path.join(run_dir, "students")
    os.makedirs(students_dir, exist_ok=True)
    student_csv = os.path.join(students_dir, "student_1.csv")
    df = pd.DataFrame([{
        "cycle": 0,
        "overall_score": 0.95,
        "code": "def solution(x):\n    return x\n",
        "plan": float("nan"),
        "code_length": 30,
        "retry_count": 0,
    }])
    df.to_csv(student_csv, index=False)

    report_path = reporting.write_run_report(
        run_dir,
        students=None,
        metric_weights={"r2": 1.0},
    )
    self.assertEqual(report_path, os.path.join(run_dir, "report.md"))
    self.assertTrue(os.path.exists(report_path))
    content = epath.Path(report_path).read_text()
    self.assertIn("# Overview", content)
    self.assertNotIn("### Plan", content)
    self.assertNotIn("\nnan\n", content)

  def test_export_top_solutions(self):
    task_dir = os.path.join(self.tmpdir, "mock_task")
    sol_dir = os.path.join(task_dir, "solutions")
    plots_dir = os.path.join(sol_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    epath.Path(sol_dir, "obsolete_old.md").write_text("old content")
    epath.Path(plots_dir, "obsolete_old.png").write_bytes(b"fake_png")
    epath.Path(sol_dir, "run_curated_20261013.md").write_text("curated content")

    run_a = os.path.join(self.tmpdir, "run_a")
    os.makedirs(os.path.join(run_a, "students"), exist_ok=True)
    run_b = os.path.join(self.tmpdir, "run_b")
    os.makedirs(os.path.join(run_b, "students"), exist_ok=True)

    lb_path = os.path.join(task_dir, "leaderboard.tsv")
    lines = [
        "timestamp\ttest_r2\ttest_mape\texperiment_name\trun_dir\ttags",
        f"10/17/2026\t0.999\t0.010\tmissing_run\t{self.tmpdir}/nonexistent\t",
        f"10/16/2026\t0.990\t0.050\trun_mock\t{self.tmpdir}/mock\tmock",
        f"10/16/2026\t0.980\t0.060\trun_ref\t{self.tmpdir}/ref\treference",
        f"10/15/2026\t0.900\t0.100\trun_a\t{run_a}\t",
        f"10/13/2026\t0.850\t0.150\trun_curated\t{self.tmpdir}/no_run\t",
        f"10/14/2026\t0.800\t0.200\trun_b\t{run_b}\t",
    ]
    epath.Path(lb_path).write_text("\n".join(lines) + "\n")

    # 1. write=False: returns IDs without writing or pruning (skips mock/ref
    # tags).
    ids = reporting.export_top_solutions(
        task_dir,
        sort_by=["test_r2", "test_mape"],
        ascending=[False, True],
        metric_weights={"r2": 1.0},
        experiments_root=self.tmpdir,
        top_n=3,
        write=False,
    )
    self.assertEqual(
        ids,
        ["missing_run_20261017", "run_a_20261015", "run_curated_20261013"],
    )
    self.assertFalse(os.path.exists(os.path.join(sol_dir, "run_a_20261015.md")))
    self.assertTrue(os.path.exists(os.path.join(sol_dir, "obsolete_old.md")))
    self.assertTrue(os.path.exists(os.path.join(plots_dir, "obsolete_old.png")))
    self.assertTrue(
        os.path.exists(os.path.join(sol_dir, "run_curated_20261013.md"))
    )

    # 2. write=True: writes .md files for live runs, preserves existing
    # curated .md even when its run_dir is gone, skips un-renderable runs
    # lacking both, and prunes obsolete.
    diagnosed = []
    ids_written = reporting.export_top_solutions(
        task_dir,
        sort_by=["test_r2", "test_mape"],
        ascending=[False, True],
        metric_weights={"r2": 1.0},
        experiments_root=self.tmpdir,
        top_n=2,
        write=True,
        diagnose=diagnosed.extend,
    )
    self.assertEqual(ids_written, ["run_a_20261015", "run_curated_20261013"])
    self.assertTrue(os.path.exists(os.path.join(sol_dir, "run_a_20261015.md")))
    self.assertTrue(
        os.path.exists(os.path.join(sol_dir, "run_curated_20261013.md"))
    )
    self.assertEqual(
        epath.Path(sol_dir, "run_curated_20261013.md").read_text(),
        "curated content",
    )
    self.assertFalse(os.path.exists(os.path.join(sol_dir, "obsolete_old.md")))
    self.assertFalse(
        os.path.exists(os.path.join(plots_dir, "obsolete_old.png"))
    )
    self.assertEqual(diagnosed, [os.path.join(sol_dir, "run_a_20261015.md")])


if __name__ == "__main__":
  absltest.main()
