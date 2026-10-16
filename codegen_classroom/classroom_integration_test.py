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
from codegen_classroom import reporting
from codegen_classroom import student_learning
from etils import epath
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


if __name__ == "__main__":
  absltest.main()
