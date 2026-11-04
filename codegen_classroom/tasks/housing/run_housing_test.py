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

"""Unit tests for housing runner, data loader, and student instructions."""

import os
from absl.testing import absltest
from codegen_classroom.tasks.housing import config
from codegen_classroom.tasks.housing import data_loader
from codegen_classroom.tasks.housing import run_housing
import numpy as np
import pandas as pd


class RunHousingTest(absltest.TestCase):

  def test_load_housing_data_shapes(self):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")

    if not os.path.exists(os.path.join(data_dir, "housing.csv")):
      self.skipTest("housing.csv not found in data dir")

    x_train, x_val, x_test, y_train, y_val, y_test = (
        data_loader.load_housing_data(data_dir=data_dir)
    )

    self.assertEqual(x_train.shape[0], y_train.shape[0])
    self.assertEqual(x_val.shape[0], y_val.shape[0])
    self.assertEqual(x_test.shape[0], y_test.shape[0])
    self.assertNotIn("median_house_value", x_train.columns)
    self.assertIn("median_house_value", y_train.columns)

  def test_load_preprocessed_housing_data(self):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")

    if not os.path.exists(os.path.join(data_dir, "housing.csv")):
      self.skipTest("housing.csv not found in data dir")

    x, y = data_loader.load_preprocessed_housing_data(data_dir=data_dir)
    self.assertEqual(x.shape[0], y.shape[0])

  def test_section_order_and_titles(self):
    prompt = config.build_student_instructions(
        feature_names=config.DEFAULT_FEATURE_NAMES,
        timeout_s=60,
        final_eval_timeout_s=180,
    )
    self.assertTrue(
        prompt.startswith("# Task: California Housing Price Regression\n\n")
    )
    last_pos = -1
    for title in config.SECTION_TITLES:
      pos = prompt.find(f"## {title}")
      self.assertNotEqual(pos, -1, f"Missing section ## {title}")
      self.assertGreater(pos, last_pos, f"Out of order: {title}")
      last_pos = pos

  def test_features_rendered_without_dtype_suffix(self):
    prompt = config.build_student_instructions(
        feature_names=config.DEFAULT_FEATURE_NAMES,
    )
    self.assertIn("`total_rooms`", prompt)
    self.assertNotIn("total_rooms (float64)", prompt)
    self.assertNotIn(
        "float64", prompt.split("## 2. Data Contract")[1].split("## 3.")[0]
    )

  def test_numeric_budgets_match_parameters(self):
    prompt = config.build_student_instructions(
        feature_names=config.DEFAULT_FEATURE_NAMES,
        timeout_s=45,
        final_eval_timeout_s=120,
    )
    self.assertIn("45 s", prompt)
    self.assertIn("120 s", prompt)

  def test_assert_no_recipes_on_instructions_and_refine_options(self):
    prompt = config.build_student_instructions(
        feature_names=config.DEFAULT_FEATURE_NAMES,
    )
    config.assert_no_recipes(prompt)

    config.assert_no_recipes(config.STUDENT_INITIAL_PROMPT)

    for option in config.STUDENT_REFINE_INSTRUCTION_OPTIONS:
      config.assert_no_recipes(option)
    with self.assertRaises(AssertionError):
      config.assert_no_recipes("C = 1.0")

  def test_probed_modules_respected(self):
    prompt_with_lgb = config.build_student_instructions(
        feature_names=config.DEFAULT_FEATURE_NAMES,
        available_modules=("numpy", "pandas", "sklearn", "lightgbm"),
    )
    self.assertIn(
        "`lightgbm`",
        prompt_with_lgb.split("## 4. Environment")[1].split("- **Forbidden")[0],
    )

    prompt_without_lgb = config.build_student_instructions(
        feature_names=config.DEFAULT_FEATURE_NAMES,
        available_modules=("numpy", "pandas", "sklearn"),
    )
    env_sec = prompt_without_lgb.split("## 4. Environment")[1].split("## 5.")[0]
    self.assertIn("`lightgbm`", env_sec.split("- **Forbidden")[1])

  def test_synthetic_csv_preprocessing_and_splits(self):
    tmp_dir = self.create_tempdir().full_path
    csv_path = os.path.join(tmp_dir, "housing.csv")
    rows = [
        "longitude,latitude,total_bedrooms,ocean_proximity,median_house_value"
    ]
    for i in range(20):
      prox = "INLAND" if i % 2 == 0 else "<1H OCEAN"
      rows.append(f"-122.{i},37.{i},{100 + i},{prox},{200000 + i * 1000}")
    rows.append("-121.0,38.0,,NEAR BAY,300000")  # NaN row to be dropped
    with open(csv_path, "w", encoding="utf-8") as f:
      f.write("\n".join(rows) + "\n")

    self.assertEqual(data_loader.ensure_data(tmp_dir), csv_path)
    x, y = data_loader.load_preprocessed_housing_data(data_dir=tmp_dir)
    self.assertLen(x, 20)
    self.assertLen(y, 20)
    self.assertIn("ocean_proximity_INLAND", x.columns)
    self.assertNotIn("ocean_proximity", x.columns)

    x_tr, x_val, x_te, y_tr, _, y_te = data_loader.load_housing_data(tmp_dir)
    self.assertEqual(len(x_tr) + len(x_val) + len(x_te), 20)
    self.assertEmpty(set(x_tr.index) & set(x_val.index))
    self.assertEmpty((set(x_tr.index) | set(x_val.index)) & set(x_te.index))
    self.assertEqual(list(x_tr.index), list(y_tr.index))
    self.assertEqual(list(x_te.index), list(y_te.index))

    with self.assertRaises(ValueError):
      data_loader.load_preprocessed_housing_data(
          tmp_dir, y_column="missing_col"
      )
    with self.assertRaises(FileNotFoundError):
      data_loader.load_preprocessed_housing_data(
          os.path.join(tmp_dir, "absent")
      )

  def test_export_top_solutions_filters_and_orders(self):
    tmp_dir = self.create_tempdir().full_path
    lb_path = os.path.join(tmp_dir, "leaderboard.tsv")
    lines = [
        "timestamp\ttest_r2\ttest_mape\texperiment_name\trun_dir\ttags",
        "10/14/2026\t0.850\t0.140\trun_b\t/tmp/b\t",
        "10/15/2026\t0.890\t0.120\trun_a\t/tmp/a\t",
        "10/16/2026\t0.990\t0.050\trun_mock\t/tmp/m\tmock",
        "10/16/2026\t0.950\t0.080\trun_ref\t/tmp/s\treference",
        "10/13/2026\t0.810\t0.160\trun_c\t/tmp/c\t",
    ]
    with open(lb_path, "w", encoding="utf-8") as f:
      f.write("\n".join(lines) + "\n")

    exported = run_housing.export_top_solutions(tmp_dir, top_n=2, write=False)
    self.assertEqual(exported, ["run_a_20261015", "run_b_20261014"])

  def test_mock_student_and_run_defaults(self):
    self.assertGreaterEqual(
        config.RUN_DEFAULTS["final_eval_timeout_s"],
        2 * config.RUN_DEFAULTS["timeout_s"],
    )
    tmp_dir = self.create_tempdir().full_path
    student = run_housing._MockStudent(
        config=run_housing.agents.AgentConfig(name="mock_1"),
        run_base_dir=tmp_dir,
    )
    resp = student.generate_solution("initial prompt")
    self.assertIn("def solution(X_train, y_train, X_val):", resp)
    self.assertEqual(student.generate_code_count, 1)

    default_student = run_housing._MockStudent(
        config=run_housing.agents.AgentConfig(name="mock_default")
    )
    self.assertTrue(os.path.isdir(default_student.run_base_dir))

  def test_download_housing_data_existing(self):
    tmp_dir = self.create_tempdir().full_path
    csv_path = os.path.join(tmp_dir, "housing.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
      f.write("dummy,data\n1,2\n")
    data_loader.download_housing_data(tmp_dir)
    self.assertTrue(os.path.isfile(csv_path))

  def test_housing_kde_curve(self):
    grid = np.linspace(0.0, 100.0, 50)
    empty_kde = run_housing._housing_kde_curve(np.array([10.0]), grid)
    self.assertEqual(empty_kde.shape, grid.shape)
    self.assertTrue(np.all(empty_kde == 0.0))

    samples = np.array([20.0, 25.0, 30.0, 35.0, 50.0])
    kde = run_housing._housing_kde_curve(samples, grid)
    self.assertEqual(kde.shape, grid.shape)
    self.assertTrue(np.all(kde >= 0.0))
    self.assertGreater(float(np.sum(kde)), 0.0)

  def test_generate_housing_dataset_overview(self):
    tmp_dir = self.create_tempdir().full_path
    plots_dir = os.path.join(tmp_dir, "plots")
    x = pd.DataFrame({
        "longitude": [-122.2, -122.3, -118.2, -117.1],
        "latitude": [37.8, 37.9, 34.0, 32.7],
        "median_income": [3.5, 4.0, 5.5, 2.8],
        "households": [300.0, 400.0, 500.0, 200.0],
        "total_rooms": [1200.0, 1600.0, 2000.0, 800.0],
        "total_bedrooms": [300.0, 400.0, 500.0, 200.0],
        "population": [800.0, 1000.0, 1200.0, 600.0],
        "housing_median_age": [30.0, 25.0, 40.0, 15.0],
    })
    y = pd.DataFrame(
        {"median_house_value": [200000.0, 250000.0, 300000.0, 150000.0]}
    )
    rel_path = run_housing._generate_housing_dataset_overview(
        x_learning=x, y_learning=y, x_test=x, y_test=y, plots_dir=plots_dir
    )
    self.assertEqual(rel_path, "plots/dataset_overview.png")
    self.assertTrue(
        os.path.isfile(os.path.join(plots_dir, "dataset_overview.png"))
    )

  def test_resolve_task_dir(self):
    task_dir = run_housing.resolve_task_dir()
    self.assertTrue(os.path.isdir(task_dir))


if __name__ == "__main__":
  absltest.main()
