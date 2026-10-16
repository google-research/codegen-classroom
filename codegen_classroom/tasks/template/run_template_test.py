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

"""Unit tests for template runner, dataset hill-climbing, and prompt contract."""

import pathlib
from absl.testing import absltest
from codegen_classroom.tasks.template import data_loader
from codegen_classroom.tasks.template import run_template
import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn import impute
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn import pipeline


class RunTemplateTest(absltest.TestCase):

  def test_load_data(self):
    x, y = data_loader.load_data("nonexistent_sample_path")
    self.assertIsInstance(x, pd.DataFrame)
    self.assertIsInstance(y, pd.DataFrame)
    self.assertLen(x, data_loader.DEFAULT_NUM_SAMPLES)
    self.assertLen(y, data_loader.DEFAULT_NUM_SAMPLES)
    self.assertEqual(tuple(x.columns), data_loader.FEATURE_COLUMNS)
    self.assertEqual(list(y.columns), [data_loader.TARGET_COLUMN])
    self.assertTrue(x["temp_sensor_proxy"].isna().any())

  def test_dataset_supports_meaningful_hill_climbing(self):
    x, y = data_loader.load_data("nonexistent_sample_path")
    x_train, x_val, y_train, y_val = model_selection.train_test_split(
        x, y["thermal_load"], test_size=0.2, random_state=42
    )

    # Tier 1: Raw imputed Ridge regression baseline
    ridge = pipeline.make_pipeline(
        impute.SimpleImputer(strategy="median"),
        linear_model.Ridge(alpha=1.0),
    )
    ridge.fit(x_train, y_train)
    r2_linear = metrics.r2_score(y_val, ridge.predict(x_val))

    # Tier 2: Out-of-the-box HistGradientBoostingRegressor on raw features
    hgbr = ensemble.HistGradientBoostingRegressor(random_state=42)
    hgbr.fit(x_train, y_train)
    r2_tree = metrics.r2_score(y_val, hgbr.predict(x_val))

    # Tier 3: Feature-engineered HistGradientBoostingRegressor + Ridge blend
    def engineer(df: pd.DataFrame) -> pd.DataFrame:
      out = df.drop(columns=["aux_noise_1", "aux_noise_2"]).copy()
      out["temp_sensor_proxy"] = out["temp_sensor_proxy"].fillna(
          out["ambient_temp"]
      )
      diff = out["ambient_temp"] - out["workload_intensity"]
      tsum = out["ambient_temp"] + out["workload_intensity"]
      fbal = out["cooling_flow"] - 0.8 * out["pressure_delta"]
      out["thermal_diff"] = diff
      out["thermal_sum"] = tsum
      out["thermal_diff_sq"] = diff**2
      out["temp_x_work"] = out["ambient_temp"] * out["workload_intensity"]
      out["cool_x_press"] = out["cooling_flow"] * out["pressure_delta"]
      out["flow_balance"] = fbal
      out["flow_x_sum_tanh"] = np.tanh(fbal * tsum)
      out["sin_phase_diff"] = np.sin(out["phase_angle"] + 0.5 * diff)
      out["cos_phase_diff"] = np.cos(out["phase_angle"] + 0.5 * diff)
      out["log_surge"] = np.log1p(out["surge_index"])
      for r in (0.0, 1.0, 2.0):
        mask = (out["operating_regime"] == r).astype(float)
        out[f"r{int(r)}_sum"] = mask * tsum
        out[f"r{int(r)}_prod"] = (
            mask * out["ambient_temp"] * out["workload_intensity"]
        )
        out[f"r{int(r)}_diff_sq"] = mask * (diff**2)
      return out

    hgbr_fe = ensemble.HistGradientBoostingRegressor(
        max_iter=250, learning_rate=0.08, random_state=42
    )
    hgbr_fe.fit(engineer(x_train), y_train)
    r2_fe = metrics.r2_score(y_val, hgbr_fe.predict(engineer(x_val)))

    self.assertLess(r2_linear, 0.55)
    self.assertGreater(r2_tree, r2_linear + 0.15)
    self.assertGreater(r2_fe, r2_tree + 0.10)
    self.assertGreater(r2_fe, 0.92)

  def test_build_student_instructions_contains_all_eight_sections(self):
    prompt = run_template.build_student_instructions(
        feature_names=["f0", "f1", "f2"],
        target_column="thermal_load",
        train_rows=960,
        val_rows=240,
        final_eval_rows=1200,
        timeout_s=60,
        final_eval_timeout_s=180,
    )

    last_index = -1
    for title in run_template.SECTION_TITLES:
      header = f"## {title}"
      self.assertIn(header, prompt)
      idx = prompt.index(header)
      self.assertGreater(idx, last_index, f"Section {title} is out of order")
      last_index = idx

  def test_instructions_contain_numeric_budgets_and_features(self):
    prompt = run_template.build_student_instructions(
        feature_names=["feature_alpha", "feature_beta"],
        target_column="custom_target",
        train_rows=100,
        val_rows=25,
        final_eval_rows=125,
        timeout_s=45,
        final_eval_timeout_s=120,
    )
    self.assertIn("`feature_alpha`", prompt)
    self.assertIn("`feature_beta`", prompt)
    self.assertIn("custom_target", prompt)
    self.assertIn("45 s", prompt)
    self.assertIn("120 s", prompt)
    self.assertIn("100", prompt)
    self.assertIn("125", prompt)

  def test_no_forbidden_hyperparameter_recipes(self):
    prompt = run_template.build_student_instructions()
    run_template.assert_no_recipes(prompt)

    run_template.assert_no_recipes(run_template.STUDENT_INITIAL_PROMPT)

    refine_options = run_template.STUDENT_REFINE_INSTRUCTION_OPTIONS
    self.assertGreaterEqual(len(refine_options), 3)
    for opt in refine_options:
      run_template.assert_no_recipes(opt)

  def test_customize_annotations_present_for_task_authors(self):
    prompt = run_template.build_student_instructions()
    self.assertNotIn("# CUSTOMIZE:", prompt)
    config_src = pathlib.Path(run_template.config.__file__).read_text(
        encoding="utf-8"
    )
    self.assertIn("# CUSTOMIZE:", config_src)

  def test_metric_weights_include_exec_time(self):
    weights = run_template.DEFAULT_METRIC_WEIGHTS
    self.assertIn("r2", weights)
    self.assertIn("mae", weights)
    self.assertIn("exec_time", weights)


if __name__ == "__main__":
  absltest.main()
