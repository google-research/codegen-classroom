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

"""Unit tests for wearable runner, data loader, and student instructions."""

import os
from absl.testing import absltest
from codegen_classroom.tasks.wearable import config
from codegen_classroom.tasks.wearable import data_loader
from codegen_classroom.tasks.wearable import run_wearable


class RunWearableTest(absltest.TestCase):

  def test_load_wearable_data_shapes(self):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")

    if not os.path.exists(os.path.join(data_dir, "aw_fb_data.csv")):
      self.skipTest("aw_fb_data.csv not found in data dir")

    x_train, x_val, x_test, y_train, y_val, y_test = (
        data_loader.load_wearable_data(data_dir=data_dir)
    )

    self.assertEqual(x_train.shape[0], y_train.shape[0])
    self.assertEqual(x_val.shape[0], y_val.shape[0])
    self.assertEqual(x_test.shape[0], y_test.shape[0])
    self.assertNotIn("activity_encoded", x_train.columns)
    self.assertIn("activity_encoded", y_train.columns)

  def test_load_preprocessed_wearable_data(self):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")

    if not os.path.exists(os.path.join(data_dir, "aw_fb_data.csv")):
      self.skipTest("aw_fb_data.csv not found in data dir")

    x, y, groups = data_loader.load_preprocessed_wearable_data(
        data_dir=data_dir
    )
    self.assertEqual(x.shape[0], y.shape[0])
    self.assertEqual(x.shape[0], groups.shape[0])

  def test_split_indices_match_split_wearable_data(self):
    """Verifies run_wearable and data_loader produce identical splits."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")

    if not os.path.exists(os.path.join(data_dir, "aw_fb_data.csv")):
      self.skipTest("aw_fb_data.csv not found in data dir")

    x, y, groups = data_loader.load_preprocessed_wearable_data(
        data_dir=data_dir
    )
    val_seed = 42

    (
        x_train,
        x_val,
        x_test,
        _,
        _,
        _,
        groups_train,
        groups_val,
        groups_test,
        x_learning,
        _,
        _,
    ) = data_loader.split_wearable_data(x, y, groups, random_state=val_seed)

    # Invariants
    self.assertLen(x_learning, len(x_train) + len(x_val))
    self.assertLen(x, len(x_learning) + len(x_test))
    # Unseen participant invariant
    train_users = set(groups_train.unique())
    val_users = set(groups_val.unique())
    test_users = set(groups_test.unique())
    self.assertTrue(train_users.isdisjoint(val_users))
    self.assertTrue(train_users.isdisjoint(test_users))
    self.assertTrue(val_users.isdisjoint(test_users))

  def test_section_order_and_titles(self):
    prompt = config.build_student_instructions(
        timeout_s=60,
        final_eval_timeout_s=180,
    )
    self.assertTrue(
        prompt.startswith(
            "# Task: Wearable Sensor Physical Activity Classification\n\n"
        )
    )
    last_pos = -1
    for title in config.SECTION_TITLES:
      pos = prompt.find(f"## {title}")
      self.assertNotEqual(pos, -1, f"Missing section ## {title}")
      self.assertGreater(pos, last_pos, f"Out of order: {title}")
      last_pos = pos

  def test_feature_names_and_classes_present(self):
    prompt = config.build_student_instructions()
    self.assertIn("`age`", prompt)
    self.assertIn("`intensity_karvonen`", prompt)
    self.assertIn("`corr_heart_steps`", prompt)
    self.assertIn("Lying (resting / sedentary)", prompt)
    self.assertIn("Running at 7 METs", prompt)

  def test_numeric_budgets_match_parameters(self):
    prompt = config.build_student_instructions(
        timeout_s=50,
        final_eval_timeout_s=150,
    )
    self.assertIn("50 s", prompt)
    self.assertIn("150 s", prompt)

  def test_no_temporal_or_deprecated_sklearn_tips(self):
    prompt = config.build_student_instructions()
    self.assertNotIn("multi_class='multinomial'", prompt)
    refine_options = config.STUDENT_REFINE_INSTRUCTION_OPTIONS
    for opt in refine_options:
      self.assertNotIn("rolling window", opt.lower())
      self.assertNotIn("datetime", opt.lower())

  def test_assert_no_recipes_on_instructions_and_refine_options(self):
    prompt = config.build_student_instructions()
    config.assert_no_recipes(prompt)

    initial_prompt = config.STUDENT_INITIAL_PROMPT
    config.assert_no_recipes(initial_prompt)

    refine_options = config.STUDENT_REFINE_INSTRUCTION_OPTIONS
    for option in refine_options:
      config.assert_no_recipes(option)

  def test_synthetic_csv_preprocessing_and_splits(self):
    tmp_dir = self.create_tempdir().full_path
    csv_path = os.path.join(tmp_dir, "aw_fb_data.csv")
    cols = [
        "age",
        "gender",
        "height",
        "weight",
        "steps",
        "heart_rate",
        "calories",
        "distance",
        "entropy_heart",
        "entropy_steps",
        "resting_heart",
        "corr_heart_steps",
        "norm_heart",
        "intensity_karvonen",
        "sd_norm_heart",
        "steps_times_distance",
        "device_encoded",
        "activity_encoded",
    ]
    rows = [",".join(cols)]
    for p in range(5):
      for row_idx in range(6):
        act = row_idx % 6
        dev = p % 2
        vals = [
            f"{20 + p * 5}",
            f"{p % 2}",
            f"{160 + p * 5}",
            f"{60 + p * 5}",
            f"{100 * (act + 1)}",
            f"{70 + act * 10}",
            f"{10 * (act + 1)}",
            f"{0.5 * (act + 1)}",
            "0.5",
            "0.5",
            "60.0",
            "0.8",
            f"{10.0 * act}",
            f"{0.1 * act}",
            "5.0",
            f"{50.0 * (act + 1)}",
            f"{dev}",
            f"{act}",
        ]
        rows.append(",".join(vals))
    with open(csv_path, "w", encoding="utf-8") as f:
      f.write("\n".join(rows) + "\n")

    self.assertEqual(data_loader.ensure_data(tmp_dir), csv_path)
    x, y, groups = data_loader.load_preprocessed_wearable_data(data_dir=tmp_dir)
    self.assertLen(x, 30)
    self.assertLen(y, 30)
    self.assertLen(groups, 30)
    self.assertEqual(groups.nunique(), 5)
    self.assertIn("device_encoded", x.columns)
    self.assertNotIn("activity_encoded", x.columns)

    (
        x_tr,
        x_val,
        x_te,
        _,
        _,
        _,
        g_tr,
        g_val,
        g_te,
        x_learning,
        _,
        _,
    ) = data_loader.split_wearable_data(x, y, groups, random_state=42)
    self.assertEqual(len(x_tr) + len(x_val) + len(x_te), 30)
    self.assertLen(x_learning, len(x_tr) + len(x_val))
    train_users = set(g_tr.unique())
    val_users = set(g_val.unique())
    test_users = set(g_te.unique())
    self.assertTrue(train_users.isdisjoint(val_users))
    self.assertTrue(train_users.isdisjoint(test_users))
    self.assertTrue(val_users.isdisjoint(test_users))

    with self.assertRaises(ValueError):
      data_loader.load_preprocessed_wearable_data(
          tmp_dir, y_column="missing_col"
      )
    with self.assertRaises(FileNotFoundError):
      data_loader.load_preprocessed_wearable_data(
          os.path.join(tmp_dir, "absent")
      )

  def test_export_top_solutions_and_mock_student(self):
    tmp_dir = self.create_tempdir().full_path
    student = run_wearable._MockStudent(
        config=run_wearable.agents.AgentConfig(name="student_0"),
        run_base_dir=tmp_dir,
    )
    code_block = student.generate_solution("test")
    self.assertIn("def solution(X_train, y_train, X_val):", code_block)
    self.assertEqual(student.generate_code_count, 1)
    run_wearable.pd.DataFrame([
        {
            "timestamp": "10/22/2026",
            "experiment_name": "wearable_6s_12c",
            "test_f1_macro": 0.605,
            "test_accuracy": 0.616,
            "run_dir": "wearable_6s_12c/run1",
            "tags": "",
        },
        {
            "timestamp": "10/22/2026",
            "experiment_name": "wearable_mock_run",
            "test_f1_macro": 0.990,
            "test_accuracy": 0.990,
            "run_dir": "mock/run",
            "tags": "mock",
        },
        {
            "timestamp": "10/15/2026",
            "experiment_name": "wearable_4s_10c",
            "test_f1_macro": 0.597,
            "test_accuracy": 0.608,
            "run_dir": "wearable_4s_10c/run1",
            "tags": "",
        },
    ]).to_csv(os.path.join(tmp_dir, "leaderboard.tsv"), sep="\t", index=False)
    exported = run_wearable.export_top_solutions(tmp_dir, top_n=1, write=False)
    self.assertEqual(exported, ["wearable_6s_12c_20261022"])


if __name__ == "__main__":
  absltest.main()
