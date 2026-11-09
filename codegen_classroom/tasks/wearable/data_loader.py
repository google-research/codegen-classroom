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

"""Data loader, preprocessor, and download manager for Wearable Activity dataset."""

from __future__ import annotations

import os
import sys
import urllib.request

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder


def download_wearable_data(data_dir: str = "data") -> None:
  """Downloads and processes the Wearable Activity dataset from GitHub mirror."""
  os.makedirs(data_dir, exist_ok=True)
  target_file = os.path.join(data_dir, "aw_fb_data.csv")

  if os.path.exists(target_file):
    print(f"Data already exists at {target_file}")
    return

  print("Downloading Wearable Activity dataset from GitHub mirror...")
  url = "https://raw.githubusercontent.com/kairess/toy-datasets/master/aw_fb_data.csv"
  urllib.request.urlretrieve(url, target_file)
  print("Download complete.")

  df, _, _, _ = clean_and_encode(pd.read_csv(target_file))
  df.to_csv(target_file, index=False)
  print(f"Data processed and saved to {target_file}")


def ensure_data(data_dir: str = "data") -> str:
  """Ensures aw_fb_data.csv exists, downloading if necessary."""
  csv_path = (
      data_dir
      if data_dir.endswith(".csv")
      else os.path.join(data_dir, "aw_fb_data.csv")
  )
  if not os.path.exists(csv_path):
    pkg_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "aw_fb_data.csv"
    )
    if os.path.exists(pkg_path):
      return pkg_path
    download_wearable_data(os.path.dirname(os.path.abspath(csv_path)))
  return csv_path


def clean_and_encode(
    raw_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], LabelEncoder, LabelEncoder]:
  """Cleans and encodes raw wearable DataFrame."""
  df = raw_df.copy()
  df = df.rename(
      columns={"hear_rate": "heart_rate", "entropy_setps": "entropy_steps"}
  )
  df = df.drop(columns=["Unnamed: 0", "X1", "A"], errors="ignore")

  numeric_cols = [
      "age",
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
      "gender",
  ]
  df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

  le_dev = LabelEncoder()
  df["device_encoded"] = le_dev.fit_transform(df["device"].astype(str))

  le_act = LabelEncoder()
  df["activity_encoded"] = le_act.fit_transform(df["activity"].astype(str))

  df.dropna(inplace=True)

  feature_cols = numeric_cols + ["device_encoded"]
  return df, feature_cols, le_act, le_dev


def load_preprocessed_wearable_data(
    data_dir: str = "data", y_column: str = "activity_encoded"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
  """Loads and preprocesses the Wearable Activity dataset, returning full X, y, and groups."""
  if not os.path.isabs(data_dir):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, data_dir)

  csv_path = os.path.join(data_dir, "aw_fb_data.csv")
  if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Dataset not found at {csv_path}")

  dataset = pd.read_csv(csv_path)
  dataset = dataset.dropna()

  # Reconstruct participant_id as proxy if not present
  if "participant_id" not in dataset.columns:
    demo_cols = ["age", "gender", "height", "weight"]
    if all(col in dataset.columns for col in demo_cols):
      dataset["participant_id"] = dataset.groupby(demo_cols).ngroup()
    else:
      raise ValueError(
          "Missing demographic columns to reconstruct participant_id"
      )

  if y_column not in dataset.columns:
    raise ValueError(f"Target column {y_column} not found in dataset.")

  y = dataset[[y_column]]

  cols_to_drop = [y_column]
  if "participant_id" in dataset.columns:
    cols_to_drop.append("participant_id")

  x = dataset.drop(columns=cols_to_drop)
  x = x.select_dtypes(include=["number"])

  return x, y, dataset["participant_id"]


def split_wearable_data(
    x: pd.DataFrame,
    y: pd.DataFrame,
    groups: pd.Series,
    random_state: int = 42,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
]:
  """Splits Wearable data into train, val, and test splits with groups preserved."""
  sgkf = StratifiedGroupKFold(
      n_splits=5, shuffle=True, random_state=random_state
  )
  learn_idx, test_idx = next(sgkf.split(x, y, groups))

  x_learning, y_learning = x.iloc[learn_idx], y.iloc[learn_idx]
  x_test_final, y_test_final = x.iloc[test_idx], y.iloc[test_idx]
  groups_learning, groups_test = groups.iloc[learn_idx], groups.iloc[test_idx]

  sgkf_val = StratifiedGroupKFold(
      n_splits=4, shuffle=True, random_state=random_state
  )
  train_idx, val_idx = next(
      sgkf_val.split(x_learning, y_learning, groups_learning)
  )

  x_train, y_train = x_learning.iloc[train_idx], y_learning.iloc[train_idx]
  x_val, y_val = x_learning.iloc[val_idx], y_learning.iloc[val_idx]
  groups_train = groups_learning.iloc[train_idx]
  groups_val = groups_learning.iloc[val_idx]

  return (
      x_train,
      x_val,
      x_test_final,
      y_train,
      y_val,
      y_test_final,
      groups_train,
      groups_val,
      groups_test,
      x_learning,
      y_learning,
      groups_learning,
  )


def load_wearable_data(
    data_dir: str = "data",
    y_column: str = "activity_encoded",
    random_state: int = 42,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
  """Loads, preprocesses and splits the Wearable Activity dataset using StratifiedGroupKFold."""
  x, y, groups = load_preprocessed_wearable_data(data_dir, y_column)
  res = split_wearable_data(x, y, groups, random_state=random_state)
  return res[0], res[1], res[2], res[3], res[4], res[5]


if __name__ == "__main__":
  d_dir = "data"
  if len(sys.argv) > 1:
    d_dir = sys.argv[1]
  download_wearable_data(d_dir)
