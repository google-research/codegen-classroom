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

"""Data loader and download manager for California Housing dataset."""

from __future__ import annotations

import os
import sys
import urllib.request

import pandas as pd
from sklearn.model_selection import train_test_split

# ===== DATA DOWNLOAD =====


def download_housing_data(data_dir: str = "data") -> None:
  """Downloads the California Housing dataset from GitHub."""
  os.makedirs(data_dir, exist_ok=True)
  target_file = os.path.join(data_dir, "housing.csv")

  if os.path.exists(target_file):
    print(f"Data already exists at {target_file}")
    return

  print("Downloading California Housing dataset...")
  url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
  urllib.request.urlretrieve(url, target_file)
  print(f"Download complete. Saved to {target_file}")


def ensure_data(data_dir: str = "data") -> str:
  """Ensures housing.csv exists, downloading if necessary."""
  csv_path = os.path.join(data_dir, "housing.csv")
  if not os.path.exists(csv_path):
    pkg_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), data_dir, "housing.csv"
    )
    if os.path.exists(pkg_path):
      return pkg_path
    download_housing_data(data_dir)
  return csv_path


# ===== DATA LOADING =====


def load_preprocessed_housing_data(
    data_dir: str = "data", y_column: str = "median_house_value"
) -> tuple[pd.DataFrame, pd.DataFrame]:
  """Loads and preprocesses the Housing dataset, returning full X and y."""
  csv_path = os.path.join(data_dir, "housing.csv")
  if not os.path.exists(csv_path):
    pkg_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), data_dir, "housing.csv"
    )
    if os.path.exists(pkg_path):
      csv_path = pkg_path
    else:
      raise FileNotFoundError(f"Dataset not found at {csv_path}")

  dataset = pd.read_csv(csv_path).dropna()

  if y_column not in dataset.columns:
    raise ValueError(f"Target column {y_column} not found in dataset.")

  y = dataset[[y_column]]
  x = dataset.drop(y_column, axis=1)

  # One-hot encode text columns
  if "ocean_proximity" in x.columns:
    x = pd.get_dummies(x, columns=["ocean_proximity"], drop_first=True)

  return x, y


def load_housing_data(
    data_dir: str = "data",
    y_column: str = "median_house_value",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
  """Loads, preprocesses and splits the Housing dataset."""
  x, y = load_preprocessed_housing_data(data_dir, y_column)

  # Partition dataset into learning set (80%) and held-out test set (20%).
  x_learning, x_test, y_learning, y_test = train_test_split(
      x, y, test_size=test_size, random_state=random_state
  )

  # Further partition learning set into train (80%) and val (20%).
  x_train, x_val, y_train, y_val = train_test_split(
      x_learning, y_learning, test_size=0.2, random_state=random_state
  )

  return x_train, x_val, x_test, y_train, y_val, y_test


if __name__ == "__main__":
  download_housing_data(sys.argv[1] if len(sys.argv) > 1 else "data")
