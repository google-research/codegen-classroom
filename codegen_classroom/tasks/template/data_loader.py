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

"""Template Data Loader: Synthetic Multi-Regime Thermal Load Regression.

Demonstrates how to structure `data_loader.py` for a benchmark task with:
  - Clear `# CUSTOMIZE:` customization points for new datasets.
  - Non-trivial synthetic tabular dynamics (regime switches, periodic phase,
    log-skewed load surges, multiplicative interactions, missing sensor values,
    and uninformative distractor columns) so ML models meaningfully hill-climb
    from simple linear baselines (R² ~ 0.45) to feature-engineered ensembles
    (R² ~ 0.95+).
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

# CUSTOMIZE: Replace with your dataset's feature and target names.
FEATURE_COLUMNS: tuple[str, ...] = (
    "ambient_temp",
    "workload_intensity",
    "phase_angle",
    "surge_index",
    "operating_regime",
    "cooling_flow",
    "pressure_delta",
    "temp_sensor_proxy",
    "aux_noise_1",
    "aux_noise_2",
)
TARGET_COLUMN: str = "thermal_load"
DEFAULT_NUM_SAMPLES: int = 1500


def generate_synthetic_dataset(
    num_samples: int = DEFAULT_NUM_SAMPLES,
    seed: int = 42,
) -> pd.DataFrame:
  """Generates a multi-regime non-linear regression dataset for hill-climbing."""
  rng = np.random.default_rng(seed)

  ambient_temp = rng.normal(0.0, 1.0, size=num_samples)
  workload_intensity = rng.normal(0.0, 1.0, size=num_samples)
  phase_angle = rng.uniform(-np.pi, np.pi, size=num_samples)
  surge_index = rng.lognormal(mean=0.0, sigma=0.85, size=num_samples)
  operating_regime = rng.choice(
      [0.0, 1.0, 2.0], size=num_samples, p=[0.45, 0.35, 0.20]
  )
  cooling_flow = rng.normal(0.0, 1.0, size=num_samples)
  pressure_delta = rng.normal(0.0, 1.0, size=num_samples)

  # Collinear sensor proxy with ~4% missing values (NaN)
  temp_sensor_proxy = 0.85 * ambient_temp + rng.normal(
      0.0, 0.35, size=num_samples
  )
  missing_mask = rng.random(num_samples) < 0.04
  temp_sensor_proxy[missing_mask] = np.nan

  # Distractor features (pure noise)
  aux_noise_1 = rng.normal(0.0, 1.0, size=num_samples)
  aux_noise_2 = rng.normal(0.0, 1.0, size=num_samples)

  # Multi-feature non-axis-aligned interactions that reward feature engineering
  thermal_diff = ambient_temp - workload_intensity
  thermal_sum = ambient_temp + workload_intensity
  flow_balance = cooling_flow - 0.8 * pressure_delta

  regime_effect = np.where(
      operating_regime == 0.0,
      2.4 * thermal_sum + 1.2 * flow_balance,
      np.where(
          operating_regime == 1.0,
          2.8 * (ambient_temp * workload_intensity) - 1.8 * flow_balance,
          2.2 * (thermal_diff**2) - 1.5 * (cooling_flow * pressure_delta),
      ),
  )

  # Coupled harmonic, logarithmic, and saturation terms
  periodic_effect = 2.0 * np.sin(phase_angle + 0.5 * thermal_diff)
  surge_effect = 1.6 * np.log1p(surge_index) * (1.0 + 0.5 * operating_regime)
  cooling_saturation = 1.8 * np.tanh(flow_balance * thermal_sum)
  noise = rng.normal(0.0, 0.18, size=num_samples)

  thermal_load = (
      regime_effect
      + periodic_effect
      + surge_effect
      - cooling_saturation
      + noise
  )

  return pd.DataFrame({
      "ambient_temp": ambient_temp,
      "workload_intensity": workload_intensity,
      "phase_angle": phase_angle,
      "surge_index": surge_index,
      "operating_regime": operating_regime,
      "cooling_flow": cooling_flow,
      "pressure_delta": pressure_delta,
      "temp_sensor_proxy": temp_sensor_proxy,
      "aux_noise_1": aux_noise_1,
      "aux_noise_2": aux_noise_2,
      TARGET_COLUMN: thermal_load,
  })


def download_data(
    output_dir: str = "data",
    num_samples: int = DEFAULT_NUM_SAMPLES,
) -> str:
  """Stages or generates the task dataset under `output_dir`."""
  # CUSTOMIZE: Replace synthetic generation with dataset download/staging
  # into `output_dir` (which is gitignored so large files are never committed).
  os.makedirs(output_dir, exist_ok=True)
  csv_path = os.path.join(output_dir, "data.csv")
  df = generate_synthetic_dataset(num_samples=num_samples)
  df.to_csv(csv_path, index=False)
  return csv_path


def load_data(
    data_path: str = "data",
    num_samples: int = DEFAULT_NUM_SAMPLES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
  """Loads features `X` and target `y` as pandas DataFrames."""
  csv_path = (
      os.path.join(data_path, "data.csv")
      if os.path.isdir(data_path)
      else data_path
  )
  if os.path.isfile(csv_path):
    df = pd.read_csv(csv_path)
    if TARGET_COLUMN in df.columns and len(df) == num_samples:
      feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
      return df[feature_cols], df[[TARGET_COLUMN]]

  df = generate_synthetic_dataset(num_samples=num_samples)
  feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
  return df[feature_cols], df[[TARGET_COLUMN]]


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Stage dataset for the template benchmark task."
  )
  parser.add_argument(
      "output_dir",
      nargs="?",
      default="data",
      type=str,
      help="Directory to save the dataset.",
  )
  args = parser.parse_args()
  download_data(output_dir=args.output_dir)
