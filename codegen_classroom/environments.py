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

"""Task environments: pluggable execution of student code for scoring.

By default the learning loop runs student code in-process (or in a forked
child) as `solution(x_train, y_train, [aux...], x_val)` on pandas DataFrames.
A `TaskEnvironment` replaces that step for tasks whose data does not fit the
DataFrame contract (e.g. PyHealth datasets trained on GPUs). The environment
owns how the code is run and returns ground truth and predictions as
index-aligned DataFrames, so scoring, refinement and mutation are unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
from typing import Protocol

import pandas as pd

# Phases passed to `TaskEnvironment.execute`.
LEARNING = "learning"  # Learning cycles: scored on the validation split.
FINAL = "final"  # Final evaluation: scored on the held-out test split.


@dataclasses.dataclass
class EnvExecution:
  """Output of running student code in an environment."""

  y_true: pd.DataFrame  # Ground truth aligned to `y_pred.index`.
  y_pred: pd.DataFrame  # Predictions to score.
  execution_feedback: str = ""  # Captured stdout / feedback for the student.
  exec_time: float = 0.0  # Wall-clock execution time in seconds.
  auxiliary: Mapping[str, pd.DataFrame] | None = None  # Extra ground truths.


class TaskEnvironment(Protocol):
  """Runs student code for a task and returns scorable predictions."""

  def execute(
      self,
      code: str,
      *,
      phase: str,
      label: str,
      timeout_seconds: float | None = None,
  ) -> EnvExecution:
    """Executes `code` for `phase` (LEARNING or FINAL).

    Args:
      code: The extracted student code.
      phase: LEARNING or FINAL.
      label: Unique, filesystem-safe label for this execution (student, cycle,
        attempt), used for working directories and logs.
      timeout_seconds: Optional timeout override. When None, the environment
        applies its own default for the phase.

    Returns:
      An EnvExecution with index-aligned ground truth and predictions.

    Raises:
      code_execution.ExecutionError, code_execution.ExecutionTimeoutError or
      scoring.ValidatorError on failures that the student should fix.
    """
    ...
