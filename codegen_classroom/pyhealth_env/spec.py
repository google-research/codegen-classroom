"""Specification of a PyHealth training environment (dataset + task)."""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import importlib
from typing import Any

BINARY = "binary"
MULTICLASS = "multiclass"
TASK_TYPES = (BINARY, MULTICLASS)


@dataclasses.dataclass
class EnvSpec:
  """A PyHealth dataset + task that students write model code for.

  Students write `solution(train_dataset, val_dataset, device)` returning a
  trained model; the environment runs inference on the evaluation split and
  scores the probabilities.
  """

  name: str
  # Label field in the task's output_schema, e.g. "mortality".
  label_key: str
  task_type: str = BINARY
  # Fully qualified classes, e.g. "pyhealth.datasets.MIMIC4Dataset".
  dataset_class: str | None = None
  dataset_kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)
  task_class: str | None = None
  task_kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)
  # Use a pre-built SampleDataset directory (containing schema.pkl) instead of
  # building one from dataset_class/task_class.
  sample_dataset_path: str | None = None
  description: str = ""
  # Patient-level train/val/test split.
  split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
  split_seed: int = 42
  # Caps on samples used during learning cycles (None = use all). Final
  # evaluation always trains on the full train split and scores the full test
  # split.
  max_learning_train_samples: int | None = None
  max_learning_val_samples: int | None = None
  # Wall-clock limits per execution, in seconds.
  learning_timeout_seconds: float = 1800
  final_timeout_seconds: float = 7200
  # Suggested epoch budget mentioned in the student prompt.
  epochs_hint: int = 5
  # Weighted metrics used for the overall score.
  metric_weights: Mapping[str, float] | None = None
  # Batch size used by the runner for inference on the evaluation split.
  inference_batch_size: int = 256

  def __post_init__(self):
    if self.task_type not in TASK_TYPES:
      raise ValueError(
          f"task_type must be one of {TASK_TYPES}, got {self.task_type!r}."
      )
    if self.sample_dataset_path is None and not (
        self.dataset_class and self.task_class
    ):
      raise ValueError(
          "EnvSpec needs either sample_dataset_path or both dataset_class and"
          " task_class."
      )
    self.split_ratios = tuple(self.split_ratios)
    if self.metric_weights is None:
      self.metric_weights = default_metric_weights(self.task_type)

  @property
  def metric_binary_threshold(self) -> float | None:
    """Threshold for binarizing probabilities in binary-label metrics."""
    return 0.5 if self.task_type == BINARY else None


def default_metric_weights(task_type: str) -> dict[str, float]:
  if task_type == BINARY:
    return {"auroc": 0.5, "auprc": 0.5}
  return {"accuracy": 0.5, "f1_macro": 0.5}


def import_object(qualified_name: str) -> Any:
  """Imports `pkg.module.Name` and returns `Name`."""
  module_name, _, attr = qualified_name.rpartition(".")
  if not module_name:
    raise ValueError(f"Expected a fully qualified name, got {qualified_name!r}")
  return getattr(importlib.import_module(module_name), attr)
