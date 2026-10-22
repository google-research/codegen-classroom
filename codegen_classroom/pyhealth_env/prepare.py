"""One-time preparation of a PyHealth environment.

Builds (or reuses) the task's litdata SampleDataset, computes deterministic
patient-level splits and summarizes the schema and label distribution for the
student prompt. Everything is written to `env_dir` so that student executions
(separate processes) can reload the exact same splits.

Runs on CPU in the main process (PyHealth's `set_task` refuses to run in a
non-main process).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import collections
import dataclasses
import hashlib
import json
import logging
import os
from typing import Any

from codegen_classroom.pyhealth_env import spec as spec_lib
import numpy as np

logger = logging.getLogger(__name__)

# Bump when the prepared artifacts change so cached env dirs are rebuilt.
PREPARE_VERSION = 2
PREPARED_FILE = "prepared.json"
SPLITS_FILE = "splits.npz"
# Max samples read to estimate the label distribution for the prompt.
_LABEL_STATS_MAX_SAMPLES = 20000

# Split names stored in SPLITS_FILE.
TRAIN, VAL, TEST = "train", "val", "test"
LEARNING_TRAIN, LEARNING_VAL = "learning_train", "learning_val"


@dataclasses.dataclass
class PreparedEnv:
  """Artifacts of a prepared environment, stored in `env_dir`."""

  env_dir: str
  sample_dataset_path: str
  split_sizes: Mapping[str, int]
  label_distribution: Mapping[str, int]
  schema_summary: Mapping[str, Any]
  # Per-key description of one real batch from `get_dataloader`.
  batch_summary: Mapping[str, Any] = dataclasses.field(default_factory=dict)

  @property
  def splits_path(self) -> str:
    return os.path.join(self.env_dir, SPLITS_FILE)


def _split_fingerprint(spec: spec_lib.EnvSpec) -> str:
  """Hash of every spec field that affects the data or splits."""
  keys = {
      "version": PREPARE_VERSION,
      "dataset_class": spec.dataset_class,
      "dataset_kwargs": spec.dataset_kwargs,
      "task_class": spec.task_class,
      "task_kwargs": spec.task_kwargs,
      "sample_dataset_path": spec.sample_dataset_path,
      "split_ratios": spec.split_ratios,
      "split_seed": spec.split_seed,
      "max_learning_train_samples": spec.max_learning_train_samples,
      "max_learning_val_samples": spec.max_learning_val_samples,
      "label_key": spec.label_key,
  }
  blob = json.dumps(keys, sort_keys=True, default=str).encode()
  return hashlib.sha256(blob).hexdigest()[:16]


def build_sample_dataset_path(spec: spec_lib.EnvSpec) -> str:
  """Returns the litdata SampleDataset directory for the spec, building it."""
  if spec.sample_dataset_path:
    return spec.sample_dataset_path
  dataset_cls = spec_lib.import_object(spec.dataset_class)
  task_cls = spec_lib.import_object(spec.task_class)
  logger.info("Loading %s(%s)", spec.dataset_class, dict(spec.dataset_kwargs))
  base_dataset = dataset_cls(**spec.dataset_kwargs)
  task = task_cls(**spec.task_kwargs)
  logger.info("Setting task %s (cached after the first run)", spec.task_class)
  sample_dataset = base_dataset.set_task(task)
  return str(sample_dataset.path)


def split_by_patient_indices(
    patient_to_index: Mapping[Any, Sequence[int]],
    ratios: Sequence[float],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Patient-level split into sorted sample index arrays.

  Same procedure as `pyhealth.datasets.split_by_patient`, but returns indices
  so they can be persisted and reused by child processes.

  Args:
    patient_to_index: Mapping from patient id to sample indices.
    ratios: Train/val/test ratios summing to 1.
    seed: Shuffle seed.

  Returns:
    Sorted (train, val, test) sample index arrays.
  """
  if not np.isclose(sum(ratios), 1.0):
    raise ValueError(f"Split ratios must sum to 1, got {ratios}.")
  patients = sorted(patient_to_index, key=str)
  rng = np.random.default_rng(seed)
  rng.shuffle(patients)
  n = len(patients)
  cut1 = int(n * ratios[0])
  cut2 = int(n * (ratios[0] + ratios[1]))
  groups = (patients[:cut1], patients[cut1:cut2], patients[cut2:])

  def _indices(group):
    idx = [i for p in group for i in patient_to_index[p]]
    return np.array(sorted(idx), dtype=np.int64)

  return tuple(_indices(g) for g in groups)


def _subsample(
    indices: np.ndarray, max_samples: int | None, seed: int
) -> np.ndarray:
  if max_samples is None or len(indices) <= max_samples:
    return indices
  rng = np.random.default_rng(seed)
  return np.sort(rng.choice(indices, size=max_samples, replace=False))


def _label_value(label: Any) -> str:
  """Converts a processed label (tensor/array/scalar) to a hashable string."""
  arr = np.asarray(label.detach().cpu() if hasattr(label, "detach") else label)
  if arr.size == 1:
    return str(int(arr.reshape(-1)[0]))
  return str(int(arr.argmax()))


def _processor_summary(processor: Any) -> dict[str, Any]:
  summary: dict[str, Any] = {"processor": type(processor).__name__}
  size = getattr(processor, "size", None)
  if callable(size):
    try:
      summary["size"] = int(size())
    except Exception:  # pylint: disable=broad-exception-caught
      pass
  for attr in ("schema", "is_token"):
    fn = getattr(processor, attr, None)
    if callable(fn):
      try:
        value = fn()
        summary[attr] = list(value) if isinstance(value, tuple) else value
      except Exception:  # pylint: disable=broad-exception-caught
        pass
  return summary


def summarize_schema(sample_dataset: Any) -> dict[str, Any]:
  """Summarizes input/output schemas and fitted processors for the prompt."""
  return {
      "inputs": {
          key: {
              "schema": str(sample_dataset.input_schema.get(key)),
              **_processor_summary(proc),
          }
          for key, proc in sample_dataset.input_processors.items()
      },
      "outputs": {
          key: {
              "schema": str(sample_dataset.output_schema.get(key)),
              **_processor_summary(proc),
          }
          for key, proc in sample_dataset.output_processors.items()
      },
  }


def summarize_batch(sample_dataset: Any, batch_size: int = 4) -> dict:
  """Describes each field of one real batch (type, dtype, shape)."""
  from pyhealth.datasets import get_dataloader  # pylint: disable=g-import-not-at-top

  batch = next(iter(get_dataloader(sample_dataset, batch_size=batch_size)))
  summary = {}
  for key, value in batch.items():
    if hasattr(value, "shape"):
      summary[key] = {
          "type": type(value).__name__,
          "dtype": str(value.dtype).replace("torch.", ""),
          "shape": list(value.shape),
      }
    elif isinstance(value, (tuple, list)) and value and hasattr(
        value[0], "shape"
    ):
      summary[key] = {
          "type": f"{type(value).__name__} of tensors",
          "shapes": [list(v.shape) for v in value],
          "dtypes": [str(v.dtype).replace("torch.", "") for v in value],
      }
    else:
      summary[key] = {
          "type": f"{type(value).__name__}[{type(value[0]).__name__}]"
          if isinstance(value, (tuple, list)) and value
          else type(value).__name__
      }
  return summary


def prepare_env(spec: spec_lib.EnvSpec, env_dir: str) -> PreparedEnv:
  """Prepares (or loads the cached) environment artifacts in `env_dir`."""
  from pyhealth.datasets import SampleDataset  # pylint: disable=g-import-not-at-top

  # Absolute paths: the artifacts are read by subprocesses with another cwd.
  env_dir = os.path.abspath(env_dir)
  os.makedirs(env_dir, exist_ok=True)
  prepared_path = os.path.join(env_dir, PREPARED_FILE)
  fingerprint = _split_fingerprint(spec)
  if os.path.exists(prepared_path):
    with open(prepared_path) as f:
      cached = json.load(f)
    if cached.get("fingerprint") == fingerprint and os.path.exists(
        os.path.join(env_dir, SPLITS_FILE)
    ):
      logger.info("Using prepared PyHealth env at %s", env_dir)
      return PreparedEnv(
          env_dir=env_dir,
          sample_dataset_path=os.path.abspath(cached["sample_dataset_path"]),
          split_sizes=cached["split_sizes"],
          label_distribution=cached["label_distribution"],
          schema_summary=cached["schema_summary"],
          batch_summary=cached.get("batch_summary", {}),
      )

  sample_path = os.path.abspath(build_sample_dataset_path(spec))
  dataset = SampleDataset(path=sample_path)
  if spec.label_key not in dataset.output_processors:
    raise ValueError(
        f"label_key {spec.label_key!r} not in task outputs"
        f" {list(dataset.output_processors)}."
    )

  train, val, test = split_by_patient_indices(
      dataset.patient_to_index, spec.split_ratios, spec.split_seed
  )
  splits = {
      TRAIN: train,
      VAL: val,
      TEST: test,
      LEARNING_TRAIN: _subsample(
          train, spec.max_learning_train_samples, spec.split_seed
      ),
      LEARNING_VAL: _subsample(
          val, spec.max_learning_val_samples, spec.split_seed + 1
      ),
  }
  for name, idx in splits.items():
    if len(idx) == 0:
      raise ValueError(f"Split {name!r} is empty; check split_ratios.")
  np.savez(os.path.join(env_dir, SPLITS_FILE), **splits)

  stats_idx = _subsample(train, _LABEL_STATS_MAX_SAMPLES, spec.split_seed)
  label_counts = collections.Counter(
      _label_value(sample[spec.label_key])
      for sample in dataset.subset(stats_idx.tolist())
  )

  prepared = PreparedEnv(
      env_dir=env_dir,
      sample_dataset_path=sample_path,
      split_sizes={k: int(len(v)) for k, v in splits.items()},
      label_distribution=dict(sorted(label_counts.items())),
      schema_summary=summarize_schema(dataset),
      batch_summary=summarize_batch(
          dataset.subset(splits[LEARNING_TRAIN][:8].tolist())
      ),
  )
  with open(prepared_path, "w") as f:
    json.dump(
        {
            "fingerprint": fingerprint,
            "spec": dataclasses.asdict(spec),
            **dataclasses.asdict(prepared),
        },
        f,
        indent=2,
        default=str,
    )
  logger.info(
      "Prepared PyHealth env %s: splits=%s labels=%s",
      spec.name,
      prepared.split_sizes,
      prepared.label_distribution,
  )
  return prepared
