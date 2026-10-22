"""Child-process entrypoint that trains and evaluates one student solution.

Invoked by `PyHealthEnvironment` as

  python -m codegen_classroom.pyhealth_env.runner --job job.json

in a fresh interpreter (never forked), with CUDA_VISIBLE_DEVICES pinned to the
GPU slot assigned to this execution. The runner:

  1. Reloads the prepared SampleDataset and the train/val/eval index splits.
  2. Imports the student's code and calls
     `model = solution(train_dataset, val_dataset, device)`.
  3. Runs inference itself on the evaluation split, which the student never
     sees, collecting `y_prob` and `y_true` from the same batches so they are
     aligned regardless of dataloader ordering.
  4. Writes `y_true` and `y_score` to the job's output .npz file.

Any exception propagates to stderr and a non-zero exit code, which the parent
turns into an ExecutionError fed back to the student.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import time
from typing import Any

import numpy as np


def _set_seed(seed: int) -> None:
  import torch  # pylint: disable=g-import-not-at-top

  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def _load_solution(code_file: str, function_name: str):
  module_spec = importlib.util.spec_from_file_location(
      "student_solution", code_file
  )
  module = importlib.util.module_from_spec(module_spec)
  sys.modules["student_solution"] = module
  module_spec.loader.exec_module(module)
  fn = getattr(module, function_name, None)
  if fn is None or not callable(fn):
    raise AttributeError(
        f"Your code must define a callable `{function_name}(train_dataset,"
        " val_dataset, device)`."
    )
  return fn


def _to_numpy(x: Any) -> np.ndarray:
  if hasattr(x, "detach"):
    x = x.detach().float().cpu().numpy()
  return np.asarray(x)


def run_inference(
    model: Any,
    eval_dataset: Any,
    label_key: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
  """Returns (y_true, y_prob) over `eval_dataset` using `model(**batch)`."""
  import torch  # pylint: disable=g-import-not-at-top
  from pyhealth.datasets import get_dataloader  # pylint: disable=g-import-not-at-top

  if not callable(model):
    raise TypeError(
        "solution() must return a trained model callable as `model(**batch)`"
        f" (e.g. a pyhealth.models.BaseModel); got {type(model).__name__}."
    )
  if isinstance(model, torch.nn.Module):
    model.eval()
  loader = get_dataloader(eval_dataset, batch_size=batch_size, shuffle=False)
  y_true_parts, y_prob_parts = [], []
  with torch.no_grad():
    for batch in loader:
      output = model(**batch)
      if not isinstance(output, dict) or "y_prob" not in output:
        raise TypeError(
            "model(**batch) must return a dict containing 'y_prob' (as"
            " pyhealth.models.BaseModel does); got"
            f" {type(output).__name__} with keys"
            f" {list(output) if isinstance(output, dict) else None}."
        )
      y_true = output.get("y_true", batch[label_key])
      y_prob_parts.append(_to_numpy(output["y_prob"]))
      y_true_parts.append(_to_numpy(y_true))
  return np.concatenate(y_true_parts), np.concatenate(y_prob_parts)


def to_scores(
    y_true: np.ndarray, y_prob: np.ndarray, task_type: str
) -> tuple[np.ndarray, np.ndarray]:
  """Converts raw model outputs to 1-D (y_true, y_score) for scoring.

  Binary: y_score is the positive-class probability. Multiclass: y_score is the
  predicted class index (argmax of y_prob).

  Args:
    y_true: Labels as returned by the model or batch.
    y_prob: Probabilities as returned by the model.
    task_type: "binary" or "multiclass".

  Returns:
    Tuple of 1-D (y_true, y_score) arrays of equal length.
  """
  n = len(y_true)
  if len(y_prob) != n:
    raise ValueError(
        f"y_prob has {len(y_prob)} rows but there are {n} labels."
    )
  if task_type == "binary":
    if y_prob.ndim == 2 and y_prob.shape[1] == 2:
      y_prob = y_prob[:, 1]
    y_score = y_prob.reshape(n, -1)
    if y_score.shape[1] != 1:
      raise ValueError(
          "Binary task: y_prob must have shape (n,), (n, 1) or (n, 2); got"
          f" {y_prob.shape}."
      )
    y_score = y_score[:, 0]
    y_true = y_true.reshape(n, -1)[:, 0]
    if np.any((y_score < 0) | (y_score > 1)):
      raise ValueError("Binary task: y_prob must be probabilities in [0, 1].")
  else:
    if y_prob.ndim != 2:
      raise ValueError(
          f"Multiclass task: y_prob must have shape (n, C); got {y_prob.shape}."
      )
    y_score = y_prob.argmax(axis=1)
    y_true = (
        y_true.argmax(axis=1)
        if y_true.ndim == 2 and y_true.shape[1] > 1
        else y_true.reshape(-1)
    )
  if not np.all(np.isfinite(y_score)):
    raise ValueError("Predictions contain NaN or infinite values.")
  return y_true.astype(np.float64), y_score.astype(np.float64)


def main(argv=None) -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--job", required=True)
  args = parser.parse_args(argv)
  with open(args.job) as f:
    job = json.load(f)

  import torch  # pylint: disable=g-import-not-at-top
  from pyhealth.datasets import SampleDataset  # pylint: disable=g-import-not-at-top

  # SampleDataset.close() deletes its directory, which is the shared cache.
  SampleDataset.close = lambda self: None

  _set_seed(job.get("seed", 0))
  splits = np.load(job["splits_path"])
  dataset = SampleDataset(path=job["sample_dataset_path"])
  train_dataset = dataset.subset(splits[job["train_split"]].tolist())
  val_dataset = dataset.subset(splits[job["val_split"]].tolist())
  eval_dataset = dataset.subset(splits[job["eval_split"]].tolist())
  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(
      f"[runner] train={len(train_dataset)} val={len(val_dataset)}"
      f" device={device}"
      f" CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}",
      flush=True,
  )

  solution = _load_solution(job["code_file"], job["function_name"])
  t0 = time.time()
  model = solution(train_dataset, val_dataset, device)
  train_time = time.time() - t0

  t0 = time.time()
  y_true, y_prob = run_inference(
      model, eval_dataset, job["label_key"], job["inference_batch_size"]
  )
  y_true, y_score = to_scores(y_true, y_prob, job["task_type"])
  print(
      f"[runner] solution() took {train_time:.1f}s; inference on"
      f" {len(y_score)} samples took {time.time() - t0:.1f}s.",
      flush=True,
  )
  np.savez(job["output_file"], y_true=y_true, y_score=y_score)


if __name__ == "__main__":
  main()
