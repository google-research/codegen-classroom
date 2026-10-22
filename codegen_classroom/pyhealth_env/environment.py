"""TaskEnvironment that trains student PyHealth models in GPU subprocesses."""

from __future__ import annotations

import json
import logging
import os
import re
import sys

from codegen_classroom import code_execution
from codegen_classroom import environments
from codegen_classroom.pyhealth_env import prepare as prepare_lib
from codegen_classroom.pyhealth_env import spec as spec_lib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RUNNER_MODULE = "codegen_classroom.pyhealth_env.runner"
# Max characters of the student's stdout returned as execution feedback.
MAX_FEEDBACK_CHARS = 8000


class PyHealthEnvironment:
  """Runs `solution(train_dataset, val_dataset, device) -> model` per attempt.

  Learning cycles train on the (optionally capped) train split and are scored
  on the validation split. Final evaluation trains on the full train split and
  is scored on the held-out test split, which students never receive.
  """

  def __init__(
      self,
      spec: spec_lib.EnvSpec,
      prepared: prepare_lib.PreparedEnv,
      work_dir: str,
      gpu_pool: code_execution.GpuPool | None = None,
      function_name: str = "solution",
      python_executable: str = sys.executable,
      seed: int = 0,
  ):
    self.spec = spec
    self.prepared = prepared
    # Absolute, because each runner subprocess starts in its own directory.
    self.work_dir = os.path.abspath(work_dir)
    self.gpu_pool = gpu_pool or code_execution.GpuPool()
    self.function_name = function_name
    self.python_executable = python_executable
    self.seed = seed

  def _splits(self, phase: str) -> tuple[str, str, str]:
    if phase == environments.LEARNING:
      return (
          prepare_lib.LEARNING_TRAIN,
          prepare_lib.LEARNING_VAL,
          prepare_lib.LEARNING_VAL,
      )
    if phase == environments.FINAL:
      return prepare_lib.TRAIN, prepare_lib.VAL, prepare_lib.TEST
    raise ValueError(f"Unknown phase {phase!r}.")

  def execute(
      self,
      code: str,
      *,
      phase: str,
      label: str,
      timeout_seconds: float | None = None,
  ) -> environments.EnvExecution:
    train_split, val_split, eval_split = self._splits(phase)
    if timeout_seconds is None:
      timeout_seconds = (
          self.spec.learning_timeout_seconds
          if phase == environments.LEARNING
          else self.spec.final_timeout_seconds
      )
    safe_label = re.sub(r"[^A-Za-z0-9_.-]", "_", label)
    workdir = os.path.join(self.work_dir, "executions", safe_label)
    os.makedirs(workdir, exist_ok=True)
    output_file = os.path.join(workdir, "predictions.npz")
    if os.path.exists(output_file):
      os.remove(output_file)
    job = {
        "sample_dataset_path": self.prepared.sample_dataset_path,
        "splits_path": self.prepared.splits_path,
        "train_split": train_split,
        "val_split": val_split,
        "eval_split": eval_split,
        "label_key": self.spec.label_key,
        "task_type": self.spec.task_type,
        "function_name": self.function_name,
        "code_file": os.path.join(workdir, "solution.py"),
        "output_file": output_file,
        "inference_batch_size": self.spec.inference_batch_size,
        "seed": self.seed,
    }
    job_path = os.path.join(workdir, "job.json")
    with open(job_path, "w") as f:
      json.dump(job, f, indent=2)

    with self.gpu_pool.acquire() as device:
      env_vars = {}
      if device is not None:
        env_vars["CUDA_VISIBLE_DEVICES"] = device
      logger.debug("[%s] Running on GPU %s (%s phase)", label, device, phase)
      stdout, exec_time = code_execution.execute_code_subprocess(
          code,
          [self.python_executable, "-m", RUNNER_MODULE, "--job", job_path],
          workdir,
          timeout_seconds=timeout_seconds,
          env_vars=env_vars,
      )

    if not os.path.exists(output_file):
      raise code_execution.ExecutionError(
          "The runner finished without writing predictions."
      )
    with np.load(output_file) as data:
      y_true = pd.DataFrame({self.spec.label_key: data["y_true"]})
      y_pred = pd.DataFrame({self.spec.label_key: data["y_score"]})
    return environments.EnvExecution(
        y_true=y_true,
        y_pred=y_pred,
        execution_feedback=stdout[-MAX_FEEDBACK_CHARS:],
        exec_time=exec_time,
    )
