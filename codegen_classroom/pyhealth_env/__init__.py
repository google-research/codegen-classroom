"""PyHealth training environments for codegen classroom.

Students write `solution(train_dataset, val_dataset, device) -> model` for a
PyHealth dataset + task; each attempt is trained in a fresh subprocess on an
assigned GPU and scored on a held-out split.

Typical use:

  spec = registry.get_spec("mimic4_mortality")
  env, prepared = make_environment(spec, env_dir=..., work_dir=run_dir,
                                   gpus=[1, 2])
  prompt = prompts.build_task_prompt(spec, prepared)
  student_learning.run_parallel_classroom_learning_cycle(..., env=env)
"""

from __future__ import annotations

from collections.abc import Sequence

from codegen_classroom import code_execution
from codegen_classroom.pyhealth_env import environment
from codegen_classroom.pyhealth_env import prepare
from codegen_classroom.pyhealth_env import prompts
from codegen_classroom.pyhealth_env import registry
from codegen_classroom.pyhealth_env import spec

EnvSpec = spec.EnvSpec
PyHealthEnvironment = environment.PyHealthEnvironment


def make_environment(
    env_spec: spec.EnvSpec,
    env_dir: str,
    work_dir: str,
    gpus: Sequence[int | str] | None = None,
    jobs_per_gpu: int = 1,
    seed: int = 0,
) -> tuple[environment.PyHealthEnvironment, prepare.PreparedEnv]:
  """Prepares the env (cached in `env_dir`) and returns it with its artifacts."""
  prepared = prepare.prepare_env(env_spec, env_dir)
  env = environment.PyHealthEnvironment(
      env_spec,
      prepared,
      work_dir=work_dir,
      gpu_pool=code_execution.GpuPool(gpus, jobs_per_gpu=jobs_per_gpu),
      seed=seed,
  )
  return env, prepared
