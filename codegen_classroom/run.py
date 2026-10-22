"""Headless end-to-end classroom run driven by a YAML config.

Usage (see scripts/run_classroom.py and configs/):

  python scripts/run_classroom.py --config configs/mimic4_mortality_vllm.yaml \
      --override run.num_cycles=2 task.overrides.dataset_kwargs.dev=true

Steps: set up the run directory and logging, prepare the task, build the
students, run the learning cycles, run the final evaluation, analyze the best
solution, and write `report.md` (+ plots) into the run directory.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import dataclasses
import logging
import os
from typing import Any

from codegen_classroom import agent_config
from codegen_classroom import agents
from codegen_classroom import data_utils
from codegen_classroom import environments
from codegen_classroom import reporting
from codegen_classroom import scoring
from codegen_classroom import student_learning
import numpy as np
import pandas as pd

logger = logging.getLogger("codegen_classroom.run")

DEFAULT_CONFIG: dict[str, Any] = {
    "run": {
        "output_dir": "runs",
        "name": None,  # Defaults to the task name.
        "seed": 0,
        "num_cycles": 5,
        "final_strategies": ["first", "last", "best"],
        "leaderboard_path": None,
        "analyze_best": True,
    },
    "llm": {
        "backend": agent_config.DEFAULT_BACKEND,
        "base_url": None,
        "api_key": None,
        # One student per entry: a model name, or {backend, model, base_url}.
        "models": None,
        "temperature": 0.8,
        "max_output_tokens": 16384,
        "extra_body": None,
        "max_code_generation_retries": None,
        "retry_sleep_seconds": None,
        "analysis_model": None,  # Defaults to the first student's model.
    },
    "classroom": {
        "early_stopping_patience": 2,
        "max_mutation_count": 2,
        "mutation_strategies": ["self_best_code", "best_peer_crossover"],
        "code_execution_max_retries": 2,
        "classroom_feedback": True,
        # Seed with the baseline solution: "none", "first" or "all" students.
        "seed_solution": "first",
        "num_workers": None,
    },
    "task": {
        "type": "pyhealth",  # "pyhealth" or "toy_regression".
        # pyhealth:
        "env": None,  # Registered env name or "custom".
        "overrides": {},  # EnvSpec field overrides.
        "env_cache_dir": None,  # Defaults to <output_dir>/envs/<env>.
        "gpus": [],
        "jobs_per_gpu": 1,
        # toy_regression:
        "n_samples": 300,
        "code_execution_timeout_seconds": 60,
    },
}

_TOY_SYSTEM_INSTRUCTION = (
    "You are an expert ML engineer. Answer with a short plan followed by one"
    " complete ```python code block defining the requested function."
)
_TOY_PROMPT = """\
Write a Python function `solution(x_train, y_train, x_val)` that trains a
regression model on (x_train, y_train) and returns a tuple
`(y_pred, feedback)`, where `y_pred` is a pandas DataFrame of predictions for
x_val with the same index as x_val and the single column 'target', and
`feedback` is a short string describing what you did. pandas is available as
`pd`, numpy as `np`; scikit-learn may be imported.
"""
_TOY_REFINE = (
    "Improve your solution, e.g. with feature engineering or regularization.",
    "Try a different model family and compare.",
)


def deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict:
  """Recursively merges `update` into a copy of `base`."""
  merged = copy.deepcopy(dict(base))
  for key, value in (update or {}).items():
    if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
      merged[key] = deep_merge(merged[key], value)
    else:
      merged[key] = copy.deepcopy(value)
  return merged


def apply_overrides(config: dict, overrides: Sequence[str]) -> dict:
  """Applies `a.b.c=value` overrides; values are parsed as YAML."""
  import yaml  # pylint: disable=g-import-not-at-top

  config = copy.deepcopy(config)
  for item in overrides or []:
    if "=" not in item:
      raise ValueError(f"Override must look like key.path=value, got {item!r}")
    path, raw = item.split("=", 1)
    keys = path.strip().split(".")
    node = config
    for key in keys[:-1]:
      if not isinstance(node.get(key), dict):
        node[key] = {}  # Missing or None-valued defaults become mappings.
      node = node[key]
    node[keys[-1]] = yaml.safe_load(raw)
  return config


def load_config(
    path: str | None, overrides: Sequence[str] | None = None
) -> dict:
  import yaml  # pylint: disable=g-import-not-at-top

  user = {}
  if path:
    with open(path) as f:
      user = yaml.safe_load(f) or {}
  return apply_overrides(deep_merge(DEFAULT_CONFIG, user), overrides or [])


@dataclasses.dataclass
class TaskSetup:
  """Everything the learning loop needs for one task."""

  name: str
  system_instruction: str
  initial_prompt: str
  refine_instructions: Sequence[str]
  metric_weights: Mapping[str, float]
  metric_binary_threshold: float | None = None
  seed_solution: str | None = None
  env: environments.TaskEnvironment | None = None
  timeout_seconds: float | None = None
  # DataFrame tasks only.
  data: Mapping[str, pd.DataFrame | None] = dataclasses.field(
      default_factory=dict
  )


def _setup_pyhealth_task(config: dict, run_dir: str) -> TaskSetup:
  from codegen_classroom import pyhealth_env  # pylint: disable=g-import-not-at-top

  task_cfg = config["task"]
  if not task_cfg.get("env"):
    raise ValueError("task.env must name a PyHealth env (or 'custom').")
  spec = pyhealth_env.registry.get_spec(
      task_cfg["env"], task_cfg.get("overrides")
  )
  env_dir = task_cfg.get("env_cache_dir") or os.path.join(
      config["run"]["output_dir"], "envs", spec.name
  )
  env, prepared = pyhealth_env.make_environment(
      spec,
      env_dir=env_dir,
      work_dir=run_dir,
      gpus=task_cfg.get("gpus") or [],
      jobs_per_gpu=task_cfg.get("jobs_per_gpu", 1),
      seed=config["run"]["seed"],
  )
  return TaskSetup(
      name=spec.name,
      system_instruction=pyhealth_env.prompts.SYSTEM_INSTRUCTION,
      initial_prompt=pyhealth_env.prompts.build_task_prompt(spec, prepared),
      refine_instructions=pyhealth_env.prompts.REFINE_INSTRUCTIONS,
      metric_weights=spec.metric_weights,
      metric_binary_threshold=spec.metric_binary_threshold,
      seed_solution=pyhealth_env.prompts.build_seed_solution(spec),
      env=env,
  )


def _setup_toy_regression_task(config: dict) -> TaskSetup:
  rng = np.random.default_rng(config["run"]["seed"])
  n = int(config["task"]["n_samples"])
  x = pd.DataFrame(rng.normal(size=(n, 3)), columns=["x1", "x2", "x3"])
  y = pd.DataFrame({
      "target": 2 * x["x1"] - x["x2"] ** 2 + 0.1 * rng.normal(size=n)
  })
  i_val, i_test = int(0.6 * n), int(0.8 * n)
  return TaskSetup(
      name="toy_regression",
      system_instruction=_TOY_SYSTEM_INSTRUCTION,
      initial_prompt=_TOY_PROMPT,
      refine_instructions=_TOY_REFINE,
      metric_weights={"r2": 1.0},
      timeout_seconds=config["task"]["code_execution_timeout_seconds"],
      data={
          "x_train": x.iloc[:i_val],
          "y_train": y.iloc[:i_val],
          "x_val": x.iloc[i_val:i_test],
          "y_val": y.iloc[i_val:i_test],
          "x_test": x.iloc[i_test:],
          "y_test": y.iloc[i_test:],
      },
  )


def setup_task(config: dict, run_dir: str) -> TaskSetup:
  task_type = config["task"]["type"]
  if task_type == "pyhealth":
    return _setup_pyhealth_task(config, run_dir)
  if task_type == "toy_regression":
    return _setup_toy_regression_task(config)
  raise ValueError(f"Unknown task.type {task_type!r}.")


def build_student_configs(
    config: dict, system_instruction: str
) -> list[agents.AgentConfig]:
  """One AgentConfig per entry of llm.models (plain names or dicts)."""
  llm = config["llm"]
  models = llm.get("models")
  if not models:
    raise ValueError("llm.models must list at least one model.")
  configs = []
  for entry in models:
    if isinstance(entry, str):
      entry = {"model": entry}
    cfg = student_learning.get_classroom_student_configs(
        system_instruction=system_instruction,
        api_key=entry.get("api_key", llm.get("api_key")),
        model_names=[entry["model"]],
        temperature=entry.get("temperature", llm["temperature"]),
        max_output_tokens=entry.get(
            "max_output_tokens", llm["max_output_tokens"]
        ),
        max_code_generation_retries=llm.get("max_code_generation_retries"),
        retry_sleep_seconds=llm.get("retry_sleep_seconds"),
        backend=entry.get("backend", llm["backend"]),
        base_url=entry.get("base_url", llm.get("base_url")),
        extra_body=entry.get("extra_body", llm.get("extra_body")),
    )[0]
    configs.append(cfg)
  # Names are student_1_<model>; renumber so they stay unique.
  for i, cfg in enumerate(configs):
    cfg.name = cfg.name.replace("student_1_", f"student_{i + 1}_", 1)
  return configs


def _analyze_best(
    config: dict,
    task: TaskSetup,
    run_dir: str,
    students: Mapping[str, agents.StudentAgent],
    eval_results: Sequence[Mapping[str, Any]],
    student_configs: Sequence[agents.AgentConfig],
) -> None:
  best_name, best_data = student_learning.get_best_solution_from_eval_results(
      eval_results
  )
  if not best_name or best_name not in students:
    logger.warning("No best solution to analyze.")
    return
  student = students[best_name]
  best_result = student.cycle_results[student.best_cycle]
  base = student_configs[0]
  analysis_cfg = dataclasses.replace(
      base,
      name="solution_analyst",
      system_instruction="",
      model_name=config["llm"].get("analysis_model") or base.model_name,
      temperature=0.2,
      response_mime_type=None,
  )
  analyst = agents.SolutionAnalysisAgent(analysis_cfg, run_base_dir=run_dir)
  analysis = analyst.analyze(
      student_name=best_name,
      best_code=best_result.code,
      best_plan=best_result.plan,
      best_test_metrics=best_data,
      best_val_metrics=best_result.metrics,
      student_instructions=task.initial_prompt,
      metrics_description=scoring.get_metrics_description(task.metric_weights),
      task_description_for_prompt=task.name,
  )
  logger.info("Best solution analysis: %s", analysis)


def run(config: dict) -> str:
  """Runs a full classroom experiment. Returns the run directory."""
  run_cfg, cls_cfg = config["run"], config["classroom"]
  run_name = run_cfg.get("name") or config["task"].get("env") or "run"
  run_dir = os.path.join(
      data_utils.resolve_workspace_dir(run_cfg["output_dir"]),
      f"{run_name}_{data_utils.generate_run_id()}",
  )
  data_utils.create_dir(run_dir)
  reporting.setup_logger(
      "codegen_classroom", log_file=os.path.join(run_dir, "run.log")
  )
  reporting.suppress_common_warnings()
  import yaml  # pylint: disable=g-import-not-at-top

  with open(os.path.join(run_dir, "config.yaml"), "w") as f:
    yaml.safe_dump(config, f, sort_keys=False)
  logger.info("Run directory: %s", run_dir)

  task = setup_task(config, run_dir)
  student_configs = build_student_configs(config, task.system_instruction)
  seed_mode = cls_cfg.get("seed_solution", "none")
  classroom: student_learning.Classroom = []
  for i, cfg in enumerate(student_configs):
    seeded = seed_mode == "all" or (seed_mode == "first" and i == 0)
    classroom.append(
        agents.StudentAgent(
            cfg,
            run_dir,
            seed_solution=task.seed_solution if seeded else None,
        )
    )
  students = {s.name: s for s in classroom}
  history = {s.name: [] for s in classroom}
  mutation_strategies = [
      agents.MutationStrategy[name.upper()]
      for name in cls_cfg["mutation_strategies"]
  ]
  metric_keys = ["overall_score", *task.metric_weights]

  active = list(classroom)
  for cycle in range(int(run_cfg["num_cycles"])):
    if not active:
      logger.info("All students excluded; stopping after %d cycles.", cycle)
      break
    feedback = (
        scoring.get_classroom_feedback_string(
            cycle, active, history, metric_keys
        )
        if cls_cfg.get("classroom_feedback")
        else None
    )
    results = student_learning.run_parallel_classroom_learning_cycle(
        current_classroom=active,
        x_train=task.data.get("x_train"),
        y_train=task.data.get("y_train"),
        x_val=task.data.get("x_val"),
        y_val=task.data.get("y_val"),
        current_cycle=cycle,
        student_initial_prompt=task.initial_prompt,
        student_function_name="solution",
        metric_weights=task.metric_weights,
        early_stopping_patience=cls_cfg["early_stopping_patience"],
        student_refine_instruction_options=list(task.refine_instructions),
        max_mutation_count=cls_cfg["max_mutation_count"],
        mutation_strategy_options=mutation_strategies,
        code_execution_max_retries=cls_cfg["code_execution_max_retries"],
        code_execution_timeout_seconds=task.timeout_seconds,
        classroom_feedback=feedback or None,
        num_workers=cls_cfg.get("num_workers"),
        metric_binary_threshold=task.metric_binary_threshold,
        env=task.env,
    )
    active = student_learning.process_cycle_results(
        active, results, history, cycle
    )

  final_results_path = os.path.join(run_dir, "final_eval_results.json")
  eval_results, _ = student_learning.run_parallel_classroom_final_evaluation(
      students=students,
      x_learning=task.data.get("x_train"),
      y_learning=task.data.get("y_train"),
      x_test_final=task.data.get("x_test"),
      y_test_final=task.data.get("y_test"),
      student_function_name="solution",
      metric_weights=task.metric_weights,
      metric_binary_threshold=task.metric_binary_threshold,
      code_execution_timeout_seconds=task.timeout_seconds,
      enforce_timeout=task.env is None,
      strategies=run_cfg["final_strategies"],
      num_workers=cls_cfg.get("num_workers"),
      final_results_path=final_results_path,
      predictions_dir=os.path.join(run_dir, "predictions"),
      env=task.env,
  )

  if run_cfg.get("analyze_best"):
    try:
      _analyze_best(
          config, task, run_dir, students, eval_results, student_configs
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      logger.warning("Solution analysis failed: %s", e)

  sections = reporting.build_run_report(
      run_dir, students=classroom, metric_weights=task.metric_weights
  )
  reporting.write_report(run_dir, sections, metric_weights=task.metric_weights)
  reporting.render_report(sections)

  if run_cfg.get("leaderboard_path"):
    best_name, best_data = student_learning.get_best_solution_from_eval_results(
        eval_results
    )
    if best_name:
      row = {
          "run_id": os.path.basename(run_dir),
          "task": task.name,
          "student_name": best_name,
          "model_name": students[best_name].model_name,
          **{k: v for k, v in (best_data or {}).items() if k != "traceback"},
      }
      data_utils.submit_results_to_local_leaderboard(
          run_cfg["leaderboard_path"], pd.DataFrame([row])
      )
  logger.info("Done. Report: %s", os.path.join(run_dir, "report.md"))
  return run_dir
