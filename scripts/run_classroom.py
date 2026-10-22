#!/usr/bin/env python
"""Runs a codegen classroom experiment from a YAML config (no notebook needed).

Examples:
  # PyHealth MIMIC-IV mortality with students served by a local vLLM server.
  python scripts/run_classroom.py --config configs/mimic4_mortality_vllm.yaml

  # Quick smoke test against the vLLM server on a toy regression task.
  python scripts/run_classroom.py --config configs/regression_toy.yaml

  # Override any config value (values are parsed as YAML).
  python scripts/run_classroom.py --config configs/mimic4_mortality_vllm.yaml \
      --override run.num_cycles=2 'task.gpus=[1]' \
      task.overrides.dataset_kwargs.dev=true

  # List the built-in PyHealth environments.
  python scripts/run_classroom.py --list-envs
"""

import argparse
import sys


def main() -> int:
  # Score lines are print()ed; keep them in order with the logs when redirected.
  sys.stdout.reconfigure(line_buffering=True)
  parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  parser.add_argument("--config", help="Path to a YAML run config.")
  parser.add_argument(
      "--override",
      nargs="*",
      default=[],
      metavar="KEY=VALUE",
      help="Dotted config overrides, e.g. run.num_cycles=3.",
  )
  parser.add_argument(
      "--list-envs",
      action="store_true",
      help="Print the built-in PyHealth environments and exit.",
  )
  parser.add_argument(
      "--print-config",
      action="store_true",
      help="Print the resolved config and exit.",
  )
  args = parser.parse_args()

  if args.list_envs:
    from codegen_classroom.pyhealth_env import registry  # pylint: disable=g-import-not-at-top

    for name in registry.list_envs():
      spec = registry.get_spec(name)
      print(f"{name:24s} {spec.task_type:10s} {spec.description}")
    return 0

  from codegen_classroom import run  # pylint: disable=g-import-not-at-top

  config = run.load_config(args.config, args.override)
  if args.print_config:
    import yaml  # pylint: disable=g-import-not-at-top

    print(yaml.safe_dump(config, sort_keys=False))
    return 0
  run_dir = run.run(config)
  print(f"Run directory: {run_dir}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
