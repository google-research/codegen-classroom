"""Built-in PyHealth environments mirroring `benchmark_tasks/`.

Dataset roots and cache directories default to the paths used by the
benchmark scripts on this cluster; override any field from the run config
(e.g. `env.dataset_kwargs.ehr_root`).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import copy
import dataclasses
from typing import Any

from codegen_classroom.pyhealth_env import spec as spec_lib

_MIMIC4_ROOT = "/shared/rsaas/physionet.org/files/mimiciv/2.2"
_MIMIC4_CACHE = "/shared/eng/pyhealth_agent/cache"
_MIMIC3_ROOT = "/srv/local/data/MIMIC-III"
_MIMIC3_CACHE = "/shared/eng/pyhealth/mimic3_dataset"
_EICU_ROOT = "/srv/local/data/physionet.org/files/eicu-crd/2.0"
_EICU_CACHE = "/shared/eng/pyhealth/eicu_dataset"


def _mimic4(tables, task_class, label_key, task_type, description):
  return spec_lib.EnvSpec(
      name="",
      dataset_class="pyhealth.datasets.MIMIC4Dataset",
      dataset_kwargs={
          "ehr_root": _MIMIC4_ROOT,
          "ehr_tables": tables,
          "cache_dir": _MIMIC4_CACHE,
          "num_workers": 4,
      },
      task_class=task_class,
      label_key=label_key,
      task_type=task_type,
      description=description,
      max_learning_train_samples=20000,
  )


_REGISTRY: Mapping[str, Callable[[], spec_lib.EnvSpec]] = {
    "mimic4_mortality": lambda: _mimic4(
        [
            "patients",
            "admissions",
            "diagnoses_icd",
            "procedures_icd",
            "prescriptions",
        ],
        "pyhealth.tasks.MortalityPredictionMIMIC4",
        "mortality",
        spec_lib.BINARY,
        "In-hospital mortality prediction from the next admission on"
        " MIMIC-IV (diagnoses, procedures, prescriptions).",
    ),
    "mimic4_readmission": lambda: _mimic4(
        [
            "patients",
            "admissions",
            "diagnoses_icd",
            "procedures_icd",
            "prescriptions",
        ],
        "pyhealth.tasks.ReadmissionPredictionMIMIC4",
        "readmission",
        spec_lib.BINARY,
        "Hospital readmission prediction on MIMIC-IV.",
    ),
    "mimic4_los": lambda: _mimic4(
        ["admissions", "diagnoses_icd", "procedures_icd", "prescriptions"],
        "pyhealth.tasks.LengthOfStayPredictionMIMIC4",
        "los",
        spec_lib.MULTICLASS,
        "Length-of-stay category prediction (multiclass) on MIMIC-IV.",
    ),
    "mimic3_mortality": lambda: spec_lib.EnvSpec(
        name="",
        dataset_class="pyhealth.datasets.MIMIC3Dataset",
        dataset_kwargs={
            "root": _MIMIC3_ROOT,
            "tables": ["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
            "cache_dir": _MIMIC3_CACHE,
        },
        task_class="pyhealth.tasks.MortalityPredictionMIMIC3",
        label_key="mortality",
        task_type=spec_lib.BINARY,
        description="Mortality prediction from the next visit on MIMIC-III.",
    ),
    "eicu_mortality": lambda: spec_lib.EnvSpec(
        name="",
        dataset_class="pyhealth.datasets.eICUDataset",
        dataset_kwargs={
            "root": _EICU_ROOT,
            "tables": ["diagnosis", "medication", "physicalexam"],
            "cache_dir": _EICU_CACHE,
        },
        task_class="pyhealth.tasks.MortalityPredictionEICU",
        label_key="mortality",
        task_type=spec_lib.BINARY,
        description="ICU mortality prediction on eICU.",
    ),
}


def list_envs() -> list[str]:
  return sorted(_REGISTRY)


def get_spec(
    name: str, overrides: Mapping[str, Any] | None = None
) -> spec_lib.EnvSpec:
  """Returns the named built-in spec, with optional field overrides.

  `dataset_kwargs` and `task_kwargs` overrides are merged into the defaults
  rather than replacing them.

  Args:
    name: Registered environment name (see `list_envs()`), or "custom" to build
      the spec entirely from `overrides`.
    overrides: EnvSpec field overrides.

  Returns:
    The EnvSpec.
  """
  overrides = dict(overrides or {})
  if name == "custom":
    return spec_lib.EnvSpec(name=overrides.pop("name", "custom"), **overrides)
  if name not in _REGISTRY:
    raise KeyError(f"Unknown PyHealth env {name!r}. Available: {list_envs()}")
  spec = _REGISTRY[name]()
  spec.name = name
  for key in ("dataset_kwargs", "task_kwargs"):
    if key in overrides:
      merged = copy.deepcopy(dict(getattr(spec, key)))
      merged.update(overrides.pop(key) or {})
      overrides[key] = merged
  return dataclasses.replace(spec, **overrides)
