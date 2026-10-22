"""Tests for the headless run driver."""

import json
import os
from unittest import mock

from absl.testing import absltest
from codegen_classroom import agents
from codegen_classroom import run

_RIDGE_SOLUTION = """Plan: ridge regression.

```python
from sklearn.linear_model import Ridge

def solution(x_train, y_train, x_val):
    model = Ridge().fit(x_train, y_train)
    y_pred = pd.DataFrame(model.predict(x_val), index=x_val.index, columns=y_train.columns)
    return y_pred, "Trained Ridge"
```
"""


class ConfigTest(absltest.TestCase):

  def test_load_config_merges_and_overrides(self):
    path = self.create_tempfile(
        content='run:\n  num_cycles: 7\nllm:\n  models: [a]\n'
    ).full_path
    config = run.load_config(
        path, ["run.num_cycles=2", "task.gpus=[1, 2]", "llm.extra_body.top_p=0.9"]
    )
    self.assertEqual(config["run"]["num_cycles"], 2)
    self.assertEqual(config["run"]["seed"], 0)  # Default kept.
    self.assertEqual(config["task"]["gpus"], [1, 2])
    self.assertEqual(config["llm"]["extra_body"], {"top_p": 0.9})
    self.assertEqual(config["llm"]["models"], ["a"])

  def test_bad_override(self):
    with self.assertRaises(ValueError):
      run.apply_overrides({}, ["no_equals_sign"])

  def test_build_student_configs_mixed_backends(self):
    config = run.load_config(None)
    config["llm"].update(
        backend="vllm",
        base_url="http://h:1/v1",
        models=["Qwen/Coder", "Qwen/Coder", {"backend": "gemini", "model": "g"}],
    )
    cfgs = run.build_student_configs(config, "sys")
    self.assertEqual(
        [c.name for c in cfgs],
        ["student_1_Qwen_Coder", "student_2_Qwen_Coder", "student_3_g"],
    )
    self.assertEqual([c.backend for c in cfgs], ["vllm", "vllm", "gemini"])
    self.assertEqual(cfgs[0].base_url, "http://h:1/v1")
    self.assertEqual(cfgs[0].system_instruction, "sys")


class ToyRunTest(absltest.TestCase):

  def test_end_to_end_toy_regression(self):
    mock_backend = mock.Mock()
    mock_backend.retryable_exceptions = ()
    mock_backend.fatal_exceptions = ()
    mock_backend.generate.side_effect = lambda history, prompt, **kw: (
        json.dumps({"model_tags": "ridge"})
        if kw.get("json_mode")
        else _RIDGE_SOLUTION
    )
    output_dir = self.create_tempdir().full_path
    config = run.load_config(
        None,
        [
            f"run.output_dir={output_dir}",
            "run.num_cycles=2",
            "llm.models=[m1, m2]",
            "llm.backend=vllm",
            "task.type=toy_regression",
            "classroom.seed_solution=none",
        ],
    )
    with mock.patch.object(
        agents.backends, "make_backend", return_value=mock_backend
    ):
      run_dir = run.run(config)

    for name in (
        "config.yaml",
        "run.log",
        "final_eval_results.json",
        "best_analysis.json",
        "report.md",
    ):
      self.assertTrue(
          os.path.exists(os.path.join(run_dir, name)), f"missing {name}"
      )
    with open(os.path.join(run_dir, "final_eval_results.json")) as f:
      results = json.load(f)
    self.assertLen(results, 2)
    self.assertGreater(results[0]["best"]["overall_score"], 0.0)


if __name__ == "__main__":
  absltest.main()
