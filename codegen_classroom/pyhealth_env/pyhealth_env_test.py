"""Tests for PyHealth environments using a small synthetic SampleDataset."""

import os
from unittest import mock

from absl.testing import absltest
from codegen_classroom import agents
from codegen_classroom import code_execution
from codegen_classroom import environments
from codegen_classroom import scoring
from codegen_classroom import student_learning
from codegen_classroom.pyhealth_env import environment as environment_lib
from codegen_classroom.pyhealth_env import prepare as prepare_lib
from codegen_classroom.pyhealth_env import prompts
from codegen_classroom.pyhealth_env import registry
from codegen_classroom.pyhealth_env import runner
from codegen_classroom.pyhealth_env import spec as spec_lib
import numpy as np
from pyhealth.datasets import create_sample_dataset

_NUM_PATIENTS = 60


def _make_samples():
  rng = np.random.default_rng(0)
  samples = []
  for p in range(_NUM_PATIENTS):
    for v in range(2):
      label = int(rng.random() < 0.4)
      conditions = [f"cond-{rng.integers(0, 20)}" for _ in range(3)]
      if label:
        conditions.append("cond-risk")
      samples.append({
          "patient_id": f"patient-{p}",
          "visit_id": f"visit-{p}-{v}",
          "conditions": conditions,
          "procedures": [f"proc-{rng.integers(0, 10)}"],
          "label": label,
      })
  return samples


_BAD_RETURN_CODE = """
def solution(train_dataset, val_dataset, device):
    return 42
"""

_ERROR_CODE = """
def solution(train_dataset, val_dataset, device):
    raise RuntimeError("boom from student")
"""


class PyHealthEnvTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.sample_dataset = create_sample_dataset(
        samples=_make_samples(),
        input_schema={"conditions": "sequence", "procedures": "sequence"},
        output_schema={"label": "binary"},
        dataset_name="synthetic",
        in_memory=False,
    )

  def setUp(self):
    super().setUp()
    # Run student code on CPU for determinism and to leave GPUs alone.
    self.enter_context(
        mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""})
    )
    self.spec = spec_lib.EnvSpec(
        name="synthetic",
        label_key="label",
        sample_dataset_path=str(self.sample_dataset.path),
        description="Synthetic binary task.",
        split_ratios=(0.6, 0.2, 0.2),
        max_learning_train_samples=40,
        epochs_hint=2,
        learning_timeout_seconds=300,
    )
    self.env_dir = self.create_tempdir().full_path
    self.prepared = prepare_lib.prepare_env(self.spec, self.env_dir)
    self.env = environment_lib.PyHealthEnvironment(
        self.spec, self.prepared, work_dir=self.create_tempdir().full_path
    )

  def test_split_by_patient_indices(self):
    p2i = {f"p{i}": [2 * i, 2 * i + 1] for i in range(10)}
    train, val, test = prepare_lib.split_by_patient_indices(
        p2i, (0.6, 0.2, 0.2), seed=1
    )
    all_idx = np.concatenate([train, val, test])
    self.assertCountEqual(all_idx.tolist(), list(range(20)))
    self.assertLen(train, 12)
    # Both samples of a patient land in the same split.
    for split in (train, val, test):
      self.assertTrue(all((i ^ 1) in set(split) for i in split))
    again = prepare_lib.split_by_patient_indices(p2i, (0.6, 0.2, 0.2), seed=1)
    np.testing.assert_array_equal(train, again[0])

  def test_prepare_env_contents_and_cache(self):
    sizes = self.prepared.split_sizes
    self.assertEqual(
        sizes["train"] + sizes["val"] + sizes["test"], 2 * _NUM_PATIENTS
    )
    self.assertEqual(sizes["learning_train"], 40)
    self.assertIn("conditions", self.prepared.schema_summary["inputs"])
    self.assertEqual(set(self.prepared.label_distribution), {"0", "1"})
    with mock.patch.object(
        prepare_lib, "split_by_patient_indices"
    ) as mock_split:
      cached = prepare_lib.prepare_env(self.spec, self.env_dir)
    mock_split.assert_not_called()
    self.assertEqual(cached.split_sizes, sizes)

  def test_execute_seed_solution_learning_phase(self):
    code, _ = code_execution.extract_python_code(
        prompts.build_seed_solution(self.spec), "solution"
    )
    execution = self.env.execute(
        code, phase=environments.LEARNING, label="student_1_cycle_0"
    )
    n_val = self.prepared.split_sizes["learning_val"]
    self.assertEqual(execution.y_pred.shape, (n_val, 1))
    self.assertTrue(execution.y_true.index.equals(execution.y_pred.index))
    self.assertTrue(((execution.y_pred >= 0) & (execution.y_pred <= 1)).all().all())
    self.assertIn("[runner]", execution.execution_feedback)
    score, metrics, _, _ = scoring.score_predictions(
        execution.y_true,
        execution.y_pred,
        {},
        self.spec.metric_weights,
        binary_threshold=self.spec.metric_binary_threshold,
    )
    self.assertTrue(np.isfinite(score))
    self.assertIn("auroc", metrics)

  def test_execute_basemodel_template(self):
    execution = self.env.execute(
        prompts.BASEMODEL_TEMPLATE,
        phase=environments.LEARNING,
        label="basemodel_template",
    )
    self.assertLen(execution.y_pred, self.prepared.split_sizes["learning_val"])
    self.assertIn("epoch 0", execution.execution_feedback)

  def test_task_prompt_documents_batch_and_templates(self):
    prompt = prompts.build_task_prompt(self.spec, self.prepared)
    self.assertIn("| `conditions` | input feature | SequenceProcessor", prompt)
    self.assertIn("| `label` | label |", prompt)
    self.assertIn("class BagOfCodesModel(BaseModel)", prompt)
    self.assertIn("from pyhealth.models import RNN", prompt)
    self.assertEqual(prompt.count("def solution("), 3)  # Contract + 2 templates.

  def test_execute_final_phase_uses_test_split(self):
    code, _ = code_execution.extract_python_code(
        prompts.build_seed_solution(self.spec), "solution"
    )
    execution = self.env.execute(
        code, phase=environments.FINAL, label="student_1_best_final"
    )
    self.assertLen(execution.y_pred, self.prepared.split_sizes["test"])

  def test_relative_dirs_are_resolved(self):
    cwd = os.getcwd()
    base = self.create_tempdir().full_path
    os.chdir(base)
    try:
      prepared = prepare_lib.prepare_env(self.spec, "envs/rel")
      env = environment_lib.PyHealthEnvironment(
          self.spec, prepared, work_dir="runs/rel"
      )
      self.assertTrue(os.path.isabs(prepared.splits_path))
      execution = env.execute(
          prompts.BASEMODEL_TEMPLATE, phase=environments.LEARNING, label="rel"
      )
    finally:
      os.chdir(cwd)
    self.assertLen(execution.y_pred, prepared.split_sizes["learning_val"])

  def test_execute_student_error_raises_execution_error(self):
    with self.assertRaisesRegex(
        code_execution.ExecutionError, "boom from student"
    ):
      self.env.execute(_ERROR_CODE, phase=environments.LEARNING, label="err")

  def test_execute_bad_return_raises_execution_error(self):
    with self.assertRaisesRegex(code_execution.ExecutionError, "trained model"):
      self.env.execute(
          _BAD_RETURN_CODE, phase=environments.LEARNING, label="bad"
      )

  def test_learning_cycle_with_env(self):
    mock_backend = mock.Mock()
    mock_backend.retryable_exceptions = ()
    mock_backend.fatal_exceptions = ()
    mock_backend.generate.return_value = "Try again."
    with mock.patch.object(
        agents.backends, "make_backend", return_value=mock_backend
    ):
      student = agents.StudentAgent(
          agents.AgentConfig(name="student_1"),
          self.create_tempdir().full_path,
          seed_solution=prompts.build_seed_solution(self.spec),
      )
    results = student_learning.run_parallel_classroom_learning_cycle(
        current_classroom=[student],
        x_train=None,
        y_train=None,
        x_val=None,
        y_val=None,
        current_cycle=0,
        student_initial_prompt=prompts.build_task_prompt(
            self.spec, self.prepared
        ),
        student_function_name="solution",
        metric_weights=self.spec.metric_weights,
        early_stopping_patience=2,
        student_refine_instruction_options=list(prompts.REFINE_INSTRUCTIONS),
        max_mutation_count=1,
        mutation_strategy_options=[agents.MutationStrategy.SELF_BEST_CODE],
        code_execution_max_retries=0,
        metric_binary_threshold=self.spec.metric_binary_threshold,
        env=self.env,
    )
    self.assertLen(results, 1)
    self.assertEqual(results[0].exclude_reason, "")
    self.assertTrue(np.isfinite(results[0].overall_score))
    self.assertIn("auroc", results[0].metrics)
    self.assertEqual(student.best_cycle, 0)


class RunnerScoresTest(absltest.TestCase):

  def test_binary_two_column_probs(self):
    y_true, y_score = runner.to_scores(
        np.array([[0.0], [1.0]]),
        np.array([[0.8, 0.2], [0.1, 0.9]]),
        "binary",
    )
    np.testing.assert_allclose(y_true, [0, 1])
    np.testing.assert_allclose(y_score, [0.2, 0.9])

  def test_binary_rejects_non_probabilities(self):
    with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
      runner.to_scores(np.array([0, 1]), np.array([[-1.0], [2.0]]), "binary")

  def test_multiclass_argmax(self):
    y_true, y_score = runner.to_scores(
        np.array([2, 0]),
        np.array([[0.1, 0.2, 0.7], [0.6, 0.3, 0.1]]),
        "multiclass",
    )
    np.testing.assert_allclose(y_true, [2, 0])
    np.testing.assert_allclose(y_score, [2, 0])

  def test_length_mismatch(self):
    with self.assertRaisesRegex(ValueError, "rows"):
      runner.to_scores(np.array([0, 1, 1]), np.array([0.5, 0.5]), "binary")


class RegistryTest(absltest.TestCase):

  def test_get_spec_merges_dataset_kwargs(self):
    spec = registry.get_spec(
        "mimic4_mortality",
        {"dataset_kwargs": {"dev": True}, "epochs_hint": 3},
    )
    self.assertEqual(spec.name, "mimic4_mortality")
    self.assertTrue(spec.dataset_kwargs["dev"])
    self.assertIn("ehr_root", spec.dataset_kwargs)
    self.assertEqual(spec.epochs_hint, 3)
    self.assertEqual(spec.metric_weights, {"auroc": 0.5, "auprc": 0.5})

  def test_all_registered_specs_build(self):
    for name in registry.list_envs():
      self.assertEqual(registry.get_spec(name).name, name)

  def test_unknown_env(self):
    with self.assertRaises(KeyError):
      registry.get_spec("nope")

  def test_spec_validation(self):
    with self.assertRaises(ValueError):
      spec_lib.EnvSpec(name="x", label_key="y")
    with self.assertRaises(ValueError):
      spec_lib.EnvSpec(
          name="x", label_key="y", sample_dataset_path="/p", task_type="bad"
      )


if __name__ == "__main__":
  absltest.main()
