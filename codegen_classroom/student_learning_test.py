# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for student_learning cycle and evaluation functions."""

import collections
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from codegen_classroom import agents
from codegen_classroom import code_execution
from codegen_classroom import data_utils
from codegen_classroom import scoring
from codegen_classroom import student_learning
import numpy as np
import pandas as pd

MOCK_GOOD_CODE = """
def solution(x_train, y_train, x_val):
    # Mock good execution
    predictions = pd.DataFrame({'prediction': np.arange(len(x_val))})
    return predictions, "Mock execution success"
"""

MOCK_BAD_CODE = """
def solution(x_train, y_train, x_val):
    raise ValueError("Mock execution error")
"""


class StudentLearningTestBase(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.base_dir = self.create_tempdir().full_path

    # Mock the LLM backend for agents
    self.mock_make_backend = self.enter_context(
        mock.patch.object(agents.backends, 'make_backend', autospec=True)
    )

    # Common data
    self.x_train = pd.DataFrame({'a': range(10)})
    self.y_train = pd.DataFrame({'b': range(10)})
    self.aux_init = {
        'positive_label': pd.DataFrame({'c': [0] * 10}),
        'negative_label': pd.DataFrame({'d': [1] * 10}),
    }
    self.x_val = pd.DataFrame({'a': range(5)})
    self.y_val = pd.DataFrame({'b': range(5)})
    self.aux_val = {
        'positive_label': pd.DataFrame({'c': [0] * 5}),
        'negative_label': pd.DataFrame({'d': [1] * 5}),
    }
    self.metric_weights = {'mae': 1.0}
    self.cv_splits = [
        (np.array([0, 1]), np.array([2, 3])),
        (np.array([2, 3]), np.array([0, 1])),
    ]

    # Default mocks for successful execution path
    self.mock_extract = self.enter_context(
        mock.patch.object(
            code_execution,
            'extract_python_code',
            return_value=(MOCK_GOOD_CODE, 'Plan'),
        )
    )
    mock_y_pred = pd.DataFrame({'prediction': np.arange(len(self.x_val))})
    self.mock_execute = self.enter_context(
        mock.patch.object(
            code_execution,
            'execute_code',
            return_value=((mock_y_pred, 'Feedback'), 0.1),
        )
    )
    self.patcher_score = mock.patch.object(
        scoring,
        'score_predictions',
        return_value=(
            0.9,
            {'mae': 0.1, 'exec_time': 0.1},
            'Good score',
            {},
        ),
    )
    self.mock_score = self.enter_context(self.patcher_score)
    self.mock_validator = mock.MagicMock(return_value=None)
    self.mock_time = self.enter_context(
        mock.patch.object(student_learning, 'time', autospec=True)
    )
    self.mock_time.time.return_value = 0
    self.mock_traceback = self.enter_context(
        mock.patch(
            'codegen_classroom.student_learning.traceback.format_exc',
            return_value='Mock Traceback',
        )
    )

  def create_student(self, name='s1'):
    config = agents.AgentConfig(name=name, api_key='mock_key')
    student = agents.StudentAgent(config, self.base_dir)
    student.generate_solution = mock.MagicMock(return_value='Mock Solution')
    student.mutate = mock.MagicMock()
    student.cycle_results = [
        agents.StudentCycleResult(cycle=0, code=MOCK_GOOD_CODE)
    ]
    return student

  def _run_cycle(self, **kwargs):
    """Helper to call run_learning_cycle with defaults."""
    student = kwargs.get('student') or self.create_student()
    defaults = {
        'student': student,
        'x_train': self.x_train,
        'y_train': self.y_train,
        'x_val': self.x_val,
        'y_val': self.y_val,
        'current_cycle': 0,
        'classroom': [student],
        'student_initial_prompt': 'Initial',
        'student_function_name': 'solution',
        'metric_weights': self.metric_weights,
        'early_stopping_patience': 3,
        'student_refine_instruction_options': ['Refine'],
        'max_mutation_count': 1,
        'mutation_strategy_options': [agents.MutationStrategy.SELF_BEST_CODE],
        'code_execution_max_retries': 1,
    }
    defaults.update(kwargs)
    return student_learning.run_learning_cycle(**defaults)

  def _run_final(self, **kwargs):
    """Helper to call run_final_evaluation with defaults."""
    student = kwargs.get('student') or self.create_student()
    defaults = {
        'student_name': student.name,
        'x_learning': self.x_train,
        'y_learning': self.y_train,
        'x_test_final': self.x_val,
        'y_test_final': self.y_val,
        'strategies': ['last'],
        'students': {student.name: student},
        'student_function_name': 'solution',
        'metric_weights': self.metric_weights,
    }
    defaults.update(kwargs)
    # Remove student if in kwargs but not accepted by run_final_evaluation
    if 'student' in kwargs and 'student' not in defaults:
      pass  # already handled by student_name and students

    # run_final_evaluation doesn't take 'student' as arg if in 'students'.
    kwargs_cleaned = defaults.copy()
    if 'student' in kwargs_cleaned:
      del kwargs_cleaned['student']

    return student_learning.run_final_evaluation(**kwargs_cleaned)


class ValidationTests(StudentLearningTestBase):

  @parameterized.named_parameters(
      dict(
          testcase_name='valid',
          inputs=dict(
              df1=pd.DataFrame({'a': [1]}),
              df2=pd.DataFrame({'b': [2]}),
              none_val=None,
          ),
          expected_regex=None,
      ),
      dict(
          testcase_name='invalid_series',
          inputs=dict(df1=pd.DataFrame({'a': [1]}), df2=pd.Series([1, 2])),
          expected_regex="Input 'df2' must be a pandas DataFrame",
      ),
  )
  def test_validate_dataframe_inputs(self, inputs, expected_regex):
    if expected_regex:
      with self.assertRaisesRegex(TypeError, expected_regex):
        student_learning._validate_dataframe_inputs(**inputs)
    else:
      student_learning._validate_dataframe_inputs(**inputs)


class ExecuteWithCVSplitsTests(StudentLearningTestBase):
  """Direct tests for _execute_with_cv_splits."""

  @parameterized.named_parameters(
      dict(
          testcase_name='with_val_with_cv',
          x_val_in=True,
          cv_splits_in=True,
          expected_len=4,
          expected_call_count=2,
          y_train_in=True,
      ),
      dict(
          testcase_name='no_val_with_cv',
          x_val_in=False,
          cv_splits_in=True,
          expected_len=4,
          expected_call_count=2,
          y_train_in=True,
      ),
      dict(
          testcase_name='with_val_no_cv',
          x_val_in=True,
          cv_splits_in=False,
          expected_len=5,
          expected_call_count=1,
          y_train_in=True,
      ),
      dict(
          testcase_name='no_val_no_cv',
          x_val_in=False,
          cv_splits_in=False,
          expected_len=10,
          expected_call_count=1,
          y_train_in=True,
      ),
      dict(
          testcase_name='no_y_train_with_cv',
          x_val_in=False,
          cv_splits_in=True,
          expected_len=4,
          expected_call_count=2,
          # Testing our semi-supervised digital twin mode bugfix explicitly
          y_train_in=False,
      ),
  )
  def test_execute_with_cv_splits_modes(
      self,
      x_val_in,
      cv_splits_in,
      expected_len,
      expected_call_count,
      y_train_in,
  ):
    """Verifies execution with different x_val and cv_splits combinations."""

    def side_effect(code, name, fn_args, **kwargs):
      del code, name, kwargs
      x_val_fold = fn_args[-1]
      y_pred = pd.DataFrame(
          {'prediction': np.arange(len(x_val_fold))}, index=x_val_fold.index
      )
      return (y_pred, 'Feedback'), 0.1

    self.mock_execute.side_effect = side_effect

    x_val = self.x_val if x_val_in else None
    cv_splits = self.cv_splits if cv_splits_in else None
    y_train = self.y_train if y_train_in else None

    y_pred_df, _, total_time = student_learning._execute_with_cv_splits(
        extracted_code=MOCK_GOOD_CODE,
        target_fn_name='solution',
        x_train=self.x_train,
        y_train=y_train,
        x_val=x_val,
        cv_splits=cv_splits,
        initial_namespace={'pd': pd, 'np': np},
        validator=self.mock_validator,
    )

    self.assertLen(y_pred_df, expected_len)
    self.assertGreater(total_time, 0)
    self.assertEqual(self.mock_execute.call_count, expected_call_count)

  def test_execute_with_cv_splits_feedback_concatenation(self):
    """Verifies that multiple feedbacks are joined with a separator."""
    self.mock_execute.side_effect = [
        ((pd.DataFrame({'p': [0]}), 'FB1'), 0.1),
        ((pd.DataFrame({'p': [0]}), 'FB2'), 0.1),
    ]

    _, feedback, _ = student_learning._execute_with_cv_splits(
        extracted_code=MOCK_GOOD_CODE,
        target_fn_name='solution',
        x_train=self.x_train,
        y_train=self.y_train,
        x_val=self.x_val,
        cv_splits=self.cv_splits,
        cv_sample_feedback_count=2,
        initial_namespace={'pd': pd, 'np': np},
        validator=self.mock_validator,
    )

    self.assertIn('FB1', feedback)
    self.assertIn('FB2', feedback)
    self.assertIn('\n\n---\n\n', feedback)
    self.assertIn('Executed on 2 folds. Feedback from fold 1 shown.', feedback)
    self.assertIn('Executed on 2 folds. Feedback from fold 2 shown.', feedback)

  def test_execute_with_cv_splits_zero_feedback(self):
    """Verifies feedback is empty when cv_sample_feedback_count is 0."""
    _, feedback, _ = student_learning._execute_with_cv_splits(
        extracted_code=MOCK_GOOD_CODE,
        target_fn_name='solution',
        x_train=self.x_train,
        y_train=self.y_train,
        x_val=self.x_val,
        cv_splits=self.cv_splits,
        cv_sample_feedback_count=0,
    )
    self.assertEqual(feedback, '')

  def test_execute_with_cv_splits_auxiliary_data(self):
    """Tests that auxiliary data is correctly sliced and passed to folds."""

    def side_effect(code, name, fn_args, **kwargs):
      del code, name, kwargs
      # x_train_fold, y_train_fold, aux1, aux2, x_val_fold
      self.assertLen(fn_args, 5)
      self.assertLen(fn_args[2], 2)  # aux1 fold len
      return (pd.DataFrame({'p': [0, 0]}, index=fn_args[-1].index), 'Ok'), 0.1

    self.mock_execute.side_effect = side_effect
    student_learning._execute_with_cv_splits(
        extracted_code=MOCK_GOOD_CODE,
        target_fn_name='solution',
        x_train=self.x_train,
        y_train=self.y_train,
        x_val=self.x_val,
        cv_splits=self.cv_splits,
        auxiliary_train_data=self.aux_init,
    )

  def test_execute_with_cv_splits_fold_failure(self):
    """Verifies that a single fold failure propagates the exception."""
    self.mock_execute.side_effect = code_execution.ExecutionError('Fold fail')
    with self.assertRaises(code_execution.ExecutionError):
      student_learning._execute_with_cv_splits(
          extracted_code=MOCK_GOOD_CODE,
          target_fn_name='fn',
          x_train=self.x_train,
          y_train=self.y_train,
          cv_splits=self.cv_splits,
      )

  def test_align_to_predictions_and_mase_scale(self):
    """Tests _align_to_predictions with non-contiguous subset index and aux MASE scale."""
    y_true = pd.DataFrame({'target': [1.0, 2.0, 3.0]}, index=[101, 205, 309])
    y_pred = pd.DataFrame({'target': [1.5, 2.5]}, index=[205, 309])
    aux = {
        'mase_scale': pd.DataFrame(
            {'mase_scale': [0.5, 0.5, 0.5]}, index=[101, 205, 309]
        )
    }

    aligned_y, aligned_aux = student_learning._align_to_predictions(
        y_pred, y_true, aux
    )
    pd.testing.assert_frame_equal(aligned_y, y_true.loc[[205, 309]])
    pd.testing.assert_frame_equal(
        aligned_aux['mase_scale'], aux['mase_scale'].loc[[205, 309]]
    )
    noop_y, noop_aux = student_learning._align_to_predictions(
        y_true, y_true, aux
    )
    self.assertIs(noop_y, y_true)
    self.assertIs(noop_aux, aux)
    none_y, none_aux = student_learning._align_to_predictions(
        y_pred, None, None
    )
    self.assertIsNone(none_y)
    self.assertIsNone(none_aux)

  def test_partial_cv_end_to_end_and_non_cv_rejection(self):
    """Verifies partial CV alignment with aux data and non-CV rejection."""
    student = self.create_student('s1')
    student.best_solution = MOCK_GOOD_CODE
    student.best_cycle = 0
    partial_splits = [(np.array([0, 1]), np.array([2]))]
    x_test = pd.DataFrame({'f': [1, 2, 3]}, index=[101, 205, 309])
    y_test = pd.DataFrame({'t': [10, 20, 30]}, index=[101, 205, 309])
    aux_test = {'scale': pd.DataFrame({'s': [1, 1, 1]}, index=[101, 205, 309])}

    self.mock_execute.return_value = (
        (pd.DataFrame({'p': [99]}, index=[309]), 'fb'),
        0.1,
    )
    res, preds = student_learning.run_final_evaluation(
        's1',
        self.x_train,
        self.y_train,
        x_test,
        y_test,
        ['best'],
        {'s1': student},
        'solution',
        self.metric_weights,
        cv_splits=partial_splits,
        auxiliary_test_data=aux_test,
        validator=scoring.prediction_validator,
    )
    self.assertIn('best', res)
    self.assertIn('s1_best_y_predictions', preds)

    self.mock_score.side_effect = self.patcher_score.temp_original
    with self.assertRaisesRegex(
        scoring.ScoreError, 'Indices of y_true and y_pred do not match'
    ):
      scoring.score_predictions(
          y_test,
          pd.DataFrame({'p': [99]}, index=[309]),
          {},
          self.metric_weights,
      )


class LearningCycleTests(StudentLearningTestBase):
  """Tests related to the main learning cycle functions."""

  @parameterized.named_parameters(
      dict(
          testcase_name='with_val_no_cv',
          x_val_in=True,
          cv_splits_in=False,
          expected_scoring_y_val='y_val',
      ),
      dict(
          testcase_name='with_val_with_cv',
          x_val_in=True,
          cv_splits_in=True,
          expected_scoring_y_val='y_train',
      ),
      dict(
          testcase_name='no_val_no_cv',
          x_val_in=False,
          cv_splits_in=False,
          expected_scoring_y_val='y_train',
      ),
      dict(
          testcase_name='no_val_with_cv',
          x_val_in=False,
          cv_splits_in=True,
          expected_scoring_y_val='y_train',
      ),
  )
  def test_run_learning_cycle_success(
      self, x_val_in, cv_splits_in, expected_scoring_y_val
  ):
    """Verifies run_learning_cycle successful paths across CV and data modes."""
    student = self.create_student()

    def side_effect(code, name, fn_args, **kwargs):
      del code, name, kwargs
      x_val_fold = fn_args[-1]
      y_pred = pd.DataFrame(
          {'prediction': np.arange(len(x_val_fold))}, index=x_val_fold.index
      )
      return (y_pred, 'Feedback'), 0.1

    self.mock_execute.side_effect = side_effect

    x_val = self.x_val if x_val_in else self.x_train
    y_val = self.y_val if x_val_in else self.y_train
    cv_splits = self.cv_splits if cv_splits_in else None

    result = student_learning.run_learning_cycle(
        student=student,
        x_train=self.x_train,
        y_train=self.y_train,
        x_val=x_val,
        y_val=y_val,
        current_cycle=0,
        classroom=[student],
        student_initial_prompt='Initial',
        student_function_name='solution',
        metric_weights=self.metric_weights,
        early_stopping_patience=3,
        student_refine_instruction_options=['Refine'],
        max_mutation_count=1,
        mutation_strategy_options=[agents.MutationStrategy.SELF_BEST_CODE],
        code_execution_max_retries=1,
        validator=self.mock_validator,
        cv_splits=cv_splits,
    )

    self.assertIsInstance(result, agents.StudentCycleResult)
    self.assertEqual(result.exclude_reason, '')
    self.mock_score.assert_called()
    call_args = self.mock_score.call_args
    expected_y = (
        self.y_val if expected_scoring_y_val == 'y_val' else self.y_train
    )
    if cv_splits_in:
      pd.testing.assert_frame_equal(
          call_args.args[0], self.y_train.loc[[0, 1, 2, 3]]
      )
    else:
      self.assertIs(call_args.args[0], expected_y)

  @parameterized.named_parameters(
      dict(
          testcase_name='init_gen_fail',
          side_effect=agents.GenerationError('Init Fail'),
          expected_reason='Initial solution generation failed.',
          expected_traceback='Mock Traceback',
      ),
      dict(
          testcase_name='retry_gen_fail',
          side_effect=[
              code_execution.ExecutionError('Fail'),
              agents.GenerationError('GenFail'),
          ],
          expected_reason='Solution generation for fix failed in retry.',
          expected_traceback='Mock Traceback',
          preset_solution=True,
          retries=1,
      ),
      dict(
          testcase_name='validator_error',
          side_effect=scoring.ValidatorError('Invalid output'),
          expected_reason='Scoring failed to complete.',
          expected_traceback='Mock Traceback',
          tb_val='Mock Traceback: Invalid output',
      ),
      dict(
          testcase_name='timeout_error',
          side_effect=code_execution.ExecutionTimeoutError('Timed out'),
          expected_reason='Scoring failed to complete.',
          expected_traceback='Mock Traceback',
          tb_val='Mock Traceback: Timed out',
      ),
  )
  def test_run_learning_cycle_errors(
      self,
      side_effect,
      expected_reason,
      expected_traceback,
      preset_solution=False,
      retries=0,
      tb_val=None,
  ):
    """Verifies run_learning_cycle error handling paths."""
    student = self.create_student()
    if preset_solution:
      student.current_solution = 'Initial solution'

    if tb_val:
      mock_tb_call = self.mock_traceback
      mock_tb_call.return_value = tb_val
    else:
      self.mock_traceback.return_value = 'Mock Traceback'

    if isinstance(side_effect, list):
      # Handle mixed side effects: first execute fails, then generate fix fails
      self.mock_execute.side_effect = [side_effect[0]]
      student.generate_solution.side_effect = side_effect[1:]
    elif isinstance(side_effect, agents.GenerationError):
      student.generate_solution.side_effect = side_effect
    else:
      self.mock_execute.side_effect = side_effect

    result = self._run_cycle(
        student=student, code_execution_max_retries=retries
    )
    self.assertEqual(result.exclude_reason, expected_reason)
    self.assertIn(expected_traceback, result.traceback or '')

  def test_run_learning_cycle_retry_and_success(self):
    """Tests that a student can fail once and succeed on retry."""
    self.mock_execute.side_effect = [
        code_execution.ExecutionError('Fail'),
        ((pd.DataFrame({'p': np.arange(len(self.x_val))}), 'Feedback'), 0.1),
    ]
    result = self._run_cycle(code_execution_max_retries=1)
    self.assertEqual(result.retry_count, 1)
    self.assertEqual(result.exclude_reason, '')
    self.assertGreater(result.overall_score, 0)

  def test_run_learning_cycle_aux_val_provided(self):
    self.mock_execute.return_value = ((pd.DataFrame({'p': [0]}), 'Ok'), 0.1)
    self.mock_score.return_value = (1.0, {}, 'Feedback', {})
    aux_val = {'val_aux': pd.DataFrame({'a': [1]})}

    self._run_cycle(auxiliary_val_data=aux_val)
    self.assertEqual(
        self.mock_score.call_args.kwargs['auxiliary_ground_truths'], aux_val
    )

  def test_run_learning_cycle_seed_solution(self):
    """Verifies that seed solution is used if provided in cycle 0."""
    student = self.create_student()
    student.current_solution = 'print("Seed")'
    with mock.patch.object(student, 'generate_solution') as mock_gen:
      result = self._run_cycle(student=student)
      # generate_solution should only be called for refinement (1 time)
      self.assertEqual(mock_gen.call_count, 1)
    self.assertGreater(result.overall_score, 0)

  def test_run_learning_cycle_multiple_retries_success(self):
    self.mock_execute.side_effect = [
        code_execution.ExecutionError('Fail 1'),
        code_execution.ExecutionError('Fail 2'),
        (
            (pd.DataFrame({'prediction': np.arange(len(self.x_val))}), 'Good'),
            0.1,
        ),
    ]
    result = self._run_cycle(code_execution_max_retries=2)
    self.assertEqual(result.retry_count, 2)
    self.assertEqual(result.exclude_reason, '')

  def test_run_learning_cycle_max_retries_exceeded(self):
    self.mock_execute.side_effect = code_execution.ExecutionError('Fail')
    result = self._run_cycle(code_execution_max_retries=1)
    self.assertEqual(result.retry_count, 1)
    self.assertEqual(result.exclude_reason, 'Scoring failed to complete.')

  def test_run_learning_cycle_retry_generation_error(self):
    """Tests GenerationError during retry fix."""
    student = self.create_student()
    student.current_solution = 'Initial solution'
    self.mock_execute.side_effect = code_execution.ExecutionError('Fail')
    with mock.patch.object(
        student,
        'generate_solution',
        side_effect=agents.GenerationError('GenFail'),
    ):
      result = student_learning.run_learning_cycle(
          student=student,
          x_train=self.x_train,
          y_train=self.y_train,
          x_val=self.x_val,
          y_val=self.y_val,
          current_cycle=0,
          classroom=[student],
          student_initial_prompt='Init',
          student_function_name='fn',
          metric_weights=self.metric_weights,
          early_stopping_patience=3,
          student_refine_instruction_options=['Refine'],
          max_mutation_count=1,
          mutation_strategy_options=[],
          code_execution_max_retries=1,
      )
    self.assertEqual(
        result.exclude_reason, 'Solution generation for fix failed in retry.'
    )

  def test_run_learning_cycle_refine_generation_error(self):
    """Tests GenerationError during refinement."""
    student = self.create_student()
    with mock.patch.object(
        student,
        'generate_solution',
        side_effect=agents.GenerationError('GenFail'),
    ):
      result = self._run_cycle(student=student, current_cycle=1)
    self.assertEqual(
        result.exclude_reason, 'Solution generation failed in refine/mutate.'
    )
    self.assertIn('Mock Traceback', result.traceback)

  def test_run_learning_cycle_max_mutations_reached(self):
    """Tests student exclusion when max mutations are reached."""
    student = self.create_student()
    student.best_cycle = 0
    student.best_score = 1.0
    student.mutation_count = 1
    # Use a mock execute that doesn't improve the score
    self.mock_execute.return_value = (
        (pd.DataFrame({'prediction': [0]}), 'Ok'),
        0.1,
    )
    result = self._run_cycle(
        student=student, current_cycle=5, early_stopping_patience=3
    )
    self.assertIn('Max mutations reached.', result.exclude_reason)

  def test_run_learning_cycle_retry_loop_interrupt(self):
    """Tests that an error during retry stops the attempt loop."""
    student = self.create_student()
    student.current_solution = 'Initial solution'
    # First execution fails, then fix generation fails
    self.mock_execute.side_effect = code_execution.ExecutionError('Fail')
    with mock.patch.object(
        student,
        'generate_solution',
        side_effect=agents.GenerationError('StopLoop'),
    ):
      result = self._run_cycle(student=student, code_execution_max_retries=5)
    # retry_count should be 0 because first retry attempt failed with GenError
    self.assertEqual(result.retry_count, 0)
    self.assertEqual(
        result.exclude_reason, 'Solution generation for fix failed in retry.'
    )

  def test_run_learning_cycle_with_aux_data(self):
    """Tests that auxiliary data is correctly passed to code execution."""
    aux_train = {'aux1': pd.DataFrame({'c': [1]})}
    with mock.patch.object(
        code_execution,
        'execute_code',
        return_value=((pd.DataFrame({'p': [0]}), 'Ok'), 0.1),
    ) as mock_exec:
      self._run_cycle(auxiliary_train_data=aux_train)
      fn_args = mock_exec.call_args.kwargs['fn_args']
      self.assertLen(fn_args, 4)
      pd.testing.assert_frame_equal(fn_args[2], aux_train['aux1'])

  @parameterized.named_parameters(
      dict(
          testcase_name='with_cv_splits_success',
          cv_splits_in=True,
          expected_error=None,
      ),
      dict(
          testcase_name='no_cv_splits_no_error_fallback',
          cv_splits_in=False,
          expected_error=None,
      ),
  )
  def test_run_learning_cycle_no_val(self, cv_splits_in, expected_error):
    """Tests run_learning_cycle when x_val/y_val are None with and without cv_splits."""
    student = self.create_student()
    if cv_splits_in:
      self.mock_execute.return_value = (
          (
              pd.DataFrame({'prediction': np.arange(len(self.x_train))}),
              'Feedback',
          ),
          0.1,
      )

    cv_splits = self.cv_splits if cv_splits_in else None

    def call_fn():
      return student_learning.run_learning_cycle(
          student=student,
          x_train=self.x_train,
          y_train=self.y_train,
          x_val=None,
          y_val=None,
          current_cycle=0,
          classroom=[student],
          student_initial_prompt='Initial',
          student_function_name='solution',
          metric_weights=self.metric_weights,
          early_stopping_patience=3,
          student_refine_instruction_options=['Refine'],
          max_mutation_count=1,
          mutation_strategy_options=[agents.MutationStrategy.SELF_BEST_CODE],
          code_execution_max_retries=1,
          validator=self.mock_validator,
          cv_splits=cv_splits,
      )

    if expected_error:
      with self.assertRaisesRegex(ValueError, expected_error):
        call_fn()
    else:
      result = call_fn()
      self.assertIsInstance(result, agents.StudentCycleResult)


class HelperFunctionsTests(StudentLearningTestBase):

  @parameterized.named_parameters(
      dict(
          testcase_name='success',
          attempt=0,
          expected_continue=True,
          expected_tb='',
      ),
      dict(
          testcase_name='max_attempts',
          attempt=1,
          expected_continue=False,
          expected_tb='Error',
      ),
  )
  def test_retry_with_traceback(self, attempt, expected_continue, expected_tb):
    student = self.create_student()
    result = agents.StudentCycleResult(cycle=0)
    student.generate_solution.return_value = 'Fixed'
    should_continue, tb = student_learning._retry_with_traceback(
        student=student,
        attempt=attempt,
        max_retries=1,
        last_error_traceback='Error',
        result=result,
    )
    self.assertEqual(should_continue, expected_continue)
    self.assertEqual(tb, expected_tb)

  @parameterized.named_parameters(
      dict(testcase_name='refine', early_stop_in=False, expect_mut=False),
      dict(testcase_name='mutate', early_stop_in=True, expect_mut=True),
  )
  def test_refine_solution_logic(self, early_stop_in, expect_mut):
    student = self.create_student()
    # Set up student to trigger early stop calculation correctly
    if early_stop_in:
      student.best_cycle = 0
      current_cycle = 5
      patience = 1
    else:
      student.best_cycle = -1
      current_cycle = 0
      patience = 3

    result = agents.StudentCycleResult(cycle=current_cycle, overall_score=0.5)
    student.generate_solution.reset_mock()

    student_learning._refine_solution(
        student=student,
        result=result,
        classroom=[student],
        score_feedback_str='FB',
        student_refine_instruction_options=['Refine'],
        max_mutation_count=1,
        mutation_strategy_options=[agents.MutationStrategy.SELF_BEST_CODE],
        early_stopping_patience=patience,
        current_cycle=current_cycle,
    )

    if expect_mut:
      student.mutate.assert_called_once()
    else:
      student.generate_solution.assert_called_once()

  @parameterized.named_parameters(
      dict(
          testcase_name='raw_text',
          render_feedback_as_md=False,
          expected_header=(
              "**Your Code's Execution Output:**\n```\nmy_feedback\n```"
          ),
      ),
      dict(
          testcase_name='markdown',
          render_feedback_as_md=True,
          expected_header=(
              "**Your Code's Execution Output Feedback:**\n\nmy_feedback\n"
          ),
      ),
  )
  def test_refine_solution_markdown_toggle(
      self, render_feedback_as_md, expected_header
  ):
    student = self.create_student()
    result = agents.StudentCycleResult(
        cycle=0, overall_score=0.5, execution_feedback='my_feedback'
    )
    student.generate_solution.reset_mock()

    student_learning._refine_solution(
        student=student,
        result=result,
        classroom=[student],
        score_feedback_str='FB',
        student_refine_instruction_options=['Refine'],
        max_mutation_count=1,
        mutation_strategy_options=[agents.MutationStrategy.SELF_BEST_CODE],
        early_stopping_patience=3,
        current_cycle=0,
        render_feedback_as_md=render_feedback_as_md,
    )

    student.generate_solution.assert_called_once()
    called_prompt = student.generate_solution.call_args[0][0]
    self.assertIn(expected_header, called_prompt)

  def test_process_cycle_results_saving(self):
    s1 = self.create_student('s1')
    result = agents.StudentCycleResult(cycle=0, overall_score=0.9, code='ok')
    history = collections.defaultdict(list)

    with (
        mock.patch.object(
            student_learning.os.path, 'exists', return_value=False
        ),
        mock.patch.object(data_utils, 'save_dataframe_to_csv') as mock_save,
        mock.patch.object(data_utils, 'create_dir') as mock_create,
    ):
      student_learning.process_cycle_results(
          current_classroom=[s1],
          students_cycle_results=[result],
          results_history=history,
          current_cycle=0,
      )
      mock_save.assert_called_once()
      mock_create.assert_called_once()

  def test_get_best_solution_from_eval_results(self):
    eval_results = [
        {
            'student_name': 's1',
            'best': {
                'overall_score': 0.8,
                'metrics_exec_time': 0.5,
                'best_cycle': 2,
            },
        },
        {
            'student_name': 's2',
            'best': {
                'overall_score': 0.9,
                'metrics_exec_time': 1.5,
                'best_cycle': 1,
            },
        },
    ]
    # Basic
    name, data = student_learning.get_best_solution_from_eval_results(
        eval_results
    )
    self.assertEqual(name, 's2')
    self.assertEqual(data['overall_score'], 0.9)

    # Max exec time
    name, _ = student_learning.get_best_solution_from_eval_results(
        eval_results, max_exec_time=1.0
    )
    self.assertEqual(name, 's1')

    # Min cycle
    name, _ = student_learning.get_best_solution_from_eval_results(
        eval_results, min_cycle=2
    )
    self.assertEqual(name, 's1')

    # No matching
    name, _ = student_learning.get_best_solution_from_eval_results(
        eval_results, max_exec_time=0.1
    )
    self.assertIsNone(name)

  def test_get_classroom_student_configs(self):
    configs = student_learning.get_classroom_student_configs(
        api_key='key', system_instruction='instr', model_names=['m1', 'm2']
    )
    self.assertLen(configs, 2)
    self.assertEqual(configs[0].name, 'student_1_m1')
    self.assertEqual(configs[1].model_name, 'm2')

    # Default models
    configs_default = student_learning.get_classroom_student_configs(
        api_key='key', system_instruction='instr'
    )
    self.assertNotEmpty(configs_default)


class FinalEvaluationTests(StudentLearningTestBase):

  def test_run_final_evaluation_success(self):
    student = self.create_student('s1')
    student.current_solution = MOCK_GOOD_CODE
    student.best_solution = MOCK_GOOD_CODE
    student.best_cycle = 0
    student.cycle_results.append(
        agents.StudentCycleResult(cycle=1, code=MOCK_GOOD_CODE)
    )

    eval_results, preds = self._run_final(
        student=student,
        strategies=['first', 'last', 'best', 'cycle_0', 'cycle_1'],
        validator=self.mock_validator,
    )

    with self.subTest('Eval results contain all strategies'):
      self.assertIn('first', eval_results)
      self.assertIn('last', eval_results)
      self.assertIn('best', eval_results)
      self.assertIn('cycle_0', eval_results)
      self.assertIn('cycle_1', eval_results)
    self.assertLen(preds, 5)

  def test_run_final_evaluation_cv_splits(self):
    student = self.create_student('s1')
    student.current_solution = MOCK_GOOD_CODE

    eval_results, preds = self._run_final(
        student=student, cv_splits=self.cv_splits
    )

    self.assertIn('last', eval_results)
    # 2 folds
    self.assertEqual(self.mock_execute.call_count, 2)
    self.assertIn('s1_last_y_predictions', preds)

  @parameterized.named_parameters(
      dict(
          testcase_name='no_results',
          strategies=['last'],
          setup_student=lambda s: setattr(s, 'cycle_results', []),
          expected_traceback='Mock Traceback',
      ),
      dict(
          testcase_name='unknown_strategy',
          strategies=['nonsense'],
          expected_error='Unknown strategy',
      ),
      dict(
          testcase_name='broad_exception',
          strategies=['last'],
          patch_target='_execute_with_cv_splits',
          patch_side_effect=RuntimeError('Boom'),
          expected_traceback='Mock Traceback: Boom',
          tb_val='Mock Traceback: Boom',
          cv_splits=True,
      ),
      dict(
          testcase_name='best_no_best_cycle',
          strategies=['best'],
          setup_student=lambda s: setattr(s, 'best_cycle', -1),
          expected_traceback='No best solution found',
          tb_val='No best solution found',
      ),
      dict(
          testcase_name='empty_extracted_code',
          strategies=['last'],
          setup_student=lambda s: setattr(s.cycle_results[0], 'code', ''),
          expected_traceback='Extracted code is empty',
          tb_val='Extracted code is empty',
      ),
      dict(
          testcase_name='cycle_invalid_format',
          strategies=['cycle_abc'],
          expected_traceback='Invalid or out-of-range cycle index',
          tb_val='Invalid or out-of-range cycle index',
      ),
      dict(
          testcase_name='cycle_out_of_range_too_high',
          strategies=['cycle_10'],
          expected_traceback=(
              'Cycle index 10 out of range for student s1 with 1 cycles.'
          ),
          tb_val='Cycle index 10 out of range for student s1 with 1 cycles.',
      ),
      dict(
          testcase_name='cycle_index_missing',
          strategies=['cycle_'],
          expected_traceback='Invalid or out-of-range cycle index',
          tb_val='Invalid or out-of-range cycle index',
      ),
  )
  def test_run_final_evaluation_errors(
      self,
      strategies,
      expected_traceback=None,
      expected_error=None,
      setup_student=None,
      patch_target=None,
      patch_side_effect=None,
      tb_val=None,
      cv_splits=None,
  ):
    """Verifies run_final_evaluation error handling paths."""
    student = self.create_student('s1')
    if setup_student:
      setup_student(student)

    if tb_val:
      self.mock_traceback.return_value = tb_val
    else:
      self.mock_traceback.return_value = 'Mock Traceback'

    cv_splits_arg = self.cv_splits if cv_splits else None

    if patch_target:
      with mock.patch.object(
          student_learning, patch_target, side_effect=patch_side_effect
      ):
        eval_results, _ = self._run_final(
            student=student, strategies=strategies, cv_splits=cv_splits_arg
        )
    else:
      eval_results, _ = self._run_final(
          student=student, strategies=strategies, cv_splits=cv_splits_arg
      )

    strategy = strategies[0]
    if expected_traceback:
      self.assertIn(
          expected_traceback, eval_results[strategy].get('traceback', '')
      )
    if expected_error:
      self.assertEqual(eval_results[strategy].get('error'), expected_error)

  def test_run_final_evaluation_with_aux_data(self):
    """Tests auxiliary data passing in final evaluation."""
    student = self.create_student('s1')
    student.cycle_results = [agents.StudentCycleResult(cycle=0, code='ok')]
    aux_learning = {'aux_l': pd.DataFrame({'l': [1]})}

    with mock.patch.object(
        code_execution,
        'execute_code',
        return_value=((pd.DataFrame({'p': [0]}), 'Ok'), 0.1),
    ) as mock_exec:
      student_learning.run_final_evaluation(
          student_name='s1',
          x_learning=self.x_train,
          y_learning=self.y_train,
          x_test_final=self.x_val,
          y_test_final=self.y_val,
          strategies=['last'],
          students={'s1': student},
          student_function_name='fn',
          metric_weights={},
          auxiliary_learning_data=aux_learning,
      )
      fn_args = mock_exec.call_args.kwargs['fn_args']
      # [x_learning, y_learning, aux_l, x_test_final]
      self.assertLen(fn_args, 4)
      pd.testing.assert_frame_equal(fn_args[2], aux_learning['aux_l'])

  @parameterized.named_parameters(
      dict(
          testcase_name='with_cv_splits_success',
          cv_splits_in=True,
          expected_error=None,
      ),
      dict(
          testcase_name='no_cv_splits_no_error_fallback',
          cv_splits_in=False,
          expected_error=None,
      ),
  )
  def test_run_final_evaluation_no_test(self, cv_splits_in, expected_error):
    """Tests run_final_evaluation when test set is None with and without cv_splits."""
    student = self.create_student('s1')
    student.current_solution = MOCK_GOOD_CODE

    cv_splits = self.cv_splits if cv_splits_in else None

    if cv_splits_in:
      # Mock execute to return predictions matching x_train size
      self.mock_execute.return_value = (
          (
              pd.DataFrame({'prediction': np.arange(len(self.x_train))}),
              'Feedback',
          ),
          0.1,
      )

    def call_fn():
      return student_learning.run_final_evaluation(
          student_name=student.name,
          x_learning=self.x_train,
          y_learning=self.y_train,
          x_test_final=None,
          y_test_final=None,
          strategies=['last'],
          students={student.name: student},
          student_function_name='solution',
          metric_weights=self.metric_weights,
          cv_splits=cv_splits,
      )

    if expected_error:
      with self.assertRaisesRegex(ValueError, expected_error):
        call_fn()
    else:
      eval_results, preds = call_fn()
      self.assertIn('last', eval_results)
      self.assertIn('s1_last_y_predictions', preds)


class ClassroomParallelTests(StudentLearningTestBase):
  """Tests for parallel classroom execution wrappers."""

  @mock.patch.object(student_learning, 'run_learning_cycle', autospec=True)
  def test_run_parallel_classroom_learning_cycle_success(self, mock_run_cycle):
    """Verifies successful parallel learning cycle execution."""
    s1 = self.create_student('s1')
    s2 = self.create_student('s2')
    students = [s1, s2]

    # Map student name to result for deterministic parallel execution
    mock_run_cycle.side_effect = (
        lambda student, *args, **kwargs: 'res1'
        if student.name == 's1'
        else 'res2'
    )

    results = student_learning.run_parallel_classroom_learning_cycle(
        current_classroom=students,
        x_train=self.x_train,
        y_train=self.y_train,
        x_val=self.x_val,
        y_val=self.y_val,
        current_cycle=0,
        student_initial_prompt='Prompt',
        student_function_name='fn',
        metric_weights=self.metric_weights,
        early_stopping_patience=3,
        student_refine_instruction_options=['Refine'],
        max_mutation_count=1,
        mutation_strategy_options=[],
        code_execution_max_retries=1,
        num_workers=2,
    )

    self.assertEqual(results, ['res1', 'res2'])

  def test_run_parallel_classroom_learning_cycle_empty(self):
    """Verifies that an empty classroom returns an empty list immediately."""
    results = student_learning.run_parallel_classroom_learning_cycle(
        current_classroom=[],
        x_train=self.x_train,
        y_train=self.y_train,
        x_val=self.x_val,
        y_val=self.y_val,
        current_cycle=0,
        student_initial_prompt='Prompt',
        student_function_name='fn',
        metric_weights=self.metric_weights,
        early_stopping_patience=3,
        student_refine_instruction_options=[],
        max_mutation_count=1,
        mutation_strategy_options=[],
        code_execution_max_retries=1,
    )
    self.assertEqual(results, [])

  @parameterized.named_parameters(
      dict(
          testcase_name='y_train_series',
          kwargs_to_override={'y_train': pd.Series([1, 2])},
          expected_regex="Input 'y_train' must be a pandas DataFrame",
      ),
      dict(
          testcase_name='aux_train_series',
          kwargs_to_override={'auxiliary_train_data': {'aux': pd.Series([1])}},
          expected_regex="Input 'aux' must be a pandas DataFrame",
      ),
  )
  def test_run_parallel_classroom_learning_cycle_validation(
      self, kwargs_to_override, expected_regex
  ):
    """Verifies input validation for parallel learning cycle."""
    s1 = self.create_student('s1')
    kwargs = {
        'current_classroom': [s1],
        'x_train': self.x_train,
        'y_train': self.y_train,
        'x_val': self.x_val,
        'y_val': self.y_val,
        'current_cycle': 0,
        'student_initial_prompt': 'Prompt',
        'student_function_name': 'fn',
        'metric_weights': self.metric_weights,
        'early_stopping_patience': 3,
        'student_refine_instruction_options': [],
        'max_mutation_count': 1,
        'mutation_strategy_options': [],
        'code_execution_max_retries': 1,
    }
    kwargs.update(kwargs_to_override)
    with self.assertRaisesRegex(TypeError, expected_regex):
      student_learning.run_parallel_classroom_learning_cycle(**kwargs)

  @mock.patch.object(student_learning, 'run_final_evaluation', autospec=True)
  def test_run_parallel_classroom_final_evaluation_success(
      self, mock_run_final
  ):
    """Verifies successful parallel final evaluation."""
    s1 = self.create_student('s1')
    students = {'s1': s1}

    mock_run_final.return_value = (
        {'s1': {'last': {'score': 0.9}}},
        {'s1_last_preds': pd.DataFrame({'p': np.arange(len(self.x_val))})},
    )

    results, preds = student_learning.run_parallel_classroom_final_evaluation(
        students=students,
        x_learning=self.x_train,
        y_learning=self.y_train,
        x_test_final=self.x_val,
        y_test_final=self.y_val,
        student_function_name='fn',
        metric_weights=self.metric_weights,
        strategies=['last'],
    )

    self.assertLen(results, 1)
    self.assertIn('s1_last_preds', preds)

  def test_run_parallel_classroom_final_evaluation_persistence(self):
    """Verifies that final evaluation results and predictions are saved to file."""
    s1 = self.create_student('s1')
    students = {'s1': s1}

    eval_res = {'student_name': 's1', 'last': {'score': 0.9}}
    eval_preds = {
        's1_last_y_predictions': pd.DataFrame(
            {'prediction': np.arange(len(self.x_val))}
        )
    }

    with mock.patch.object(
        student_learning,
        'run_final_evaluation',
        return_value=(eval_res, eval_preds),
    ):
      with mock.patch.object(data_utils, 'save_dict_to_json') as mock_json:
        with mock.patch.object(data_utils, 'save_dataframe_to_csv') as mock_csv:
          with mock.patch.object(data_utils, 'create_dir') as mock_mkdir:
            student_learning.run_parallel_classroom_final_evaluation(
                students=students,
                x_learning=self.x_train,
                y_learning=self.y_train,
                x_test_final=self.x_val,
                y_test_final=self.y_val,
                student_function_name='fn',
                metric_weights=self.metric_weights,
                final_results_path='/path/results.json',
                predictions_dir='/path/preds',
            )
            mock_json.assert_called_once()
            mock_csv.assert_called_once()
            mock_mkdir.assert_called_once_with('/path/preds')

  @mock.patch.object(student_learning, 'run_final_evaluation', autospec=True)
  def test_run_parallel_classroom_final_evaluation_empty_preds(
      self, mock_final_eval
  ):
    """Verifies that empty or None predictions are skipped during saving."""
    s1 = self.create_student('s1')

    # Map student name to result for deterministic parallel execution
    def side_effect(student_name, *args, **kwargs):
      del args, kwargs
      if student_name == 's1':
        return ({'s1': {}}, {'empty': pd.DataFrame()})
      elif student_name == 's2':
        return ({'s2': {}}, {'none': None})
      return ({}, {})

    mock_final_eval.side_effect = side_effect

    with mock.patch.object(data_utils, 'save_dataframe_to_csv') as mock_csv:
      with mock.patch.object(data_utils, 'create_dir'):
        student_learning.run_parallel_classroom_final_evaluation(
            students={'s1': s1, 's2': s1},
            x_learning=self.x_train,
            y_learning=self.y_train,
            x_test_final=self.x_val,
            y_test_final=self.y_val,
            student_function_name='fn',
            metric_weights=self.metric_weights,
            predictions_dir='/path/preds',
        )
        mock_csv.assert_not_called()

  @parameterized.named_parameters(
      dict(
          testcase_name='y_learning_series',
          kwargs_to_override={'y_learning': pd.Series([1])},
          expected_regex="Input 'y_learning' must be a pandas DataFrame",
      ),
  )
  def test_run_parallel_classroom_final_evaluation_validation(
      self, kwargs_to_override, expected_regex
  ):
    """Verifies input validation for parallel final evaluation."""
    s1 = self.create_student('s1')
    kwargs = {
        'students': {'s1': s1},
        'x_learning': self.x_train,
        'y_learning': self.y_train,
        'x_test_final': self.x_val,
        'y_test_final': self.y_val,
        'student_function_name': 'fn',
        'metric_weights': self.metric_weights,
    }
    kwargs.update(kwargs_to_override)
    with self.assertRaisesRegex(TypeError, expected_regex):
      student_learning.run_parallel_classroom_final_evaluation(**kwargs)

  @parameterized.named_parameters(
      dict(
          testcase_name='with_cv_splits_success',
          cv_splits_in=True,
          expected_error=None,
      ),
      dict(
          testcase_name='no_cv_splits_no_error_fallback',
          cv_splits_in=False,
          expected_error=None,
      ),
  )
  def test_run_parallel_classroom_no_val(self, cv_splits_in, expected_error):
    """Tests parallel learning cycle when x_val/y_val are None with and without cv_splits."""
    s1 = self.create_student('s1')
    students = [s1]

    cv_splits = self.cv_splits if cv_splits_in else None

    def call_fn():
      return student_learning.run_parallel_classroom_learning_cycle(
          current_classroom=students,
          x_train=self.x_train,
          y_train=self.y_train,
          x_val=None,
          y_val=None,
          current_cycle=0,
          student_initial_prompt='Prompt',
          student_function_name='fn',
          metric_weights=self.metric_weights,
          early_stopping_patience=3,
          student_refine_instruction_options=['Refine'],
          max_mutation_count=1,
          mutation_strategy_options=[],
          code_execution_max_retries=1,
          cv_splits=cv_splits,
      )

    if expected_error:
      with self.assertRaisesRegex(ValueError, expected_error):
        call_fn()
    else:
      with mock.patch.object(
          student_learning, 'run_learning_cycle', return_value='res1'
      ):
        results = call_fn()
        self.assertEqual(results, ['res1'])

  @parameterized.named_parameters(
      dict(
          testcase_name='with_cv_splits_success',
          cv_splits_in=True,
          expected_error=None,
      ),
      dict(
          testcase_name='no_cv_splits_no_error_fallback',
          cv_splits_in=False,
          expected_error=None,
      ),
  )
  def test_run_parallel_classroom_final_evaluation_no_test(
      self, cv_splits_in, expected_error
  ):
    """Tests parallel final evaluation when test set is None with and without cv_splits."""
    s1 = self.create_student('s1')
    students = {'s1': s1}

    cv_splits = self.cv_splits if cv_splits_in else None

    def call_fn():
      return student_learning.run_parallel_classroom_final_evaluation(
          students=students,
          x_learning=self.x_train,
          y_learning=self.y_train,
          x_test_final=None,
          y_test_final=None,
          student_function_name='fn',
          metric_weights=self.metric_weights,
          strategies=['last'],
          cv_splits=cv_splits,
      )

    if expected_error:
      with self.assertRaisesRegex(ValueError, expected_error):
        call_fn()
    else:
      mock_eval_res = {'student_name': 's1', 'last': {'score': 0.9}}
      mock_preds = {
          's1_last_preds': pd.DataFrame({'p': np.arange(len(self.x_train))})
      }
      with mock.patch.object(
          student_learning,
          'run_final_evaluation',
          return_value=(mock_eval_res, mock_preds),
      ):
        results, preds = call_fn()
        self.assertLen(results, 1)
        self.assertIn('s1_last_preds', preds)


if __name__ == '__main__':
  absltest.main()
