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

from collections.abc import Mapping
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from codegen_classroom import agents
from codegen_classroom import metrics as metric_lib
from codegen_classroom import scoring
import numpy as np
import pandas as pd


class ScoringValidatorsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='valid_tuple',
          result=(pd.DataFrame({'p': [1.0, 2.0]}), 'Done'),
          should_pass=True,
      ),
      dict(
          testcase_name='invalid_just_df',
          result=pd.DataFrame({'p': [1.0, 2.0]}),
          should_pass=False,
          error_regex=r'must return a tuple of \(pd.DataFrame, str\)',
      ),
      dict(
          testcase_name='invalid_tuple_length',
          result=(pd.DataFrame({'p': [1.0]}),),
          should_pass=False,
          error_regex=r'must have exactly 2 elements',
      ),
      dict(
          testcase_name='invalid_first_element',
          result=('not a df', 'Done'),
          should_pass=False,
          error_regex=(
              r'first element of the return tuple must be a pandas DataFrame'
          ),
      ),
      dict(
          testcase_name='invalid_second_element',
          result=(pd.DataFrame({'p': [1.0]}), 123),
          should_pass=False,
          error_regex=r'second element of the return tuple must be a string',
      ),
  )
  def test_execution_validator(
      self,
      result,
      should_pass,
      error_regex='',
  ):
    if should_pass:
      self.assertIsNone(scoring.execution_validator({}, {}, result))

    else:
      with self.assertRaisesRegex(scoring.ValidatorError, error_regex):
        scoring.execution_validator({}, {}, result)

  @parameterized.named_parameters(
      dict(
          testcase_name='valid',
          result=(pd.DataFrame({'p': [1.0, 2.0]}), 'Done'),
          fn_args=[None, None, None, None, pd.DataFrame({'a': range(2)})],
          should_pass=True,
      ),
      dict(
          testcase_name='fail_structure',
          result='not a tuple',
          fn_args=[None, None, None, None, pd.DataFrame({'a': range(2)})],
          should_pass=False,
          error_regex='must return a tuple',
      ),
      dict(
          testcase_name='incorrect_rows',
          result=(pd.DataFrame({'p': [1.0]}), ''),
          fn_args=[None, None, None, None, pd.DataFrame({'a': range(2)})],
          should_pass=False,
          error_regex='incorrect number of rows',
      ),
      dict(
          testcase_name='mismatched_index',
          result=(
              pd.DataFrame({'p': [1.0, 2.0]}, index=[10, 11]),
              '',
          ),
          fn_args=[None, None, None, None, pd.DataFrame({'a': range(2)})],
          should_pass=False,
          error_regex='index does not match X_val index',
      ),
      dict(
          testcase_name='empty_df',
          result=(pd.DataFrame({'p': []}, columns=['p']), ''),
          fn_args=[None, None, None, None, pd.DataFrame({'a': range(2)})],
          should_pass=False,
          error_regex='Predictions DataFrame is empty',
      ),
      dict(
          testcase_name='too_few_fn_args',
          result=(pd.DataFrame({'p': [1.0]}), ''),
          fn_args=[None, None],
          should_pass=False,
          error_regex='Expected at least 3 fn_args',
      ),
      dict(
          testcase_name='x_val_not_df',
          result=(pd.DataFrame({'p': [1.0]}), ''),
          fn_args=[None, None, None, None, 'not a df'],
          should_pass=False,
          error_regex='X_val is not a DataFrame',
      ),
      dict(
          testcase_name='wrong_num_columns',
          result=(pd.DataFrame({'p1': [1.0], 'p2': [2.0]}), ''),
          fn_args=[None, None, None, None, pd.DataFrame({'a': range(1)})],
          should_pass=False,
          error_regex='must have 1 column',
      ),
      dict(
          testcase_name='contains_nan',
          result=(pd.DataFrame({'p': [1.0, np.nan]}), ''),
          fn_args=[None, None, None, None, pd.DataFrame({'a': range(2)})],
          should_pass=False,
          error_regex='contains NaN values',
      ),
      dict(
          testcase_name='non_numeric',
          result=(pd.DataFrame({'p': [1.0, 'a']}), ''),
          fn_args=[None, None, None, None, pd.DataFrame({'a': range(2)})],
          should_pass=False,
          error_regex='All columns in Predictions DataFrame must be numeric',
      ),
      dict(
          testcase_name='valid_int_dtype',
          result=(pd.DataFrame({'p': [1, 2]}), 'Done'),
          fn_args=[None, None, None, None, pd.DataFrame({'a': range(2)})],
          should_pass=True,
          expected_dtype='int',
      ),
      dict(
          testcase_name='invalid_int_dtype_floats',
          result=(pd.DataFrame({'p': [1.0, 2.0]}), 'Done'),
          fn_args=[None, None, None, None, pd.DataFrame({'a': range(2)})],
          should_pass=False,
          error_regex='Predictions DataFrame must contain only integers',
          expected_dtype='int',
      ),
  )
  def test_prediction_validator(
      self,
      result,
      fn_args,
      should_pass,
      error_regex='',
      expected_dtype=None,
  ):
    if should_pass:
      scoring.prediction_validator(
          fn_args, {}, result, expected_dtype=expected_dtype
      )
    else:
      with self.assertRaisesRegex(scoring.ValidatorError, error_regex):
        scoring.prediction_validator(
            fn_args, {}, result, expected_dtype=expected_dtype
        )

  @parameterized.named_parameters(
      dict(
          testcase_name='valid',
          result=(pd.DataFrame({'p': [1, 2]}), 'Done'),
          fn_args=[None, None, None, None, pd.DataFrame({'a': range(2)})],
          should_pass=True,
      ),
      dict(
          testcase_name='invalid_floats',
          result=(pd.DataFrame({'p': [1.0, 2.0]}), 'Done'),
          fn_args=[None, None, None, None, pd.DataFrame({'a': range(2)})],
          should_pass=False,
          error_regex='Predictions DataFrame must contain only integers',
      ),
  )
  def test_classification_prediction_validator(
      self,
      result,
      fn_args,
      should_pass,
      error_regex='',
  ):
    if should_pass:
      scoring.classification_prediction_validator(fn_args, {}, result)
    else:
      with self.assertRaisesRegex(scoring.ValidatorError, error_regex):
        scoring.classification_prediction_validator(fn_args, {}, result)


def _create_default_aux_gt(
    y_true_df: pd.DataFrame,
) -> Mapping[str, pd.DataFrame]:
  """Creates default auxiliary ground truths based on y_true_df."""
  return {
      'positive_label': pd.DataFrame(
          {'label': [0] * len(y_true_df)}, index=y_true_df.index
      ),
      'negative_label': pd.DataFrame(
          {'label': [1] * len(y_true_df)}, index=y_true_df.index
      ),
  }


class ScoringPredictionsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.y_true_df = pd.DataFrame({'target': [1, 2, 3, 4, 5]})
    self.aux_gt = {
        'positive_label': pd.DataFrame({'label': [0, 0, 0, 0, 0]}),
        'negative_label': pd.DataFrame({'label': [1, 1, 1, 1, 1]}),
    }
    self.y_pred_good_df = pd.DataFrame({'target': [1.1, 1.9, 3.1, 4.1, 4.9]})

  @parameterized.named_parameters(
      dict(
          testcase_name='basic_scoring',
          metric_weights={'mae': 1.0, 'r2': 1.0},
          pre_computed={'exec_time': 0.1},
          y_pred_df_in=pd.DataFrame({'target': [1.1, 1.9, 3.1, 4.1, 4.9]}),
          aux_gt_in=_create_default_aux_gt(
              pd.DataFrame({'target': [1, 2, 3, 4, 5]})
          ),
          expected_score=0.952045,  # ( (1/1.1) + 0.995 ) / 2
          expected_metrics={
              'mae': 0.1,
              'r2': 0.995,
              'exec_time': 0.1,
          },
          expected_components={
              'mae': 1.0 * metric_lib.inverse_normalize(0.1),
              'r2': 1.0 * metric_lib.clip_negative(0.995),
              # exec_time is not in metric_weights, so not in score_components
          },
      ),
      dict(
          testcase_name='zero_weights',
          metric_weights={},
          pre_computed={},
          y_pred_df_in=pd.DataFrame({'target': [1.1, 1.9, 3.1, 4.1, 4.9]}),
          aux_gt_in=_create_default_aux_gt(
              pd.DataFrame({'target': [1, 2, 3, 4, 5]})
          ),
          expected_score=0.0,
          expected_metrics={},
          expected_components={},
      ),
      dict(
          testcase_name='only_precomputed_no_weight',
          metric_weights={},
          pre_computed={'exec_time': 4.0},
          y_pred_df_in=pd.DataFrame({'target': [1.1, 1.9, 3.1, 4.1, 4.9]}),
          aux_gt_in=_create_default_aux_gt(
              pd.DataFrame({'target': [1, 2, 3, 4, 5]})
          ),
          expected_score=0.0,
          expected_metrics={'exec_time': 4.0},
          expected_components={},
      ),
      dict(
          testcase_name='only_precomputed_with_weight',
          metric_weights={'exec_time': 0.5},
          pre_computed={'exec_time': 4.0},
          y_pred_df_in=pd.DataFrame({'target': [1.1, 1.9, 3.1, 4.1, 4.9]}),
          aux_gt_in=_create_default_aux_gt(
              pd.DataFrame({'target': [1, 2, 3, 4, 5]})
          ),
          expected_score=0.2,  # 1 / (1 + 4.0) = 0.2
          expected_metrics={'exec_time': 4.0},
          expected_components={
              'exec_time': 0.5 * metric_lib.inverse_normalize(4.0)
          },
      ),
      dict(
          testcase_name='mape_with_zeros',
          metric_weights={'mape': 1.0},
          pre_computed={},
          y_true_df_in=pd.DataFrame({'target': [0, 20, 30]}),
          y_pred_df_in=pd.DataFrame({'target': [0, 22, 33]}),
          aux_gt_in=_create_default_aux_gt(
              pd.DataFrame({'target': [0, 20, 30]})
          ),
          expected_score=1 / 11,
          expected_metrics={'mape': 10.0},
          expected_components={
              'mape': 1.0 * metric_lib.inverse_normalize(10.0)
          },
      ),
      dict(
          testcase_name='r2_const_pred',
          metric_weights={'r2': 1.0},
          pre_computed={},
          y_true_df_in=pd.DataFrame({'target': [1, 2, 3]}),
          y_pred_df_in=pd.DataFrame({'target': [2, 2, 2]}),
          aux_gt_in=_create_default_aux_gt(pd.DataFrame({'target': [1, 2, 3]})),
          expected_score=0.0,
          expected_metrics={'r2': 0.0},
          expected_components={'r2': 1.0 * metric_lib.clip_negative(0.0)},
      ),
      dict(
          testcase_name='specificity_binary_with_threshold',
          metric_weights={'specificity_binary': 1.0},
          pre_computed={},
          y_true_df_in=pd.DataFrame({'target': [1, 3, 1, 3]}),
          y_pred_df_in=pd.DataFrame({'target': [0.5, 2.8, 1.2, 3.1]}),
          aux_gt_in={
              'binary_ground_truth': pd.DataFrame({'l': [0, 1, 0, 1]}),
              'binary_threshold': pd.DataFrame({'t': [2.5, 2.5, 2.5, 2.5]}),
          },
          expected_score=1.0,
          expected_metrics={'specificity_binary': 1.0},
          expected_components={'specificity_binary': 1.0},
      ),
      dict(
          testcase_name='auroc_with_binary_gt',
          metric_weights={'auroc': 1.0},
          pre_computed={},
          y_true_df_in=pd.DataFrame({'target': [1, 3, 1, 3]}),
          y_pred_df_in=pd.DataFrame({'target': [0.1, 0.4, 0.35, 0.8]}),
          aux_gt_in={
              'binary_ground_truth': pd.DataFrame({'l': [0, 0, 1, 1]}),
          },
          expected_score=0.75,
          expected_metrics={'auroc': 0.75},
          expected_components={'auroc': 0.75},
      ),
      dict(
          testcase_name='binary_metrics_with_first_class_threshold',
          metric_weights={'specificity_binary': 1.0, 'auroc': 1.0},
          pre_computed={},
          y_true_df_in=pd.DataFrame({'target': [1.0, 3.0, 1.0, 3.0]}),
          y_pred_df_in=pd.DataFrame({'target': [0.1, 0.35, 0.4, 0.8]}),
          aux_gt_in={},  # binary_ground_truth is derived from binary_threshold
          binary_threshold_in=2.5,
          expected_score=0.875,  # (1.0 spec + 0.75 auroc) / 2
          expected_metrics={'specificity_binary': 1.0, 'auroc': 0.75},
          expected_components={'specificity_binary': 1.0, 'auroc': 0.75},
      ),
      dict(
          testcase_name='accuracy_with_threshold',
          metric_weights={'accuracy': 1.0},
          pre_computed={},
          y_true_df_in=pd.DataFrame({'target': [1, 3, 1, 3]}),
          y_pred_df_in=pd.DataFrame({'target': [0.5, 2.8, 1.2, 3.1]}),
          aux_gt_in={},
          binary_threshold_in=2.5,
          expected_score=1.0,
          expected_metrics={'accuracy': 1.0},
          expected_components={'accuracy': 1.0},
      ),
  )
  def test_score_predictions_cases(
      self,
      metric_weights,
      pre_computed,
      expected_score,
      expected_metrics,
      expected_components,
      aux_gt_in,
      y_true_df_in=None,
      y_pred_df_in=None,
      binary_threshold_in=None,
  ):
    y_true = y_true_df_in if y_true_df_in is not None else self.y_true_df
    y_pred = y_pred_df_in if y_pred_df_in is not None else self.y_pred_good_df

    score, metrics, feedback, score_components = scoring.score_predictions(
        y_true,
        y_pred,
        pre_computed,
        metric_weights,
        auxiliary_ground_truths=aux_gt_in,
        binary_threshold=binary_threshold_in,
    )
    self.assertAlmostEqual(score, expected_score, places=3)
    for key, val in expected_metrics.items():
      self.assertIn(key, metrics)
      self.assertAlmostEqual(metrics[key], val, places=3)
    self.assertEqual(len(score_components), len(expected_components))
    for key, val in expected_components.items():
      self.assertIn(key, score_components)
      self.assertAlmostEqual(score_components[key], val, places=3)

    if metric_weights:
      self.assertIn('Overall Score:', feedback)
      for name in metric_weights:
        if name in metric_lib.ALL_METRICS_CONFIG and metric_weights[name] > 0:
          metric_config = metric_lib.ALL_METRICS_CONFIG[name]
          self.assertIn(metric_config.description, feedback)

  @parameterized.named_parameters(
      dict(
          testcase_name='mismatched_index',
          y_true=None,
          y_pred_func=lambda self: pd.DataFrame(
              {'target': self.y_pred_good_df['target'].values},
              index=range(10, 15),
          ),
          metric_weights={'mae': 1.0},
          aux_gt_func=None,
          expected_regex='Indices',
      ),
      dict(
          testcase_name='mismatched_shape_rows',
          y_true=None,
          y_pred_func=lambda self: pd.DataFrame({'target': [1, 2, 3]}),
          metric_weights={'mae': 1.0},
          aux_gt_func=None,
          expected_regex='Indices',
      ),
      dict(
          testcase_name='mismatched_shape_cols',
          y_true=None,
          y_pred_func=lambda self: pd.DataFrame({
              'target1': self.y_true_df['target'],
              'target2': self.y_true_df['target'],
          }),
          metric_weights={'mae': 1.0},
          aux_gt_func=None,
          expected_regex='Shapes',
      ),
      dict(
          testcase_name='nan_in_y_true',
          y_true=pd.DataFrame({'target': [1, 2, np.nan, 4, 5]}),
          y_pred_func=None,
          metric_weights={'mae': 1.0},
          aux_gt_func=None,
          expected_regex='y_true contains NaN',
      ),
      dict(
          testcase_name='y_true_not_dataframe',
          y_true=np.array([[1], [2], [3], [4], [5]]),
          y_pred_func=None,
          metric_weights={'mae': 1.0},
          aux_gt_func=None,
          expected_regex='y_true must be a pandas DataFrame',
      ),
      dict(
          testcase_name='typo_in_metric_lists_registry',
          y_true=None,
          y_pred_func=None,
          metric_weights={'msae': 1.0},
          aux_gt_func=None,
          expected_regex=(
              r'Unknown metric name\(s\) in metric_weights:'
              r" \['msae'\].*Registered metrics:"
          ),
      ),
      dict(
          testcase_name='aux_not_dataframe',
          y_true=None,
          y_pred_func=None,
          metric_weights={'mae': 1.0},
          aux_gt_func=lambda self: {'bad': [1, 2, 3]},
          expected_regex='must be a DataFrame',
      ),
      dict(
          testcase_name='aux_index_mismatch',
          y_true=None,
          y_pred_func=None,
          metric_weights={'mae': 1.0},
          aux_gt_func=lambda self: {
              'positive_label': pd.DataFrame(
                  {'positive_label': [1] * 5}, index=range(10, 15)
              )
          },
          expected_regex='Indices of auxiliary',
      ),
      dict(
          testcase_name='aux_contains_nan',
          y_true=None,
          y_pred_func=None,
          metric_weights={'mae': 1.0},
          aux_gt_func=lambda self: {
              'positive_label': pd.DataFrame(
                  {'positive_label': [np.nan, 1, 1, 1, 1]}
              )
          },
          expected_regex='contains NaN values',
      ),
  )
  def test_score_predictions_invalid_inputs(
      self, y_true, y_pred_func, metric_weights, aux_gt_func, expected_regex
  ):
    y_true_in = self.y_true_df if y_true is None else y_true
    y_pred_in = (
        self.y_pred_good_df if y_pred_func is None else y_pred_func(self)
    )
    aux_gt_in = self.aux_gt if aux_gt_func is None else aux_gt_func(self)
    with self.assertRaisesRegex(scoring.ScoreError, expected_regex):
      scoring.score_predictions(
          y_true_in,
          y_pred_in,
          {},
          metric_weights,
          auxiliary_ground_truths=aux_gt_in,
      )

  @mock.patch.object(metric_lib.Metric, 'compute', autospec=True)
  def test_score_predictions_metric_compute_error(self, mock_compute):
    mock_compute.side_effect = ValueError('Mock compute error')
    with self.assertRaisesRegex(
        scoring.ScoreError, "Error calculating metric 'mae'"
    ):
      scoring.score_predictions(
          self.y_true_df,
          self.y_pred_good_df,
          {},
          {'mae': 1.0},
          auxiliary_ground_truths=self.aux_gt,
      )

  @mock.patch.object(metric_lib.Metric, 'compute', return_value=np.nan)
  def test_score_predictions_metric_non_finite(self, mock_compute):  # pylint: disable=unused-argument
    # This will log a warning and use 0.0 for the metric value
    _, metrics, _, _ = scoring.score_predictions(
        self.y_true_df,
        self.y_pred_good_df,
        {},
        {'mae': 1.0},
        auxiliary_ground_truths=self.aux_gt,
    )
    self.assertTrue(np.isnan(metrics['mae']))

  @mock.patch.object(metric_lib.Metric, 'normalize_score', autospec=True)
  def test_score_predictions_metric_normalize_error(self, mock_normalize):
    mock_normalize.side_effect = ValueError('Mock normalize error')
    with self.assertRaisesRegex(
        scoring.ScoreError, "Error during normalization of metric 'mae'"
    ):
      scoring.score_predictions(
          self.y_true_df,
          self.y_pred_good_df,
          {},
          {'mae': 1.0},
          auxiliary_ground_truths=self.aux_gt,
      )

  def test_score_predictions_default_metric_case(self):
    # Test the 'else' block in the metric calculation loop
    custom_metric = mock.Mock(spec=metric_lib.Metric)
    custom_metric.func = True
    custom_metric.compute.return_value = 0.5
    custom_metric.normalize_score.return_value = 0.5
    custom_metric.goal_string = 'maximize'
    custom_metric.description = 'Custom Metric'

    with mock.patch.dict(
        metric_lib.ALL_METRICS_CONFIG, {'custom_metric': custom_metric}
    ):
      _, metrics, _, _ = scoring.score_predictions(
          self.y_true_df,
          self.y_pred_good_df,
          {},
          {'custom_metric': 1.0},
      )
      self.assertIn('custom_metric', metrics)
      self.assertEqual(metrics['custom_metric'], 0.5)
      # Ensure compute was called with (y_true, y_pred)
      custom_metric.compute.assert_called_once()
      args, _ = custom_metric.compute.call_args
      # Flattened numpy arrays are passed
      np.testing.assert_array_equal(args[0], self.y_true_df.values.flatten())
      np.testing.assert_array_equal(
          args[1], self.y_pred_good_df.values.flatten()
      )

  def test_unknown_metric_zero_weight_and_empty_weights(self):
    _, metrics, _, _ = scoring.score_predictions(
        self.y_true_df, self.y_pred_good_df, {}, {'mae': 1.0, 'unknown': 0.0}
    )
    self.assertAlmostEqual(metrics['mae'], 0.1)
    self.assertNotIn('unknown', metrics)

    score, empty_m, _, empty_c = scoring.score_predictions(
        self.y_true_df, self.y_pred_good_df, {}, {}
    )
    self.assertEqual(score, 0.0)
    self.assertEqual(empty_m, {})
    self.assertEqual(empty_c, {})

  def test_late_registered_metric_accepted_at_scoring(self):
    custom = metric_lib.Metric(
        name='custom_late',
        func=lambda yt, yp: 0.5,
        description='Custom late metric',
        higher_is_better=True,
        normalize=metric_lib.clip_negative,
    )
    weights = {'custom_late': 1.0}
    with self.assertRaises(scoring.ScoreError):
      scoring.score_predictions(
          self.y_true_df, self.y_pred_good_df, {}, weights
      )
    with mock.patch.dict(
        metric_lib.ALL_METRICS_CONFIG, {'custom_late': custom}
    ):
      score, metrics, _, _ = scoring.score_predictions(
          self.y_true_df, self.y_pred_good_df, {}, weights
      )
      self.assertEqual(metrics['custom_late'], 0.5)
      self.assertEqual(score, 0.5)

  # --- Non-finite scoring tests ---
  @parameterized.named_parameters(
      dict(
          testcase_name='computed_nan',
          mode='computed',
          metric='mae',
          value=float('nan'),
      ),
      dict(
          testcase_name='precomputed_inf',
          mode='precomputed',
          metric='r2',
          value=float('inf'),
      ),
  )
  def test_nonfinite_metric_zeroes_component_safely(self, mode, metric, value):
    if mode == 'computed':
      with mock.patch.object(metric_lib.Metric, 'compute', return_value=value):
        score, metrics, _, components = scoring.score_predictions(
            self.y_true_df, self.y_pred_good_df, {}, {metric: 1.0}
        )
    else:
      score, metrics, _, components = scoring.score_predictions(
          self.y_true_df, self.y_pred_good_df, {metric: value}, {metric: 1.0}
      )
    if np.isnan(value):
      self.assertTrue(np.isnan(metrics[metric]))
    else:
      self.assertEqual(metrics[metric], value)
    self.assertEqual(components[metric], 0.0)
    self.assertEqual(score, 0.0)

  def test_nonfinite_one_metric_preserves_finite_metric(self):
    nan_mae = metric_lib.Metric(
        'mae',
        'MAE',
        False,
        metric_lib.inverse_normalize,
        lambda yt, yp: float('nan'),
    )
    with mock.patch.dict(metric_lib.ALL_METRICS_CONFIG, {'mae': nan_mae}):
      score, metrics, _, components = scoring.score_predictions(
          self.y_true_df, self.y_pred_good_df, {}, {'mae': 1.0, 'r2': 1.0}
      )
    self.assertTrue(np.isnan(metrics['mae']))
    self.assertEqual(components['mae'], 0.0)
    self.assertAlmostEqual(metrics['r2'], 0.995)
    self.assertAlmostEqual(components['r2'], 0.995)
    self.assertAlmostEqual(score, 0.995 / 2.0)

  # --- MASE scoring tests ---
  def test_mase_scoring(self):
    scale_1d = pd.DataFrame({'mase_scale': [0.5] * len(self.y_true_df)})
    score, metrics, _, _ = scoring.score_predictions(
        self.y_true_df,
        pd.DataFrame({'target': self.y_true_df['target'] + 0.5}),
        {},
        {'mase': 1.0},
        auxiliary_ground_truths={'mase_scale': scale_1d},
    )
    self.assertAlmostEqual(metrics['mase'], 1.0)
    self.assertAlmostEqual(score, 0.5)

    y_2d = pd.DataFrame({'c1': [1.0, 2.0], 'c2': [3.0, 4.0]})
    pred_2d = pd.DataFrame({'c1': [1.5, 2.5], 'c2': [3.5, 4.5]})
    scale_2d = pd.DataFrame({'c1': [0.5, 0.5], 'c2': [1.0, 1.0]})
    score_2d, metrics_2d, _, _ = scoring.score_predictions(
        y_2d,
        pred_2d,
        {},
        {'mase': 1.0},
        auxiliary_ground_truths={'mase_scale': scale_2d},
    )
    self.assertAlmostEqual(metrics_2d['mase'], 0.75)
    self.assertAlmostEqual(score_2d, 1.0 / 1.75)

    # Missing aux data raises ScoreError
    with self.assertRaises(scoring.ScoreError):
      scoring.score_predictions(
          self.y_true_df, self.y_pred_good_df, {}, {'mase': 1.0}
      )


class ScoringFeedbackTest(parameterized.TestCase):

  def test_get_metrics_description_returns_str(self):
    self.assertIsInstance(scoring.get_metrics_description({}), str)
    weights = {'mae': 1.0, 'r2': 0.5, 'exec_time': 0.0}
    self.assertIsInstance(scoring.get_metrics_description(weights), str)
    with self.assertRaises(scoring.ScoreError):
      scoring.get_metrics_description({'mae': 1.0, 'unknown': 1.0})

  def test_prediction_validator_feedback_length_coverage(self):
    with mock.patch.object(scoring.logger, 'warning'):
      fn_args = [None, None, None, None, pd.DataFrame({'a': range(1)})]
      # Case 1: Feedback OK
      scoring.prediction_validator(
          fn_args, {}, (pd.DataFrame({'p': [1.0]}), 'OK'), max_feedback_len=100
      )
      # Case 2: Feedback too long
      scoring.prediction_validator(
          fn_args,
          {},
          (pd.DataFrame({'p': [1.0]}), 'L' * 200),
          max_feedback_len=100,
      )

  # Just ensure it runs without error for coverage

  def test_get_classroom_feedback_string_returns_str(self):
    metric_keys = ['overall_score', 'mae']

    # Case 1: Initial call
    self.assertIsInstance(
        scoring.get_classroom_feedback_string(0, [], {}, []), str
    )

    # Case 2: History with no results
    history_empty = {'s1': []}
    self.assertIsInstance(
        scoring.get_classroom_feedback_string(
            1, ['s1'], history_empty, metric_keys
        ),
        str,
    )

    # Case 3: History with some results
    history_with_stats = {
        's1': [
            agents.StudentCycleResult(
                cycle=0, overall_score=0.1, metrics={'mae': 1}
            )
        ]
    }
    self.assertIsInstance(
        scoring.get_classroom_feedback_string(
            1, ['s1'], history_with_stats, metric_keys
        ),
        str,
    )

  def test_get_classroom_feedback_string_empty_stat_lines(self):
    # Case: num_successful_results > 0 but no metrics requested (metric_keys=[])
    # Should trigger "if not stat_lines: return summary"
    history = {
        's1': [
            agents.StudentCycleResult(
                cycle=0, overall_score=0.1, metrics={'mae': 1}
            )
        ]
    }
    # Passing empty metric_keys
    result = scoring.get_classroom_feedback_string(1, ['s1'], history, [])
    self.assertIn('Completed **1 cycles**', result)
    self.assertNotIn('Stats from', result)  # No stats header
    self.assertNotIn('No successful results yet', result)

  def test_get_classroom_feedback_string_iterator_metric_keys(self):
    # Case: metric_keys is an iterator that gets exhausted
    # Should trigger "if key not in metric_keys: continue"
    history = {
        's1': [
            agents.StudentCycleResult(
                cycle=0, overall_score=0.1, metrics={'mae': 1}
            )
        ]
    }
    # Pass iterator as metric_keys
    metric_keys_iter = iter(['mae'])
    result = scoring.get_classroom_feedback_string(
        1, ['s1'], history, metric_keys_iter
    )
    self.assertIn('Completed **1 cycles**', result)
    self.assertNotIn('Stats from', result)

  def test_get_classroom_feedback_string_exclude_reason(self):
    # Case: result has exclude_reason
    # Should trigger "if result.exclude_reason: continue"
    history = {
        's1': [
            agents.StudentCycleResult(cycle=0, exclude_reason='Error'),
        ]
    }
    result = scoring.get_classroom_feedback_string(1, ['s1'], history, ['mae'])
    # Since the only result is excluded, num_successful_results stays 0.
    self.assertIn('No successful results yet', result)


if __name__ == '__main__':
  absltest.main()
