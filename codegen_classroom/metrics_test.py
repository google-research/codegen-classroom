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

from absl.testing import absltest
from absl.testing import parameterized
from codegen_classroom import metrics as metric_lib
import numpy as np


class MetricsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.y_true = np.array([1, 2, 3, 4, 5])
    self.y_pred_perfect = np.array([1, 2, 3, 4, 5])
    self.y_pred_good = np.array([1.1, 1.9, 3.1, 4.1, 4.9])
    self.y_pred_const = np.array([3, 3, 3, 3, 3])

  @parameterized.named_parameters(
      dict(
          testcase_name='mae',
          metric_name='mae',
          y_true=np.array([1, 2, 3, 4, 5]),
          y_pred=np.array([1, 2, 3, 4, 6]),
          expected_val=0.2,
      ),
      dict(
          testcase_name='rmse',
          metric_name='rmse',
          y_true=np.array([1, 2, 3, 4, 5]),
          y_pred=np.array([1, 2, 3, 4, 6]),
          expected_val=0.447,
      ),
      dict(
          testcase_name='r2_perfect',
          metric_name='r2',
          y_pred=np.array([1, 2, 3, 4, 5]),
          expected_val=1.0,
      ),
      dict(
          testcase_name='r2_good',
          metric_name='r2',
          y_pred=np.array([1.1, 1.9, 3.1, 4.1, 4.9]),
          expected_val=0.995,
      ),
      dict(
          testcase_name='accuracy',
          metric_name='accuracy',
          y_true=np.array([0, 1, 2, 0, 1, 2]),
          y_pred=np.array([0, 2, 1, 0, 0, 1]),
          expected_val=2 / 6,
      ),
      dict(
          testcase_name='balanced_accuracy',
          metric_name='balanced_accuracy',
          y_true=np.array([0, 0, 1, 1]),
          y_pred=np.array([0, 1, 1, 1]),
          expected_val=0.75,
      ),
      dict(
          testcase_name='f1_macro',
          metric_name='f1_macro',
          y_true=np.array([0, 1, 2, 1]),
          y_pred=np.array([0, 0, 2, 1]),
          expected_val=7 / 9,
      ),
      dict(
          testcase_name='f1_binary',
          metric_name='f1_binary',
          y_true=np.array([0, 1, 1, 0, 1]),
          y_pred=np.array([0, 1, 0, 0, 1]),
          expected_val=0.8,
      ),
      dict(
          testcase_name='precision_binary',
          metric_name='precision_binary',
          y_true=np.array([0, 1, 1, 0, 1]),
          y_pred=np.array([0, 1, 0, 0, 1]),
          expected_val=1.0,
      ),
      dict(
          testcase_name='recall_binary',
          metric_name='recall_binary',
          y_true=np.array([0, 1, 1, 0, 1]),
          y_pred=np.array([0, 1, 0, 0, 1]),
          expected_val=2 / 3,
      ),
      dict(
          testcase_name='precision_macro',
          metric_name='precision_macro',
          y_true=np.array([0, 1, 2]),
          y_pred=np.array([0, 0, 0]),
          expected_val=1 / 9,
      ),
      dict(
          testcase_name='recall_macro',
          metric_name='recall_macro',
          y_true=np.array([0, 1, 2]),
          y_pred=np.array([0, 0, 0]),
          expected_val=1 / 3,
      ),
      dict(
          testcase_name='f1_weighted',
          metric_name='f1_weighted',
          y_true=np.array([0, 1, 2, 1]),
          y_pred=np.array([0, 0, 2, 1]),
          expected_val=0.75,
      ),
      dict(
          testcase_name='precision_weighted',
          metric_name='precision_weighted',
          y_true=np.array([0, 1, 2, 1]),
          y_pred=np.array([0, 0, 2, 1]),
          expected_val=0.875,
      ),
      dict(
          testcase_name='recall_weighted',
          metric_name='recall_weighted',
          y_true=np.array([0, 1, 2, 1]),
          y_pred=np.array([0, 0, 2, 1]),
          expected_val=0.75,
      ),
      dict(
          testcase_name='mape_basic',
          metric_name='mape',
          y_true=np.array([10, 20]),
          y_pred=np.array([11, 22]),
          expected_val=10.0,
      ),
      dict(
          testcase_name='mape_zero_true',
          metric_name='mape',
          y_true=np.array([0, 20]),
          y_pred=np.array([1, 22]),
          expected_val=10.0,
      ),
      dict(
          testcase_name='mape_all_zeros_perfect',
          metric_name='mape',
          y_true=np.array([0, 0, 0]),
          y_pred=np.array([0, 0, 0]),
          expected_val=0.0,
      ),
      dict(
          testcase_name='mape_all_zeros_bad',
          metric_name='mape',
          y_true=np.array([0, 0, 0]),
          y_pred=np.array([0, 1, 0]),
          expected_val=100.0,
      ),
      dict(
          testcase_name='mape_mixed_zeros',
          metric_name='mape',
          y_true=np.array([0, 100]),
          y_pred=np.array([1, 110]),
          expected_val=10.0,
      ),
      dict(
          testcase_name='wasserstein_distance',
          metric_name='wasserstein_distance',
          y_true=np.array([1, 2, 3]),
          y_pred=np.array([1, 2, 3]),
          expected_val=0.0,
      ),
      dict(
          testcase_name='wasserstein_distance_diff',
          metric_name='wasserstein_distance',
          y_true=np.array([1, 2, 3]),
          y_pred=np.array([2, 3, 4]),
          expected_val=1.0,
      ),
      dict(
          testcase_name='specificity_binary',
          metric_name='specificity_binary',
          y_true=np.array([0, 1, 1, 0, 1]),
          y_pred=np.array([0, 1, 0, 0, 1]),
          expected_val=1.0,
      ),
      dict(
          testcase_name='auroc',
          metric_name='auroc',
          y_true=np.array([0, 0, 1, 1]),
          y_pred=np.array([0.1, 0.4, 0.35, 0.8]),
          expected_val=0.75,
      ),
      dict(
          testcase_name='auprc',
          metric_name='auprc',
          y_true=np.array([0, 0, 1, 1]),
          y_pred=np.array([0.1, 0.4, 0.35, 0.8]),
          expected_val=5 / 6,  # average_precision_score
      ),
  )
  def test_metric_compute(
      self,
      metric_name,
      expected_val,
      y_true=None,
      y_pred=None,
  ):
    y_true_np = y_true if y_true is not None else self.y_true
    y_pred_np = y_pred if y_pred is not None else self.y_pred_good

    metric = metric_lib.ALL_METRICS_CONFIG[metric_name]
    val = metric.compute(y_true_np, y_pred_np)
    self.assertAlmostEqual(val, expected_val, places=3)

  @parameterized.named_parameters(
      dict(testcase_name='accuracy', name='accuracy'),
      dict(testcase_name='balanced_accuracy', name='balanced_accuracy'),
      dict(testcase_name='f1_macro', name='f1_macro'),
      dict(testcase_name='f1_binary', name='f1_binary'),
      dict(testcase_name='precision_macro', name='precision_macro'),
      dict(testcase_name='precision_binary', name='precision_binary'),
      dict(testcase_name='recall_macro', name='recall_macro'),
      dict(testcase_name='recall_binary', name='recall_binary'),
      dict(testcase_name='f1_weighted', name='f1_weighted'),
      dict(testcase_name='precision_weighted', name='precision_weighted'),
      dict(testcase_name='recall_weighted', name='recall_weighted'),
      dict(testcase_name='mae', name='mae', higher=False),
      dict(testcase_name='mase', name='mase', higher=False),
      dict(testcase_name='mse', name='mse', higher=False),
      dict(testcase_name='rmse', name='rmse', higher=False),
      dict(
          testcase_name='wasserstein_distance',
          name='wasserstein_distance',
          higher=False,
      ),
      dict(testcase_name='specificity_binary', name='specificity_binary'),
      dict(testcase_name='auroc', name='auroc'),
      dict(testcase_name='auprc', name='auprc'),
      dict(testcase_name='worst_group_accuracy', name='worst_group_accuracy'),
  )
  def test_metric_configuration(self, name, higher=True):
    metric = metric_lib.ALL_METRICS_CONFIG[name]
    self.assertEqual(metric.higher_is_better, higher)
    expected_msg = 'Higher is better' if higher else 'Lower is better'
    self.assertEqual(metric.goal_string, expected_msg)

  @parameterized.named_parameters(
      dict(testcase_name='pearson', metric_name='pearson'),
      dict(testcase_name='spearman', metric_name='spearman'),
      dict(testcase_name='kendall', metric_name='kendall'),
  )
  def test_correlations(self, metric_name):
    metric = metric_lib.ALL_METRICS_CONFIG[metric_name]
    y_true = self.y_true
    self.assertAlmostEqual(metric.compute(y_true, self.y_pred_perfect), 1.0)
    self.assertAlmostEqual(metric.compute(y_true, y_true * -1), -1.0)
    self.assertAlmostEqual(metric.compute(y_true, self.y_pred_const), 0.0)
    self.assertEqual(metric.compute(np.array([]), np.array([])), 0.0)
    self.assertEqual(metric.compute(np.array([1]), np.array([1])), 0.0)
    self.assertEqual(metric.compute(np.array([1, 1]), np.array([1, 2])), 0.0)

  def test_metric_compute_precomputed_error(self):
    metric = metric_lib.ALL_METRICS_CONFIG['exec_time']
    with self.assertRaisesRegex(TypeError, 'is pre-computed'):
      metric.compute(self.y_true, self.y_pred_good)

  def test_auc_metrics_single_class_behavior(self):
    y_true_single = np.array([0, 0, 0, 0])
    y_score = np.array([0.1, 0.2, 0.3, 0.4])

    # AUROC returns NaN for single class in current sklearn version.
    val_auroc = metric_lib.auroc(y_true_single, y_score)
    self.assertTrue(np.isnan(val_auroc))

    # AUPRC returns 0.0 for single class in current sklearn version.
    val_auprc = metric_lib.auprc(y_true_single, y_score)
    self.assertEqual(val_auprc, 0.0)

  def test_worst_group_accuracy(self):
    y_true = np.array([0, 0, 1, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    groups = np.array([0, 0, 0, 1, 1, 2, 2, 2])

    acc_no_groups = metric_lib.worst_group_accuracy(y_true, y_pred)
    self.assertAlmostEqual(acc_no_groups, 5 / 8)

    acc = metric_lib.worst_group_accuracy(
        y_true, y_pred, subgroup_labels=groups
    )
    self.assertAlmostEqual(acc, 1 / 3)

    metric = metric_lib.ALL_METRICS_CONFIG['worst_group_accuracy']
    self.assertAlmostEqual(metric.compute(y_true, y_pred), 5 / 8)
    self.assertAlmostEqual(
        metric.compute(y_true, y_pred, subgroup_labels=groups), 1 / 3
    )

  def test_normalization(self):
    self.assertAlmostEqual(metric_lib.clip_negative(-1.0), 0.0)
    self.assertAlmostEqual(metric_lib.clip_negative(1.0), 1.0)
    # Binary metrics should have None for normalization
    self.assertIsNone(metric_lib.ALL_METRICS_CONFIG['accuracy'].normalize)
    self.assertIsNone(metric_lib.ALL_METRICS_CONFIG['auroc'].normalize)


class ForecastMetricsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='mase_scalar',
          y_true=np.array([[10.0, 12.0], [20.0, 22.0]]),
          y_pred=np.array([[12.0, 14.0], [22.0, 24.0]]),
          metric='mase',
          kwargs={'seasonal_scale': 2.0},
          expected=1.0,
      ),
      dict(
          testcase_name='mase_matrix',
          y_true=np.array([[10.0, 12.0], [20.0, 22.0]]),
          y_pred=np.array([[12.0, 14.0], [22.0, 24.0]]),
          metric='mase',
          kwargs={'seasonal_scale': np.array([[2.0, 2.0], [4.0, 4.0]])},
          expected=0.75,
      ),
      dict(
          testcase_name='mase_perfect_forecast_is_zero',
          y_true=np.array([1.0, 2.0, 3.0]),
          y_pred=np.array([1.0, 2.0, 3.0]),
          metric='mase',
          kwargs={'seasonal_scale': 2.0},
          expected=0.0,
      ),
      dict(
          testcase_name='mase_constant_series_wrong_is_inf',
          y_true=np.array([5.0, 5.0, 5.0]),
          y_pred=np.array([5.0, 5.0, 6.0]),
          metric='mase',
          kwargs={'seasonal_scale': 0.0},
          expected=float('inf'),
      ),
      dict(
          testcase_name='mase_inf_input_is_inf',
          y_true=np.array([1.0, 2.0]),
          y_pred=np.array([1.0, float('inf')]),
          metric='mase',
          kwargs={'seasonal_scale': 1.0},
          expected=float('inf'),
      ),
      dict(
          testcase_name='mase_mixed_zero_and_positive_scale_is_inf',
          y_true=np.array([1.0, 2.0]),
          y_pred=np.array([2.0, 3.0]),
          metric='mase',
          kwargs={'seasonal_scale': np.array([1.0, 0.0])},
          expected=float('inf'),
      ),
  )
  def test_forecast_metrics(self, y_true, y_pred, metric, kwargs, expected):
    self.assertAlmostEqual(
        metric_lib.ALL_METRICS_CONFIG[metric].compute(y_true, y_pred, **kwargs),
        expected,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='missing_scale',
          kwargs={},
          err_pattern='mase requires `seasonal_scale`',
      ),
      dict(
          testcase_name='size_mismatch',
          kwargs={'seasonal_scale': np.array([1.0, 2.0])},
          err_pattern='seasonal_scale has 2 entries but 3 were expected',
      ),
  )
  def test_mase_invalid_scale_raises(self, kwargs, err_pattern):
    y = np.array([1.0, 2.0, 3.0])
    with self.assertRaisesRegex(ValueError, err_pattern):
      metric_lib.mase(y, y, **kwargs)

  @parameterized.named_parameters(
      dict(testcase_name='empty_input', y_true=[], y_pred=[], scale=[]),
      dict(
          testcase_name='nan_input',
          y_true=[1.0, 2.0],
          y_pred=[1.0, float('nan')],
          scale=1.0,
      ),
  )
  def test_mase_nan_cases(self, y_true, y_pred, scale):
    val = metric_lib.mase(
        np.array(y_true), np.array(y_pred), seasonal_scale=scale
    )
    self.assertTrue(np.isnan(val))

  def test_register_metric(self):
    custom_metric = metric_lib.Metric(
        name='custom_test_metric',
        func=lambda yt, yp: 42.0,
        description='Custom test metric',
        higher_is_better=True,
        normalize=None,
    )
    self.addCleanup(
        metric_lib.ALL_METRICS_CONFIG.pop, 'custom_test_metric', None
    )
    metric_lib.register_metric(custom_metric)
    self.assertIn('custom_test_metric', metric_lib.ALL_METRICS_CONFIG)
    self.assertEqual(
        metric_lib.ALL_METRICS_CONFIG['custom_test_metric'], custom_metric
    )


if __name__ == '__main__':
  absltest.main()
