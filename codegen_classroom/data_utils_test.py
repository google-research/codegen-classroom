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

import dataclasses
import datetime
import json
import os
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from codegen_classroom import agents
from codegen_classroom import data_utils
import pandas as pd


@dataclasses.dataclass
class AnotherConfig:
  name: str
  model_name: Any = None
  value: int = 1


class DataUtilsTest(parameterized.TestCase):
  """Tests for general utility functions in data_utils."""

  def setUp(self):
    super().setUp()
    self.test_dir = self.create_tempdir().full_path

  def _create_file(self, filename, content):
    path = os.path.join(self.test_dir, filename)
    with open(path, 'w') as f:
      f.write(content)
    return path

  def _read_file(self, filename):
    with open(os.path.join(self.test_dir, filename), 'r') as f:
      return f.read()

  @mock.patch.object(datetime, 'datetime', autospec=True)
  def test_generate_run_id(self, mock_datetime):
    """Tests generating a run identifier."""
    mock_datetime.now.return_value.strftime.return_value = '20231027_103000'
    self.assertEqual(data_utils.generate_run_id(), '20231027_103000')

  def test_create_dir(self):
    """Tests creating directories."""
    new_path = os.path.join(self.test_dir, 'new_dir/sub_dir')
    self.assertFalse(os.path.exists(new_path))
    data_utils.create_dir(new_path)
    self.assertTrue(os.path.exists(new_path))
    # Test existing directory
    data_utils.create_dir(new_path)
    self.assertTrue(os.path.exists(new_path))

  @parameterized.named_parameters(
      dict(
          testcase_name='list_of_dicts',
          data_to_save=[{'key1': 'value1', 'key2': 2}, {'item': 'data'}],
      ),
      dict(
          testcase_name='single_dict',
          data_to_save={'config_param': 'value', 'level': 5},
      ),
  )
  def test_save_and_load_dict_to_json(self, data_to_save):
    """Tests saving and loading a list of dicts or a single dict to/from JSON."""

    # Test save
    json_path = os.path.join(self.test_dir, 'data.json')
    data_utils.save_dict_to_json(data_to_save, json_path)
    self.assertTrue(os.path.exists(json_path))
    self.assertEqual(json.loads(self._read_file('data.json')), data_to_save)

    # Test load
    loaded_data = data_utils.load_json_from_file(json_path)
    self.assertEqual(loaded_data, data_to_save)

  @parameterized.named_parameters(
      dict(
          testcase_name='not_found',
          filename='nonexistent.json',
          content=None,
          expected_error=OSError,
      ),
      dict(
          testcase_name='invalid_json',
          filename='invalid.json',
          content='{"a": 1,}',
          expected_error=json.JSONDecodeError,
      ),
  )
  def test_load_json_from_file_errors(self, filename, content, expected_error):
    """Tests loading invalid or non-existent JSON files."""
    if content is not None:
      json_path = self._create_file(filename, content)
    else:
      json_path = os.path.join(self.test_dir, filename)
    with self.assertRaises(expected_error):
      data_utils.load_json_from_file(json_path)

  @parameterized.named_parameters(
      dict(
          testcase_name='write',
          mode='w',
          index=False,
          init=None,
          exp='h\n1,a\n',
      ),
      dict(
          testcase_name='write_idx',
          mode='w',
          index=True,
          init=None,
          exp=',h\n0,1,a\n',
      ),
      dict(
          testcase_name='append_new',
          mode='a',
          index=False,
          init=None,
          exp='h\n1,a\n',
      ),
      dict(
          testcase_name='append_exist',
          mode='a',
          index=False,
          init='h\n0,z\n',
          exp='h\n0,z\n1,a\n',
      ),
      dict(
          testcase_name='append_exist_idx',
          mode='a',
          index=True,
          init=',h\n0,0,z\n',
          exp=',h\n0,0,z\n0,1,a\n',
      ),
  )
  def test_save_dataframe_to_csv(self, mode, index, init, exp):
    """Tests saving/appending DataFrame to CSV."""
    csv_path = os.path.join(self.test_dir, 'df_data.csv')
    df = pd.DataFrame({'col1': [1], 'col2': ['a']})
    header = 'col1,col2'
    if init:
      with open(csv_path, 'w') as f:
        f.write(init.replace('h', header))

    data_utils.save_dataframe_to_csv(df, csv_path, mode=mode, index=index)
    expected_content = exp.replace('h', header)
    self.assertEqual(self._read_file('df_data.csv'), expected_content)
    self.assertTrue(os.path.exists(os.path.dirname(csv_path)))

  @parameterized.named_parameters(
      dict(
          testcase_name='valid_csv',
          filename='df_read_data.csv',
          content='col1,col2\n1,a\n2,b',
          expected_data=[{'col1': 1, 'col2': 'a'}, {'col1': 2, 'col2': 'b'}],
      ),
      dict(
          testcase_name='warn_if_bad_lines',
          filename='bad_lines.csv',
          content='col1,col2\n1,a\n2,b,c\n3,d',
          expected_data=[{'col1': 1, 'col2': 'a'}, {'col1': 3, 'col2': 'd'}],
      ),
      dict(
          testcase_name='parser_error',
          filename='parser_error.csv',
          content='col1,col2\n1,"a',
          expected_data=[],
      ),
  )
  def test_read_dict_from_csv(self, filename, content, expected_data):
    """Tests reading a DataFrame from CSV."""
    csv_path = self._create_file(filename, content)
    data = data_utils.read_dict_from_csv(csv_path)
    self.assertEqual(data, expected_data)

  @parameterized.named_parameters(
      dict(
          testcase_name='agent_config',
          cfg=agents.AgentConfig(
              name='test',
              api_key='mock_api_key',
              model_name='gemini-2.5-flash',
          ),
          expected_mutations={
              'model_name': 'gemini-2.5-flash',
          },
      ),
      dict(
          testcase_name='another_config_gemini_s',
          cfg=AnotherConfig(
              name='another',
              model_name='gemini-2.5-flash',
          ),
          expected_mutations={'model_name': 'gemini-2.5-flash'},
      ),
      dict(
          testcase_name='another_config_none',
          cfg=AnotherConfig(name='another_none', model_name=None),
          expected_mutations={'model_name': None},
      ),
  )
  def test_config_to_serializable_dict(self, cfg, expected_mutations):
    """Tests config to dict conversion for various dataclasses."""
    s_dict = data_utils.config_to_serializable_dict(cfg)
    expected_dict = dataclasses.asdict(cfg)
    expected_dict.update(expected_mutations)
    self.assertEqual(s_dict, expected_dict)

  @mock.patch('codegen_classroom.data_utils.genai')
  def test_config_to_serializable_dict_with_model_entry(self, mock_genai):
    """Tests conversion when config includes a ModelEntry."""
    mock_entry = mock.MagicMock()
    mock_entry.__str__.return_value = "MockModel(name='test_model')"
    mock_genai.types.Model = type(mock_entry)  # Make isinstance work

    cfg = AnotherConfig(
        name='with_model',
        model_name=mock_entry,
    )
    s_dict = data_utils.config_to_serializable_dict(cfg)
    expected_dict = dataclasses.asdict(cfg)
    expected_dict['model_name'] = "MockModel(name='test_model')"
    self.assertEqual(s_dict, expected_dict)

  def test_config_to_serializable_dict_non_dataclass(self):
    """Tests non-dataclass input raises TypeError."""
    with self.assertRaisesRegex(
        TypeError, 'Input must be a dataclass instance.'
    ):
      data_utils.config_to_serializable_dict({'a': 1})

  def test_submit_results_to_local_leaderboard(self):
    """Tests leaderboard appending and parent dir creation via epath."""
    leaderboard_file = os.path.join(
        self.test_dir, 'nested', 'dir', 'leaderboard.tsv'
    )

    # First write (appends to new file, creates parent dirs)
    df1 = pd.DataFrame({'score': [10.0], 'name': ['Alice']})
    data_utils.submit_results_to_local_leaderboard(leaderboard_file, df1)

    self.assertTrue(os.path.exists(leaderboard_file))

    # Second write (appends to existing file)
    df2 = pd.DataFrame({'score': [12.0], 'name': ['Bob']})
    data_utils.submit_results_to_local_leaderboard(leaderboard_file, df2)

    # Third write with new column (preserves history, older rows get NaN)
    df3 = pd.DataFrame({'score': [15.0], 'name': ['Charlie'], 'tags': ['xgb']})
    data_utils.submit_results_to_local_leaderboard(leaderboard_file, df3)

    # Expect tab separated values and correct rows with NaN for older runs
    df_read = pd.read_csv(leaderboard_file, sep='\t')
    self.assertLen(df_read, 3)
    self.assertEqual(list(df_read['name']), ['Alice', 'Bob', 'Charlie'])
    self.assertTrue(pd.isna(df_read.loc[0, 'tags']))
    self.assertEqual(df_read.loc[2, 'tags'], 'xgb')

    # Fourth write with run_dir path normalization
    df4 = pd.DataFrame({
        'score': [18.0],
        'name': ['David'],
        'run_dir': ['/tmp/absolute/parent_dir/my_run_dir'],
    })
    data_utils.submit_results_to_local_leaderboard(leaderboard_file, df4)

    df_read2 = pd.read_csv(leaderboard_file, sep='\t')
    self.assertLen(df_read2, 4)
    self.assertEqual(df_read2.loc[3, 'run_dir'], 'parent_dir/my_run_dir')

    # Fifth write with None/empty run_dir
    df5 = pd.DataFrame({
        'score': [20.0],
        'name': ['Eve'],
        'run_dir': [None],
    })
    data_utils.submit_results_to_local_leaderboard(leaderboard_file, df5)

    df_read3 = pd.read_csv(leaderboard_file, sep='\t')
    self.assertLen(df_read3, 5)
    self.assertTrue(pd.isna(df_read3.loc[4, 'run_dir']))

    # Sixth write with solution_dir normalization and header-only existing file
    header_only_lb = os.path.join(self.test_dir, 'leaderboard_custom.txt')
    with open(header_only_lb, 'w') as f:
      f.write('score\tname\tsolution_dir\n')
    df6 = pd.DataFrame({
        'score': [25.0],
        'name': ['Frank'],
        'solution_dir': ['/workspace/tasks/housing/solutions/sol_1'],
    })
    with self.assertLogs(data_utils.logger, level='WARNING') as cm:
      data_utils.submit_results_to_local_leaderboard(header_only_lb, df6)
    self.assertTrue(any('does not end with .tsv' in msg for msg in cm.output))
    df_read4 = pd.read_csv(header_only_lb, sep='\t')
    self.assertLen(df_read4, 1)
    self.assertEqual(df_read4.loc[0, 'solution_dir'], 'solutions/sol_1')

  def test_round_leaderboard_metrics(self):
    df = pd.DataFrame({
        'test_accuracy': [0.1234567, 0.987654],
        'overall_score': [0.777777, 0.888888],
        'train_cycles': [10, 5],
        'student': ['s1', 's2'],
    })
    rounded = data_utils.round_leaderboard_metrics(df, precision=3)
    self.assertEqual(list(rounded['test_accuracy']), [0.123, 0.988])
    self.assertEqual(list(rounded['overall_score']), [0.778, 0.889])
    self.assertEqual(list(rounded['train_cycles']), [10, 5])
    self.assertEqual(list(rounded['student']), ['s1', 's2'])

  @parameterized.named_parameters(
      dict(
          testcase_name='with_runfiles',
          env={'BUILD_WORKSPACE_DIRECTORY': '/fake/workspace/dir'},
          path='/fake/sandbox/.runfiles/my_workspace/data/file.csv',
          expected='/fake/workspace/dir/data/file.csv',
      ),
      dict(
          testcase_name='without_runfiles',
          env={'BUILD_WORKSPACE_DIRECTORY': '/fake/workspace/dir'},
          path='/tmp/other/file.csv',
          expected='/tmp/other/file.csv',
      ),
      dict(
          testcase_name='outside_bazel',
          env={},
          path='/fake/sandbox/.runfiles/my_workspace/data/file.csv',
          expected='/fake/sandbox/.runfiles/my_workspace/data/file.csv',
      ),
  )
  def test_resolve_workspace_dir(self, env, path, expected):
    """Tests path resolution inside and outside bazel runfiles."""
    with mock.patch.dict(os.environ, env, clear=True):
      self.assertEqual(data_utils.resolve_workspace_dir(path), expected)

  @parameterized.named_parameters(
      dict(
          testcase_name='experiment_path',
          path='/workspace/experiments/housing_exp/20261014_120000',
          expected='housing_exp/20261014_120000',
      ),
      dict(
          testcase_name='nested_task_path',
          path='/workspace/repo/tasks/template/exp/run_2',
          expected='exp/run_2',
      ),
      dict(
          testcase_name='single_segment',
          path='run_99',
          expected='run_99',
      ),
  )
  def test_sanitize_path(self, path, expected):
    """Tests dirname/basename path sanitization."""
    self.assertEqual(data_utils.sanitize_path(path), expected)

  @parameterized.named_parameters(
      dict(
          testcase_name='empty_dir',
          files={},
          subfolder=None,
          students=None,
          expected={},
      ),
      dict(
          testcase_name='nonexistent_dir',
          files=None,
          subfolder=None,
          students=None,
          expected={},
      ),
      dict(
          testcase_name='multiple_students',
          files={
              'alice.csv': 'cycle,score\n0,10\n1,20',
              'bob.csv': 'cycle,score\n0,15',
          },
          subfolder=None,
          students=None,
          expected={
              'alice': [{'cycle': 0, 'score': 10}, {'cycle': 1, 'score': 20}],
              'bob': [{'cycle': 0, 'score': 15}],
          },
      ),
      dict(
          testcase_name='filter_students',
          files={
              'alice.csv': 'cycle,score\n0,10',
              'bob.csv': 'cycle,score\n0,15',
          },
          subfolder=None,
          students=['bob'],
          expected={'bob': [{'cycle': 0, 'score': 15}]},
      ),
      dict(
          testcase_name='filter_missing_student',
          files={'alice.csv': 'cycle,score\n0,10'},
          subfolder=None,
          students=['nonexistent'],
          expected={},
      ),
      dict(
          testcase_name='malformed_csv_returns_empty_list',
          files={'broken.csv': 'col1,col2\n1,"unclosed'},
          subfolder=None,
          students=None,
          expected={'broken': []},
      ),
      dict(
          testcase_name='nested_students_subfolder',
          files={'alice.csv': 'cycle,score\n0,10'},
          subfolder='students',
          students=None,
          expected={'alice': [{'cycle': 0, 'score': 10}]},
      ),
  )
  def test_load_student_results(self, files, subfolder, students, expected):
    """Tests loading student results from a directory with various layouts."""
    run_dir = os.path.join(self.test_dir, 'student_test_run')
    if files is None:
      res = data_utils.load_student_results(
          os.path.join(self.test_dir, 'does_not_exist'), students=students
      )
      self.assertEqual(res, expected)
      return

    target_dir = os.path.join(run_dir, subfolder) if subfolder else run_dir
    os.makedirs(target_dir, exist_ok=True)
    for fname, content in files.items():
      with open(os.path.join(target_dir, fname), 'w') as f:
        f.write(content)

    res = data_utils.load_student_results(run_dir, students=students)
    self.assertEqual(res, expected)

  def test_load_student_results_oserror_handling(self):
    """Tests graceful fallback when glob or CSV reading raises OSError."""
    run_dir = os.path.join(self.test_dir, 'oserror_run')
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'alice.csv'), 'w') as f:
      f.write('cycle,score\n0,10\n')

    with mock.patch.object(
        data_utils, 'read_dict_from_csv', side_effect=OSError('read failed')
    ):
      self.assertEqual(data_utils.load_student_results(run_dir), {})

    mock_path = mock.MagicMock()
    mock_path.__truediv__.return_value.is_dir.return_value = False
    mock_path.exists.return_value = True
    mock_path.glob.side_effect = OSError('glob failed')
    with mock.patch.object(data_utils.epath, 'Path', return_value=mock_path):
      self.assertEqual(data_utils.load_student_results(run_dir), {})

  @parameterized.named_parameters(
      dict(
          testcase_name='valid_tsv',
          content='rank\tstudent_name\tscore\n1\tAlice\t99.5\n2\tBob\t88.0\n',
          expected_len=2,
          expected_cols=['rank', 'student_name', 'score'],
      ),
      dict(
          testcase_name='single_row',
          content='student_name\tscore\nAlice\t100\n',
          expected_len=1,
          expected_cols=['student_name', 'score'],
      ),
  )
  def test_load_leaderboard_success(self, content, expected_len, expected_cols):
    """Tests loading existing TSV leaderboard file."""
    lb_path = self._create_file('test_leaderboard.tsv', content)
    df = data_utils.load_leaderboard(lb_path)
    self.assertIsInstance(df, pd.DataFrame)
    self.assertLen(df, expected_len)
    self.assertEqual(list(df.columns), expected_cols)

  @parameterized.named_parameters(
      dict(
          testcase_name='missing_file',
          filename='missing_leaderboard.tsv',
          content=None,
          expected_error=OSError,
      ),
      dict(
          testcase_name='empty_file',
          filename='empty_leaderboard.tsv',
          content='',
          expected_error=pd.errors.EmptyDataError,
      ),
  )
  def test_load_leaderboard_error_cases(
      self, filename, content, expected_error
  ):
    """Tests error handling for missing and empty leaderboard files."""
    if content is not None:
      path = self._create_file(filename, content)
    else:
      path = os.path.join(self.test_dir, filename)

    with self.assertRaises(expected_error):
      data_utils.load_leaderboard(path)


if __name__ == '__main__':
  absltest.main()
