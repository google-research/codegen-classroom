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

import io
import linecache
import math
import multiprocessing
import os
import queue
import re
import signal
import sys
import textwrap
import threading
import time
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from codegen_classroom import code_execution

# Enums/Exceptions
ExtractionError = code_execution.ExtractionError
ExecutionError = code_execution.ExecutionError
ExecutionTimeoutError = code_execution.ExecutionTimeoutError


class ExtractPythonCodeTest(parameterized.TestCase):

  @parameterized.named_parameters(
      # Success Cases
      dict(
          testcase_name='single_block_with_plan',
          code_string='Plan A\n```python\nprint("OK")\n```\nPlan B',
          expected_code='print("OK")',
          expected_plan='Plan A Plan B',
      ),
      dict(
          testcase_name='single_block_no_plan',
          code_string='```python\nx = 1\n```',
          expected_code='x = 1',
          expected_plan='',
      ),
      dict(
          testcase_name='single_block_extra_whitespace',
          code_string='  ```python  \n  a = 1  \n  ```  ',
          expected_code='a = 1',
          expected_plan='',
      ),
      dict(
          testcase_name='multiple_blocks_longest_middle',
          code_string=(
              'A```python\n1\n```B```python\n12345\n```C```python\n12\n```D'
          ),
          expected_code='12345',
          expected_plan='A```python\n1\n```B C```python\n12\n```D',
      ),
      dict(
          testcase_name='empty_code_block',
          code_string='Text ```python\n``` More Text',
          expected_code='',
          expected_plan='Text More Text',
      ),
      dict(
          testcase_name='empty_code_block_no_newline',
          code_string='text ```python ``` text',
          expected_code='',
          expected_plan='text text',
      ),
      dict(
          testcase_name='single_empty_block_no_plan',
          code_string='```python\n```',
          expected_code='',
          expected_plan='',
      ),
      dict(
          testcase_name='malformed_extra_stuff_after_close',
          code_string='```python\nx=1\n``` extramaterial```',
          expected_code='x=1',
          expected_plan='extramaterial```',
      ),
      dict(
          testcase_name='other_language_blocks_ignored',
          code_string='```bash\necho hi\n```\n```python\n#python\n```',
          expected_code='#python',
          expected_plan='```bash\necho hi\n```',
      ),
      dict(
          testcase_name='plan_ends_with_open_marker',
          code_string="```python\nprint('ok')\n``` plan starts ```python",
          expected_code="print('ok')",
          expected_plan='plan starts ```python',
      ),
  )
  def test_extract_python_code_valid(
      self, code_string, expected_code, expected_plan
  ):
    code, plan = code_execution.extract_python_code(code_string)
    self.assertEqual(code, expected_code)
    self.assertEqual(plan, expected_plan)

  @parameterized.named_parameters(
      # Error Cases for extract_python_code
      dict(
          testcase_name='none_input',
          code_string=None,
          expected_error_msg=(
              r'Input code_string cannot be None, empty or type: NoneType\.\.'
          ),
      ),
      dict(
          testcase_name='integer_input',
          code_string=123,
          expected_error_msg=(
              r'Input code_string cannot be None, empty or type: int\.\.'
          ),
      ),
      dict(
          testcase_name='empty_string',
          code_string='',
          expected_error_msg=(
              r'Input code_string cannot be None, empty or type: str\.\.'
          ),
      ),
      dict(
          testcase_name='whitespace_string',
          code_string='   \n\t  ',
          expected_error_msg=(
              r'Input code_string cannot be None, empty or type: str\.\.'
          ),
      ),
      dict(
          testcase_name='no_markers',
          code_string='No code here',
          expected_error_msg='No Python code block start marker',
      ),
      dict(
          testcase_name='incomplete_cutoff',
          code_string='```python\nx=1',
          expected_error_msg=(
              "Python code block start marker '```python' found, but no valid"
              ' complete block could be extracted.'
          ),
      ),
      dict(
          testcase_name='malformed_no_proper_close',
          code_string='```python and some text',
          expected_error_msg=(
              "Python code block start marker '```python' found, but no valid"
              ' complete block could be extracted.'
          ),
      ),
      dict(
          testcase_name='only_opening_marker',
          code_string='text ```python text',
          expected_error_msg=(
              "Python code block start marker '```python' found, but no valid"
              ' complete block could be extracted.'
          ),
      ),
  )
  def test_extract_python_code_errors(self, code_string, expected_error_msg):
    with self.assertRaisesRegex(ExtractionError, expected_error_msg):
      code_execution.extract_python_code(code_string)

  @parameterized.named_parameters(
      dict(
          testcase_name='target_fn_present_unique',
          code_string=textwrap.dedent("""
              ```python
              print("short")
              ```

              Plan

              ```python
              def test_func():
                print("long block")
              ```
          """),
          target_fn_name='test_func',
          expected_code=textwrap.dedent("""
              def test_func():
                print("long block")
          """).strip(),
          expected_plan=textwrap.dedent("""
              ```python
              print("short")
              ```

              Plan
          """).strip(),
      ),
      dict(
          testcase_name='target_fn_present_multiple_chooses_longest',
          code_string=textwrap.dedent("""
              ```python
              def test_func():
                print("short func")
              ```

              Plan

              ```python
              def test_func():
                print("long block")
                print("more lines")
              ```
          """),
          target_fn_name='test_func',
          expected_code=textwrap.dedent("""
              def test_func():
                print("long block")
                print("more lines")
          """).strip(),
          expected_plan=textwrap.dedent("""
              ```python
              def test_func():
                print("short func")
              ```

              Plan
          """).strip(),
      ),
      dict(
          testcase_name='target_fn_not_found_fallback',
          code_string=textwrap.dedent("""
              ```python
              print("short")
              ```

              Plan

              ```python
              def other_func():
                print("long block")
              ```
          """),
          target_fn_name='test_func',
          expected_code=textwrap.dedent("""
              def other_func():
                print("long block")
          """).strip(),
          expected_plan=textwrap.dedent("""
              ```python
              print("short")
              ```

              Plan
          """).strip(),
      ),
      dict(
          testcase_name='target_fn_in_string_not_def',
          code_string=textwrap.dedent("""
              ```python
              print("def test_func(): pass")
              ```

              Plan

              ```python
              def real_func():
                print("This is the one to pick")
              ```
          """),
          target_fn_name='test_func',
          expected_code=textwrap.dedent("""
              def real_func():
                print("This is the one to pick")
          """).strip(),
          expected_plan=textwrap.dedent("""
              ```python
              print("def test_func(): pass")
              ```

              Plan
          """).strip(),
      ),
      dict(
          testcase_name='target_fn_in_shorter_block_is_selected',
          code_string=textwrap.dedent("""
              This is plan A.
              ```python
              print("this is the longest block")
              print("it has multiple lines")
              print("really, it is long")
              ```

              This is plan B.

              ```python
              def my_func():
                return 1
              ```

              This is plan C.
          """),
          target_fn_name='my_func',
          expected_code=textwrap.dedent("""
              def my_func():
                return 1
          """).strip(),
          expected_plan=textwrap.dedent("""
              This is plan A.
              ```python
              print("this is the longest block")
              print("it has multiple lines")
              print("really, it is long")
              ```

              This is plan B. This is plan C.
          """).strip(),
      ),
  )
  def test_extract_python_code_with_target_fn(
      self, code_string, target_fn_name, expected_code, expected_plan
  ):
    code, plan = code_execution.extract_python_code(
        code_string, target_fn_name=target_fn_name
    )
    self.assertEqual(code, expected_code)
    self.assertEqual(plan, expected_plan)


class ExecuteTargetTest(absltest.TestCase):

  def test_execute_target_success(self):
    q = multiprocessing.Queue()
    extracted_code = 'def f(a): return a + 1'
    target_fn_name = 'f'
    fn_args = [1]
    fn_kwargs = {}
    initial_namespace = {}
    local_filename = '<test_file>'

    # Use a mock to catch the signals/redirects without breaking the
    # test process
    with (
        mock.patch.object(signal, 'signal') as mock_signal,
        mock.patch('sys.stdout', new_callable=io.StringIO),
        mock.patch('sys.stderr', new_callable=io.StringIO),
    ):
      code_execution._execute_target(
          q,
          extracted_code,
          target_fn_name,
          fn_args,
          fn_kwargs,
          initial_namespace,
          local_filename,
      )
      mock_signal.assert_called_with(signal.SIGINT, signal.SIG_IGN)

    status, result = q.get()
    self.assertEqual(status, 'success')
    self.assertEqual(result, 2)
    self.assertNotIn(local_filename, linecache.cache)

  def test_execute_target_error(self):
    q = multiprocessing.Queue()
    extracted_code = "def f(): raise ValueError('intentional error')"
    target_fn_name = 'f'
    fn_args = []
    fn_kwargs = {}
    initial_namespace = {}
    local_filename = '<test_file_error>'

    with (
        mock.patch.object(signal, 'signal'),
        mock.patch('sys.stdout', new_callable=io.StringIO),
        mock.patch('sys.stderr', new_callable=io.StringIO),
    ):
      code_execution._execute_target(
          q,
          extracted_code,
          target_fn_name,
          fn_args,
          fn_kwargs,
          initial_namespace,
          local_filename,
      )

    status, result = q.get()
    self.assertEqual(status, 'error')
    self.assertIsInstance(result, ExecutionError)
    self.assertIn('ValueError - intentional error', str(result))
    self.assertNotIn(local_filename, linecache.cache)


class ExecuteCodeTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    linecache.clearcache()
    self.mock_time = self.enter_context(
        mock.patch.object(time, 'time', autospec=True)
    )
    # Default side effect for most tests
    self.mock_time.side_effect = [1.0, 2.5, 4.0, 5.5, 7.0, 8.5]

  # Use 'f' as the target function name by default.
  @parameterized.named_parameters(
      dict(
          testcase_name='success',
          code='def f(a, b=2): return a*b',
          fn_args=[3],
          fn_kwargs={'b': 4},
          expected_result=12,
          expected_exec_time=1.5,
      ),
      dict(
          testcase_name='default_kwargs',
          code='def f(a, b=2): return a*b',
          fn_args=[3],
          expected_result=6,
          expected_exec_time=1.5,
      ),
      dict(
          testcase_name='initial_namespace',
          code='def f(): return my_variable',
          initial_namespace={'my_variable': 55},
          expected_result=55,
          expected_exec_time=1.5,
      ),
      dict(
          testcase_name='time_import',
          code='import time\ndef f(): return time.time() > 0',
          expected_result=True,
          expected_exec_time=3.0,  # 4.0 - 1.0 (time.time() called inside)
          time_side_effect=[1.0, 2.5, 4.0],
      ),
  )
  def test_execute_code_success(
      self,
      code,
      expected_result,
      expected_exec_time,
      target_fn_name='f',
      fn_args=None,
      fn_kwargs=None,
      initial_namespace=None,
      time_side_effect=None,
  ):
    if time_side_effect:
      self.mock_time.side_effect = time_side_effect
    else:  # Default for tests without internal time calls
      self.mock_time.side_effect = [
          1.0,
          2.5,
      ]

    code_output, exec_time = code_execution.execute_code(
        code, target_fn_name, fn_args, fn_kwargs, initial_namespace
    )
    self.assertEqual(code_output, expected_result)
    self.assertIsInstance(exec_time, float)
    self.assertAlmostEqual(exec_time, expected_exec_time)
    self.assertNotIn('<dynamic_code>', linecache.cache)
    self.assertNotIn(target_fn_name, linecache.cache)

  @parameterized.named_parameters(
      dict(
          testcase_name='syntax_error',
          code='def f(): bad syntax',
          expected_error_contains='SyntaxError',
      ),
      dict(
          testcase_name='runtime_error',
          code='def f(): return 1/0',
          expected_error_contains='ZeroDivisionError',
      ),
      dict(
          testcase_name='name_error',
          code='def f(): return undefined_var',
          expected_error_contains='NameError',
      ),
      dict(
          testcase_name='function_not_found',
          code='def other_func(): pass',
          expected_error_contains="Target function 'f' not found",
      ),
      dict(
          testcase_name='target_not_callable',
          code='f = 123',
          expected_error_contains="Target 'f' was found but is not callable",
      ),
  )
  def test_execute_code_errors(
      self,
      code,
      expected_error_contains,
      target_fn_name='f',
      fn_args=None,
      fn_kwargs=None,
      initial_namespace=None,
  ):
    with self.assertRaisesRegex(ExecutionError, expected_error_contains):
      code_execution.execute_code(
          code, target_fn_name, fn_args, fn_kwargs, initial_namespace
      )
    self.assertNotIn('<dynamic_code>', linecache.cache)
    self.assertNotIn(target_fn_name, linecache.cache)

  @parameterized.named_parameters(
      dict(
          testcase_name='within_timeout',
          code='import time\ndef f():\n  time.sleep(0.1)\n  return 42',
          timeout_seconds=2.0,
          expected_result=42,
          expect_timeout=False,
      ),
      dict(
          testcase_name='exceeds_timeout',
          code='import time\ndef f():\n  time.sleep(1)\n  return 42',
          timeout_seconds=0.1,
          expected_result=None,
          expect_timeout=True,
      ),
  )
  def test_execute_code_timeout(
      self, code, timeout_seconds, expected_result, expect_timeout
  ):
    if expect_timeout:
      with self.assertRaisesRegex(
          ExecutionTimeoutError, 'Code execution timed out'
      ):
        code_execution.execute_code(code, 'f', timeout_seconds=timeout_seconds)
    else:
      code_output, _ = code_execution.execute_code(
          code, 'f', timeout_seconds=timeout_seconds
      )
      self.assertEqual(code_output, expected_result)

  def test_execute_code_error_with_timeout(self):
    code = 'def f(): return 1/0'
    with self.assertRaisesRegex(ExecutionError, 'ZeroDivisionError'):
      code_execution.execute_code(code, 'f', timeout_seconds=1)

  def test_execute_code_no_timeout_option(self):
    code = 'def f(): return 99'
    # Mock time to check exec_time calculation
    self.mock_time.side_effect = [1.0, 3.5]
    code_output, exec_time = code_execution.execute_code(
        code, 'f', timeout_seconds=None
    )
    self.assertEqual(code_output, 99)
    self.assertAlmostEqual(exec_time, 2.5)

  def test_execute_code_with_module_in_namespace_and_timeout(self):
    code = 'def my_func(): return math.sqrt(16)'
    initial_namespace = {'math': math}
    code_output, _ = code_execution.execute_code(
        code,
        'my_func',
        initial_namespace=initial_namespace,
        timeout_seconds=2.0,
    )

    self.assertEqual(code_output, 4.0)

  @parameterized.named_parameters(
      dict(
          testcase_name='value_error',
          code='def f(): raise ValueError("test error")',
          expected_exception=ValueError,
          expected_regex='test error',
      ),
      dict(
          testcase_name='type_error',
          code='def f(): return 1 + "a"',
          expected_exception=TypeError,
          expected_regex=re.escape('unsupported operand type(s)'),
      ),
      dict(
          testcase_name='name_error',
          code='def f(): return unknown_variable',
          expected_exception=NameError,
          expected_regex='unknown_variable',
      ),
      dict(
          testcase_name='import_error',
          code='def f(): import non_existent_module',
          expected_exception=ModuleNotFoundError,
          expected_regex='No module named',
      ),
  )
  @mock.patch.object(code_execution, 'multiprocessing', autospec=True)
  def test_execute_code_child_process_exceptions(
      self, mock_mp, code, expected_exception, expected_regex
  ):
    mock_process = mock.MagicMock()
    mock_queue = mock.MagicMock()
    mock_mp.get_context.return_value.Process.return_value = mock_process
    mock_mp.get_context.return_value.Queue.return_value = mock_queue

    exception_instance = expected_exception(expected_regex)
    mock_queue.get.return_value = ('error', exception_instance)
    mock_process.is_alive.return_value = False
    mock_process.join.return_value = None

    with self.assertRaises(expected_exception) as cm:
      code_execution.execute_code(code, 'f', timeout_seconds=1)
    if expected_exception == TypeError:
      self.assertIn(
          'unsupported operand type', str(cm.exception).replace('\\', '')
      )
    else:
      self.assertIn(expected_regex, str(cm.exception))

  @parameterized.named_parameters(
      dict(
          testcase_name='queue_empty',
          get_side_effect=queue.Empty,
          expected_regex='Failed to get result from process queue',
      ),
      dict(
          testcase_name='queue_other_error',
          get_side_effect=ValueError('test queue error'),
          expected_regex='Error retrieving from queue: test queue error',
      ),
      dict(
          testcase_name='queue_bad_status',
          get_side_effect=[('unexpected', 'foo')],
          expected_regex='Unknown error during code execution: foo',
      ),
  )
  @mock.patch.object(code_execution, 'multiprocessing', autospec=True)
  def test_execute_code_queue_get_errors(
      self, mock_mp, get_side_effect, expected_regex
  ):
    mock_process = mock.MagicMock()
    mock_queue = mock.MagicMock()
    mock_mp.get_context.return_value.Process.return_value = mock_process
    mock_mp.get_context.return_value.Queue.return_value = mock_queue

    mock_process.join.return_value = None
    mock_process.is_alive.return_value = False
    mock_queue.get.side_effect = get_side_effect

    with self.assertRaisesRegex(ExecutionError, expected_regex):
      code_execution.execute_code('def f(): return 1', 'f', timeout_seconds=1)

  @parameterized.named_parameters(
      dict(
          testcase_name='timeout_kill',
          is_alive_side_effect=[True, True],
          timeout_seconds=0.001,
          expected_exception=ExecutionTimeoutError,
      ),
      dict(
          testcase_name='cleanup_kill',
          is_alive_side_effect=[False, True],
          timeout_seconds=10,
          expected_exception=ExecutionError,
      ),
  )
  @mock.patch.object(os, 'kill', autospec=True)
  @mock.patch.object(code_execution, 'multiprocessing', autospec=True)
  def test_execute_code_kill_lookup_error(
      self,
      mock_mp,
      mock_kill,
      is_alive_side_effect,
      timeout_seconds,
      expected_exception,
  ):
    mock_process = mock.MagicMock()
    mock_queue = mock.MagicMock()
    mock_mp.get_context.return_value.Process.return_value = mock_process
    mock_mp.get_context.return_value.Queue.return_value = mock_queue
    mock_queue.get.side_effect = queue.Empty()
    mock_process.is_alive.side_effect = is_alive_side_effect
    mock_kill.side_effect = ProcessLookupError()
    mock_process.join.return_value = None

    with self.assertRaises(expected_exception):
      code_execution.execute_code(
          'def f(): pass', 'f', timeout_seconds=timeout_seconds
      )

  @mock.patch.object(code_execution, 'multiprocessing', autospec=True)
  @mock.patch.object(time, 'monotonic', autospec=True)
  def test_execute_code_polling_loop_process_dies_sigabrt(
      self, mock_monotonic, mock_mp
  ):
    mock_process = mock.MagicMock(pid=12345, exitcode=-6)
    mock_queue = mock.MagicMock()
    mock_mp.get_context.return_value.Process.return_value = mock_process
    mock_mp.get_context.return_value.Queue.return_value = mock_queue

    mock_process.is_alive.return_value = False
    mock_queue.get.side_effect = queue.Empty()
    mock_monotonic.side_effect = [10.0, 10.1]

    with self.assertRaisesRegex(ExecutionError, 'exitcode=-6; if using JAX'):
      code_execution.execute_code('def f(): return 1', 'f', timeout_seconds=2.0)

  @mock.patch.object(code_execution, 'multiprocessing', autospec=True)
  @mock.patch.object(time, 'monotonic', autospec=True)
  def test_execute_code_polling_loop_timeout(self, mock_monotonic, mock_mp):
    mock_process = mock.MagicMock(pid=12345)
    mock_queue = mock.MagicMock()
    mock_mp.get_context.return_value.Process.return_value = mock_process
    mock_mp.get_context.return_value.Queue.return_value = mock_queue

    mock_process.is_alive.return_value = True
    mock_queue.get.side_effect = queue.Empty()
    mock_monotonic.side_effect = [10.0, 10.1, 12.1, 12.1]

    with self.assertRaisesRegex(ExecutionTimeoutError, 'timed out'):
      code_execution.execute_code('def f(): return 1', 'f', timeout_seconds=2.0)

  @mock.patch.object(os, 'kill', autospec=True)
  @mock.patch.object(code_execution, 'multiprocessing', autospec=True)
  def test_execute_code_background_threads_block(self, mock_mp, mock_kill):
    mock_process = mock.MagicMock(pid=12345)
    mock_queue = mock.MagicMock()
    mock_mp.get_context.return_value.Process.return_value = mock_process
    mock_mp.get_context.return_value.Queue.return_value = mock_queue

    mock_process.is_alive.return_value = True
    mock_queue.get.return_value = ('success', 42)

    result, _ = code_execution.execute_code(
        'def f(): pass', 'f', timeout_seconds=1.0
    )

    self.assertEqual(result, 42)
    mock_kill.assert_called_with(12345, signal.SIGKILL)


class ExecuteCodeSubprocessTest(absltest.TestCase):

  def _run(self, code, timeout=None, env_vars=None):
    workdir = self.create_tempdir().full_path
    command = [sys.executable, 'solution.py']
    return workdir, code_execution.execute_code_subprocess(
        code, command, workdir, timeout_seconds=timeout, env_vars=env_vars
    )

  def test_success_returns_stdout_and_writes_files(self):
    code = textwrap.dedent("""
        import os
        print('hello', os.environ['MY_VAR'])
        open('out.txt', 'w').write('done')
    """)
    workdir, (stdout, exec_time) = self._run(code, env_vars={'MY_VAR': 'x'})
    self.assertEqual(stdout.strip(), 'hello x')
    self.assertGreater(exec_time, 0)
    with open(os.path.join(workdir, 'out.txt')) as f:
      self.assertEqual(f.read(), 'done')

  def test_nonzero_exit_raises_with_traceback(self):
    with self.assertRaisesRegex(ExecutionError, 'ZeroDivisionError'):
      self._run('1 / 0')

  def test_timeout_kills_process_group(self):
    code = textwrap.dedent("""
        import subprocess, sys, time
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
        open('child.pid', 'w').write(str(child.pid))
        time.sleep(60)
    """)
    workdir = self.create_tempdir().full_path
    start = time.time()
    with self.assertRaises(ExecutionTimeoutError):
      code_execution.execute_code_subprocess(
          code, [sys.executable, 'solution.py'], workdir, timeout_seconds=2
      )
    self.assertLess(time.time() - start, 20)
    with open(os.path.join(workdir, 'child.pid')) as f:
      child_pid = int(f.read())
    time.sleep(0.5)
    with self.assertRaises(ProcessLookupError):
      os.kill(child_pid, 0)


class GpuPoolTest(absltest.TestCase):

  def test_no_devices_yields_none(self):
    pool = code_execution.GpuPool()
    with pool.acquire() as device:
      self.assertIsNone(device)

  def test_slots_round_robin_and_block(self):
    pool = code_execution.GpuPool([1, 2], jobs_per_gpu=1)
    with pool.acquire() as d1, pool.acquire() as d2:
      self.assertCountEqual([d1, d2], ['1', '2'])
      acquired = threading.Event()

      def _worker():
        with pool.acquire():
          acquired.set()

      t = threading.Thread(target=_worker)
      t.start()
      self.assertFalse(acquired.wait(timeout=0.3))  # All slots taken.
    t.join(timeout=5)
    self.assertTrue(acquired.is_set())

  def test_invalid_jobs_per_gpu(self):
    with self.assertRaises(ValueError):
      code_execution.GpuPool([0], jobs_per_gpu=0)


if __name__ == '__main__':
  absltest.main()
