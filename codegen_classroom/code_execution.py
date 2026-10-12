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

"""Code utils for extracting and executing python code in string format."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import io
import linecache
import multiprocessing
import os
import queue
import re
import signal
import sys
import time
from typing import Any


class ExtractionError(Exception):
  """Custom exception for errors during code extraction."""


class ExecutionError(Exception):
  """Custom exception for errors during code execution."""


class ExecutionTimeoutError(Exception):
  """Custom exception for code execution timing out."""


def extract_python_code(
    code_string: str | None,
    target_fn_name: str | None = None,
) -> tuple[str, str]:
  """Extracts the most relevant Python code block and the surrounding plan.

  Finds all Markdown-style Python code blocks (```python ... ```).
  - Prioritizes blocks containing the definition of `target_fn_name` if given.
  - If multiple blocks match,  the one with the most lines is chosen.
  - The Markdown `python_plan` consists of all text from the original string,
    excluding the content of the selected `python_code` block.

  Args:
      code_string: The string potentially containing Python code blocks.
      target_fn_name: The name of the function to look for in code blocks.

  Returns:
    A tuple of:
      - python_code: The Python code from the selected block, stripped.
      - python_plan: Text before and after the selected code block, stripped and
        joined by a space. Includes any other code blocks.

  Raises:
    ExtractionError: If the input is not a valid string, or if no valid
      complete Python code blocks (```python ... ```) are found.
  """
  if not isinstance(code_string, str) or not code_string.strip():
    raise ExtractionError(
        "Input code_string cannot be None, empty or type: "
        f"{type(code_string).__name__}.."
    )

  pattern = r"```python\s*(.*?)\s*```"
  matches = list(re.finditer(pattern, code_string, flags=re.DOTALL))

  if not matches:  # No complete ```python ... ``` blocks found.
    if "```python" not in code_string:
      raise ExtractionError(
          "No Python code block start marker (```python) found."
      )
    else:
      #  "```python" was found, but no regex match, indicating an issue.
      raise ExtractionError(
          "Python code block start marker '```python' found, but no valid"
          " complete block could be extracted."
      )

  best_match = None
  if target_fn_name:
    # Look for blocks containing the target Python function definition.
    target_def_pattern = rf"(^|\n)def\s+{target_fn_name}\s*\("
    fn_matches = [
        m for m in matches if re.search(target_def_pattern, m.group(1).strip())
    ]
    if fn_matches:
      best_match = max(fn_matches, key=lambda m: len(m.group(1).strip()))

  if not best_match:
    # Fallback to the longest block overall
    best_match = max(matches, key=lambda m: len(m.group(1).strip()))

  start_text = code_string[: best_match.start()].strip()
  end_text = code_string[best_match.end() :].strip()
  python_code = best_match.group(1).strip()
  # The plan is everything before the longest match and everything after code.
  python_plan = " ".join([text for text in [start_text, end_text] if text])
  return python_code, python_plan


def _run_compiled_code(
    extracted_code: str,
    target_fn_name: str,
    fn_args: Sequence[Any],
    fn_kwargs: Mapping[str, Any],
    initial_namespace: Mapping[str, Any] | None,
    local_filename: str,
) -> Any:
  """Compiles and executes the extracted code, and calls the target function."""
  local_namespace = dict(initial_namespace) if initial_namespace else {}
  try:
    compiled_code = compile(
        extracted_code, filename=local_filename, mode="exec"
    )
    exec(compiled_code, local_namespace)  # pylint: disable=exec-used
    target_fn = local_namespace.get(target_fn_name)
    if target_fn is None:
      raise ExecutionError(
          f"Target function '{target_fn_name}' not found in executed code."
      )
    if not callable(target_fn):
      raise ExecutionError(
          f"Target '{target_fn_name}' was found but is not callable."
      )
    return target_fn(*fn_args, **fn_kwargs)
  except Exception as e:
    raise ExecutionError(
        f"Error during code execution: {type(e).__name__} - {e}"
    ) from e


def _execute_target(
    q: multiprocessing.Queue,
    extracted_code: str,
    target_fn_name: str,
    fn_args: Sequence[Any],
    fn_kwargs: Mapping[str, Any],
    initial_namespace: Mapping[str, Any],
    local_filename: str,
):
  """Target function to run in a separate process."""

  # Prevent child from receiving SIGINT and dumping tracebacks onto Colab.
  signal.signal(signal.SIGINT, signal.SIG_IGN)

  # Redirect stdout/stderr to prevent concurrent writes to Colab ZMQ sockets,
  # which instantly crashes the Colab kernel.
  sys.stdout = io.StringIO()
  sys.stderr = io.StringIO()

  try:
    linecache.cache[local_filename] = (
        len(extracted_code),
        None,
        extracted_code.splitlines(True),
        local_filename,
    )
    code_output = _run_compiled_code(
        extracted_code,
        target_fn_name,
        fn_args,
        fn_kwargs,
        initial_namespace,
        local_filename,
    )
    q.put(("success", code_output))
  except Exception as e:  # pylint: disable=broad-exception-caught
    # Catching all exceptions is intentional here to report any error in the
    # provided code back to the mainprocess.
    q.put(("error", e))
  finally:
    linecache.cache.pop(local_filename, None)


def execute_code(
    extracted_code: str,
    target_fn_name: str,
    fn_args: Sequence[Any] | None = None,
    fn_kwargs: Mapping[str, Any] | None = None,
    initial_namespace: Mapping[str, Any] | None = None,
    local_filename: str = "<dynamic_code>",
    timeout_seconds: float | None = None,
) -> tuple[Any, float]:
  """Compiles, executes code, and calls a target function, with timeout.

  Args:
      extracted_code: The Python code string to execute.
      target_fn_name: The name of the function to call within the executed code.
      fn_args: Positional arguments for the target function.
      fn_kwargs: Keyword arguments for the target function.
      initial_namespace: Mapping to pre-populate the execution namespace.
      local_filename: Filename for tracebacks.
      timeout_seconds: Optional maximum time to allow for execution.

  Returns:
      tuple[Any, float]: A tuple containing:
        - code_output: Return value of the target_fn.
        - exec_time: Execution time in seconds.

  Raises:
      ExecutionError: Wraps any exception raised during code compilation,
          execution, target function lookup, or the target function call.
      ExecutionTimeoutError: If the execution time exceeds timeout_seconds.
  """
  fn_args = fn_args or []
  fn_kwargs = fn_kwargs or {}
  start_time = time.time()

  if timeout_seconds is None:
    linecache.cache[local_filename] = (
        len(extracted_code),
        None,
        extracted_code.splitlines(True),
        local_filename,
    )
    try:
      code_output = _run_compiled_code(
          extracted_code,
          target_fn_name,
          fn_args,
          fn_kwargs,
          initial_namespace,
          local_filename,
      )
    finally:
      linecache.cache.pop(local_filename, None)
      exec_time = time.time() - start_time
    return code_output, exec_time
  else:
    # Sandbox execution timeout handling:
    # Using 'multiprocessing' with 'fork' context for isolated execution.
    # 'fork' is preferred here because it avoids pickling issues with modules
    # (e.g., pandas, numpy) and has lower overhead than 'spawn' or 'forkserver'.
    #
    # Risk Mitigation:
    #   1. Child runs isolated code.
    #   2. Child ignores SIGINT.
    #   3. Child stdout/stderr redirected to prevent socket corruption.
    #   4. Cleanup: Parent uses SIGKILL in finally block to prevent zombies.
    #
    # Note: Forkserver could be considered later if process stability degrades.

    ctx = multiprocessing.get_context("fork")
    q = ctx.Queue()

    args = (
        q,
        extracted_code,
        target_fn_name,
        fn_args,
        fn_kwargs,
        initial_namespace,
        local_filename,
    )

    p = ctx.Process(target=_execute_target, args=args)
    p.start()

    try:
      # Wait on queue with timeout to prevent deadlock from full pipe buffers
      try:
        if (
            isinstance(p.pid, int)
            and timeout_seconds is not None
            and timeout_seconds > 1.0
        ):
          deadline = time.monotonic() + timeout_seconds
          while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
              raise queue.Empty()
            try:
              status, result = q.get(timeout=min(0.2, remaining))
              break
            except queue.Empty:
              if not p.is_alive():
                raise
        else:
          status, result = q.get(timeout=timeout_seconds)
        exec_time = time.time() - start_time
      except queue.Empty as e:
        if not p.is_alive():
          exit_info = ""
          if isinstance(p.exitcode, int) and p.exitcode < 0:
            exit_info = (
                f" (exitcode={p.exitcode}; if using JAX/XLA, the C++ runtime"
                " aborted in this environment—fall back to NumPy/scikit-learn"
                " such as MLPClassifier)"
            )
          raise ExecutionError(
              "Failed to get result from process queue: process died"
              f" unexpectedly.{exit_info}"
          ) from e

        # If the process is still alive after queue timeout, it means the
        # execution itself timed out.
        try:
          os.kill(p.pid, signal.SIGKILL)  # pyrefly: ignore[bad-argument-type]
        except ProcessLookupError:
          pass
        p.join(timeout=1.0)
        raise ExecutionTimeoutError(
            f"Code execution timed out after {timeout_seconds} seconds."
        ) from e
      except Exception as e:
        raise ExecutionError(f"Error retrieving from queue: {e}") from e

      # Join with a short timeout in case background worker threads (e.g.
      # joblib/loky/OpenMP) in the forked child block interpreter shutdown.
      p.join(timeout=2.0)
      if p.is_alive():
        try:
          os.kill(p.pid, signal.SIGKILL)  # pyrefly: ignore[bad-argument-type]
        except ProcessLookupError:
          pass
        p.join(timeout=1.0)

      if status == "success":
        return result, exec_time
      else:
        if isinstance(result, Exception):
          # Re-raise the exception from the subprocess
          raise result
        else:
          raise ExecutionError(f"Unknown error during code execution: {result}")
    finally:
      # Clean up child process regardless of exceptions like KeyboardInterrupt.
      if p.is_alive():
        try:
          os.kill(p.pid, signal.SIGKILL)  # pyrefly: ignore[bad-argument-type]
        except ProcessLookupError:
          pass
        p.join()
