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

"""Tests for LLM backends."""

from unittest import mock

from absl.testing import absltest
from codegen_classroom import agents
from codegen_classroom import backends
import httpx
import openai


def _chat_response(text):
  message = mock.Mock(content=text)
  return mock.Mock(choices=[mock.Mock(message=message)])


class OpenAICompatBackendTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_client = mock.Mock()
    self.mock_client_cls = self.enter_context(
        mock.patch.object(openai, 'OpenAI', return_value=self.mock_client)
    )
    self.create = self.mock_client.chat.completions.create
    self.create.return_value = _chat_response('ok')

  def test_client_construction(self):
    backends.OpenAICompatBackend(
        model_name='m', base_url='http://host:8000/v1', http_timeout_ms=5000
    )
    _, kwargs = self.mock_client_cls.call_args
    self.assertEqual(kwargs['base_url'], 'http://host:8000/v1')
    self.assertEqual(kwargs['api_key'], 'EMPTY')
    self.assertEqual(kwargs['timeout'], 5.0)
    self.assertEqual(kwargs['max_retries'], 0)

  def test_generate_maps_roles_and_params(self):
    backend = backends.OpenAICompatBackend(
        model_name='qwen', extra_body={'top_p': 0.9}
    )
    text = backend.generate(
        [('user', 'u1'), ('model', 'm1')],
        'u2',
        system_instruction='sys',
        temperature=0.3,
        max_output_tokens=128,
    )
    self.assertEqual(text, 'ok')
    _, kwargs = self.create.call_args
    self.assertEqual(
        kwargs['messages'],
        [
            {'role': 'system', 'content': 'sys'},
            {'role': 'user', 'content': 'u1'},
            {'role': 'assistant', 'content': 'm1'},
            {'role': 'user', 'content': 'u2'},
        ],
    )
    self.assertEqual(kwargs['model'], 'qwen')
    self.assertEqual(kwargs['temperature'], 0.3)
    self.assertEqual(kwargs['max_tokens'], 128)
    self.assertEqual(kwargs['extra_body'], {'top_p': 0.9})
    self.assertNotIn('response_format', kwargs)

  def test_generate_json_mode_and_no_system(self):
    backend = backends.OpenAICompatBackend(model_name='m')
    backend.generate(
        [],
        'p',
        system_instruction='',
        temperature=0.0,
        max_output_tokens=8,
        json_mode=True,
    )
    _, kwargs = self.create.call_args
    self.assertEqual(kwargs['response_format'], {'type': 'json_object'})
    self.assertEqual(kwargs['messages'], [{'role': 'user', 'content': 'p'}])

  def test_strips_think_blocks(self):
    self.create.return_value = _chat_response(
        '<think>reasoning\nmore</think>\nFinal answer'
    )
    backend = backends.OpenAICompatBackend(model_name='m')
    text = backend.generate(
        [], 'p', system_instruction='', temperature=0, max_output_tokens=8
    )
    self.assertEqual(text, 'Final answer')

  def test_strips_reasoning_without_opening_tag(self):
    self.create.return_value = _chat_response(
        'reasoning only\n</think>\n\nFinal answer'
    )
    backend = backends.OpenAICompatBackend(model_name='m')
    text = backend.generate(
        [], 'p', system_instruction='', temperature=0, max_output_tokens=8
    )
    self.assertEqual(text, 'Final answer')

  def test_empty_choices_returns_empty_string(self):
    self.create.return_value = mock.Mock(choices=[])
    backend = backends.OpenAICompatBackend(model_name='m')
    text = backend.generate(
        [], 'p', system_instruction='', temperature=0, max_output_tokens=8
    )
    self.assertEqual(text, '')

  def test_retryable_exceptions(self):
    backend = backends.OpenAICompatBackend(model_name='m')
    request = httpx.Request('POST', 'http://x')
    self.assertIsInstance(
        openai.APITimeoutError(request), backend.retryable_exceptions
    )
    self.assertIsInstance(
        openai.APIConnectionError(request=request),
        backend.retryable_exceptions,
    )
    self.assertNotIsInstance(ValueError(), backend.retryable_exceptions)


  def test_context_overflow_is_fatal_not_retryable(self):
    backend = backends.OpenAICompatBackend(model_name='m')
    response = httpx.Response(400, request=httpx.Request('POST', 'http://x'))
    error = openai.BadRequestError(
        'maximum context length', response=response, body=None
    )
    self.assertNotIsInstance(error, backend.retryable_exceptions)
    self.assertIsInstance(error, backend.fatal_exceptions)


class MakeBackendTest(absltest.TestCase):

  def test_vllm_alias_and_default_base_url(self):
    with mock.patch.object(openai, 'OpenAI') as mock_cls:
      config = agents.AgentConfig(name='a', backend='vllm', model_name='m')
      agent = agents.LLMAgent(config, self.create_tempdir().full_path)
    self.assertIsInstance(agent.backend, backends.OpenAICompatBackend)
    _, kwargs = mock_cls.call_args
    self.assertEqual(kwargs['base_url'], 'http://localhost:8000/v1')

  def test_unknown_backend(self):
    config = agents.AgentConfig(name='a', backend='nope')
    with self.assertRaisesRegex(ValueError, 'Unknown backend'):
      backends.make_backend(config)

  def test_student_retries_on_backend_errors(self):
    with mock.patch.object(openai, 'OpenAI') as mock_cls:
      create = mock_cls.return_value.chat.completions.create
      create.side_effect = [
          openai.APIConnectionError(request=httpx.Request('POST', 'http://x')),
          _chat_response('solution'),
      ]
      config = agents.AgentConfig(
          name='s',
          backend='openai_compat',
          model_name='m',
          retry_sleep_seconds=0,
      )
      student = agents.StudentAgent(config, self.create_tempdir().full_path)
      self.assertEqual(student.generate_solution('p'), 'solution')
    self.assertEqual(create.call_count, 2)

  def test_student_fatal_error_becomes_generation_error(self):
    with mock.patch.object(openai, 'OpenAI') as mock_cls:
      create = mock_cls.return_value.chat.completions.create
      response = httpx.Response(400, request=httpx.Request('POST', 'http://x'))
      create.side_effect = openai.BadRequestError(
          'maximum context length', response=response, body=None
      )
      config = agents.AgentConfig(
          name='s', backend='vllm', model_name='m', retry_sleep_seconds=0
      )
      student = agents.StudentAgent(config, self.create_tempdir().full_path)
      with self.assertRaisesRegex(agents.GenerationError, 'Non-retryable'):
        student.generate_solution('p')
    self.assertEqual(create.call_count, 1)  # Not retried.


if __name__ == '__main__':
  absltest.main()
