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

import os
import sys
import time
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from codegen_classroom import agent_config
from codegen_classroom import agents
from google import genai
import numpy as np
import pandas as pd


class AgentsTestBase(parameterized.TestCase):
  """Base class for Agent tests with common mocks."""

  def setUp(self):
    super().setUp()
    self.base_dir = self.create_tempdir().full_path
    self.agent_name = 'test_agent'

    # Mock genai Client
    self.mock_model_client_instance = mock.create_autospec(
        genai.Client, instance=True
    )
    self.mock_model_client_cls = self.enter_context(
        mock.patch.object(
            genai,
            'Client',
            autospec=True,
            return_value=self.mock_model_client_instance,
        )
    )
    self.mock_generate_response = mock.Mock()
    self.mock_model_client_instance.models.generate_content.return_value = (
        self.mock_generate_response
    )

    self.config = agents.AgentConfig(
        name=self.agent_name, api_key='mock_api_key'
    )

  def create_agent(self, agent_class, config=None, **kwargs):
    """Helper to create agent instances."""
    return agent_class(config or self.config, self.base_dir, **kwargs)

  def set_mock_response(self, text_response):
    self.mock_generate_response.text = text_response

  def assert_file_exists(self, path):
    self.assertTrue(os.path.exists(path), f'File not found: {path}')

  def _add_best_cycle_result(
      self, agent, score, solution='test_solution', cycle=0
  ):
    """Helper to add a cycle result and set it as the best."""
    result = agents.StudentCycleResult(
        cycle=cycle, overall_score=score, code='code', code_hash='default_hash'
    )
    agent.add_cycle_result(result)
    agent.update_best_solution(score, cycle, solution)


class GeminiAgentTest(AgentsTestBase):
  """Tests for the base GeminiAgent."""

  def test_init_saves_config(self):
    self.create_agent(agents.GeminiAgent)
    expected_config_path = os.path.join(
        self.base_dir, 'agent_configs', f'{self.agent_name}_config.json'
    )
    self.assert_file_exists(expected_config_path)
    self.mock_model_client_cls.assert_called_once()
    _, kwargs = self.mock_model_client_cls.call_args
    self.assertEqual(kwargs['http_options'].timeout, 300_000)

  def test_init_respects_custom_http_timeout(self):
    config = agents.AgentConfig(
        name='timeout_agent',
        api_key='mock_api_key',
        http_timeout_ms=120_000,
    )
    self.create_agent(agents.GeminiAgent, config=config)
    _, kwargs = self.mock_model_client_cls.call_args
    self.assertEqual(kwargs['http_options'].timeout, 120_000)

  @parameterized.named_parameters(
      dict(
          testcase_name='with_system_instructions',
          system_instructions='Be test',
      ),
      dict(
          testcase_name='no_system_instructions',
          system_instructions='',
      ),
  )
  def test_generate(self, system_instructions):
    config = agents.AgentConfig(
        name='code_gen_agent',
        api_key='mock_api_key',
        system_instruction=system_instructions,
    )
    agent = self.create_agent(agents.GeminiAgent, config=config)
    self.set_mock_response('Response')
    response = agent.generate('Prompt')
    self.assertEqual(response, 'Response')
    self.assertEqual(agent.history, [('user', 'Prompt'), ('model', 'Response')])

    self.mock_model_client_instance.models.generate_content.assert_called_once()
    _, kwargs = (
        self.mock_model_client_instance.models.generate_content.call_args
    )
    self.assertLen(kwargs['contents'], 1)
    self.assertEqual(kwargs['contents'][0].parts[0].text, 'Prompt')
    self.assertEqual(kwargs['contents'][0].role, 'user')
    self.assertEqual(kwargs['config'].system_instruction, system_instructions)

  def test_generate_with_mime_type(self):
    mime_type = 'application/json'
    config = agents.AgentConfig(
        name='mime_agent',
        api_key='mock_api_key',
        response_mime_type=mime_type,
    )
    agent = self.create_agent(agents.GeminiAgent, config=config)
    self.set_mock_response('{"result": "ok"}')
    response = agent.generate('Prompt')
    self.assertEqual(response, '{"result": "ok"}')

    self.mock_model_client_instance.models.generate_content.assert_called_once()
    _, kwargs = (
        self.mock_model_client_instance.models.generate_content.call_args
    )
    self.assertEqual(kwargs['config'].response_mime_type, mime_type)

  def test_generate_does_not_record_history_on_empty_response(self):
    agent = self.create_agent(agents.GeminiAgent)
    self.set_mock_response('')
    response = agent.generate('Prompt')
    self.assertEqual(response, '')
    self.assertEqual(agent.history, [])

  def test_generate_does_not_record_history_on_error(self):
    agent = self.create_agent(agents.GeminiAgent)
    self.mock_model_client_instance.models.generate_content.side_effect = (
        genai.errors.ServerError('Server error', {})
    )
    with self.assertRaises(genai.errors.ServerError):
      agent.generate('Prompt')
    self.assertEqual(agent.history, [])

  def test_generate_multi_turn_history(self):
    agent = self.create_agent(agents.GeminiAgent)
    self.set_mock_response('First response')
    self.assertEqual(agent.generate('First prompt'), 'First response')

    self.set_mock_response('Second response')
    self.assertEqual(agent.generate('Second prompt'), 'Second response')

    self.assertEqual(
        agent.history,
        [
            ('user', 'First prompt'),
            ('model', 'First response'),
            ('user', 'Second prompt'),
            ('model', 'Second response'),
        ],
    )
    _, kwargs = (
        self.mock_model_client_instance.models.generate_content.call_args
    )
    self.assertLen(kwargs['contents'], 3)
    self.assertEqual(kwargs['contents'][0].parts[0].text, 'First prompt')
    self.assertEqual(kwargs['contents'][0].role, 'user')
    self.assertEqual(kwargs['contents'][1].parts[0].text, 'First response')
    self.assertEqual(kwargs['contents'][1].role, 'model')
    self.assertEqual(kwargs['contents'][2].parts[0].text, 'Second prompt')
    self.assertEqual(kwargs['contents'][2].role, 'user')

  def test_reset_history(self):
    agent = self.create_agent(agents.GeminiAgent)
    agent.history = [('user', 'p')]
    agent.reset_history()
    self.assertEqual(agent.history, [])


class StudentAgentTest(AgentsTestBase):

  def test_student_agent(self):
    agent = self.create_agent(
        agents.StudentAgent, seed_solution='print("seed")'
    )
    self.assert_file_exists(os.path.join(self.base_dir, 'students'))
    self.assertEqual(agent.current_solution, 'print("seed")')
    self.assertEqual(agent.best_cycle, -1)
    self.assertEqual(agent.best_score, -float('inf'))
    self.assertEqual(agent.mutation_count, 0)
    self.assertEqual(agent.generate_code_count, 0)

    self.set_mock_response('print("new")')
    self.assertEqual(agent.generate_solution('write code'), 'print("new")')
    self.assertEqual(agent.current_solution, 'print("new")')
    self.mock_model_client_instance.models.generate_content.assert_called_once()
    self.assertEqual(agent.generate_code_count, 1)

  def test_add_cycle_result(self):
    agent = self.create_agent(agents.StudentAgent)
    result = agents.StudentCycleResult(cycle=1, code='code', code_hash='hash1')
    agent.add_cycle_result(result)
    self.assertEqual(agent.cycle_results, [result])
    self.assertEqual(result.code_hash, 'hash1')

    result2 = agents.StudentCycleResult(
        cycle=2, code='code2', code_hash='hash2'
    )
    agent.add_cycle_result(result2)
    self.assertEqual(agent.cycle_results, [result, result2])
    self.assertEqual(result2.code_hash, 'hash2')

  @parameterized.named_parameters(
      dict(
          testcase_name='complete',
          input_metrics={'acc': 0.9, 'loss': 0.1},
          expected_metrics={'metrics_acc': 0.9, 'metrics_loss': 0.1},
      ),
      dict(
          testcase_name='nan_metric',
          input_metrics={'acc': 0.9, 'loss': np.nan},
          expected_metrics={'metrics_acc': 0.9, 'metrics_loss': np.nan},
      ),
  )
  def test_student_cycle_result_flatten(self, input_metrics, expected_metrics):
    result = agents.StudentCycleResult(
        cycle=1,
        metrics=input_metrics,
        code='some code',
        code_hash='test_hash',
        plan='some plan',
        code_length=9,
    )
    flattened = result.flatten()

    # Check that all expected metric keys are in the output
    for key in expected_metrics:
      self.assertIn(key, flattened)

    # Check the values of the metrics
    for key, expected_val in expected_metrics.items():
      actual_val = flattened[key]
      if pd.isna(expected_val):
        self.assertTrue(pd.isna(actual_val))
      else:
        self.assertEqual(actual_val, expected_val)

    # Check base fields are present and correct
    expected_base = {
        'cycle': 1,
        'code_hash': 'test_hash',
        'code': 'some code',
        'plan': 'some plan',
        'code_length': 9,
    }
    self.assertDictContainsSubset(expected_base, flattened)

  def test_student_cycle_result_flatten_empty_metrics(self):
    result = agents.StudentCycleResult(
        cycle=1, metrics={}, code_hash='empty_hash'
    )
    flattened = result.flatten()
    self.assertNotIn('metrics_', ''.join(flattened.keys()))
    self.assertEqual(flattened['cycle'], 1)
    self.assertEqual(flattened['code_hash'], 'empty_hash')

  @parameterized.named_parameters(
      dict(
          testcase_name='with_best_cycle',
          best_cycle=0,
          expected_strings=[
              'Best Cycle = 0',
              'Score = 0.9500',
              'Best Plan:\nbest_plan',
              'Best Code:\nbest_code_ever',
          ],
          unexpected_strings=['Best Solution (Full):'],
      ),
      dict(
          testcase_name='no_best_cycle',
          best_cycle=-1,
          expected_strings=['Best Cycle = N/A'],
          unexpected_strings=['Score =', 'Best Plan:', 'Best Code:'],
      ),
  )
  def test_repr(self, best_cycle, expected_strings, unexpected_strings):
    agent = self.create_agent(agents.StudentAgent)
    if best_cycle >= 0:
      agent.add_cycle_result(
          agents.StudentCycleResult(
              cycle=best_cycle,
              overall_score=0.95,
              code='best_code_ever',
              code_hash='best_hash',
              plan='best_plan',
          )
      )
      agent.update_best_solution(0.95, best_cycle, 'full solution for best')
    else:
      agent.best_cycle = -1

    representation = repr(agent)
    self.assertIsInstance(representation, str)
    self.assertIn(f"name='{self.agent_name}'", representation)
    for s in expected_strings:
      self.assertIn(s, representation)
    for s in unexpected_strings:
      self.assertNotIn(s, representation)

  @parameterized.named_parameters(
      dict(
          testcase_name='agent1_is_less',
          score1=0.3,
          score2=0.9,
          agent1_is_lt=True,
          agent1_is_eq=False,
          agent1_is_gt=False,
      ),
      dict(
          testcase_name='agents_are_equal',
          score1=0.5,
          score2=0.5,
          agent1_is_lt=False,
          agent1_is_eq=True,
          agent1_is_gt=False,
      ),
      dict(
          testcase_name='agent1_is_greater',
          score1=0.8,
          score2=0.2,
          agent1_is_lt=False,
          agent1_is_eq=False,
          agent1_is_gt=True,
      ),
      dict(
          testcase_name='agent1_none_score',
          score1=None,
          score2=0.2,
          agent1_is_lt=True,
          agent1_is_eq=False,
          agent1_is_gt=False,
      ),
      dict(
          testcase_name='both_none',
          score1=None,
          score2=None,
          agent1_is_lt=False,
          agent1_is_eq=True,
          agent1_is_gt=False,
      ),
  )
  def test_comparison_operators(
      self, score1, score2, agent1_is_lt, agent1_is_eq, agent1_is_gt
  ):
    agent1 = self.create_agent(agents.StudentAgent)
    agent2_config = agents.AgentConfig(name='agent2', api_key='mock_api_key')
    agent2 = self.create_agent(agents.StudentAgent, config=agent2_config)

    if score1 is not None:
      self._add_best_cycle_result(agent1, score1)
    if score2 is not None:
      self._add_best_cycle_result(agent2, score2)

    self.assertEqual(
        agent1 < agent2,
        agent1_is_lt,
        msg=(
            f'Agent1 < Agent2 failed for scores {agent1.get_best_score()},'
            f' {agent2.get_best_score()}'
        ),
    )
    self.assertEqual(
        agent1 == agent2,
        agent1_is_eq,
        msg=(
            f'Agent1 == Agent2 failed for scores {agent1.get_best_score()},'
            f' {agent2.get_best_score()}'
        ),
    )
    self.assertEqual(
        agent1 > agent2,
        agent1_is_gt,
        msg=(
            f'Agent1 > Agent2 failed for scores {agent1.get_best_score()},'
            f' {agent2.get_best_score()}'
        ),
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='add_operator',
          op=lambda a, b: a + b,
          expected_substring='produced a better solution',
      ),
      dict(
          testcase_name='sub_operator',
          op=lambda a, b: a - b,
          expected_substring='produced a worse solution',
      ),
  )
  def test_arithmetic_operators(self, op, expected_substring):
    agent1 = self.create_agent(agents.StudentAgent)
    agent2_config = agents.AgentConfig(name='agent2', api_key='mock_api_key')
    agent2 = self.create_agent(agents.StudentAgent, config=agent2_config)
    self._add_best_cycle_result(agent2, 0.314159, solution='test_solution')

    result = op(agent1, agent2)
    self.assertIsInstance(result, str)
    self.assertIn(expected_substring, result)
    self.assertIn('with a score of 0.3142', result)  # Formatted to .4f
    self.assertIn('test_solution', result)

    with self.assertRaises(TypeError):
      _ = op(agent1, 'not an agent')

  @parameterized.named_parameters(
      dict(
          testcase_name='initial_cycle',
          current_score=0.1,
          expected_is_better=True,
          initial_best_score=-float('inf'),
      ),
      dict(
          testcase_name='improves',
          current_score=1.0,
          expected_is_better=True,
          initial_best_score=0.5,
      ),
      dict(
          testcase_name='no_improve',
          current_score=0.0,
          expected_is_better=False,
          initial_best_score=0.5,
      ),
      dict(
          testcase_name='none_score',
          current_score=None,
          expected_is_better=False,
          initial_best_score=0.5,
      ),
      dict(
          testcase_name='nan_score',
          current_score=np.nan,
          expected_is_better=False,
          initial_best_score=0.5,
      ),
  )
  def test_update_best_solution(
      self,
      current_score,
      expected_is_better,
      initial_best_score,
  ):
    agent = self.create_agent(agents.StudentAgent)
    agent.best_score = initial_best_score
    agent.best_cycle = 0 if initial_best_score > -float('inf') else -1
    agent.best_solution = 'old_solution'

    current_cycle = 1
    current_solution = 'new_solution'
    is_better = agent.update_best_solution(
        current_score=current_score,
        current_cycle=current_cycle,
        current_solution=current_solution,
    )

    self.assertEqual(is_better, expected_is_better)
    if expected_is_better:
      self.assertEqual(agent.best_cycle, current_cycle)
      self.assertEqual(agent.best_score, current_score)
      self.assertEqual(agent.best_solution, current_solution)
    else:
      self.assertNotEqual(agent.best_cycle, current_cycle)
      self.assertEqual(agent.best_score, initial_best_score)
      self.assertEqual(agent.best_solution, 'old_solution')

  def test_update_best_solution_sequence(self):
    agent = self.create_agent(agents.StudentAgent)
    self.assertTrue(agent.update_best_solution(0.5, 0, 'sol0'))
    self.assertEqual(agent.best_score, 0.5)
    self.assertFalse(agent.update_best_solution(0.4, 1, 'sol1'))
    self.assertEqual(agent.best_score, 0.5)
    self.assertTrue(agent.update_best_solution(0.6, 2, 'sol2'))
    self.assertEqual(agent.best_score, 0.6)
    self.assertEqual(agent.best_cycle, 2)
    self.assertEqual(agent.best_solution, 'sol2')

  @parameterized.named_parameters(
      dict(
          testcase_name='with_best_score',
          score=0.75,
          add_result=True,
          expected_score=0.75,
      ),
      dict(
          testcase_name='none_score',
          score=None,
          add_result=True,
          expected_score=float('-inf'),
      ),
      dict(
          testcase_name='nan_score',
          score=np.nan,
          add_result=True,
          expected_score=float('-inf'),
      ),
      dict(
          testcase_name='empty_agent',
          score=None,
          add_result=False,
          expected_score=float('-inf'),
      ),
  )
  def test_get_best_score(self, score, add_result, expected_score):
    """Tests the get_best_score method with different scenarios."""
    agent = self.create_agent(agents.StudentAgent)
    if add_result:
      self._add_best_cycle_result(agent, score)
    self.assertEqual(agent.get_best_score(), expected_score)

  @parameterized.named_parameters(
      dict(
          testcase_name='eventual_success',
          side_effects=[
              genai.errors.ServerError('Server error', {}),
              mock.Mock(text='print("new")'),
          ],
          max_retries=2,
          expected_result='print("new")',
          expected_calls=2,
          expected_sleeps=1,
      ),
      dict(
          testcase_name='fails_after_retries',
          side_effects=[
              genai.errors.ServerError('Server error', {}),
              genai.errors.ClientError('Client error', {}),
              genai.errors.ServerError('Server error', {}),
          ],
          max_retries=3,
          expected_result=None,
          expected_calls=3,
          expected_sleeps=2,
      ),
  )
  @mock.patch.object(time, 'sleep', autospec=True)
  def test_generate_solution_retries(
      self,
      mock_sleep,
      side_effects,
      max_retries,
      expected_result,
      expected_calls,
      expected_sleeps,
  ):
    config = agents.AgentConfig(
        name=self.agent_name,
        api_key='mock_api_key',
        max_code_generation_retries=max_retries,
        retry_sleep_seconds=5,
    )
    agent = self.create_agent(agents.StudentAgent, config=config)
    self.mock_model_client_instance.models.generate_content.side_effect = (
        side_effects
    )

    if expected_result is not None:
      self.assertEqual(agent.generate_solution('write code'), expected_result)
      self.assertEqual(agent.current_solution, expected_result)
    else:
      with self.assertRaises(agents.GenerationError):
        agent.generate_solution('write code')

    self.assertEqual(
        self.mock_model_client_instance.models.generate_content.call_count,
        expected_calls,
    )
    self.assertEqual(mock_sleep.call_count, expected_sleeps)
    self.assertEqual(agent.generate_code_count, 1)

  def _setup_mutation_agents(self, current_score, better_score, worse_score):
    """Sets up a current agent and mutators for mutation tests."""
    current_agent = self.create_agent(
        agents.StudentAgent,
        config=agents.AgentConfig(name='current', api_key='mock_api_key'),
    )
    if current_score is not None:
      self._add_best_cycle_result(
          current_agent, current_score, 'current_solution'
      )
      # best_solution is updated, but current_solution used as mutate base.
      current_agent.current_solution = current_agent.best_solution
    else:
      current_agent.current_solution = 'initial_current_solution'

    better_agent = self.create_agent(
        agents.StudentAgent,
        config=agents.AgentConfig(name='better', api_key='mock_api_key'),
    )
    self._add_best_cycle_result(better_agent, better_score, 'better_solution')

    worse_agent = self.create_agent(
        agents.StudentAgent,
        config=agents.AgentConfig(name='worse', api_key='mock_api_key'),
    )
    self._add_best_cycle_result(worse_agent, worse_score, 'worse_solution')

    return current_agent, better_agent, worse_agent

  @parameterized.named_parameters(
      dict(
          testcase_name='with_better_peer',
          current_score=0.5,
          expected_solution='better_solution',
      ),
      dict(
          testcase_name='no_better_peer',
          current_score=0.9,
          expected_solution='current_solution',
      ),
      dict(
          testcase_name='self_no_best',
          current_score=None,
          expected_solution='better_solution',
      ),
  )
  def test_mutate_best_peer_replacement(self, current_score, expected_solution):
    current, better, worse = self._setup_mutation_agents(
        current_score, 0.8, 0.2
    )
    current.mutate(
        prompt='test',
        mutators=[better, worse, current],
        strategy=agents.MutationStrategy.BEST_PEER_REPLACEMENT,
    )
    self.assertEqual(current.current_solution, expected_solution)
    self.assertEqual(current.mutation_count, 1)
    self.mock_model_client_instance.models.generate_content.assert_not_called()
    self.assertEqual(current.generate_code_count, 0)

  @parameterized.named_parameters(
      dict(
          testcase_name='self_best_code',
          strategy=agents.MutationStrategy.SELF_BEST_CODE,
          expected_substrings=[
              'Analyze your own best solution and try to improve it.',
              'current_solution',
              'Base Prompt\n',
          ],
      ),
      dict(
          testcase_name='best_peer_crossover',
          strategy=agents.MutationStrategy.BEST_PEER_CROSSOVER,
          expected_substrings=[
              'better produced a better solution',
              'Analyze the better solution and incorporate its techniques',
              'better_solution',
              'current_solution',
          ],
      ),
      dict(
          testcase_name='contrastive_peer_crossover',
          strategy=agents.MutationStrategy.CONTRASTIVE_PEER_CROSSOVER,
          expected_substrings=[
              'Improve your solution using a contrastive approach',
              '**Inspirational Solution**',
              'better_solution',
              '**Solution to Avoid**',
              'worse_solution',
              'current_solution',
          ],
      ),
  )
  @mock.patch.object(agents.StudentAgent, 'generate_solution', autospec=True)
  def test_mutate_strategies(
      self, mock_generate_solution, strategy, expected_substrings
  ):
    current, better, worse = self._setup_mutation_agents(0.5, 0.8, 0.2)
    mock_generate_solution.return_value = 'Mutated result'
    current.mutate(
        prompt='Base Prompt',
        mutators=[better, worse, current],
        strategy=strategy,
    )
    self.assertEqual(current.mutation_count, 1)
    mock_generate_solution.assert_called_once()
    called_prompt = mock_generate_solution.call_args[0][1]
    for substr in expected_substrings:
      self.assertIn(substr, called_prompt)

  @parameterized.named_parameters(
      dict(
          testcase_name='only_better',
          mutator_list_func=lambda c, b, w: [b, c],
          expect_positive='better_solution',
          expect_negative='None',
      ),
      dict(
          testcase_name='only_worse',
          mutator_list_func=lambda c, b, w: [w, c],
          expect_positive='None',
          expect_negative='worse_solution',
      ),
  )
  @mock.patch.object(agents.StudentAgent, 'generate_solution', autospec=True)
  def test_mutate_contrastive_peer_crossover_edges(
      self,
      mock_generate_solution,
      mutator_list_func,
      expect_positive,
      expect_negative,
  ):
    current, better, worse = self._setup_mutation_agents(0.5, 0.8, 0.2)
    mutators = mutator_list_func(current, better, worse)
    current.mutate(
        prompt='Base Prompt',
        mutators=mutators,
        strategy=agents.MutationStrategy.CONTRASTIVE_PEER_CROSSOVER,
    )
    mock_generate_solution.assert_called_once()
    called_prompt = mock_generate_solution.call_args[0][1]
    self.assertIn(expect_positive, called_prompt)
    self.assertIn(expect_negative, called_prompt)

  @parameterized.named_parameters(
      dict(
          testcase_name='no_valid_mutators',
          mutators_func=lambda agent, b, w: [
              agents.StudentAgent(agent.config, agent.run_base_dir)
          ],
          strategy=agents.MutationStrategy.BEST_PEER_CROSSOVER,
          current_score=0.5,
      ),
      dict(
          testcase_name='self_no_best_cycle',
          mutators_func=lambda agent, b, w: [
              agent,
              b,
          ],  # Need a valid mutator to avoid first fallback
          strategy=agents.MutationStrategy.BEST_PEER_CROSSOVER,
          current_score=None,
      ),
      dict(
          testcase_name='no_peers_for_crossover',
          mutators_func=lambda agent, b, w: [agent],
          strategy=agents.MutationStrategy.BEST_PEER_CROSSOVER,
          current_score=0.5,
      ),
      dict(
          testcase_name='no_peers_for_contrastive',
          mutators_func=lambda agent, b, w: [agent],
          strategy=agents.MutationStrategy.CONTRASTIVE_PEER_CROSSOVER,
          current_score=0.5,
      ),
      dict(
          testcase_name='peer_not_better',
          mutators_func=lambda agent, b, w: [b, agent],
          strategy=agents.MutationStrategy.BEST_PEER_CROSSOVER,
          current_score=0.8,
          better_score=0.5,
          unexpected_str='better produced',
      ),
  )
  @mock.patch.object(agents.StudentAgent, 'generate_solution', autospec=True)
  def test_mutate_fallback_to_self_best_code(
      self,
      mock_generate_solution,
      mutators_func,
      strategy,
      current_score,
      better_score=0.8,
      unexpected_str=None,
  ):
    current, better, worse = self._setup_mutation_agents(
        current_score, better_score, 0.2
    )
    mutators = mutators_func(current, better, worse)
    mock_generate_solution.return_value = 'Fallback result'

    current.mutate(
        prompt='Base Prompt',
        mutators=mutators,
        strategy=strategy,
    )

    mock_generate_solution.assert_called_once()
    called_prompt = mock_generate_solution.call_args[0][1]
    self.assertIn(
        'Analyze your own best solution and try to improve it.', called_prompt
    )
    if current_score is None:
      self.assertIn('This is your first attempt', called_prompt)
      self.assertIn('initial_current_solution', called_prompt)
    else:
      self.assertIn('current_solution', called_prompt)
    if unexpected_str:
      self.assertNotIn(unexpected_str, called_prompt)

  def test_mutate_missing_context_key(self):
    current, better, worse = self._setup_mutation_agents(0.5, 0.8, 0.2)
    original_prompt = agents.MutationStrategy.BEST_PEER_CROSSOVER.prompt
    agents.MutationStrategy.BEST_PEER_CROSSOVER.prompt = '{missing_key}'
    try:
      with self.assertRaisesRegex(
          ValueError,
          "Missing context for strategy BEST_PEER_CROSSOVER: 'missing_key'",
      ):
        current.mutate(
            prompt='Base',
            mutators=[
                current,
                better,
                worse,
            ],  # Ensure it tries to use BEST_PEER_CROSSOVER
            strategy=agents.MutationStrategy.BEST_PEER_CROSSOVER,
        )
    finally:
      # Restore original prompt
      agents.MutationStrategy.BEST_PEER_CROSSOVER.prompt = original_prompt


class AnalyzeBestSolutionTest(AgentsTestBase):

  def test_analyze_best_solution(self):
    # Mock LLM response text returning valid metadata JSON
    json_response = """
    {
      "model_tags": "ridge, pipeline",
      "model_description": "A simple Ridge regression model.",
      "model_insights": "BMI has high feature importance."
    }
    """
    self.set_mock_response(json_response)

    analyzer_cfg = agents.AgentConfig(
        name='solution_metadata',
        api_key='mock_api_key',
        model_name=agent_config.DEFAULT_MODEL,
    )
    analyzer = agents.SolutionAnalysisAgent(
        config=analyzer_cfg, run_base_dir=self.base_dir
    )

    res = analyzer.analyze(
        student_name='student_1',
        best_code='def solution(X, Y, X_val): return None',
        best_plan='Use Ridge.',
        best_test_metrics={'metrics_r2': 0.8},
        best_val_metrics={'metrics_r2': 0.75},
        student_instructions='Be standard.',
        metrics_description='We weight R2.',
        task_description_for_prompt='the HOMA-IR predicting task',
    )

    self.assertEqual(res['model_tags'], 'ridge, pipeline')
    self.assertEqual(
        res['model_description'], 'A simple Ridge regression model.'
    )
    self.assertEqual(res['model_insights'], 'BMI has high feature importance.')

    # Ensure the result is saved in best_analysis.json in self.base_dir
    expected_analysis_path = os.path.join(self.base_dir, 'best_analysis.json')
    self.assert_file_exists(expected_analysis_path)

  @parameterized.named_parameters(
      dict(
          testcase_name='json_parse_error_exhausted',
          side_effects=['bad json 1', 'bad json 2'],
          expected_exception=ValueError,
      ),
      dict(
          testcase_name='genai_api_error_exhausted',
          side_effects=[
              genai.errors.ServerError('Server error', {}),
              genai.errors.ClientError('Client error', {}),
          ],
          expected_exception=ValueError,
      ),
      dict(
          testcase_name='api_error_then_success',
          side_effects=[
              genai.errors.ServerError('Server error', {}),
              '{"model_tags": "success"}',
          ],
          expected_exception=None,
      ),
      dict(
          testcase_name='api_error_then_success_default_retries',
          side_effects=[
              genai.errors.ClientError('Client error', {}),
              '{"model_tags": "success"}',
          ],
          expected_exception=None,
          use_default_retries=True,
      ),
      dict(
          testcase_name='json_parse_error_then_success',
          side_effects=[
              'not valid json',
              '{"model_tags": "success"}',
          ],
          expected_exception=None,
      ),
  )
  @mock.patch.object(time, 'sleep', autospec=True)
  def test_analyze_retries_and_failures(
      self,
      mock_sleep,
      side_effects,
      expected_exception,
      use_default_retries=False,
  ):
    """Verifies SolutionAnalysisAgent iteratively retries and fails or succeeds properly."""
    side_effect_list = []
    for effect in side_effects:
      if isinstance(effect, str):
        m = mock.Mock()
        m.text = effect
        side_effect_list.append(m)
      else:
        side_effect_list.append(effect)

    self.mock_model_client_instance.models.generate_content.side_effect = (
        side_effect_list
    )

    config = agents.AgentConfig(
        name='solution_metadata',
        api_key='mock_api_key',
        max_code_generation_retries=len(side_effects),
    )
    analyzer = agents.SolutionAnalysisAgent(
        config=config, run_base_dir=self.base_dir
    )

    analyze_kwargs = dict(
        student_name='student_1',
        best_code='def test(): pass',
        best_plan='Testing retry',
        best_test_metrics={},
        best_val_metrics={},
        student_instructions='instructions',
        metrics_description='metrics',
    )
    if not use_default_retries:
      analyze_kwargs['max_retries'] = len(side_effects)

    if expected_exception:
      with self.assertRaises(expected_exception):
        analyzer.analyze(**analyze_kwargs)
    else:
      res = analyzer.analyze(**analyze_kwargs)
      self.assertEqual(res['model_tags'], 'success')

      if 'not valid json' in side_effects:
        call_args_2 = (
            self.mock_model_client_instance.models.generate_content.call_args_list[
                1
            ][
                1
            ][
                'contents'
            ][
                -1
            ]
            .parts[0]
            .text
        )
        self.assertIn('Failed to parse your response as JSON', call_args_2)

    self.assertLen(
        side_effects,
        self.mock_model_client_instance.models.generate_content.call_count,
    )

    expected_sleeps = sum(
        1 for e in side_effects[:-1] if isinstance(e, Exception)
    )
    self.assertEqual(mock_sleep.call_count, expected_sleeps)


class AgentConfigTest(parameterized.TestCase):

  def test_load_env_with_custom_path(self):
    env_file = self.create_tempfile(content='GEMINI_API_KEY=custom_path_key\n')
    with mock.patch.dict(os.environ, {}, clear=True):
      agent_config.load_env(env_path=env_file.full_path)
      self.assertEqual(os.environ.get('GEMINI_API_KEY'), 'custom_path_key')


if __name__ == '__main__':
  absltest.main()
