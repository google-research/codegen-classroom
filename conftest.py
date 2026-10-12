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

"""Pytest conftest to initialize absl flags for absltest compatibility."""

from absl import flags


def pytest_configure(config):
  # Parse absl flags so that absltest.TestCase.create_tempdir() works
  # when tests are run via pytest instead of absltest.main().
  if not flags.FLAGS.is_parsed():
    flags.FLAGS([''])
