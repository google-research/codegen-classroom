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

"""Codegen Classroom SDK."""

from codegen_classroom import agent_config
from codegen_classroom import agents
from codegen_classroom import code_execution
from codegen_classroom import data_utils
from codegen_classroom import metrics
from codegen_classroom import plotting
from codegen_classroom import reporting
from codegen_classroom import scoring
from codegen_classroom import student_learning

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "agents",
    "code_execution",
    "data_utils",
    "agent_config",
    "metrics",
    "plotting",
    "reporting",
    "scoring",
    "student_learning",
]
