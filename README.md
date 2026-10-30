# Codegen Classroom 🎓

### A framework for iterative code generation for ML and data analysis tasks using parallel LLM agents.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-research/codegen-classroom/blob/main/codegen_classroom/tasks/template/task_example.ipynb)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Gemini API](https://img.shields.io/badge/Gemini-API-8E75B2.svg)](https://ai.google.dev/gemini-api/docs)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub Stars](https://img.shields.io/github/stars/google-research/codegen-classroom?style=social)](https://github.com/google-research/codegen-classroom/stargazers)

**Authors:** [Jake Garrison](mailto:jakegarrison@google.com),
[Zhihan Zhang](mailto:zzhihan@cs.washington.edu),
[Deon Kouatchou](mailto:deonkouatchou@google.com),
[Girish Narayanswamy](mailto:girishvn@google.com)

**Created:** August 2025

--------------------------------------------------------------------------------

## Overview

Codegen Classroom **automates code generation** for tasks that can be scored
quantitatively (e.g., regression, classification, signal processing) using
**parallel [Gemini](https://ai.google.dev/gemini-api/docs/models) and
[Gemma](https://ai.google.dev/gemma) agents**.

Instead of single-shot prompting, multiple student agents **iteratively write,
execute, debug, and refine** Python solutions in a sandboxed **recursive
self-improvement (RSI)** loop. Agents learn from execution feedback (validation
scores, runtime errors, diagnostic metrics) and **collaborate via peer
crossover** to share high-performing ideas and escape local optima.

Whether you define the data splits and scoring metrics directly or prompt an
agent to set up the task, the classroom handles implementation and iterative
optimization.

--------------------------------------------------------------------------------

## Installation

Requires Python 3.9+ and a [`GEMINI_API_KEY`](https://ai.google.dev/gemini-api/docs/api-key):

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install from PyPI (or editable local clone with `pip install -e .`)
pip install codegen-classroom

# Optional extras:
pip install "codegen-classroom[lightgbm]"  # LightGBM
pip install "codegen-classroom[notebook]"  # Plotly and IPython for notebooks
pip install "codegen-classroom[torch]"     # PyTorch and TorchVision
pip install "codegen-classroom[jax]"       # JAX
pip install "codegen-classroom[all]"       # All optional dependencies and pytest

# 3. Configure GEMINI_API_KEY
export GEMINI_API_KEY="your-api-key"       # Or copy .env.example to .env
```

--------------------------------------------------------------------------------

## Quickstart

Run an iterative classroom loop on any custom `pandas` dataset:

```python
import os
from codegen_classroom import agents
from codegen_classroom import student_learning
import pandas as pd

# 1. Prepare training, validation, and held-out test splits
x_train = pd.DataFrame({"x1": [1.0, 2.0, 3.0, 4.0], "x2": [10.0, 20.0, 30.0, 40.0]})
y_train = pd.DataFrame({"target": [2.1, 4.2, 5.9, 8.1]})
x_val = pd.DataFrame({"x1": [5.0, 6.0], "x2": [50.0, 60.0]})
y_val = pd.DataFrame({"target": [10.2, 12.1]})
x_test = pd.DataFrame({"x1": [7.0], "x2": [70.0]})
y_test = pd.DataFrame({"target": [14.0]})

# 2. Configure student agents
student_configs = student_learning.get_classroom_student_configs(
    system_instruction="You are an expert ML engineer.",
    model_names=["gemini-3.5-flash"],
    api_key=os.environ.get("GEMINI_API_KEY"),
)
classroom = [agents.StudentAgent(cfg) for cfg in student_configs]

# 3. Run iterative learning cycles
prompt = """
Write a python function `solution(x_train, y_train, x_val)` that trains a regression
model on (x_train, y_train) and returns predictions on x_val as a pandas DataFrame with column 'target'.
"""
for cycle in range(3):
  student_learning.run_parallel_classroom_learning_cycle(
      current_classroom=classroom,
      x_train=x_train,
      y_train=y_train,
      x_val=x_val,
      y_val=y_val,
      current_cycle=cycle,
      student_initial_prompt=prompt,
      student_function_name="solution",
      metric_weights={"r2": 1.0},
      early_stopping_patience=2,
      student_refine_instruction_options=["Consider feature scaling or Ridge regression."],
      max_mutation_count=2,
      mutation_strategy_options=[agents.MutationStrategy.SELF_BEST_CODE],
      code_execution_max_retries=1,
  )

# 4. Final evaluation on held-out test data
eval_results, _ = student_learning.run_parallel_classroom_final_evaluation(
    students={s.name: s for s in classroom},
    x_learning=x_train,
    y_learning=y_train,
    x_test_final=x_test,
    y_test_final=y_test,
    student_function_name="solution",
    metric_weights={"r2": 1.0},
)
print("Final evaluation completed:", eval_results)
```

--------------------------------------------------------------------------------

## Tasks

Self-contained benchmark suites live under
[`codegen_classroom/tasks/<task>/`](./codegen_classroom/tasks/). Each task
includes a shared configuration (`config.py`), dataset loader
(`data_loader.py`), headless runner (`run_<task>.py`), unit tests
(`run_<task>_test.py`), companion notebook (`<task>.ipynb`), `leaderboard.tsv`,
and archived top solutions (`solutions/`).

| Task | Summary | Notebook |
| :--- | :--- | :---: |
| [`template`](./codegen_classroom/tasks/template/) | Synthetic multi-regime thermal load regression template (`N = 1,500`, 10 features) scored on R², MAE, and runtime. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-research/codegen-classroom/blob/main/codegen_classroom/tasks/template/task_example.ipynb) |

```bash
# Run a benchmark task headlessly:
python -m codegen_classroom.tasks.template.run_template

# Verify a single task or all tasks:
python -m codegen_classroom.verify_task --task template --strict
python -m codegen_classroom.verify_task --all
```

A normal run (without `--mock_students`) ends by running the held-out final
evaluation, writing `report.md` and saved predictions to the run directory,
appending a row to the task's `leaderboard.tsv`, and exporting the top runs to
`solutions/`. `--report_only=<run_dir|latest>` only rebuilds `report.md` for an
existing run and never re-executes student code.

To add a new benchmark task, copy
[`tasks/template/`](./codegen_classroom/tasks/template/) or prompt a coding
agent with [`tasks/AGENTS.md`](./codegen_classroom/tasks/AGENTS.md) to scaffold
one end-to-end.

--------------------------------------------------------------------------------

## Classroom Learning Method

Each RSI cycle evolves parallel **Student Agents** across four stages:

1.  **Generate, Self-Debug, and Execute**: Each student writes a Python
    `solution(x_train, y_train, x_val)` function executed in a timed sandbox,
    automatically repairing runtime or output contract errors before returning
    predictions and diagnostic `feedback` (e.g., CV scores or feature
    importances).
2.  **Multi-Objective Scoring**: Predictions are evaluated against weighted task
    metrics (e.g., R², MAPE, Macro-F1) and execution time, updating each
    student's `best_score` and `best_solution`.
3.  **Guided Refinement and Peer Crossover**: Students refine their code using
    domain prompts and self-logged `feedback`. After `max_mutation_count`
    stagnant cycles, a student applies a **Mutation Strategy**:
    *   `SELF_BEST_CODE`: Re-anchors on the student's own best solution.
    *   `BEST_PEER_CROSSOVER` / `BEST_PEER_REPLACEMENT`: Adopts feature
        engineering and modeling ideas from top-scoring peers.
    *   `CONTRASTIVE_PEER_CROSSOVER`: Contrasts top peer solutions against
        weaker attempts to combine winning patterns and avoid failure modes.
4.  **Final Evaluation and Analysis**: Each student's `first`, `last`, and
    `best` validation solutions are retrained on the full learning split and
    evaluated once on the held-out test set, while `SolutionAnalysisAgent`
    synthesizes results into `report.md` and `leaderboard.tsv`.

--------------------------------------------------------------------------------

## Library Structure

*   [agent_config.py](./codegen_classroom/agent_config.py): Default model
    configurations, generation parameters, environment resolution, and execution
    timeouts.
*   [agents.py](./codegen_classroom/agents.py): `StudentAgent` and
    `SolutionAnalysisAgent` state management, refinement prompts, and peer
    mutation strategies.
*   [code_execution.py](./codegen_classroom/code_execution.py): Python code
    extraction and timed subprocess execution capturing `stdout`, `stderr`, and
    return values.
*   [data_utils.py](./codegen_classroom/data_utils.py): Data loading, file
    operations (`etils.epath`), run ID generation, and `leaderboard.tsv`
    tracking.
*   [metrics.py](./codegen_classroom/metrics.py): Performance metrics (MAE, R²,
    Pearson, Macro-F1, Worst-Group Accuracy) and score normalization.
*   [plotting.py](./codegen_classroom/plotting.py): Progress and final
    evaluation plots via Matplotlib (headless PNG export) or Plotly (notebooks).
*   [reporting.py](./codegen_classroom/reporting.py): Structured logging,
    Markdown table formatting, and report export (`report.md`).
*   [scoring.py](./codegen_classroom/scoring.py): Prediction validation (shape,
    index, NaN checks) and weighted composite scoring.
*   [student_learning.py](./codegen_classroom/student_learning.py): Parallel
    classroom cycle orchestration and final test evaluation across `first`,
    `last`, and `best` strategies.
*   [verify_task.py](./codegen_classroom/verify_task.py): Automated structural,
    notebook, solution report, and prompt hygiene verifier.
*   [tasks/](./codegen_classroom/tasks/): Benchmark task suites,
    [Task Authoring Guide](./codegen_classroom/tasks/AGENTS.md), and
    [canonical template](./codegen_classroom/tasks/template/).

--------------------------------------------------------------------------------

## Disclaimer

**This is not an officially supported Google product.**

The agents powered by Gemini are experimental and may generate inaccurate,
hallucinated, or unoptimized code. Always execute the agent loop inside a
sandboxed environment to prevent unexpected side effects from code execution.
