# Codegen Classroom 🎓

### A framework for iterative code generation for ML and data analysis tasks
### using parallel LLM agents in Colab.

--------------------------------------------------------------------------------

## Overview

Codegen Classroom **automates code generation** for tasks that can be scored
(e.g., regression, classification, signal processing) using **parallel
[Gemini](https://ai.google.dev/gemini-api/docs/models) agents** in Colab. The
library operates as an **autonomous LLM agent loop**; it
evaluates the output of a language model iteratively. It is best suited for
tasks where solution quality can be **scored numerically**. Agents
**iteratively write, test, and refine** solutions in a learning loop where code
is **executed and scored** against quantitative metrics.

Agents **improve based on performance** feedback (scores, errors) and can
**collaborate** with other agents to share insights and escape local optima.

This framework elevates the human role from **developer to supervisor**. You
define the **task, data, constraints, and scoring** criteria, while agents
perform the **implementation and iterative refinement**.

--------------------------------------------------------------------------------

## Getting Started

### Prerequisites

*   Python 3.9 or higher
*   Google Colab or Jupyter Notebook environment

### Installation & Setup

We recommend using a virtual environment to avoid dependency conflicts.

1.  **Create a virtual environment**:
    ```bash
    python3 -m venv .venv
    ```

2.  **Activate the virtual environment**:
    *   On Linux/macOS:
        ```bash
        source .venv/bin/activate
        ```
    *   On Windows:
        ```bash
        .venv\Scripts\activate
        ```

3.  **Install dependencies in editable mode**:
    ```bash
    # Standard installation (core library - matplotlib included)
    pip install -e .

    # For interactive notebooks (Plotly and IPython):
    pip install -e ".[notebook]"  # or pip install codegen_classroom[notebook]

    # With LightGBM (recommended for tabular tasks):
    pip install -e ".[lightgbm]"

    # With PyTorch:
    pip install -e ".[torch]"

    # With JAX:
    pip install -e ".[jax]"

    # All dependencies (dev tools, torch, jax, lightgbm):
    pip install -e ".[all]"
    ```
    *Note*: `matplotlib` is a core dependency, while `plotly` and `ipython` live
    in the `[notebook]` extra (`pip install codegen_classroom[notebook]`).
    `lightgbm` is the approved gradient boosting framework; `xgboost` is not
    supported.

4.  **Configure Environment Variables**:
    Copy `.env.example` to `.env` and fill in your `GEMINI_API_KEY`.
    ```bash
    cp .env.example .env
    ```

5.  **Run Tests**:
    Verify the local installation by running the test suite:
    ```bash
    pytest
    ```

6.  **Set up the Colab/Jupyter Kernel** (optional):
    ```bash
    pip install ipykernel
    python -m ipykernel install --user --name=codegen-classroom --display-name "Python (Codegen Classroom)"
    ```

--------------------------------------------------------------------------------

## Quickstart

The core library provides a modular agent loop that can be imported and
customized for any scorable machine learning or data science task.

```python
import os
from codegen_classroom import agent_config
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
    model_names=["gemini-2.5-flash"],
    api_key=os.environ.get("GEMINI_API_KEY"),
)
classroom = [agents.StudentAgent(cfg) for cfg in student_configs]

# 3. Run iterative learning cycles
prompt = """
Write a python function `solution(x_train, y_train, x_val)` that trains a regression
model on (x_train, y_train) and returns predictions on x_val as a pandas DataFrame with column 'target'.
"""
for cycle in range(3):
  results = student_learning.run_parallel_classroom_learning_cycle(
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

## Library Structure

The core library is organized into the following modules:

*   [agent_config.py](./codegen_classroom/agent_config.py):
    Global configuration defaults for models, generation parameters, and
    execution timeouts.
*   [agents.py](./codegen_classroom/agents.py): Defines the `StudentAgent`,
    including **state management** (`best_score`, `best_solution`),
    **refinement logic**, and **mutation strategies** (self-reflection, peer
    crossover).
*   [code_execution.py](./codegen_classroom/code_execution.py): Handles
    **extraction of Python code** blocks from LLM responses and **executes the
    code** in a sandboxed environment, capturing `stdout`, `stderr`, and
    results.
*   [data_utils.py](./codegen_classroom/data_utils.py): Helpers for **data
    loading**, generic file operations via `etils.epath`, **run ID
    generation**, and local leaderboard tracking.
*   [metrics.py](./codegen_classroom/metrics.py): Contains definitions for
    individual **performance metrics** (e.g., MAE, R², Pearson, Worst-Group
    Accuracy), including computation and **normalization logic**.
*   [plotting.py](./codegen_classroom/plotting.py): Progress and final
    evaluation charts using Matplotlib (headless image export) or Plotly
    (interactive notebooks).
*   [reporting.py](./codegen_classroom/reporting.py): Centralized logging setup,
    markdown table rendering with configurable decimal precision,
    `ReportSection` structured models, and dual sinks: interactive inline
    display and disk export (`report.md`).
*   [scoring.py](./codegen_classroom/scoring.py): Implements **solution
    validation** (e.g., checking prediction shapes, NaNs) and calculates the
    overall **weighted score** based on metrics.
*   [student_learning.py](./codegen_classroom/student_learning.py):
    **Orchestrates** the entire agent learning process. Contains the main
    `run_learning_cycle` function and helpers to **run agents in parallel** and
    perform final evaluations across strategies (`first`, `last`, `best`).

--------------------------------------------------------------------------------

## Disclaimer

**This is not an officially supported Google product.**

The agents powered by Gemini are experimental and may generate inaccurate,
hallucinated, or unoptimized code. Always execute the agent loop inside a
sandboxed environment (like Google Colab) to prevent unexpected side effects
from code execution.
