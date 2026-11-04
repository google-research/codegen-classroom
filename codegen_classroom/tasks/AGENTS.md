# Agent Guide: Authoring Codegen Classroom Benchmark Tasks

Use `tasks/template/` as the executable blueprint when creating or updating a
benchmark task in `codegen_classroom`. Every task must be self-contained inside
`tasks/<task>/` without modifying core library files (`codegen_classroom/*.py`).

## Task Structure

Keep each task directory minimal and self-contained:

-   `__init__.py`: Package marker.
-   `README.md`: Problem overview, dataset source/citation, evaluation metrics,
    and portable `python3 -m` / `pytest` usage commands using standard sections
    (`## Overview`, `## Files`, `## Dataset`, `## Scoring`, `## Configuration`,
    `## Results`, `## Quickstart`).
-   `config.py`: Shared task configuration (`RUN_DEFAULTS`,
    `DEFAULT_STUDENT_MODELS`, `DEFAULT_METRIC_WEIGHTS`) and student prompts
    (`STUDENT_INSTRUCTIONS_TEMPLATE`, `STUDENT_INITIAL_PROMPT`,
    `STUDENT_REFINE_INSTRUCTION_OPTIONS`, `build_student_instructions()`)
    imported by both `run_<task>.py` and `<task>.ipynb`.
-   `data_loader.py`: Dataset download/generation (`ensure_data()`),
    preprocessing, and train/validation/test split helpers.
-   `run_<task>.py`: Headless runner importing settings and prompts from
    `config.py`, executing the learning loop, submitting to the leaderboard, and
    exporting `solutions/`.
-   `run_<task>_test.py`: Fast, hermetic unit tests verifying data shapes, split
    invariants, student prompt sections, and `_MockStudent`.
-   `<task>.ipynb`: Companion Jupyter/Colab notebook importing settings and
    prompts from `config.py`, with all cell outputs cleared before committing.
-   `leaderboard.tsv` (or `leaderboards/<variant>/leaderboard.tsv` for
    multi-variant suites): Tab-separated evaluation log recording performance,
    `total_tokens`, and `est_cost_usd` (populated automatically by
    `data_utils.submit_results_to_local_leaderboard()`).
-   `solutions/`: Curated top-performing solution reports (`<solution_id>.md`)
    and progress plots (`plots/<solution_id>.png`). Solution IDs must start
    with a letter (e.g. `<task>_<descriptor>_<YYYYMMDD>`), place dates at the
    end, and never contain dots or start with a digit.

## Conventions

### Data and Artifacts

-   **Never commit raw datasets or binary weights**: Store downloaded archives
    (`.zip`, `.tar.gz`), arrays (`.npy`, `.npz`, `.parquet`), tabular datasets
    (`.csv`), audio/images (`.wav`), and model weights (`.pt`, `.pth`, `.pkl`,
    `.h5`) strictly inside gitignored `data/`, `cache/`, `models/`, or
    `experiments/` directories (or external cache directories symlinked into
    `data/`).
-   **Leakage-free splits**: Hold out a final test split (`X_test`, `y_test`)
    that student agents never access during iterative learning cycles. Use group
    or chronological splits when observations share subjects or time series.
-   **Experiment routing**: Write per-run logs and intermediate checkpoints to
    `CODEGEN_EXPERIMENTS_DIR` (defaulting to
    `<tempdir>/codegen_classroom/<task>`), never inside the tracked task
    directory.

### Student Instructions

Render student system instructions in `config.py` using the 8-section structure
demonstrated in `tasks/template/config.py` (keep section headings and bullet
labels concise, 1–3 words):

1.  `1. Objective` (task goal and exact metric weights)
2.  `2. Data Contract` (input DataFrames, feature types, `(y_pred_df, feedback)`
    return signature)
3.  `3. Evaluation` (per-cycle and final evaluation timeouts, validation
    protocol)
4.  `4. Environment` (importable packages, single-threaded execution limits)
5.  `5. Principles` (domain modeling guidance and inductive biases—no hardcoded
    hyperparameter recipes or repetitive phrasing)
6.  `6. Pitfalls` (index alignment, data leakage, degenerate predictions)
7.  `7. Iteration Discipline` (fast cycle-0 baseline, single-axis refinement,
    experiment logging via `feedback`)
8.  `8. Output Contract` (pre-return verification checklist)

### Notebooks, Runners, and Markdown

-   **Notebooks**: Start setup in `<task>.ipynb` with a `# @title Install` cell
    (`try` / `except ImportError:`
    `!pip install -q "codegen-classroom[notebook]"`), and only place Colab
    `# @param` annotations on literal values (never on
    `config.RUN_DEFAULTS[...]` expressions).
-   **Hermetic `_MockStudent`**: In `run_<task>.py`, `_MockStudent` should
    default `config.api_key = "mock_key"` and
    `run_base_dir = tempfile.gettempdir()`, and skip writing "mock" runs to
    `leaderboard.tsv` or `solutions/`.
-   **Markdown Tables**: Use single-line GitHub Flavored Markdown pipe tables
    (`| col | col |`) in `README.md` and `solutions/*.md`.

### Concurrency and Thread Limits

-   Call `agent_config.load_env()` before running classroom cycles to load API
    keys and cap worker BLAS/OpenMP thread pools.
-   Avoid importing heavy native/GPU frameworks (e.g., `torch`, `jax`) in the
    parent process prior to subprocess forking.
-   Set `final_eval_timeout_s` to 2x–3x `timeout_s` so final evaluation on the
    combined learning split does not time out.

## Creating a New Task

1.  Copy `tasks/template/` to `tasks/<new_task>/` and replace all `# CUSTOMIZE:`
    placeholders.
2.  Implement `data_loader.py` and verify splits with `pytest
    codegen_classroom/tasks/<new_task>/run_<new_task>_test.py`.
3.  Run a headless classroom benchmark to populate `leaderboard.tsv` and
    `solutions/`:

    ```bash
    python -m codegen_classroom.tasks.<new_task>.run_<new_task>
    ```
4.  Run the task verifier and ensure all checks pass with zero warnings:

    ```bash
    python -m codegen_classroom.verify_task --task <new_task> --strict
    ```
