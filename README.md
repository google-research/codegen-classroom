# Codegen Classroom 🎓 (PyHealth fork)

### Iterative code generation for ML tasks with parallel LLM agents, run as plain Python scripts.

--------------------------------------------------------------------------------

## Overview

Codegen Classroom **automates code generation** for tasks that can be scored
numerically. A "classroom" of LLM **students** writes solutions, which are
**executed and scored**; students then **refine** their code from scores,
tracebacks and execution output, and **learn from peers** (mutation /
crossover) when they stall.

This fork adds:

*   **PyHealth training environments**: students write PyTorch / PyHealth model
    code for a clinical dataset + task (e.g. MIMIC-IV mortality). Each attempt
    trains in a fresh subprocess on an assigned GPU and is scored with
    AUROC/AUPRC (or accuracy/F1 for multiclass) on a held-out split.
*   **Local LLM inference**: students can be served by a local **vLLM** server
    (any OpenAI-compatible endpoint) as well as the Gemini API, including mixed
    classrooms.
*   **No notebooks**: experiments run from `scripts/run_classroom.py` with a
    YAML config; logs, results, `report.md` and plots go to a run directory.

--------------------------------------------------------------------------------

## Setup

Two separate environments are required. vLLM 0.31 ships torch 2.13 (CUDA 13),
which conflicts with PyHealth's torch 2.7.1, so the **LLM server** and the
**classroom** (which trains PyHealth models) must never share an env.

```bash
git clone https://github.com/jhnwu3/codegen-classroom-pyhealth.git
cd codegen-classroom-pyhealth

# 1. Classroom env: Python 3.12 with PyHealth installed (editable checkout,
#    see "Reproducing our results" for the exact commit).
conda activate PyHealthAgent
python -m pip install -e ".[repro,cli]"   # exact pinned versions we used
#   or, unpinned:  python -m pip install -e ".[openai,cli,dev,pyhealth]"

# 2. LLM server env (separate!)
conda create -n vllm -c conda-forge --override-channels python=3.12 -y
TMPDIR=$HOME/pip_tmp ~/miniconda3/envs/vllm/bin/python -m pip install \
    vllm==0.31.0 bitsandbytes==0.50.2
```

Pitfalls we hit on the lab machines:

*   Call each env's interpreter explicitly (`<env>/bin/python -m pip`).
    `conda run -n vllm pip install vllm` resolved `pip` from another env and
    installed vLLM's dependencies into the classroom env, breaking PyHealth's
    torch/triton/pydantic.
*   `/tmp` is only 8 GB on the sunlab servers; point `TMPDIR` at a larger disk
    when installing vLLM (multi-GB CUDA wheels) or pip fails with
    `No space left on device`.
*   The lab conda has no default channel: use `-c conda-forge --override-channels`.
*   PyHealth pins `pydantic~=2.11.7`; newer `google-genai` releases need
    pydantic>=2.12, so the `repro` extra pins `google-genai==2.8.0`.

Optional: copy `.env.example` to `.env` to set `GEMINI_API_KEY`,
`CODEGEN_BACKEND` or `VLLM_BASE_URL`.

Run the tests (366 tests, ~2 min, no GPU or LLM needed) from the repository
root: `CUDA_VISIBLE_DEVICES="" python -m pytest -q`.

--------------------------------------------------------------------------------

## Running

1.  **Start the LLM server** (with the `vllm` env's binaries on `PATH`, on
    the GPUs you reserve for it; exact recipes per machine are in
    [Reproducing our results](#reproducing-our-results)):

    ```bash
    PATH=~/miniconda3/envs/vllm/bin:$PATH MODEL=<hf model id> GPUS=0 \
        bash scripts/serve_vllm.sh
    curl http://localhost:8000/v1/models      # check it is up
    ```

2.  **Smoke test** the pipeline on a toy regression task (no PyHealth data):

    ```bash
    python scripts/run_classroom.py --config configs/regression_toy.yaml
    ```

3.  **Run a PyHealth environment**:

    ```bash
    python scripts/run_classroom.py --list-envs
    python scripts/run_classroom.py --config configs/mimic4_mortality_vllm.yaml

    # Quick run on the MIMIC-IV dev subset with 2 cycles on GPU 1 only
    python scripts/run_classroom.py --config configs/mimic4_mortality_vllm.yaml \
        --override run.num_cycles=2 'task.gpus=[1]' \
        task.overrides.dataset_kwargs.dev=true
    ```

    `--override key.path=value` accepts any config field (values are parsed as
    YAML); `--print-config` shows the resolved config.
    `configs/run_classroom.sub` is an HTCondor submit example.

Each run writes `<run.output_dir>/<name>_<timestamp>/` containing
`config.yaml`, `run.log`, per-student cycle CSVs (`students/`), every
execution's code, stdout/stderr and predictions (`executions/`),
`final_eval_results.json`, `best_analysis.json`, `report.md` and plots.
The first run of an environment also prepares and caches the PyHealth
SampleDataset and patient-level splits under `envs/<env>/`.

--------------------------------------------------------------------------------

## Reproducing our results

### Software

| Component | Version used |
|---|---|
| Classroom env | Python 3.12.12, `pip install -e ".[repro,cli]"` (all pins in `pyproject.toml`) |
| PyHealth | editable install of [evanfebrianto/PyHealth](https://github.com/evanfebrianto/PyHealth) branch `feat/ehrmamba`, commit `c232681` (reports 2.0.1). The checkout had local edits only in FHIR-related files and temp-dir cleanup, none on the code paths used here |
| vLLM env | Python 3.12.15, vllm 0.31.0, torch 2.13.0 (CUDA 13), bitsandbytes 0.50.2 |
| NVIDIA driver | 595.71 (CUDA 13.2) on sunlab-serv-02 |

```bash
git clone https://github.com/evanfebrianto/PyHealth.git && cd PyHealth
git checkout c232681 && python -m pip install -e .   # into the classroom env
```

### Data

MIMIC-IV v2.2 mortality (`pyhealth.tasks.MortalityPredictionMIMIC4`, tables
patients, admissions, diagnoses_icd, procedures_icd, prescriptions). We reuse
an existing PyHealth task cache instead of rebuilding it:

```
/shared/eng/pyhealth_agent/cache/b77effea-2e99-5b1b-90c3-0fa6d2540af1/tasks/
  MortalityPredictionMIMIC4_187a839c-4f67-585a-bbb3-355429e27594/
  samples_cdbbc602-34e2-5a41-8643-4c76b08829f6.ld
```

128,236 visits from 57,701 patients; inputs `conditions` (vocab 18,356),
`procedures` (9,781), `drugs` (6,071); ~2.3% positive. The env splits patients
80/10/10 with seed 42 (102,276 / 12,990 / 12,970 samples) and caps learning
cycles at 20,000 training samples. Without the cache, set
`task.overrides.dataset_kwargs.ehr_root` to a MIMIC-IV 2.2 root and drop
`sample_dataset_path` to rebuild it (slow, and writes to `cache_dir`).

### LLM checkpoints

| Model | HF revision | Size | Where it fits |
|---|---|---|---|
| [QuantTrio/Qwen3.5-9B-AWQ](https://huggingface.co/QuantTrio/Qwen3.5-9B-AWQ) | `938f8e3` | 12.4 GB | 2× RTX 2080 Ti (TP=2) |
| [unsloth/Qwen3.8-27B-unsloth-bnb-4bit](https://huggingface.co/unsloth/Qwen3.8-27B-unsloth-bnb-4bit) | `8aa5f05` | 21 GB | one free A6000 (~25-30 GB); not yet run |

Download: `hf download <model> --revision <rev>`.

### Run on 8× RTX 2080 Ti (sunlab-serv-02)

```bash
# Terminal 1: LLM on GPUs 0-1 (startup ~6 min, mostly a ~5 min warmup)
PATH=~/miniconda3/envs/vllm/bin:$PATH MODEL=QuantTrio/Qwen3.5-9B-AWQ \
  GPUS=0,1 TP=2 MAX_MODEL_LEN=32768 \
  EXTRA_ARGS='--dtype half --max-num-seqs 16 --limit-mm-per-prompt {"image":0,"video":0}' \
  bash scripts/serve_vllm.sh

# Terminal 2: classroom, students train on GPUs 2-3
conda activate PyHealthAgent
python scripts/run_classroom.py --config configs/mimic4_mortality_qwen9b_2080ti.yaml
```

Pick any two idle GPUs for the server (`GPUS=...`) and point `task.gpus` at
others (e.g. `--override 'task.gpus=[0, 1]'`). Student models need only 1-3 GB,
so they can share a card with other light jobs; the server needs whole cards
(`GPU_MEM_UTIL=0.90` of 11 GB). The first start on a machine spends ~5 min on
warmup; later starts reuse the compile cache in `~/.cache/vllm` and are ready
in about a minute.

Turing (sm75) specifics: no bf16 (`--dtype half`); the default
`--max-num-seqs 256` exceeds Qwen3.5's linear-attention cache blocks and vLLM
refuses to start; a 16k context overflowed with within-cycle retry history
(use 32k: the KV cache holds ~155k tokens); thinking is disabled in the config
to save context. Throughput is ~70-90 tokens/s per request.

### Run on A6000s (sunlab-serv-03)

Use `configs/mimic4_mortality_cached_qwen.yaml` (4-bit Qwen3.8-27B) with one
free A6000 for the server; set `task.gpus` to the student GPUs. On
2026-10-16 every GPU there had only ~7 GB free (resident SGLang servers), so
this configuration has not been run yet.

### Reference results (held-out test split, MIMIC-IV mortality)

Hand-written templates through the same runner (no LLM), full training split,
5 epochs, RTX A6000:

| Solution | AUROC | AUPRC | Train time |
|---|---|---|---|
| Template A: PyHealth `RNN` + `Trainer` (the seed solution) | 0.781 | 0.088 | 241 s |
| Template B: custom `BaseModel` (bag-of-codes MLP) | 0.774 | 0.078 | 91 s |

LLM classroom runs (`configs/mimic4_mortality_qwen9b_2080ti.yaml`: 2 students,
2 cycles, no seed solution, RTX 2080 Ti):

| Run | Students | Best solution | Val AUROC (cycle 0 → 1) | Test AUROC | Test AUPRC |
|---|---|---|---|---|---|
| `mimic4_mortality_20261019_161455` | Qwen3.5-9B-AWQ ×2 | student 1, cycle 1: custom `BaseModel` with learned attention-weighted pooling per code type, PyHealth `Trainer`, 5 epochs | 0.614 → 0.683 | **0.792** | **0.097** |
| `mimic4_mortality_20261022_155823` (rerun of the recipe above, `task.gpus=[0, 1]`, LLM on GPUs 2,6) | Qwen3.5-9B-AWQ ×2 | student 2, cycle 1: custom `BaseModel`, mean pooling + MLP | 0.614 → 0.699 | **0.785** | **0.090** |

Notes on this run:
*   Each run is a single seed with only 2 cycles; treat them as smoke tests,
    not a benchmark. Across the two runs the best test AUROC was 0.785-0.792,
    and in both runs one student was dropped after cycle 0.
*   Student 2 failed all three cycle-0 attempts with the same error (batch
    tensors not moved to `model.device`). As in upstream, a student whose cycle
    fails is removed from the classroom, so student 2 took no further part.
*   Common 9B-model errors, each fixed from traceback feedback within a cycle:
    using `self.label_key` without setting it, label/logit shape mismatches,
    missing imports, and CPU/GPU device mismatches.

Run artifacts (code, logs and predictions for every attempt) are written to
`runs/<run>/`, and the best result of each run is appended to
`runs/leaderboard.tsv`.

--------------------------------------------------------------------------------

## Configuration

Key fields (see `codegen_classroom/run.py:DEFAULT_CONFIG` for all of them):

| Field | Meaning |
|---|---|
| `llm.backend` | `vllm` / `openai_compat` or `gemini` (default for all students) |
| `llm.base_url` | OpenAI-compatible server URL (default `http://localhost:8000/v1`) |
| `llm.models` | One student per entry: a model name, or `{backend, model, base_url}` to mix backends |
| `llm.extra_body` | Extra request fields for vLLM (e.g. `top_p`, `chat_template_kwargs`) |
| `run.num_cycles` | Learning cycles before the final evaluation |
| `classroom.seed_solution` | Start `first`/`all`/`none` students from the PyHealth RNN baseline |
| `task.env` | Built-in PyHealth env (`--list-envs`) or `custom` |
| `task.overrides` | `EnvSpec` overrides: dataset paths/kwargs, `max_learning_train_samples`, timeouts, `metric_weights`, ... |
| `task.gpus`, `task.jobs_per_gpu` | GPUs used to train student models, and concurrent jobs per GPU |

**GPU layout.** Keep the vLLM server and student training on different GPUs,
e.g. on 3× A6000: GPU 0 serves the LLM, and `task.gpus: [1, 2]` trains student
models. Students run concurrently; when every GPU slot is busy, the remaining
students wait for one.

--------------------------------------------------------------------------------

## PyHealth environments

Students write:

```python
def solution(train_dataset, val_dataset, device):
    ...  # build and train a model, e.g. pyhealth.models.RNN + Trainer
    return model
```

*   `train_dataset` / `val_dataset` are PyHealth `SampleDataset` subsets split
    by patient. During learning cycles the training split can be capped
    (`max_learning_train_samples`) to keep iterations fast.
*   The returned model is evaluated by the runner, which calls
    `model(**batch)` on the evaluation split and reads `"y_prob"`. Students
    never receive the evaluation split, and `y_true` comes from the same
    batches as `y_prob`, so predictions are always aligned.
*   Learning cycles are scored on the validation split. The final evaluation
    retrains the chosen code (`first` / `last` / `best` cycle) on the full
    training split and scores it on the held-out **test** split.
*   Every attempt runs as `python -m codegen_classroom.pyhealth_env.runner` in
    its own process group, with `CUDA_VISIBLE_DEVICES` set to the assigned GPU
    and a wall-clock timeout (`learning_timeout_seconds`,
    `final_timeout_seconds`). Errors and timeouts are fed back to the student.

Built-in environments mirror `benchmark_tasks/` (`mimic4_mortality`,
`mimic4_readmission`, `mimic4_los`, `mimic3_mortality`, `eicu_mortality`)
with the cluster's dataset roots and cache dirs as defaults. A custom env needs
either a `dataset_class` + `task_class` (fully qualified names and kwargs) or a
prebuilt `sample_dataset_path`:

```yaml
task:
  type: pyhealth
  env: custom
  overrides:
    name: my_task
    dataset_class: pyhealth.datasets.MIMIC4Dataset
    dataset_kwargs: {ehr_root: /path/to/mimiciv/2.2, ehr_tables: [diagnoses_icd], cache_dir: /path/cache}
    task_class: pyhealth.tasks.MortalityPredictionMIMIC4
    label_key: mortality
    task_type: binary
```

--------------------------------------------------------------------------------

## Library Structure

*   [agents.py](./codegen_classroom/agents.py): `StudentAgent` (state,
    refinement, mutation strategies) and `SolutionAnalysisAgent`, built on a
    backend-agnostic `LLMAgent`.
*   [backends.py](./codegen_classroom/backends.py): `GeminiBackend` and
    `OpenAICompatBackend` (vLLM), plus the retryable-error definitions.
*   [agent_config.py](./codegen_classroom/agent_config.py): defaults (backend,
    vLLM URL, models, generation parameters, timeouts) and `.env` loading.
*   [student_learning.py](./codegen_classroom/student_learning.py): the
    learning cycle, parallel classroom orchestration and final evaluation.
*   [environments.py](./codegen_classroom/environments.py): the
    `TaskEnvironment` interface, which lets tasks run student code their own
    way instead of the default in-process DataFrame contract.
*   [pyhealth_env/](./codegen_classroom/pyhealth_env/): PyHealth env specs and
    registry, data preparation, the subprocess runner, and student prompts.
*   [code_execution.py](./codegen_classroom/code_execution.py): code
    extraction, the forked in-process executor (DataFrame tasks), the
    subprocess executor and `GpuPool`.
*   [scoring.py](./codegen_classroom/scoring.py) /
    [metrics.py](./codegen_classroom/metrics.py): validation, metric registry
    and weighted overall score.
*   [reporting.py](./codegen_classroom/reporting.py) /
    [plotting.py](./codegen_classroom/plotting.py): logging, console output,
    `report.md` and Matplotlib plots.
*   [run.py](./codegen_classroom/run.py): the config-driven experiment loop
    behind `scripts/run_classroom.py`.

The DataFrame API from upstream still works for tabular tasks:
`solution(x_train, y_train, x_val) -> (y_pred_df, feedback)` via
`student_learning.run_parallel_classroom_learning_cycle(...)` without `env`
(see `configs/regression_toy.yaml`).

--------------------------------------------------------------------------------

## Disclaimer

**This is not an officially supported Google product.**

LLM-generated code may be inaccurate or unoptimized. Student code is executed
on this machine: subprocesses isolate crashes, GPUs and timeouts, but they are
not a security sandbox. Run experiments on machines and accounts where that is
acceptable.
