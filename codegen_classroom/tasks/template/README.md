# Template Task: Multi-Regime Thermal Load Regression

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-research/codegen-classroom/blob/main/codegen_classroom/tasks/template/task_example.ipynb)

This directory serves as the executable reference template for adding new
benchmark tasks to `codegen_classroom`. It implements a synthetic **Multi-Regime
Thermal Load Regression** benchmark (`N = 1,500` rows, `10` features) scored on
R² (`0.60`), MAE (`0.40`), and execution time (`0.01`) so
student agents can meaningfully hill-climb across modeling tiers.

## Dataset

Unlike a trivial linear toy dataset, `data_loader.py` simulates a physical
system with three distinct operating regimes (`operating_regime` $\in \{0, 1,
2\}$), periodic phase harmonics (`phase_angle`), log-normal surge loads
(`surge_index`), cooling saturation (`cooling_flow`, `pressure_delta`), missing
sensor observations (`temp_sensor_proxy` with ~4% `NaN`), and two pure noise
distractors (`aux_noise_1`, `aux_noise_2`).

## Results

| Modeling Tier | Representative Approach | Expected Test R² | Expected Test MAE |
| :--- | :--- | :---: | :---: |
| **Tier 0: Naive** | Constant mean / median predictor | `0.00` | `~2.95` |
| **Tier 1: Linear Baseline** | Median imputation + `Ridge` on raw features | `~0.48` | `~1.62` |
| **Tier 2: Tree Baseline** | Default `HistGradientBoostingRegressor` on raw features | `~0.83` | `~0.88` |
| **Tier 3: Feature Engineering + Blending** | Regime-crossed interactions, harmonic/log basis, noise pruning + stacked GBDT / MLP / ExtraTrees | **`0.95 - 0.98+`** | **`0.22 - 0.42`** |

## Files

When creating a new task (`cp -r tasks/template tasks/<your_task>`), search for
`# CUSTOMIZE:` across the files below:

```text
tasks/template/
├── README.md              # Task overview, dataset description, and baseline table
├── config.py              # CUSTOMIZE: Shared task settings, metric weights, and 8-section prompt
├── data_loader.py         # CUSTOMIZE: Replace synthetic generator with your dataset loader
├── run_template.py        # Headless classroom runner importing from config.py
├── run_template_test.py   # Offline unit tests for data loader and prompt invariants
├── task_example.ipynb     # Interactive Jupyter / Colab walkthrough importing from config.py
├── leaderboard.tsv        # Tab-separated evaluation log updated by run_template.py
└── solutions/             # Top-performing run reports (*.md) and trajectory plots
```

## Quickstart

```bash
# 1. Run offline unit tests:
pytest codegen_classroom/tasks/template/run_template_test.py

# 2. Run a headless classroom benchmark (updates leaderboard.tsv and solutions/):
python -m codegen_classroom.tasks.template.run_template \
  --num_students=3 \
  --num_cycles=4

# 3. Rebuild report.md for the latest run (no code re-execution):
python -m codegen_classroom.tasks.template.run_template --report_only=latest

# 4. Verify task structure, runner contract, notebook, and solutions:
python -m codegen_classroom.verify_task --task template --strict
```
