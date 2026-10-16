# Template Task: Multi-Regime Thermal Load Regression

This directory serves as the executable reference template for adding new
benchmark tasks to `codegen_classroom`. It implements a synthetic **Multi-Regime
Thermal Load Regression** benchmark (`N = 1,500` rows, `10` features) scored on
R² (`0.60`), MAE (`0.40`), and execution time (`-0.05` over `[0, 30]s`) so
student agents can meaningfully hill-climb across modeling tiers.

## 1. Benchmark Task Design & Hill-Climbing Signal

Unlike a trivial linear toy dataset, `data_loader.py` simulates a physical
system with three distinct operating regimes (`operating_regime` $\in \{0, 1,
2\}$), periodic phase harmonics (`phase_angle`), log-normal surge loads
(`surge_index`), cooling saturation (`cooling_flow`, `pressure_delta`), missing
sensor observations (`temp_sensor_proxy` with ~4% `NaN`), and two pure noise
distractors (`aux_noise_1`, `aux_noise_2`).

| Modeling Tier | Representative Approach         | Expected  | Expected Test |
:               :                                 : Test R²   : MAE           :
| :------------ | :------------------------------ | :-------: | :-----------: |
| **Tier 0:     | Constant mean / median          | `0.00`    | `~2.95`       |
: Naive**       : predictor                       :           :               :
| **Tier 1:     | Median imputation + `Ridge` on  | `~0.48`   | `~1.62`       |
: Linear        : raw features                    :           :               :
: Baseline**    :                                 :           :               :
| **Tier 2:     | Default                         | `~0.83`   | `~0.88`       |
: Tree          : `HistGradientBoostingRegressor` :           :               :
: Baseline**    : on raw features                 :           :               :
| **Tier 3:     | Regime-crossed interactions,    | **`0.95 - | **`0.22 -     |
: Feature       : harmonic/log basis, noise       : 0.98+`**  : 0.42`**       :
: Engineering + : pruning + stacked GBDT / MLP /  :           :               :
: Blending**    : ExtraTrees                      :           :               :

## 2. Directory Structure & Customization Points

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

## 3. Running and Verifying the Task

```bash
# 1. Run offline unit tests:
pytest codegen_classroom/tasks/template/run_template_test.py

# 2. Run a headless classroom benchmark (updates leaderboard.tsv and solutions/):
python -m codegen_classroom.tasks.template.run_template \
  --num_students=3 \
  --num_cycles=4

# 3. Verify task structure, runner contract, notebook, and solutions:
python -m codegen_classroom.verify_task --task template --strict
```
