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

"""Standalone script to run the Housing Regression task.

Demonstrates:
  - Tabular regression on 1990 California census blocks
  - Prompt generation & student instruction rendering
  - RUN CONSTANTS with empirical rationale
  - Top solution archive export
"""

from __future__ import annotations

import collections
import functools
import json
import os
import tempfile
import time
from typing import MutableMapping

from absl import app
from absl import flags
from codegen_classroom import agent_config
from codegen_classroom import agents
from codegen_classroom import data_utils
from codegen_classroom import plotting
from codegen_classroom import reporting
from codegen_classroom import scoring
from codegen_classroom import student_learning
from codegen_classroom.tasks.housing import config
from codegen_classroom.tasks.housing import data_loader
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn import metrics as sk_metrics
from sklearn.model_selection import train_test_split

# ==============================================================================
# === 1. TASK CONFIGURATION ===
# ==============================================================================


FLAGS = flags.FLAGS

flags.DEFINE_integer(
    "num_cycles",
    config.RUN_DEFAULTS["num_cycles"],
    "Number of learning cycles to run.",
)
flags.DEFINE_string(
    "experiment_name", "housing_headless", "Name of the experiment."
)
flags.DEFINE_list(
    "model_names",
    list(config.DEFAULT_STUDENT_MODELS),
    "Model names to use for students in the classroom.",
)
flags.DEFINE_integer(
    "num_students",
    config.RUN_DEFAULTS["num_students"],
    "Number of students in the classroom (up to len(model_names)).",
)
flags.DEFINE_integer(
    "early_stopping_patience",
    config.RUN_DEFAULTS["early_stopping_patience"],
    "Early stopping patience in cycles without improvement.",
)
flags.DEFINE_integer(
    "max_mutation_count",
    config.RUN_DEFAULTS["max_mutation_count"],
    "Max mutations before student exclusion.",
)
flags.DEFINE_string(
    "run_dir",
    None,
    "Explicit directory for this run's outputs. If not set, generated"
    " automatically under experiments/<experiment_name>/.",
)
flags.DEFINE_string(
    "report_only",
    None,
    "Regenerate report.md for an existing run directory (pass path or 'latest')"
    " and exit without running any learning cycles.",
)
flags.DEFINE_bool(
    "mock_students",
    False,
    "Use mock student producing baseline model for hermetic unit testing.",
)
flags.DEFINE_integer(
    "timeout_s",
    config.RUN_DEFAULTS["timeout_s"],
    "Per-solution execution timeout in seconds for training cycles.",
)
flags.DEFINE_integer(
    "final_eval_timeout_s",
    config.RUN_DEFAULTS["final_eval_timeout_s"],
    "Per-solution execution timeout in seconds for final evaluation.",
)
flags.DEFINE_integer(
    "max_output_tokens",
    config.RUN_DEFAULTS["max_output_tokens"],
    "Maximum output tokens for student completions.",
)
flags.DEFINE_integer(
    "export_top_n",
    3,
    "Number of top solutions to export to solutions/.",
)
flags.DEFINE_string(
    "seed_from",
    None,
    "Optional seed solution ('best', solution ID, or path to report .md).",
)
flags.DEFINE_enum(
    "action",
    "run",
    ["run", "diagnostics"],
    "Action to perform: 'run' (default) or 'diagnostics'.",
)

agent_config.load_env()
reporting.suppress_common_warnings()

# ==============================================================================
# === EXECUTION HELPERS ===
# ==============================================================================


def _housing_kde_curve(vals: np.ndarray, grid: np.ndarray) -> np.ndarray:
  """Computes a smooth Gaussian KDE curve on `grid`."""
  v = np.asarray(vals, dtype=float)
  v = v[np.isfinite(v)]
  if len(v) < 2:
    return np.zeros_like(grid)
  std = float(np.std(v))
  iqr = float(np.percentile(v, 75) - np.percentile(v, 25))
  sigma = min(std, iqr / 1.34) if iqr > 0 else std
  bw = max(0.9 * max(sigma, 1e-3) * (len(v) ** (-0.2)), 4.0)
  diffs = (grid[:, None] - v[None, :]) / bw
  return np.mean(np.exp(-0.5 * diffs * diffs), axis=1) / (
      bw * np.sqrt(2.0 * np.pi)
  )


def _generate_housing_dataset_overview(
    x_learning: pd.DataFrame,
    y_learning: pd.DataFrame,
    x_test: pd.DataFrame,
    y_test: pd.DataFrame,
    plots_dir: str,
) -> str:
  """Generates a 4-panel California housing dataset and spatial overview figure."""
  os.makedirs(plots_dir, exist_ok=True)
  overview_abs = os.path.join(plots_dir, "dataset_overview.png")
  y_tr_k = y_learning.iloc[:, 0].to_numpy(dtype=float).ravel() / 1000.0
  y_te_k = y_test.iloc[:, 0].to_numpy(dtype=float).ravel() / 1000.0
  x_all = pd.concat([x_learning, x_test], axis=0)
  y_all_k = np.concatenate([y_tr_k, y_te_k])
  lons_all = pd.to_numeric(x_all["longitude"], errors="coerce").to_numpy()
  lats_all = pd.to_numeric(x_all["latitude"], errors="coerce").to_numpy()
  inc_all = pd.to_numeric(x_all["median_income"], errors="coerce").to_numpy()

  with plt.style.context("default"):
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(19.2, 4.2), dpi=150)

    # Panel 1: California Geographic Valuation Map ($k)
    sc = ax0.scatter(
        lons_all,
        lats_all,
        c=np.clip(y_all_k, 30, 500),
        cmap="viridis",
        s=5,
        alpha=0.58,
        edgecolors="none",
    )
    cbar = fig.colorbar(sc, ax=ax0, fraction=0.042, pad=0.03)
    cbar.set_label("Median Value ($k)", fontsize=8.5)
    cbar.ax.tick_params(labelsize=8.0)
    hubs = {
        "SF": (-122.42, 37.77),
        "LA": (-118.24, 34.05),
        "SD": (-117.16, 32.72),
        "SAC": (-121.49, 38.58),
    }
    for h_name, (h_lon, h_lat) in hubs.items():
      ax0.scatter(
          [h_lon],
          [h_lat],
          color="#ef4444",
          marker="*",
          s=65,
          edgecolors="white",
          linewidth=0.6,
          zorder=5,
      )
      ax0.text(
          h_lon + 0.22,
          h_lat + 0.08,
          h_name,
          fontsize=8,
          fontweight="bold",
          color="#111827",
      )
    plotting.style_axes(
        ax0,
        f"California Tract Valuation Map (n={len(y_all_k):,})",
        "Longitude (°W)",
        "Latitude (°N)",
    )

    # Panel 2: Target Valuation Distribution & $500k Top-Coding Spike
    grid_k = np.linspace(15.0, 515.0, 220)
    ax1.hist(y_tr_k, bins=32, density=True, alpha=0.22, color="#1f77b4")
    ax1.hist(y_te_k, bins=32, density=True, alpha=0.22, color="#f97316")
    ax1.plot(
        grid_k,
        _housing_kde_curve(y_tr_k, grid_k),
        color="#1f77b4",
        lw=2.0,
        label=f"Train (n={len(y_tr_k):,}, Med=${np.median(y_tr_k):.0f}k)",
    )
    ax1.plot(
        grid_k,
        _housing_kde_curve(y_te_k, grid_k),
        color="#f97316",
        lw=2.0,
        linestyle="--",
        label=f"Test (n={len(y_te_k):,}, Med=${np.median(y_te_k):.0f}k)",
    )
    cap_pct = float(np.mean(y_all_k >= 499.9) * 100.0)
    ax1.axvline(
        500.0,
        color="#dc2626",
        linestyle=":",
        lw=1.4,
        label=f"$500k Top-Code Cap ({cap_pct:.1f}%)",
    )
    ax1.set_xlim(10, 525)
    plotting.style_axes(
        ax1,
        "Valuation Distribution & $500k Census Cap",
        "Median House Value ($k)",
        "Density",
    )
    ax1.legend(fontsize=7.3, loc="upper right", framealpha=0.92)

    # Panel 3: Median Income vs. Valuation Colored by Latitude
    sc2 = ax2.scatter(
        inc_all,
        y_all_k,
        c=lats_all,
        cmap="coolwarm_r",
        s=5,
        alpha=0.40,
        edgecolors="none",
    )
    cbar2 = fig.colorbar(sc2, ax=ax2, fraction=0.042, pad=0.03)
    cbar2.set_label("Latitude (°N)", fontsize=8.5)
    cbar2.ax.tick_params(labelsize=8.0)
    ax2.axhline(
        500.0, color="#dc2626", linestyle=":", lw=1.2, label="$500k Ceiling"
    )
    r_inc = float(np.corrcoef(inc_all, y_all_k)[0, 1])
    ax2.set_xlim(0.4, 15.2)
    ax2.set_ylim(10, 525)
    plotting.style_axes(
        ax2,
        f"Median Income vs. Valuation (r = {r_inc:.2f})",
        "Median Household Income ($10k USD)",
        "Median House Value ($k)",
    )
    ax2.legend(fontsize=7.5, loc="lower right", framealpha=0.92)

    # Panel 4: Socioeconomic & Engineered Density Correlations with Valuation
    hh = np.maximum(pd.to_numeric(x_all["households"], errors="coerce"), 1.0)
    rooms = np.maximum(
        pd.to_numeric(x_all["total_rooms"], errors="coerce"), 1.0
    )
    beds = np.maximum(
        pd.to_numeric(x_all["total_bedrooms"], errors="coerce"), 1.0
    )
    pop = np.maximum(pd.to_numeric(x_all["population"], errors="coerce"), 1.0)
    feat_df = pd.DataFrame(
        {
            "Median Income": inc_all,
            "Income / Room": inc_all / (rooms / hh),
            "Rooms / Household": rooms / hh,
            "Bedrooms / Room": beds / rooms,
            "Pop / Household": pop / hh,
            "Housing Age": pd.to_numeric(
                x_all["housing_median_age"], errors="coerce"
            ),
            "Latitude (°N)": lats_all,
        },
        index=x_all.index,
    )
    y_ser = pd.Series(y_all_k, index=x_all.index)
    p_corrs = [
        float(feat_df[c].corr(y_ser, method="pearson")) for c in feat_df.columns
    ]
    s_corrs = [
        float(feat_df[c].corr(y_ser, method="spearman"))
        for c in feat_df.columns
    ]
    order = np.argsort(np.abs(s_corrs))
    f_names = [list(feat_df.columns)[i] for i in order]
    p_vals = [p_corrs[i] for i in order]
    s_vals = [s_corrs[i] for i in order]
    y_pos = np.arange(len(f_names))
    ax3.barh(
        y_pos - 0.17, p_vals, height=0.34, color="#94a3b8", label="Pearson r"
    )
    ax3.barh(
        y_pos + 0.17, s_vals, height=0.34, color="#0d9488", label="Spearman ρ"
    )
    ax3.axvline(0.0, color="#111827", lw=0.9)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(f_names, fontsize=8.0)
    ax3.set_xlim(-0.42, 0.78)
    plotting.style_axes(
        ax3,
        "Feature & Ratio Correlations with Value",
        "Correlation Coefficient (r / ρ)",
        "",
    )
    ax3.legend(fontsize=7.5, loc="lower right", framealpha=0.92)

    plt.tight_layout(w_pad=1.4)
    plt.savefig(overview_abs, bbox_inches="tight", dpi=150)
    plt.close(fig)
  return "plots/dataset_overview.png"


def _generate_housing_diagnostics(
    task_dir: str,
    md_files: list[str],
    force: bool = False,
) -> None:
  """Generates spatial, price-tier MAPE, parity, Bland-Altman, KDE, and CAPE diagnostic plots."""
  if not md_files:
    return
  data_dir = os.path.join(task_dir, "data")
  data_loader.ensure_data(data_dir)
  x, y = data_loader.load_preprocessed_housing_data(
      data_dir=data_dir, y_column="median_house_value"
  )
  x_learning, x_test, y_learning, y_test = train_test_split(
      x, y, test_size=0.2, random_state=config.VAL_SEED
  )
  y_true = y_test.iloc[:, 0].to_numpy(dtype=float).ravel()
  lats = x_test["latitude"].to_numpy(dtype=float).ravel()
  lons = x_test["longitude"].to_numpy(dtype=float).ravel()
  plots_dir = os.path.join(task_dir, "solutions", "plots")
  os.makedirs(plots_dir, exist_ok=True)
  matplotlib.use("Agg")
  overview_rel = _generate_housing_dataset_overview(
      x_learning, y_learning, x_test, y_test, plots_dir
  )

  def _baseline_fn(
      x_tr: pd.DataFrame, y_tr: pd.DataFrame, x_te: pd.DataFrame
  ) -> np.ndarray:
    m = ensemble.HistGradientBoostingRegressor(
        max_iter=60, max_depth=5, random_state=42
    )
    m.fit(x_tr, y_tr.iloc[:, 0])
    return m.predict(x_te)

  for md_path in md_files:
    sol_id = os.path.splitext(os.path.basename(md_path))[0]
    plot_path = os.path.join(plots_dir, f"{sol_id}_diagnostics.png")
    if not force and os.path.isfile(plot_path):
      continue
    preds = reporting.resolve_solution_predictions(
        md_path,
        x_learning,
        y_learning,
        x_test,
        baseline_fn=_baseline_fn,
    )
    if preds is None:
      continue
    y_best, y_first = preds
    r2_0 = float(sk_metrics.r2_score(y_true, y_first))
    r2_b = float(sk_metrics.r2_score(y_true, y_best))
    mape_0 = float(
        sk_metrics.mean_absolute_percentage_error(y_true, y_first) * 100.0
    )
    mape_b = float(
        sk_metrics.mean_absolute_percentage_error(y_true, y_best) * 100.0
    )
    ape_first = 100.0 * np.abs(y_first - y_true) / np.maximum(y_true, 1.0)
    ape_best = 100.0 * np.abs(y_best - y_true) / np.maximum(y_true, 1.0)

    with plt.style.context("default"):
      fig, axes = plt.subplots(2, 3, figsize=(18.0, 8.2), dpi=150)
      (ax0, ax1, ax2), (ax3, ax4, ax5) = axes

      # Panel 1: Spatial APE Map across California
      sc = ax0.scatter(
          lons,
          lats,
          c=np.clip(ape_best, 0, 35),
          cmap="YlOrRd",
          s=7,
          alpha=0.70,
          edgecolors="none",
      )
      cbar = fig.colorbar(sc, ax=ax0, fraction=0.042, pad=0.03)
      cbar.set_label("APE (%)", fontsize=9)
      cbar.ax.tick_params(labelsize=8.5)
      hubs = {
          "SF": (-122.42, 37.77),
          "LA": (-118.24, 34.05),
          "SD": (-117.16, 32.72),
          "SAC": (-121.49, 38.58),
      }
      for h_name, (h_lon, h_lat) in hubs.items():
        ax0.scatter(
            [h_lon],
            [h_lat],
            color="#1f77b4",
            marker="*",
            s=65,
            edgecolors="k",
            zorder=5,
        )
        ax0.text(
            h_lon + 0.22, h_lat + 0.08, h_name, fontsize=8, fontweight="bold"
        )
      plotting.style_axes(
          ax0,
          f"Spatial Error Map (Test MAPE: {mape_b:.2f}%)",
          "Longitude (°W)",
          "Latitude (°N)",
      )

      # Panel 2: MAPE across Housing Value Brackets
      tiers = [
          ("<$100k\n(Low)", y_true < 100000),
          ("$100–200k\n(Mid-Low)", (y_true >= 100000) & (y_true < 200000)),
          ("$200–350k\n(Mid-High)", (y_true >= 200000) & (y_true < 350000)),
          ("≥$350k\n(High/Cap)", y_true >= 350000),
          ("Overall\nTest", np.ones_like(y_true, dtype=bool)),
      ]
      t_labels = [t[0] for t in tiers]
      m0_tiers = [
          float(
              sk_metrics.mean_absolute_percentage_error(y_true[m], y_first[m])
              * 100.0
          )
          for _, m in tiers
      ]
      mb_tiers = [
          float(
              sk_metrics.mean_absolute_percentage_error(y_true[m], y_best[m])
              * 100.0
          )
          for _, m in tiers
      ]
      plotting.plot_grouped_bar_panel(
          ax1,
          t_labels,
          m0_tiers,
          mb_tiers,
          label_first=f"Cycle 0 ({mape_0:.1f}%)",
          label_best=f"Best ({mape_b:.1f}%)",
          title="MAPE by Valuation Tier (Cycle 0 vs. Best)",
          xlabel="Census Tract Valuation Bracket",
          ylabel="MAPE (%)",
          fmt="{:.1f}%",
          color_best="#ff7f0e",
      )

      # Panel 3: Valuation Parity & $500k Top-Coding Ceiling
      yt_k, yf_k, yb_k = y_true / 1000.0, y_first / 1000.0, y_best / 1000.0
      ax2.scatter(
          yt_k,
          yf_k,
          color="#7f7f7f",
          alpha=0.16,
          s=10,
          label=f"Cycle 0 (R²={r2_0:.3f})",
      )
      ax2.scatter(
          yt_k,
          yb_k,
          color="#1f77b4",
          alpha=0.25,
          s=10,
          label=f"Best (R²={r2_b:.3f})",
      )
      ax2.plot(
          [0, 520], [0, 520], "k--", lw=1.2, alpha=0.8, label="Ideal (y = x)"
      )
      ax2.axvline(
          500.0,
          color="#d62728",
          linestyle=":",
          lw=1.3,
          label="$500k Census Cap",
      )
      ax2.set_xlim(0, 535)
      ax2.set_ylim(0, 535)
      plotting.style_axes(
          ax2,
          f"Valuation Parity (R²: {r2_0:.3f} → {r2_b:.3f})",
          "Observed Median Value ($k)",
          "Predicted Median Value ($k)",
      )
      ax2.legend(loc="upper left", fontsize=8, framealpha=0.9)

      # Panel 4: Bland-Altman Valuation Agreement ($k)
      plotting.plot_bland_altman_panel(
          ax3,
          y_true,
          y_best,
          y_first,
          scale=0.001,
          unit="$k",
          title="Bland–Altman Agreement (Pred − True)",
          xlabel="Mean Valuation ($k)",
          ylabel="Valuation Error: Pred − True ($k)",
          fmt="+.1f",
          sd_fmt=".1f",
      )

      # Panel 5: Predicted vs. Observed Valuation Density (KDE)
      grid_k = np.linspace(15.0, 515.0, 220)
      d_true = _housing_kde_curve(yt_k, grid_k)
      d_first = _housing_kde_curve(yf_k, grid_k)
      d_best = _housing_kde_curve(yb_k, grid_k)
      ax4.fill_between(grid_k, d_true, color="#64748b", alpha=0.15)
      ax4.plot(
          grid_k,
          d_true,
          color="#334155",
          lw=1.8,
          linestyle=":",
          label=f"Observed Test (σ=${np.std(yt_k):.0f}k)",
      )
      ax4.plot(
          grid_k,
          d_first,
          color="#94a3b8",
          lw=1.8,
          linestyle="--",
          label=f"Cycle 0 Preds (σ=${np.std(yf_k):.0f}k)",
      )
      ax4.fill_between(grid_k, d_best, color="#0d9488", alpha=0.18)
      ax4.plot(
          grid_k,
          d_best,
          color="#0d9488",
          lw=2.1,
          label=f"Best Preds (σ=${np.std(yb_k):.0f}k)",
      )
      ax4.axvline(500.0, color="#dc2626", linestyle=":", lw=1.2)
      ax4.set_xlim(10, 525)
      plotting.style_axes(
          ax4,
          "Predicted vs. Observed Valuation Density",
          "Median House Value ($k)",
          "Probability Density",
      )
      ax4.legend(loc="upper right", fontsize=7.5, framealpha=0.92)

      # Panel 6: Cumulative Absolute Percentage Error (CAPE Accuracy Curve)
      tau_grid = np.linspace(0.0, 40.0, 161)
      cdf_0 = [float(np.mean(ape_first <= t) * 100.0) for t in tau_grid]
      cdf_b = [float(np.mean(ape_best <= t) * 100.0) for t in tau_grid]
      within10_0 = float(np.mean(ape_first <= 10.0) * 100.0)
      within10_b = float(np.mean(ape_best <= 10.0) * 100.0)
      within20_0 = float(np.mean(ape_first <= 20.0) * 100.0)
      within20_b = float(np.mean(ape_best <= 20.0) * 100.0)
      ax5.plot(
          tau_grid,
          cdf_0,
          color="#64748b",
          linestyle="--",
          lw=1.7,
          label=f"Cycle 0 (≤10%: {within10_0:.1f}%, ≤20%: {within20_0:.1f}%)",
      )
      ax5.plot(
          tau_grid,
          cdf_b,
          color="#1f77b4",
          linestyle="-",
          lw=2.2,
          label=f"Best (≤10%: {within10_b:.1f}%, ≤20%: {within20_b:.1f}%)",
      )
      ax5.axvline(10.0, color="#0d9488", linestyle=":", lw=1.1, alpha=0.8)
      ax5.axvline(20.0, color="#f97316", linestyle=":", lw=1.1, alpha=0.8)
      ax5.scatter(
          [10.0, 20.0],
          [within10_b, within20_b],
          color="#ff7f0e",
          edgecolor="#111827",
          s=42,
          zorder=5,
      )
      ax5.set_xlim(0.0, 40.0)
      ax5.set_ylim(0.0, 102.0)
      plotting.style_axes(
          ax5,
          "Cumulative Error Accuracy (Tracts ≤ τ% APE)",
          "Absolute Percentage Error Threshold τ (%)",
          "Test Census Tracts Within ≤ τ% Error (%)",
      )
      ax5.legend(loc="lower right", fontsize=7.5, framealpha=0.92)

      plt.tight_layout(pad=1.2)
      plt.savefig(plot_path, bbox_inches="tight", dpi=150)
      plt.close(fig)

    t_red = [
        100.0 * (m0_tiers[i] - mb_tiers[i]) / max(m0_tiers[i], 1e-6)
        for i in range(4)
    ]
    tot_red = 100.0 * (mape_0 - mape_b) / max(mape_0, 1e-6)
    summary_md = (
        "| Valuation Tier | Cycle 0 Baseline MAPE | Best Solution MAPE |"
        " Relative Error Reduction |\n| --- | --- | --- | --- |\n"
        f"| Low (<\\$100k) | {m0_tiers[0]:.2f}% | {mb_tiers[0]:.2f}% |"
        f" {t_red[0]:.1f}% |\n"
        f"| Mid-Low (\\$100k–\\$200k) | {m0_tiers[1]:.2f}% |"
        f" {mb_tiers[1]:.2f}% | {t_red[1]:.1f}% |\n"
        f"| Mid-High (\\$200k–\\$350k) | {m0_tiers[2]:.2f}% |"
        f" {mb_tiers[2]:.2f}% | {t_red[2]:.1f}% |\n"
        f"| High & Top-Coded (≥\\$350k) | {m0_tiers[3]:.2f}% |"
        f" {mb_tiers[3]:.2f}% | {t_red[3]:.1f}% |\n"
        "| **Tracts Within ≤10% / ≤20% APE** |"
        f" {within10_0:.1f}% / {within20_0:.1f}% |"
        f" **{within10_b:.1f}% / {within20_b:.1f}%** |"
        f" **+{within10_b - within10_0:.1f}% /"
        f" +{within20_b - within20_0:.1f}%** |\n"
        f"| **Overall Held-Out Test** | **{mape_0:.2f}%** (R²={r2_0:.3f}) |"
        f" **{mape_b:.2f}%** (R²={r2_b:.3f}) | **{tot_red:.1f}%** |"
    )
    reporting.embed_diagnostic_section(
        md_path,
        f"plots/{sol_id}_diagnostics.png",
        summary_md,
        overview_plot_rel_path=overview_rel,
    )


def export_top_solutions(
    task_dir: str,
    top_n: int = 3,
    write: bool = True,
) -> list[str]:
  """Exports top solution reports from leaderboard.tsv to solutions/."""
  return reporting.export_top_solutions(
      task_dir,
      sort_by=["test_r2", "test_mape"],
      ascending=[False, True],
      top_n=top_n,
      write=write,
      metric_weights=config.DEFAULT_METRIC_WEIGHTS,
      experiments_root=data_utils.experiments_root(config.TASK_NAME),
      diagnose=functools.partial(_generate_housing_diagnostics, task_dir),
  )


def resolve_task_dir(base_dir: str | None = None) -> str:
  """Resolves the live task directory."""
  return data_utils.resolve_task_dir(__file__, base_dir)


class _MockStudent(agents.StudentAgent):
  """Deterministic mock student for fast offline unit and smoke testing."""

  def __init__(
      self,
      config: agents.AgentConfig,  # pylint: disable=redefined-outer-name
      run_base_dir: str = "",
      seed_solution: str | None = None,
  ):
    config.api_key = config.api_key or "mock_key"
    super().__init__(
        config=config,
        run_base_dir=run_base_dir or tempfile.gettempdir(),
        seed_solution=seed_solution,
    )

  def generate_solution(self, prompt: str) -> str:
    del prompt
    self.generate_code_count += 1
    self.current_solution = (
        "```python\n"
        "import pandas as pd\n"
        "from sklearn.ensemble import HistGradientBoostingRegressor\n"
        "def solution(X_train, y_train, X_val):\n"
        "    model = HistGradientBoostingRegressor(max_iter=50,"
        " random_state=42)\n"
        "    model.fit(X_train, y_train.iloc[:, 0])\n"
        "    preds = model.predict(X_val)\n"
        "    return pd.DataFrame({'predictions': preds},"
        " index=X_val.index), 'Mock HGBR baseline'\n"
        "```\n"
    )
    return self.current_solution


def _report_only() -> None:
  """Rebuilds report.md for an existing run without re-executing code."""
  run_dir = data_utils.resolve_run_dir(
      FLAGS.report_only,
      os.path.join(
          data_utils.experiments_root(config.TASK_NAME), FLAGS.experiment_name
      ),
  )
  reporting.write_run_report(
      run_dir,
      metric_weights=config.DEFAULT_METRIC_WEIGHTS,
  )


def main(argv: list[str]) -> None:
  del argv

  if FLAGS.report_only:
    _report_only()
    return

  if FLAGS.action == "diagnostics":
    task_dir = resolve_task_dir()
    solutions_dir = os.path.join(task_dir, "solutions")
    md_files = [
        os.path.join(solutions_dir, f)
        for f in sorted(os.listdir(solutions_dir))
        if f.endswith(".md")
    ]
    _generate_housing_diagnostics(task_dir, md_files)
    return

  # 1. Load Data
  print("Loading data...")
  y_column = "median_house_value"
  task_dir = resolve_task_dir()
  data_dir = os.path.join(task_dir, "data")
  data_loader.ensure_data(data_dir)

  x, y = data_loader.load_preprocessed_housing_data(
      data_dir=data_dir, y_column=y_column
  )

  # 2. Define Task
  metric_weights = dict(config.DEFAULT_METRIC_WEIGHTS)

  student_instructions = config.build_student_instructions(
      feature_names=list(x.columns),
      target_column=y_column,
      train_rows=int(len(x) * 0.8 * 0.8),
      val_rows=int(len(x) * 0.8 * 0.2),
      final_eval_rows=int(len(x) * 0.8),
      timeout_s=FLAGS.timeout_s,
      final_eval_timeout_s=FLAGS.final_eval_timeout_s,
      metric_weights=metric_weights,
  )

  initial_prompt = config.STUDENT_INITIAL_PROMPT
  refine_options = list(config.STUDENT_REFINE_INSTRUCTION_OPTIONS)

  # 3. Config
  runs_base_path = os.path.join(
      data_utils.experiments_root(config.TASK_NAME), FLAGS.experiment_name
  )
  os.makedirs(runs_base_path, exist_ok=True)

  if FLAGS.run_dir:
    run_dir = os.path.abspath(FLAGS.run_dir)
    run_id = os.path.basename(run_dir)
  else:
    run_id = "headless_" + time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(runs_base_path, run_id)
  os.makedirs(run_dir, exist_ok=True)

  student_function_name = config.STUDENT_FUNCTION_NAME
  num_learning_cycles = FLAGS.num_cycles
  code_execution_max_retries = 3
  code_execution_max_duration = FLAGS.timeout_s

  early_stopping_patience = FLAGS.early_stopping_patience
  max_mutation_count = FLAGS.max_mutation_count

  test_size = 0.2
  val_seed = config.VAL_SEED

  mutation_strategy_options = [
      agents.MutationStrategy.SELF_BEST_CODE,
      agents.MutationStrategy.BEST_PEER_CROSSOVER,
      agents.MutationStrategy.CONTRASTIVE_PEER_CROSSOVER,
  ]

  # 4. Split Data (learning set vs held-out final test)
  x_learning, x_test_final, y_learning, y_test_final = train_test_split(
      x, y, test_size=test_size, random_state=val_seed
  )

  # Train/val split within learning set
  x_train, x_val, y_train, y_val = train_test_split(
      x_learning, y_learning, test_size=0.2, random_state=val_seed
  )

  # Setup logging
  logger = reporting.setup_logger(
      "codegen_classroom", log_file=os.path.join(run_dir, "run.log")
  )
  logger.info("Starting housing regression headless run: %s", run_id)

  models = [
      FLAGS.model_names[i % len(FLAGS.model_names)]
      for i in range(FLAGS.num_students)
  ]
  student_cfgs = student_learning.get_classroom_student_configs(
      system_instruction=student_instructions,
      model_names=models,
      max_output_tokens=FLAGS.max_output_tokens,
  )

  agent_cls = _MockStudent if FLAGS.mock_students else agents.StudentAgent
  seed_md = data_utils.load_seed_solution(FLAGS.seed_from, task_dir=task_dir)
  students = {
      cfg.name: agent_cls(
          config=cfg,
          run_base_dir=run_dir,
          seed_solution=seed_md,
      )
      for cfg in student_cfgs
  }

  current_classroom = list(students.values())
  results_history: MutableMapping[str, student_learning.ClassroomResults] = (
      collections.defaultdict(list)
  )
  active_metrics = ["overall_score"] + list(metric_weights.keys())

  validator = functools.partial(
      scoring.prediction_validator, expected_columns=1
  )

  # Run learning cycles
  for current_cycle in range(num_learning_cycles):
    if not current_classroom:
      break
    logger.info("=== Starting Cycle %d ===", current_cycle)

    classroom_feedback = scoring.get_classroom_feedback_string(
        cycle_num=current_cycle,
        current_classroom=current_classroom,
        classroom_results_history=results_history,
        metric_keys=active_metrics,
    )

    classroom_results = student_learning.run_parallel_classroom_learning_cycle(
        current_classroom=current_classroom,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        current_cycle=current_cycle,
        student_initial_prompt=initial_prompt,
        student_function_name=student_function_name,
        code_execution_max_retries=code_execution_max_retries,
        code_execution_timeout_seconds=code_execution_max_duration,
        metric_weights=metric_weights,
        early_stopping_patience=early_stopping_patience,
        student_refine_instruction_options=refine_options,
        max_mutation_count=max_mutation_count,
        mutation_strategy_options=mutation_strategy_options,
        validator=validator,
        classroom_feedback=classroom_feedback,
        num_workers=len(current_classroom),
    )

    current_classroom = student_learning.process_cycle_results(
        current_classroom, classroom_results, results_history, current_cycle
    )
    for student in students.values():
      student.reset_history()

  # Final Evaluation on held-out test set
  logger.info("Evaluating on held-out test set...")
  test_results_path = os.path.join(run_dir, "final_eval_results.json")
  test_prediction_dir = os.path.join(run_dir, "predictions")

  student_learning.run_parallel_classroom_final_evaluation(
      students=students,
      x_learning=x_learning,
      y_learning=y_learning,
      x_test_final=x_test_final,
      y_test_final=y_test_final,
      student_function_name=student_function_name,
      metric_weights=metric_weights,
      code_execution_timeout_seconds=FLAGS.final_eval_timeout_s,
      enforce_timeout=True,
      strategies=["first", "best", "last"],
      validator=validator,
      num_workers=len(students),
      final_results_path=test_results_path,
      predictions_dir=test_prediction_dir,
  )

  row_data = None
  if os.path.isfile(test_results_path):
    with open(test_results_path, "r", encoding="utf-8") as f:
      eval_json = json.load(f)
    best_student_obj = max(
        students.values(),
        key=lambda s: float(s.best_score)
        if s.best_score is not None
        else -float("inf"),
    )
    best_eval_entry = next(
        (
            e
            for e in eval_json
            if e.get("student_name") == best_student_obj.name
        ),
        {},
    )
    best_strategy_eval = best_eval_entry.get("best", {})
    b_cycle = best_student_obj.best_cycle
    best_res = (
        best_student_obj.cycle_results[b_cycle]
        if 0 <= b_cycle < len(best_student_obj.cycle_results)
        else None
    )
    best_val_metrics = best_res.metrics if best_res else {}
    best_plan = best_res.plan if best_res else ""
    meta = {}
    if not FLAGS.mock_students and best_student_obj.best_solution:
      try:
        analyzer = agents.SolutionAnalysisAgent(
            config=agents.AgentConfig(name="solution_metadata"),
            run_base_dir=run_dir,
        )
        meta = (
            analyzer.analyze(
                student_name=best_student_obj.name,
                best_code=best_student_obj.best_solution,
                best_plan=best_plan,
                best_test_metrics=best_strategy_eval,
                best_val_metrics=best_val_metrics,
                student_instructions=student_instructions,
                metrics_description=scoring.get_metrics_description(
                    metric_weights
                ),
                task_description_for_prompt=(
                    "California housing price regression"
                ),
            )
            or {}
        )
      except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning("SolutionAnalysisAgent failed: %s", e)

    row_data = {
        "timestamp": pd.Timestamp.now().strftime("%m/%d/%Y"),
        "test_mape": best_strategy_eval.get("metrics_mape"),
        "test_r2": best_strategy_eval.get("metrics_r2"),
        "test_pearson": best_strategy_eval.get("metrics_pearson"),
        "test_exec_time_s": best_strategy_eval.get("metrics_exec_time"),
        "val_mape": best_val_metrics.get("mape"),
        "val_r2": best_val_metrics.get("r2"),
        "val_pearson": best_val_metrics.get("pearson"),
        "train_cycles": num_learning_cycles,
        "train_best_cycle": best_student_obj.best_cycle,
        "train_val_seed": val_seed,
        "train_num_students": len(students),
        "codegen_version": 2,
        "solution_type": "best_validation_solution",
        "total_tokens": None,
        "est_cost_usd": None,
        "experiment_name": FLAGS.experiment_name or "housing_headless",
        "student": best_student_obj.name,
        "run_id": run_id,
        "run_dir": run_dir,
        "user": os.environ.get("USER", ""),
        "tags": "mock" if FLAGS.mock_students else meta.get("model_tags", ""),
        "description": meta.get("model_description", ""),
        "insights": meta.get("model_insights", ""),
        "solution_dir": "",
    }

  reporting.write_run_report(
      run_dir,
      students=list(students.values()),
      metric_weights=config.DEFAULT_METRIC_WEIGHTS,
  )
  logger.info("Report generated in %s/report.md", run_dir)

  if row_data is not None and not FLAGS.mock_students:
    lb_path = os.path.join(task_dir, "leaderboard.tsv")
    data_utils.submit_results_to_local_leaderboard(
        lb_path, pd.DataFrame([row_data])
    )
    logger.info("Results appended to leaderboard: %s", lb_path)

  # Export top solutions
  if not FLAGS.mock_students:
    try:
      export_top_solutions(task_dir, top_n=FLAGS.export_top_n)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logger.warning("Solution archive export failed: %s", e)


if __name__ == "__main__":
  app.run(main)
