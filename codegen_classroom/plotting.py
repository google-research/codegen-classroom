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

"""Plotting utils for charts, decoupled from reporting and display logic.

Charts are rendered headlessly with Matplotlib (Agg backend) and saved to disk.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
from typing import Any

from codegen_classroom import data_utils
from codegen_classroom import metrics as metric_lib
from etils import epath
import matplotlib
from matplotlib import ticker
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

StudentProgressData = Sequence[Mapping[str, str | int | float | bool | None]]
FinalResultsData = Sequence[
    # Example: [{'student_name': str, 'best': {metric: score}, 'best': ...}]
    Mapping[str, str | Mapping[str, int | float | None]]
]

_PALETTE = [
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
    "#FF97FF",
    "#FECB52",
]


def get_metric_key(metric_name: str) -> str:
  """Adds 'metrics_' prefix if not overall_score."""
  return (
      f"metrics_{metric_name}"
      if metric_name != "overall_score"
      else "overall_score"
  )


def plot_progress(
    student_results_dir: str | epath.Path,
    metric_names: list[str],
    students: list[str] | None = None,
    dark_mode: bool = False,
    fig_width: int = 900,
    fig_height_per_metric: int = 400,
    title_font_size: int = 12,
    axis_font_size: int = 12,
    legend_font_size: int = 12,
    show_mutation_points: bool = False,
    save_path: str | None = None,
    show: bool = False,
):
  """Plots the progress of students.

  Args:
      student_results_dir: Directory containing student result CSV files.
      metric_names: A list of metric names to plot.
      students: A list of student names to plot. If None, all are plotted.
      dark_mode: If True, use a dark theme (ignored if save_path is set).
      fig_width: Width of the figure.
      fig_height_per_metric: Height of the figure per metric row.
      title_font_size: Font size for plot titles.
      axis_font_size: Font size for axis titles and ticks.
      legend_font_size: Font size for legend items.
      show_mutation_points: Show mutation points on the cumulative plot.
      save_path: If set, save figure here (forces theme to light).
      show: Whether to call plt.show() (no-op with the Agg backend).
  """
  if not metric_names:
    print("Metrics list cannot be empty.")
    return

  student_data_map = data_utils.load_student_results(
      student_results_dir, students
  )
  if not student_data_map:
    print("No data found.")
    return

  if save_path:
    # A committed artifact should always be light
    dark_mode = False

  plot_progress_static(
      student_data=student_data_map,
      metric_names=metric_names,
      dark_mode=dark_mode,
      single_plot_size=(fig_width / 100.0, fig_height_per_metric / 100.0),
      title_fontsize=title_font_size,
      label_fontsize=axis_font_size,
      legend_fontsize=legend_font_size,
      show_mutation_points=show_mutation_points,
      save_path=save_path,
      show=show,
  )


def plot_progress_static(
    student_data: str | epath.Path | Mapping[str, Sequence[Mapping[str, Any]]],
    metric_names: Sequence[str] | None = None,
    dark_mode: bool = False,
    single_plot_size: tuple[float, float] = (4.8, 4.2),
    title_fontsize: int = 11,
    label_fontsize: int = 10,
    legend_fontsize: int = 10,
    show_mutation_points: bool = True,
    save_path: str | epath.Path | None = None,
    save_fig: str | epath.Path | None = None,
    show: bool = False,
    show_fig: bool | None = None,
    dpi: int = 150,
    metric_weights: Mapping[str, float] | None = None,
) -> None:
  """Plots learning progress across metrics on a static Matplotlib subplot grid."""
  effective_save_path = save_path or save_fig
  if show_fig is not None:
    show = show_fig

  if isinstance(student_data, (str, epath.Path)):
    p = epath.Path(str(student_data))
    students_dir = p / "students" if (p / "students").exists() else p
    student_data_map = data_utils.load_student_results(students_dir)
  else:
    student_data_map = student_data

  if not student_data_map:
    return

  if not metric_names:
    if metric_weights:
      sorted_metrics = sorted(
          [m for m in metric_weights if m != "overall_score"],
          key=lambda m: metric_weights[m],
          reverse=True,
      )
      metric_names = ["overall_score"] + sorted_metrics
    else:
      metric_names = ["overall_score"]

  valid_metrics = {}
  for m in metric_names:
    cfg = metric_lib.ALL_METRICS_CONFIG.get(m)
    higher_is_better = cfg.higher_is_better if cfg else True
    valid_metrics[m] = {
        "higher_is_better": higher_is_better,
    }

  if not valid_metrics:
    return

  num_metrics = len(valid_metrics)
  ncols = min(num_metrics, 3)
  nrows = math.ceil(num_metrics / ncols)

  edge_color = "white" if dark_mode else "black"
  style_name = "dark_background" if dark_mode else "default"

  with plt.style.context(style_name):
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(single_plot_size[0] * ncols, single_plot_size[1] * nrows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    cmap = plt.get_cmap("tab10")
    color_map = {
        name: cmap(i % 10) for i, name in enumerate(student_data_map.keys())
    }

    any_mutations = False
    for i, (metric_name, m_info) in enumerate(valid_metrics.items()):
      ax = axes_flat[i]
      higher_is_better = m_info["higher_is_better"]
      metric_key = get_metric_key(metric_name)

      for student_name, all_cycles_data in student_data_map.items():
        student_color = color_map[student_name]
        student_metrics = []
        for record in all_cycles_data:
          if metric_key in record and "cycle" in record:
            student_metrics.append({
                "cycle": int(record["cycle"]),
                "metric_val": float(record[metric_key]),
                "mutation_strategy": str(record.get("mutation_strategy", "")),
            })

        if not student_metrics:
          continue

        student_metrics.sort(key=lambda x: x["cycle"])
        cycles = [d["cycle"] for d in student_metrics]
        metric_values = [d["metric_val"] for d in student_metrics]

        cumulative_values = (
            np.maximum.accumulate(metric_values)
            if higher_is_better
            else np.minimum.accumulate(metric_values)
        )

        ax.scatter(cycles, metric_values, color=student_color, alpha=0.35, s=25)
        ax.plot(
            cycles,
            cumulative_values,
            color=student_color,
            linewidth=2.0,
            drawstyle="steps-post",
            label=student_name,
        )

        if show_mutation_points:
          mutation_cycles = [
              d["cycle"]
              for d in student_metrics
              if d["mutation_strategy"]
              and str(d["mutation_strategy"]).lower() != "nan"
          ]
          if mutation_cycles:
            any_mutations = True
            mutation_indices = [
                cycles.index(mc) for mc in mutation_cycles if mc in cycles
            ]
            mutation_values = [cumulative_values[mi] for mi in mutation_indices]
            ax.scatter(
                mutation_cycles,
                mutation_values,
                color=student_color,
                marker="D",
                s=60,
                edgecolors=edge_color,
                zorder=5,
            )

      if metric_weights:
        weight_str = (
            "1.00"
            if metric_name == "overall_score"
            else f"{metric_weights.get(metric_name, 0):.2f}"
        )
        title = f"{metric_name} (Weight: {weight_str})"
      else:
        title = metric_name

      ax.set_box_aspect(0.88)
      ax.set_title(
          title,
          fontsize=title_fontsize,
          fontweight="bold",
          pad=10,
      )
      ax.set_xlabel("Cycle", fontsize=label_fontsize)
      ax.set_ylabel(metric_name, fontsize=label_fontsize)
      ax.grid(True, linestyle="--", alpha=0.6)
      ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
      ax.tick_params(axis="both", which="major", labelsize=label_fontsize)

    for j in range(num_metrics, nrows * ncols):
      fig.delaxes(axes_flat[j])

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
      if show_mutation_points and any_mutations:
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="D",
                color="w",
                markerfacecolor="gray",
                markersize=7,
                markeredgecolor=edge_color,
                linestyle="None",
            )
        )
        labels.append("Mutation")
      fig.legend(
          handles,
          labels,
          loc="upper center",
          bbox_to_anchor=(0.5, 0.98),
          ncol=min(
              len(student_data_map)
              + (1 if show_mutation_points and any_mutations else 0),
              6,
          ),
          fontsize=legend_fontsize,
      )

    top_rect = 0.90 if nrows == 1 else 0.93
    plt.tight_layout(rect=[0, 0, 1, top_rect], w_pad=2.0, h_pad=2.5)

    if effective_save_path:
      output_path = epath.Path(effective_save_path)
      output_path.parent.mkdir(parents=True, exist_ok=True)
      with output_path.open("wb") as f:
        plt.savefig(f, bbox_inches="tight", dpi=dpi)
    if show:
      plt.show()
    else:
      plt.close(fig)


def plot_final_test(
    metrics: list[str],
    final_results_path: str | None = None,
    final_results_data: FinalResultsData | None = None,
    dark_mode: bool = False,
    fig_width_per_metric: int = 450,
    fig_height: int = 500,
    title_font_size: int = 14,
    axis_font_size: int = 12,
    legend_font_size: int = 12,
    save_path: str | None = None,
    show: bool = False,
):
  """Plots final test evaluation results across First, Last, and Best strategies."""
  if final_results_path:
    with epath.Path(final_results_path).open("r") as f:
      final_results = json.load(f)
      if not final_results:
        print(f"**No data loaded from {final_results_path}.**")
        return
  elif final_results_data:
    final_results = final_results_data
  else:
    raise ValueError(
        "Provide exactly one of `final_results_path` or `final_results_data`."
    )

  if not metrics:
    print("Metrics list cannot be empty.")
    return

  if save_path:
    dark_mode = False

  _plot_final_test_matplotlib(
      final_results,
      metrics,
      dark_mode,
      (fig_width_per_metric / 100, fig_height / 100),
      title_font_size,
      axis_font_size,
      legend_font_size,
      save_path,
      show,
  )


def _plot_final_test_matplotlib(
    final_results,
    metrics,
    dark_mode,
    single_plot_size,
    title_font_size,
    axis_font_size,
    legend_font_size,
    save_path,
    show,
):
  """Plots final test evaluation bars using the Matplotlib backend."""

  if dark_mode:
    plt.style.use("dark_background")
  else:
    plt.style.use("default")

  student_names = [res.get("student_name", "Unknown") for res in final_results]
  plot_metrics = metrics

  fig_width = single_plot_size[0] * len(plot_metrics)
  fig, axes = plt.subplots(
      1,
      len(plot_metrics),
      figsize=(fig_width, single_plot_size[1]),
      squeeze=False,
  )
  axes = axes.flatten()

  bar_configs = {
      "First": {"color": _PALETTE[2], "data_key": "first"},
      "Last": {"color": _PALETTE[0], "data_key": "last"},
      "Best": {"color": _PALETTE[1], "data_key": "best"},
  }

  bar_width = 0.25
  x = np.arange(len(student_names))

  for i, metric in enumerate(plot_metrics):
    ax = axes[i]
    metric_key = get_metric_key(metric)

    offsets = [-bar_width, 0, bar_width]

    for idx, (name, config) in enumerate(bar_configs.items()):
      data_key = config["data_key"]
      values = []
      for res in final_results:
        result_dict = res.get(data_key) or {}
        values.append(result_dict.get(metric_key, np.nan))

      ax.bar(
          x + offsets[idx],
          values,
          bar_width,
          label=name if i == 0 else "",
          color=config["color"],
      )

    ax.set_title(f"Final Score: '{metric.upper()}'", fontsize=title_font_size)
    ax.set_xticks(x)
    ax.set_xticklabels(
        student_names, rotation=-45, ha="left"
    )  # Adjusted for negative angle
    ax.set_ylabel(metric.replace("_", " ").upper(), fontsize=axis_font_size)
    ax.grid(True, linestyle="--", alpha=0.6, axis="y")

  if show or save_path:
    fig.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=3,
        fontsize=legend_font_size,
    )
    plt.tight_layout()

  if save_path:
    epath.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with epath.Path(save_path).open("wb") as f:
      plt.savefig(f, bbox_inches="tight")
  if show:
    plt.show()
  if not show:
    plt.close(fig)  # Prevent leaking plt figure in headless mode
