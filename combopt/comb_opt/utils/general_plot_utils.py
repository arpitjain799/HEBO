import math
from typing import Optional, Dict, Any, Tuple, Union

from scipy import stats

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import FixedLocator

from comb_opt.utils.general_utils import plot_mean_std

POINT_TO_INCH = 0.0138889

# Colorblind
COLORS = [
    (0.00392156862745098, 0.45098039215686275, 0.6980392156862745),
    (0.8705882352941177, 0.5607843137254902, 0.0196078431372549),
    (0.00784313725490196, 0.6196078431372549, 0.45098039215686275),
    (0.8352941176470589, 0.3686274509803922, 0.0),
    (0.8, 0.47058823529411764, 0.7372549019607844),
    (0.792156862745098, 0.5686274509803921, 0.3803921568627451),
    (0.984313725490196, 0.6862745098039216, 0.8941176470588236),
    (0.5803921568627451, 0.5803921568627451, 0.5803921568627451),
    (0.9254901960784314, 0.8823529411764706, 0.2),
    (0.33725490196078434, 0.7058823529411765, 0.9137254901960784),
]

MARKERS = ['o', 'v', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X', '^', '<', '>']


def remove_x_ticks_beyond(ax: Axes, x_low: float, x_up: float):
    """
    Remove ticks at `z` smaller than `x_low` and greater than `x_up`
    """
    major_ticks = ax.get_xticks()
    minor_ticks = np.vstack(
        [np.linspace(major_ticks[i], major_ticks[i + 1], 7)[1:-1] for i in range(len(major_ticks) - 1)]).flatten()
    minor_ticks = [minor_tick for minor_tick in minor_ticks if x_low <= minor_tick <= x_up]
    ax.set_xticks([x for x in ax.get_xticks() if x_low <= x <= x_up])
    ax.xaxis.set_minor_locator(FixedLocator(minor_ticks))


def plot_rect(x_start: float, x_split: float, x_end: float, y_start: float,
              y_end: float, fill_start: float = 0, x_fill_max: Optional[float] = None,
              alpha: float = .3, color: Optional = None,
              linestyle=":", linewidth=1, ax: Optional[Axes] = None, marker=None, **markerkwargs):
    """
    Plot a squared line from (x_start, y_start) to (x_end, y_end), splitting the lines at `x_split`
    """
    if x_fill_max is None:
        x_fill_max = x_split
    if ax is None:
        ax = plt.subplot()

    if y_end == y_start:
        x_split = x_end
    p = ax.plot([x_start, x_split], [y_start, y_start], linestyle=linestyle, color=color, linewidth=linewidth)
    color = p[0].get_color()
    if fill_start > 0:
        ax.fill_between([x_start, min(x_fill_max, x_split)], [y_start - fill_start, y_start],
                        [y_start + fill_start, y_start],
                        alpha=alpha, color=color)
    ax.plot([x_split, x_split], [y_start, y_end], linestyle=linestyle, color=color, linewidth=linewidth)
    ax.plot([x_split, x_end], [y_end, y_end], linestyle=linestyle, color=color, linewidth=linewidth, markevery=[1],
            marker=marker, **markerkwargs)


def get_significance_intervals(data_y: Dict[str, np.ndarray]):
    final_data_y = {k: data_y[k][:, -1] for k in data_y}
    sorted_keys = sorted(data_y.keys(), key=lambda k: final_data_y[k].mean(),
                         reverse=True)
    key_to_rank = {k: i for i, k in enumerate(sorted_keys)}

    p_val = .05
    non_sig_diff_lists = []
    for i in range(len(sorted_keys)):
        ref_key = sorted_keys[i]
        non_sig_diff_list = [ref_key]
        for j in range(i + 1, len(sorted_keys)):
            compare_key = sorted_keys[j]
            # TODO: handle both cases (lower greater)
            if stats.ttest_ind(  # Welch's one-sided t-test?
                    final_data_y[ref_key],
                    final_data_y[compare_key],
                    alternative="greater",
                    equal_var=False).pvalue > p_val:
                non_sig_diff_list.append(compare_key)
        non_sig_diff_lists.append(non_sig_diff_list)

    to_plot_non_sig_diff_lists = [non_sig_diff_lists[0]]
    for i in range(1, len(non_sig_diff_lists)):
        non_sig_diff_list = non_sig_diff_lists[i]
        # check if this list matches the end of the previous list
        if not set(non_sig_diff_list).issubset(set(non_sig_diff_lists[i - 1][-len(non_sig_diff_list):])):
            if i == (len(non_sig_diff_lists) - 1):
                continue
            to_plot_non_sig_diff_lists.append(non_sig_diff_list)

    x_col = []
    max_rank_per_col = {}

    for i, to_plot_non_sig_diff_list in enumerate(to_plot_non_sig_diff_lists):
        min_rank = key_to_rank[to_plot_non_sig_diff_list[0]]
        col = 0
        while col in max_rank_per_col and max_rank_per_col[col] >= min_rank:
            col += 1
        x_col.append(col)
        max_rank_per_col[col] = key_to_rank[to_plot_non_sig_diff_list[-1]]

    return x_col, to_plot_non_sig_diff_lists


def get_split_end(sorted_ys: np.ndarray, y_min: float, y_max: float,
                  x_end_curve: float, x_start_legend: float, default_x_split: float,
                  min_dist_btw_labels: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a sorted list of y values, give the y_ends values that can be used such
    that i'th label corresponding to (x_end_curve, sorted_ys[i]) would be printed
     at the end of a line ending at (x_start_legend, y_ends[i])

    Args:
        sorted_ys: sorted list of y values
        y_min: y min limit of the ax
        y_max: y max limit of the ax
        x_end_curve: curves stop at x_end_curve
        x_start_legend: legend should be printed just after x_start legend
        default_x_split: if line is not squared, value of the x_split
        min_dist_btw_labels: minimum vertical distance between two labels
    """
    y_ends = []
    x_splits = []
    upper_y_limit = max((y_min + y_max + min_dist_btw_labels * (len(sorted_ys) - 1)) / 2, y_max)
    upper_y_limit = max((min(sorted_ys) + max(sorted_ys) + min_dist_btw_labels * (len(sorted_ys) - 1)) / 2,
                        max(sorted_ys))

    next_y = y_min
    for i in range(len(sorted_ys)):
        if sorted_ys[i] < next_y:  # cannot be straight
            y_ends.append(next_y)
        else:  # straight
            y_ends.append(sorted_ys[i])

        next_y = y_ends[-1] + min_dist_btw_labels

    if y_ends[-1] > upper_y_limit:  # readjust from largest to smallest
        next_y = upper_y_limit
        for i in range(-1, -len(sorted_ys) - 1, -1):
            if y_ends[i] > next_y:  # is not straight
                y_ends[i] = min(next_y, y_ends[i])

            next_y = y_ends[i] - min_dist_btw_labels

    y_ends = np.array(y_ends)

    if max(y_ends) > y_max or min(y_ends) < y_min:
        y_ends = y_min + (y_ends - min(y_ends)) * (y_max - y_min) / (max(y_ends) - min(y_ends))

    default_x = default_x_split
    n_to_split_x_up = 0
    n_to_split_x_down = 0
    for i in range(len(y_ends)):
        if y_ends[i] == sorted_ys[i]:
            if n_to_split_x_up > 0:
                x_splits.extend(
                    list((x_start_legend - x_end_curve) * np.linspace(0, 1, n_to_split_x_up + 2)[1:-1] + x_end_curve)[
                    ::-1])
                n_to_split_x_up = 0
            if n_to_split_x_down > 0:
                x_splits.extend(
                    list((x_start_legend - x_end_curve) * np.linspace(0, 1, n_to_split_x_down + 2)[1:-1] + x_end_curve))
                n_to_split_x_down = 0
            x_splits.append(default_x)
        elif y_ends[i] < sorted_ys[i]:
            if n_to_split_x_up > 0:
                x_splits.extend(
                    list((x_start_legend - x_end_curve) * np.linspace(0, 1, n_to_split_x_up + 2)[1:-1] + x_end_curve)[
                    ::-1])
                n_to_split_x_up = 0
            n_to_split_x_down += 1
        elif y_ends[i] > sorted_ys[i]:
            if n_to_split_x_down > 0:
                x_splits.extend(
                    list((x_start_legend - x_end_curve) * np.linspace(0, 1, n_to_split_x_down + 2)[1:-1] + x_end_curve))
                n_to_split_x_down = 0
            n_to_split_x_up += 1

    if n_to_split_x_down > 0:
        x_splits.extend(
            list((x_start_legend - x_end_curve) * np.linspace(0, 1, n_to_split_x_down + 2)[1:-1] + x_end_curve)
        )
    if n_to_split_x_up > 0:
        x_splits.extend(
            list((x_start_legend - x_end_curve) * np.linspace(0, 1, n_to_split_x_up + 2)[1:-1] + x_end_curve)[::-1]
        )

    x_splits = np.array(x_splits)
    return y_ends, x_splits


def get_ax_size(ax: Axes) -> Tuple[float, float]:
    fig = plt.gcf()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    return bbox.width, bbox.height


def plot_curves_with_ranked_legends(
        ax: Axes, data_y: Dict[str, np.ndarray], data_x: Union[np.ndarray, Dict[str, np.ndarray]],
        data_lb: Optional[Union[Dict[str, np.ndarray], np.ndarray, float]] = None,
        data_ub: Optional[Union[Dict[str, np.ndarray], np.ndarray, float]] = None,
        data_key_to_label: Optional[Dict[str, str]] = None, data_marker: Optional[Dict[str, str]] = None,
        data_color: Optional[Dict[str, str]] = None, alpha: float = .3, n_std: float = 1,
        label_fontsize: int = 18, linewidth: int = 3, marker_kwargs: Optional[Dict[str, Any]] = None
):
    """
    Plot curves with legends written vertically with position corresponding to the final values (final regrets, scores,
    ...) on the right of the plot.

    Args:
        data_lb: lower bound for confidence interval (for instance if values are known to be in [0, 1])
        data_ub: upper bound for confidence interval (for instance if values are known to be in [0, 1])

    Returns:
        ax: axis containing the plots
        y_ends: array of vertical positions of the legend
        x_start_legend: x value at which legend lines start
        x_start_legend_text: x value at which labels are written

    """
    default_marker_kwargs = dict(
        markersize=15,
        fillstyle="full",
        markeredgewidth=3,
        markerfacecolor="white",
    )
    if marker_kwargs is None:
        marker_kwargs = default_marker_kwargs

    if data_marker is None:
        data_marker = {data_key: MARKERS[i % len(MARKERS)] for i, data_key in enumerate(data_y)}

    if data_color is None:
        data_color = {data_key: COLORS[i % len(COLORS)] for i, data_key in enumerate(data_y)}

    if data_key_to_label is None:
        data_key_to_label = {data_k: data_k for data_k in data_y}

    if not isinstance(data_x, dict):
        data_x = {data_key: data_x for data_key in data_y}

    if not isinstance(data_lb, dict):
        data_lb = {data_key: data_lb for data_key in data_y}

    if not isinstance(data_ub, dict):
        data_ub = {data_key: data_ub for data_key in data_y}

    _, ax_height = get_ax_size(ax)

    max_x = -np.inf
    min_x = np.inf

    value_for_rank_1 = {}
    value_for_rank_2 = {}

    for data_key in data_y:
        y = data_y[data_key]

        if y.ndim == 1:
            y = y.reshape(1, -1)
        value_for_rank_1[data_key] = y[:, -1].mean()
        value_for_rank_2[data_key] = y.mean()

    sorted_data_keys = sorted(data_y.keys(), key=lambda label: (value_for_rank_1[label], value_for_rank_2[label]))
    rank_of_key = {k: i for i, k in enumerate(sorted_data_keys)}

    for rank, data_key in enumerate(sorted_data_keys):
        X = data_x[data_key]
        y = data_y[data_key]

        if y.ndim == 1:
            y = y.reshape(1, -1)

        max_x = max(max_x, X[-1])
        min_x = min(min_x, X[0])

        markers_on = [i for i in range(0, len(X), math.ceil(len(X) // 4))]
        if (len(X) - 1) not in markers_on:
            markers_on.append(len(X) - 1)

        marker = data_marker.get(data_key)
        color = data_color.get(data_key)
        plot_mean_std(
            X, y, lb=data_lb[data_key], ub=data_ub[data_key],
            linewidth=linewidth, ax=ax, color=color, alpha=alpha, n_std=n_std,
            marker=marker, markevery=markers_on, **marker_kwargs, zorder=rank + 1
        )

    # -------- Plot dotted lines to legend ----------
    ymin, ymax = ax.get_ylim()

    x_start_legend = min_x + (max_x - min_x) * 1.2
    min_dist_btw_labels = (ymax - ymin) / ax_height * max(label_fontsize,
                                                          marker_kwargs["markeredgewidth"] + marker_kwargs[
                                                              "markersize"] + 5) * POINT_TO_INCH * 1.5

    default_x_split = max_x * .5 + .5 * x_start_legend
    y_ends, x_splits = get_split_end(
        sorted_ys=np.array([value_for_rank_1[label] for label in sorted_data_keys]),
        y_min=ymin,
        y_max=ymax,
        x_end_curve=max_x,
        x_start_legend=x_start_legend,
        default_x_split=default_x_split,
        min_dist_btw_labels=min_dist_btw_labels
    )

    x_col_ind, non_sig_interval_lists = get_significance_intervals(data_y=data_y)

    label_offset = (max_x - min_x) * (.05 + 0.01 * len(x_col_ind))
    x_start_label = x_start_legend + label_offset

    x_col = np.linspace(x_start_legend, x_start_label, len(set(x_col_ind)) + 3)[2:-1]

    for i, to_plot_non_sig_diff_list in enumerate(non_sig_interval_lists):
        ax.plot([x_col[x_col_ind[i]] for _ in range(len(to_plot_non_sig_diff_list))],
                [y_ends[rank_of_key[k]] for k in to_plot_non_sig_diff_list], c='k',
                marker=".")

    for i, data_key in enumerate(sorted_data_keys):
        y = data_y[data_key]

        if y.ndim == 1:
            y = y.reshape(1, -1)
        fill_start = 0 if len(y) == 1 else y[:, -1].std() * n_std

        plot_rect(
            x_start=max_x,
            x_split=x_splits[i],
            x_end=x_start_legend,
            y_start=y[:, -1].mean(),
            y_end=y_ends[i],
            fill_start=fill_start,
            x_fill_max=default_x_split,
            alpha=alpha,
            color=data_color[data_key],
            ax=ax,
            marker=data_marker[data_key],
            linewidth=linewidth,
            **marker_kwargs
        )

        text = data_key_to_label[data_key]
        plt.text(x_start_label, y_ends[i], text,
                 fontsize=label_fontsize, va="center", ha="left")

    ax.set_ylim(min(ymin, min(y_ends) - min_dist_btw_labels / 2), max(ymax, max(y_ends) + min_dist_btw_labels / 2))

    # -------- Remove the ticks beyon last x --------
    xlim_min, xlim_max = ax.get_xlim()

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.tick_left()
    ax.xaxis.tick_bottom()

    remove_x_ticks_beyond(ax=ax, x_low=-np.inf, x_up=max_x)

    ax.set_xlim(xlim_min, xlim_max)

    ax.spines["bottom"].set_bounds(xlim_min, max_x)

    # -------- Plot vertical line separating plot and legend -------
    ymin, ymax = ax.get_ylim()

    ax.plot([max_x, max_x], [ymin, ymax], linestyle="--", color="k", linewidth=linewidth, zorder=0)

    ax.set_ylim(ymin, ymax)

    # -----------------------------------------------

    return ax, y_ends, x_start_legend, x_start_label
