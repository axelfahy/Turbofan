"""Plot functions."""
from typing import Dict, Optional, Sequence, Union

import bff.plot as bplt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_distribution(data: Union[pd.DataFrame, np.array], column: Optional[str] = None,
                      title: str = 'Distribution of label', n_bins: int = 10,
                      ylim: Optional[float] = None):
    """
    Plot the distribution of the DataFrame.

    Parameters
    ----------
    data : pd.DataFrame or np.array
        Data with the distribution to plot.
    column : str, optional
        Column of the DataFrame to plot.
        If not provided, use all columns and flatten the data.
    title : str, default 'Distribution of label'
        Title to use for the plot.
    n_bins : int, default 10
        Number of bins for the histogram.
    ylim : float, optional
        Set an y limit if provided.
    """
    _, ax = plt.subplots(1, 1, figsize=(12, 8))

    if column is None:
        if len(data.shape) > 1:
            data = data.flatten()
        ax.hist(data, n_bins)
        col_name = 'Value'
    else:
        ax.hist(data[column], n_bins)
        col_name = column

    ax.set_title(title, fontsize=14)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_xlabel(col_name.capitalize(), fontsize=12)

    if ylim is not None:
        ax.set_ylim(ylim)

    # Style.
    # Remove border on the top and right.
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Set alpha on remaining borders.
    ax.spines['left'].set_alpha(0.4)
    ax.spines['bottom'].set_alpha(0.4)

    # Only show ticks on the left and bottom spines
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # Style of ticks.
    plt.xticks(fontsize=10, alpha=0.7)
    plt.yticks(fontsize=10, alpha=0.7)

    ax.axes.grid(True, which='major', axis='y', color='black', alpha=0.3, linestyle='--', lw=0.5)
    bplt.set_thousands_separator(ax, which='y', nb_decimals=0)
    return ax


def plot_performance_by_threshold(scores: Dict, metrics: Sequence[str] = None,
                                  title: str = 'Performance by threshold') -> plt.axes:
    """
    Plot the performance by threshold.

    The accuracy, precision, recall and f1-score are plot for each threshold.

    Parameters
    ----------
    scores : Dict
        Dict with the scores to plot (accuracy, precision, recall, ...).
        The key is the threshold, the value a dict with the metrics.
    metrics : Sequence of str
        List with the metrics to plot.
    title : str, default 'Performance by threshold'
        Title for the plot.

    Returns
    -------
    plt.axes
        The axis of the plot.
    """
    metrics = metrics or ['accuracy_mean', 'precision_mean', 'recall_mean']
    _, ax = plt.subplots(1, 1, figsize=(16, 8))

    thresholds = scores.keys()
    colors = bplt.get_n_colors(len(metrics))

    for metric, c in zip(metrics, colors):
        values = [v[metric] for _, v in sorted(scores.items())]
        ax.plot(thresholds, values, label=metric.capitalize(), color=c,
                marker='D', linestyle='none')

    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Threshold (number of cycles defined as failure)', fontsize=12)
    ax.set_ylabel('Performance', fontsize=12)

    ax.grid(True, which='major', axis='y', alpha=0.6, linestyle='--', lw=0.5)

    # Style.
    # Remove border on the top and right.
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Set alpha on remaining borders.
    ax.spines['left'].set_alpha(0.4)
    ax.spines['bottom'].set_alpha(0.4)

    # Remove ticks on y axis.
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('none')

    ax.legend()
    return ax
