"""CLI to plot some explorations about the dataset."""
from datetime import datetime
from pathlib import Path

import bff.plot as bplt
import click
from loguru import logger
import matplotlib.pyplot as plt
import pandas as pd

logger.add(
    Path('/tmp') / f"turbofan_exploration_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log")


def plot_distribution(df: pd.DataFrame, title: str = 'Distribution of label'):
    """
    Plot the distribution of the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the distribution to plot.
    title : str, default 'Distribution of label'
        Title to use for the plot.
    """
    n_bins = 10

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.hist(df['label'], n_bins)
    ax.set_title(title, fontsize=14)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_xlabel('Label', fontsize=12)
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


@click.command()
@click.argument('input_folder', type=click.Path())
@click.argument('output_folder', type=click.Path())
def exploration(input_folder: Path, output_folder: Path):
    """
    Some exploration on the dataset.

    Parameters
    ----------
    input_folder : Path
        Input folder with the interim data.
    output_folder : Path
        Output folder to store the visualization.
    """
    logger.info(f'Visualization from {input_folder} to {output_folder}')
    Path(output_folder).mkdir(exist_ok=True)
    df_train = pd.read_parquet(Path(input_folder) / 'df_train.parquet')
    df_test = pd.read_parquet(Path(input_folder) / 'df_test.parquet')

    logger.info('Saving distribution of label from train set.')
    plot_distribution(df_train, title='Distribution of labels')
    plt.savefig(Path(output_folder) / '00_distribution_label_train.png', transparent=True)

    logger.info('Saving distribution of label from test set.')
    plot_distribution(df_test, title='Distribution of test set')
    plt.savefig(Path(output_folder) / '01_distribution_label_test.png', transparent=True)


if __name__ == '__main__':
    exploration()  # pylint: disable=no-value-for-parameter
