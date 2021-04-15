"""CLI to plot some explorations about the dataset."""
from datetime import datetime
from pathlib import Path

import click
import click_pathlib
from loguru import logger
import matplotlib.pyplot as plt
import pandas as pd

from turbofan.viz import plot_distribution

logger.add(
    Path('/tmp') / f"turbofan_exploration_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log")


@click.command()
@click.argument('input_folder', type=click_pathlib.Path(exists=True, readable=True, dir_okay=True))
@click.argument('output_folder', type=click_pathlib.Path(exists=False,
                                                         readable=True, dir_okay=True))
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
    plot_distribution(df_train, 'label', title='Distribution of labels')
    plt.savefig(Path(output_folder) / '00_distribution_label_train.png', transparent=True)

    logger.info('Saving distribution of label from test set.')
    plot_distribution(df_test, 'label', title='Distribution of test set')
    plt.savefig(Path(output_folder) / '01_distribution_label_test.png', transparent=True)


if __name__ == '__main__':
    exploration()  # pylint: disable=no-value-for-parameter
