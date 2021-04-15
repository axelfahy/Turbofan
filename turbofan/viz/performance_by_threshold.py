"""CLI to plot the performance by threshold."""
from datetime import datetime
import json
from pathlib import Path

import click
import click_pathlib
from loguru import logger
import matplotlib.pyplot as plt

from turbofan.viz import plot_performance_by_threshold

logger.add(
    Path('/tmp')
    / f"turbofan_performance_by_threshold_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log")


@click.command()
@click.argument('input_folder', type=click_pathlib.Path(exists=True, readable=True, dir_okay=True))
@click.argument('output_folder', type=click_pathlib.Path(exists=False,
                                                         readable=True, dir_okay=True))
def performance_by_threshold(input_folder: Path, output_folder: Path):
    """
    Plot the performance by threshold.

    Parameters
    ----------
    input_folder : Path
        Input folder with the interim data.
    output_folder : Path
        Output folder to store the visualization.
    """
    logger.info('Plot the performance by threshold')
    # Retrieve the performances.
    scores = {}
    for p in input_folder.glob('*/metrics/performance.json'):
        with p.open('r') as f:
            scores[int(p.parent.parent.name)] = json.load(f)

    plot_performance_by_threshold(scores)
    plt.savefig(Path(output_folder) / 'performance_by_threshold.png', transparent=False)


if __name__ == '__main__':
    performance_by_threshold()  # pylint: disable=no-value-for-parameter
