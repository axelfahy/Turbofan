"""CLI to create the dataset."""
from datetime import datetime
from pathlib import Path

import click
from loguru import logger

from turbofan.data import create_test_set, create_train_set

logger.add(
    Path('/tmp') / f"turbofan_create_dataset_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log")


@click.command()
@click.argument('input_folder', type=click.Path())
@click.argument('output_folder', type=click.Path())
def create_dataset(input_folder: Path, output_folder: Path):
    """
    Create a dataset from the raw data.

    This will merge the training and testing files as parquet files.

    Parameters
    ----------
    input_folder : Path
        Input folder with the raw data.
    output_folder : Path
        Output folder to store the processed data, as parquet.
    """
    logger.info(f'Creating dataset from {input_folder} to {output_folder}')
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    create_train_set(Path(input_folder), Path(output_folder) / 'df_train.parquet')
    create_test_set(Path(input_folder), Path(output_folder) / 'df_test.parquet')


if __name__ == '__main__':
    create_dataset()  # pylint: disable=no-value-for-parameter
