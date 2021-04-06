"""CLI to create the dataset."""
from datetime import datetime
from pathlib import Path

import click
from loguru import logger
import pandas as pd

logger.add(
    Path('/tmp') / f"turbofan_create_dataset_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log")


HEADER = ['unit_number', 'time'] + \
         [f'op_settings{i}' for i in range(1, 4)] + [f'sensor{i}' for i in range(1, 27)]


def create_train_set(input_folder: Path, output: Path):
    """
    Merge the training files from the input folder.

    Parameters
    ----------
    input_folder : Path
        Folder with the raw data.
    output : Path
        Output file (parquet file).
    """
    # Merge the training files.
    dfs = []
    for i, f in enumerate(sorted([p for p in input_folder.iterdir() if 'train' in p.as_posix()])):
        df_raw = pd.read_csv(f, sep=' ', header=None, names=HEADER)
        logger.info(f'Experiment {i}: {f} with {df_raw.shape[0]} entries.')
        # The label is the difference with the time at failure and the current time.
        df = (df_raw
              .assign(experiment_number=i)
              .assign(max_time=df_raw.groupby('unit_number')['time'].transform('max'))
              .assign(label=lambda x: x['max_time'] - x['time'])
              )
        dfs.append(df)
    df_merged = pd.concat(dfs)
    df_merged.to_parquet(str(output))


def create_test_set(input_folder: Path, output: Path):
    """
    Merge the training files from the input folder.

    Parameters
    ----------
    input_folder : Path
        Folder with the raw data.
    output : Path
        Output file (parquet file).
    """
    # Merge the testing files.
    dfs = []
    for i, f in enumerate(sorted([p for p in input_folder.iterdir() if 'test' in p.as_posix()])):
        df_raw = pd.read_csv(f, sep=' ', header=None, names=HEADER)
        df_rul = pd.read_csv(f.as_posix().replace('test', 'RUL'), header=None, names=['rul'])
        logger.info(f'Experiment {i + 1}: {f} with {df_raw.shape[0]} entries.')
        # The label is the difference with the time at failure and the current time.
        df = (df_raw
              .merge(df_rul, left_on=df_raw['unit_number'], right_on=df_rul.index)
              .drop('key_0', axis='columns')
              .assign(experiment_number=i + 1)
              .assign(max_test=df_raw.groupby('unit_number')['time'].transform('max'))
              .assign(max_time=lambda x: x['max_test'] + x['rul'])
              .assign(label=lambda x: x['max_time'] - x['time'])
              )
        dfs.append(df)
    df_merged = pd.concat(dfs)
    df_merged.to_parquet(str(output))


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
