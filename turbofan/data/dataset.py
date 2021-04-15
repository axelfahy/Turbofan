"""Dataset functions."""
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

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
