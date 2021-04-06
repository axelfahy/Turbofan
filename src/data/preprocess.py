"""CLI to apply preprocessing and create features for training."""
from datetime import datetime
from pathlib import Path

import click
from loguru import logger
from sklearn.preprocessing import StandardScaler
import pandas as pd

logger.add(
    Path('/tmp') / f"turbofan_preprocess_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log")


@click.command()
@click.argument('input_folder', type=click.Path())
@click.argument('output_folder', type=click.Path())
def preprocess(input_folder: Path, output_folder: Path):
    """
    Load the dataset and apply some preprocessing.

    - Remove the features having no variance.
    - Standardize the data.

    Parameters
    ----------
    input_folder : Path
        Input folder with the raw data.
    output_folder : Path
        Output folder to store the processed data, as parquet.
    """
    logger.info(f'Preprocessing of dataset from {input_folder} to {output_folder}')
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Load the data.
    df_train = (pd.read_parquet(Path(input_folder) / 'df_train.parquet')
                .drop('max_time', axis='columns')
                .dropna(axis='columns')
                .set_index(['unit_number', 'time', 'experiment_number'])
                )
    df_test = (pd.read_parquet(Path(input_folder) / 'df_test.parquet')
               .drop(['max_test', 'max_time'], axis='columns')
               .drop('rul', axis='columns')
               .dropna(axis='columns')
               .set_index(['unit_number', 'time', 'experiment_number'])
               )
    # Remove features having low variance.
    df_train_with_var = df_train.loc[:, df_train.std() > 0.3].drop('label', axis='columns')
    df_test_with_var = df_test.loc[:, df_train_with_var.columns]

    # Standardize the data.
    scaler = StandardScaler()
    df_train_norm = (pd.DataFrame(scaler.fit_transform(df_train_with_var),
                                  columns=df_train_with_var.columns,
                                  index=df_train_with_var.index)
                     .assign(label=df_train['label'])
                     )
    df_test_norm = (pd.DataFrame(scaler.transform(df_test_with_var),
                                 columns=df_test_with_var.columns,
                                 index=df_test_with_var.index)
                    .assign(label=df_test['label'])
                    )
    df_train_norm.to_parquet(Path(output_folder) / 'df_train_with_var_norm.parquet')
    df_test_norm.to_parquet(Path(output_folder) / 'df_test_with_var_norm.parquet')


if __name__ == '__main__':
    preprocess()  # pylint: disable=no-value-for-parameter
