"""CLI to apply regression with XGBoost."""
from datetime import datetime
from pathlib import Path
import pickle
import pprint
import json

import bff.plot as bplt
import click
import click_pathlib
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

logger.add(
    Path('/tmp') / f"turbofan_xgboost_regression_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log")


@click.command()
@click.argument('input_folder', type=click_pathlib.Path(exists=True, readable=True, dir_okay=True))
@click.argument('output_folder', type=click_pathlib.Path(exists=False,
                                                         readable=True, dir_okay=True))
@click.option('--threshold', '-t', type=int, default=400)
# pylint: disable=too-many-locals,unused-argument
def train(input_folder: Path, output_folder: Path, threshold: int = 400):
    """
    Train XGBoost to predict the Remaining Useful Life (RUL) on the test set.

    Train only on units having more than a given RUL.

    Parameters
    ----------
    input_folder : Path
        Input folder with the preocessed data.
    output_folder : Path
        Output folder to store the results of the training.
    threshold : int, default 400
        Threshold fo the parts to select for the training.
        Only part having more than the threshold will be used.
    """
    output_folder.mkdir(exist_ok=True, parents=True)
    output_figure = output_folder / 'figures'
    output_figure.mkdir(exist_ok=True)
    output_model = output_folder / 'dump'
    output_model.mkdir(exist_ok=True)
    output_metric = output_folder / 'metrics'
    output_metric.mkdir(exist_ok=True)
    df_train = pd.read_parquet(Path(input_folder) / 'df_train_with_var_norm.parquet')

    # pylint: disable=unused-variable
    units = df_train.query('label > @threshold').index.unique(level=0)
    df_X = df_train.query('unit_number in @units and experiment_number == 0')

    xgb_models = []
    scores = []

    df_predictions = (pd.DataFrame(index=range(len(df_X)))
                      .assign(prediction=None, label=None)
                      .astype({'prediction': 'float32', 'label': 'float32'})
                      )

    kfold = KFold(n_splits=5, random_state=10, shuffle=True)

    for i, (train_index, test_index) in enumerate(kfold.split(df_X.values)):
        X_train, X_test = (df_X.drop('label', axis='columns').values[train_index],
                           df_X.drop('label', axis='columns').values[test_index])
        y_train, y_test = df_X['label'].values[train_index], df_X['label'].values[test_index]

        xgb_model = XGBRegressor(max_depth=7, learning_rate=0.03, min_child_weight=2, subsample=0.7,
                                 colsample_bytree=0.7, n_estimators=500, verbosity=1, n_jobs=24)
        xgb_model.fit(X_train, y_train)

        # Save the model.
        pickle.dump(xgb_model, (output_model / f'model_{i}.pkl').open(mode='wb'))

        xgb_models.append(xgb_model)

        preds = xgb_model.predict(X_test)

        # Assign the predictions and label used in the test set.
        for k, j in enumerate(test_index):
            df_predictions.at[j, 'prediction'] = preds[k]
            df_predictions.at[j, 'label'] = y_test[k]

        score = metrics.mean_squared_error(y_test, preds)

        logger.info(f'Fold {i}: mse={score}')
        scores.append(score)

    df_predictions = df_predictions.set_index(df_X.index)

    results = {
        'mse_std': np.std(scores),
        'mse_mean': np.mean(scores),
        'params': xgb_model.get_params()
    }

    with open(output_metric / 'performance.json', 'w') as f:
        json.dump(results, f)

    logger.info(f'Results:\n{pprint.pformat(results)}')

    # Plots of predictions.
    ax = bplt.plot_true_vs_pred(df_predictions['label'], df_predictions['prediction'],
                                title='Predicted vs Actual with XGB', with_identity=True, alpha=0.3)
    bplt.set_thousands_separator(ax, nb_decimals=0)
    plt.savefig(output_figure / 'true_vs_pred_xgboost.png', transparent=False, bbox_inches='tight')


if __name__ == '__main__':
    train()  # pylint: disable=no-value-for-parameter
