"""CLI to apply classification with XGBoost."""
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
from sklearn.metrics import (accuracy_score,
                             f1_score,
                             precision_score,
                             recall_score,
                             roc_auc_score)
from sklearn.model_selection import KFold
from sklearn.utils import class_weight
from xgboost import XGBClassifier

logger.add(
    Path('/tmp')
    / f"turbofan_xgboost_classification_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log")


@click.command()
@click.argument('input_folder', type=click_pathlib.Path(exists=True, readable=True, dir_okay=True))
@click.argument('output_folder', type=click_pathlib.Path(exists=False,
                                                         readable=True, dir_okay=True))
@click.option('--threshold', '-t', type=int)
# pylint: disable=too-many-locals,unused-argument
def train(input_folder: Path, output_folder: Path, threshold: int):
    """
    Train XGBoost to predict the Remaining Useful Life (RUL) on the test set.

    Set a threshold to define if a part is good or near the failure.

    Parameters
    ----------
    input_folder : Path
        Input folder with the preocessed data.
    output_folder : Path
        Output folder to store the results of the training.
    threshold : int
        Threshold to define if a part is still good or not.
        If above, good, otherwise near failure.
    """
    output_folder.mkdir(exist_ok=True, parents=True)
    output_figure = output_folder / 'figures'
    output_figure.mkdir(exist_ok=True)
    output_model = output_folder / 'dump'
    output_model.mkdir(exist_ok=True)
    output_metric = output_folder / 'metrics'
    output_metric.mkdir(exist_ok=True)
    df_train = pd.read_parquet(Path(input_folder) / 'df_train_with_var_norm.parquet')

    df_X = df_train.drop('label', axis='columns')
    df_y = (df_train[['label']] < threshold).astype(int)
    class_weights = class_weight.compute_class_weight('balanced', classes=df_y['label'].unique(),
                                                      y=df_y['label'])
    df_y = df_y.assign(weight=lambda x: class_weights[x['label']])

    xgb_models = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    df_predictions = (pd.DataFrame(index=range(len(df_X)))
                      .assign(prediction=None, prediction_proba=None, label=None)
                      .astype({'prediction': 'float32', 'prediction_proba': 'float32',
                               'label': 'float32'})
                      )

    kfold = KFold(n_splits=5, random_state=10, shuffle=True)

    for i, (train_index, test_index) in enumerate(kfold.split(df_X.values)):
        X_train, X_test = df_X.values[train_index], df_X.values[test_index]
        y_train, y_test = df_y['label'].values[train_index], df_y['label'].values[test_index]

        xgb_model = XGBClassifier(max_depth=7, learning_rate=0.03, min_child_weight=2,
                                  subsample=0.7, colsample_bytree=0.7, n_estimators=500,
                                  verbosity=1, n_jobs=24, use_label_encoder=False)
        xgb_model.fit(X_train, y_train, sample_weight=df_y['weight'].values[train_index])

        # Save the model.
        pickle.dump(xgb_model, (output_model / f'model_{i}.pkl').open(mode='wb'))

        xgb_models.append(xgb_model)

        preds = xgb_model.predict(X_test)
        preds_bin = np.where(preds >= 0.5, 1, 0)

        # Assign the predictions and label used in the test set.
        for k, j in enumerate(test_index):
            df_predictions.at[j, 'prediction'] = preds_bin[k]
            df_predictions.at[j, 'prediction_proba'] = preds[k]
            df_predictions.at[j, 'label'] = y_test[k]

        # Save the metrics.
        accuracies.append(accuracy_score(y_test, preds_bin))
        precisions.append(precision_score(y_test, preds_bin))
        recalls.append(recall_score(y_test, preds_bin))
        f1_scores.append(f1_score(y_test, preds_bin, average=None))

        logger.info(f'Metrics fold {i}:')
        logger.info(f'Accuracy: {accuracies[-1]}')
        logger.info(f'Precision: {precisions[-1]}')
        logger.info(f'Recall: {recalls[-1]}')
        logger.info(f'f1_scores: {f1_scores[-1]}')

    df_predictions = df_predictions.set_index(df_X.index)

    results = {
        'accuracy_std': np.std(accuracies),
        'accuracy_mean': np.mean(accuracies),
        'precision_std': np.std(precisions),
        'precision_mean': np.mean(precisions),
        'recall_std': np.std(recalls),
        'recall_mean': np.mean(recalls),
        'roc_auc_score': roc_auc_score(df_predictions['label'].values,
                                       df_predictions['prediction'].values),
        'f1_score_ok_std': np.std([i[0] for i in f1_scores]),
        'f1_score_ok_mean': np.mean([i[0] for i in f1_scores]),
        'f1_score_ko_std': np.std([i[1] for i in f1_scores]),
        'f1_score_ko_mean': np.mean([i[1] for i in f1_scores]),
        'params': xgb_model.get_params()
    }

    with open(output_metric / 'performance.json', 'w') as f:
        json.dump(results, f)

    logger.info(f'Results:\n{pprint.pformat(results)}')

    # Plots of confusion matrix.
    bplt.plot_confusion_matrix(df_predictions['label'],
                               df_predictions['prediction'],
                               ticklabels=['Healthy', 'Near failure'],
                               title=f'Confusion matrix - threshold={threshold}',
                               rotation_xticks=0,
                               stats='f1-score')
    plt.savefig(output_figure / 'confusion_matrix.png', transparent=False, bbox_inches='tight')
    # Plots of ROC curve.
    bplt.plot_roc_curve(df_predictions['label'],
                        df_predictions['prediction'],
                        title=f'ROC curve - threshold={threshold}')
    plt.savefig(output_figure / 'roc_curve.png', transparent=False, bbox_inches='tight')


if __name__ == '__main__':
    train()  # pylint: disable=no-value-for-parameter
