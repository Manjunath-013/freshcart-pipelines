"""
Mage block: evaluate_model
Type: transformer
Pipeline: iris_classifier

Evaluates predictions from train_model: prints confusion matrix and
returns a metrics DataFrame (accuracy, precision, recall, f1_score).
Based on the real instructor Mage notebook (Latest/Maze_ML_PipeLine.ipynb).
"""

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def evaluate(df, *args, **kwargs):
    y_true = df['target']
    y_pred = df['predicted']

    print('Confusion matrix:')
    print(pd.crosstab(y_true, y_pred, rownames=['actual'], colnames=['predicted']))

    metrics = pd.DataFrame({
        'metric': ['accuracy', 'precision', 'recall', 'f1_score'],
        'value': [
            accuracy_score(y_true, y_pred),
            precision_score(y_true, y_pred, average='weighted', zero_division=0),
            recall_score(y_true, y_pred, average='weighted', zero_division=0),
            f1_score(y_true, y_pred, average='weighted', zero_division=0),
        ]
    })
    print(metrics.to_string(index=False))
    return metrics


@test
def test_metrics_shape(output, *args):
    assert len(output) == 4, 'Expected 4 metric rows'
    assert 'metric' in output.columns and 'value' in output.columns
