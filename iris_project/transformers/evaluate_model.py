"""
iris_project/transformers/evaluate_model.py
Mage block: evaluate_model  |  type: transformer
Pipeline: iris_classifier  (downstream branch from train_model — Part C3)

Prints confusion matrix and per-class metrics table.
Runs in parallel with save_results after train_model's @test passes.
"""

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

try:
    from mage_ai.data_preparation.decorators import test, transformer
except ImportError:
    def transformer(fn):
        return fn

    def test(fn):
        return fn


@transformer
def evaluate(df, *args, **kwargs):
    y_true = df['target']
    y_pred = df['predicted']

    print('Confusion matrix:')
    print(pd.crosstab(y_true, y_pred, rownames=['actual'], colnames=['predicted']))

    metrics = pd.DataFrame({
        'metric': ['accuracy', 'precision', 'recall', 'f1_score'],
        'value': [
            round(accuracy_score(y_true, y_pred), 3),
            round(precision_score(y_true, y_pred, average='macro'), 3),
            round(recall_score(y_true, y_pred, average='macro'), 3),
            round(f1_score(y_true, y_pred, average='macro'), 3),
        ]
    })
    print('\nMetrics:')
    print(metrics.to_string(index=False))
    return metrics
