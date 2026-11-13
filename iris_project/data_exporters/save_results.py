"""
Mage block: save_results
Type: data_exporter
Pipeline: iris_classifier

Saves test predictions to iris_predictions.csv.
Based on the real instructor Mage notebook (Latest/Maze_ML_PipeLine.ipynb).
"""

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

import pandas as pd


@data_exporter
def export_data(df, *args, **kwargs):
    df.to_csv('iris_predictions.csv', index=False)
    print('Saved', len(df), 'predictions to iris_predictions.csv')
