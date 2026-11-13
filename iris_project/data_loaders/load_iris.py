"""
Mage block: load_iris
Type: data_loader
Pipeline: iris_classifier

Loads the Iris dataset from sklearn and returns it as a DataFrame.
Guard pattern keeps it importable outside Mage (for pytest CI).
"""

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

import pandas as pd
from sklearn.datasets import load_iris


@data_loader
def load_data(*args, **kwargs):
    iris = load_iris(as_frame=True)
    df = iris.frame
    print('Rows loaded:', len(df))
    return df


@test
def test_output(output, *args):
    assert len(output) == 150, 'Iris must have 150 rows'
