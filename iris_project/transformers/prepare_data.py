"""
Mage block: prepare_data
Type: transformer
Pipeline: iris_classifier

Splits the Iris DataFrame 80/20, adds a 'split' column.
Guard pattern keeps it importable outside Mage (for pytest CI).
"""

import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from mage_ai.data_preparation.decorators import test, transformer
except ImportError:
    def transformer(fn):
        return fn

    def test(fn):
        return fn


@transformer
def transform(df, *args, **kwargs):
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df['split'] = 'train'
    test_df['split'] = 'test'
    result = pd.concat([train_df, test_df])
    print('Train rows:', len(train_df), '| Test rows:', len(test_df))
    return result


@test
def test_output(output, *args):
    assert 'split' in output.columns, 'split column missing'
