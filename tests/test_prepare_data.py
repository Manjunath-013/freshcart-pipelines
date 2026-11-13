# tests/test_prepare_data.py
"""
Part A3 — CI pytest test for the prepare_data Mage block.

The block uses the guard pattern:
    if 'transformer' not in globals():
        from mage_ai.data_preparation.decorators import transformer

so it can be imported outside Mage (the decorator becomes a no-op identity wrapper).
"""

from sklearn.datasets import load_iris
from iris_project.transformers.prepare_data import transform


def test_split_is_80_20():
    """Default test_size=0.2 must produce exactly 120 train and 30 test rows."""
    df = load_iris(as_frame=True).frame
    out = transform(df)
    assert (out['split'] == 'train').sum() == 120, 'Expected 120 train rows'
    assert (out['split'] == 'test').sum() == 30,   'Expected 30 test rows'


def test_split_column_values():
    """'split' column must only contain the values 'train' and 'test'."""
    df = load_iris(as_frame=True).frame
    out = transform(df)
    assert set(out['split'].unique()) == {'train', 'test'}


def test_no_data_loss():
    """Total rows must remain 150 after the split."""
    df = load_iris(as_frame=True).frame
    out = transform(df)
    assert len(out) == 150, f'Expected 150 rows, got {len(out)}'
