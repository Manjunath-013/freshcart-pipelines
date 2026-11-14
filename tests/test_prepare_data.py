# tests/test_prepare_data.py
"""
CI pytest test for the prepare_data Mage block.

The block can be imported outside Mage because it guards its decorator
imports with the `if 'transformer' not in globals()` pattern.
"""

from sklearn.datasets import load_iris
from iris_project.transformers.prepare_data import transform


def test_split_is_80_20():
    """Default test_size=0.2 must produce 120 train and 30 test rows."""
    df = load_iris(as_frame=True).frame
    out = transform(df)
    assert (out['split'] == 'train').sum() == 120, 'Expected 120 train rows'
    assert (out['split'] == 'test').sum() == 30, 'Expected 30 test rows'


def test_split_column_values():
    """'split' column must only contain 'train' or 'test'."""
    df = load_iris(as_frame=True).frame
    out = transform(df)
    assert set(out['split'].unique()) == {'train', 'test'}


def test_no_data_loss():
    """Total rows must remain 150 after split."""
    df = load_iris(as_frame=True).frame
    out = transform(df)
    assert len(out) == 150
