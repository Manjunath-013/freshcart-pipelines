"""
Mage block: prepare_data
Type: transformer
Pipeline: iris_classifier

Splits the Iris DataFrame into train (80 %) and test (20 %) sets,
adds a 'split' column, and shuffles with a fixed random seed.
Can be imported outside Mage (decorators are conditionally imported).
"""

try:
    from mage_ai.data_preparation.decorators import test, transformer
except ImportError:
    def transformer(fn):
        return fn

    def test(fn):
        return fn


import pandas as pd
from sklearn.model_selection import train_test_split


@transformer
def transform(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    """
    Add a 'split' column ('train' | 'test') using an 80/20 stratified split.

    Runtime variable supported:
        test_size (float): override the default 0.2 (passed via --runtime-vars)
    """
    test_size = float(kwargs.get("test_size", 0.2))  # ← FIXED back to 0.2
    random_state = int(kwargs.get("random_state", 42))

    train_idx, test_idx = train_test_split(
        df.index,
        test_size=test_size,
        random_state=random_state,
        stratify=df["target"],
    )

    df = df.copy()
    df["split"] = "train"
    df.loc[test_idx, "split"] = "test"

    n_train = (df["split"] == "train").sum()
    n_test = (df["split"] == "test").sum()
    print(f"Split: {n_train} train / {n_test} test  (test_size={test_size})")
    return df


@test
def test_split_ratio(df: pd.DataFrame) -> None:
    """With default test_size=0.2: 120 train, 30 test."""
    assert (df["split"] == "train").sum() == 120, "Expected 120 train rows"
    assert (df["split"] == "test").sum() == 30, "Expected 30 test rows"


@test
def test_split_column_exists(df: pd.DataFrame) -> None:
    """'split' column must be present."""
    assert "split" in df.columns, "Missing 'split' column"


@test
def test_no_data_loss(df: pd.DataFrame) -> None:
    """Total rows must still be 150."""
    assert len(df) == 150, f"Row count changed: {len(df)}"
