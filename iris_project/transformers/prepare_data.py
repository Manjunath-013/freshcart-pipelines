"""
Mage block: prepare_data  — INTENTIONALLY BROKEN for CI red-run demo (Part A3)
Reverts immediately in the next commit.
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
    BROKEN: test_size changed to 0.3 to trigger red CI run.
    """
    test_size = float(kwargs.get("test_size", 0.3))  # ← BROKEN: was 0.2
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
    assert "split" in df.columns, "Missing 'split' column"


@test
def test_no_data_loss(df: pd.DataFrame) -> None:
    assert len(df) == 150, f"Row count changed: {len(df)}"
