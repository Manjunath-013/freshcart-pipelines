"""
Mage block: validate_data  (C4 — custom block with @test)
Type: custom (transformer)
Pipeline: iris_classifier

Inserted after load_iris. Validates:
  - no missing values
  - exactly 3 target classes
  - feature values are non-negative
  - no duplicate rows

This is Part C4: "your own block with its own @test".
"""

try:
    from mage_ai.data_preparation.decorators import custom, test
except ImportError:
    def custom(fn):
        return fn

    def test(fn):
        return fn


import pandas as pd


@custom
def transform_custom(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    """
    Run data quality checks on the raw Iris DataFrame.
    Passes the DataFrame through unchanged (pure validation step).
    """
    # Check 1: no nulls
    null_count = df.isnull().sum().sum()
    print(f"[validate_data] null values: {null_count}")

    # Check 2: exactly 3 classes
    n_classes = df["target"].nunique()
    print(f"[validate_data] distinct classes: {n_classes}")

    # Check 3: non-negative feature values
    feature_cols = [c for c in df.columns if c != "target" and c != "split"]
    neg_count = (df[feature_cols] < 0).sum().sum()
    print(f"[validate_data] negative feature values: {neg_count}")

    # Check 4: no duplicate rows
    dup_count = df.duplicated().sum()
    print(f"[validate_data] duplicate rows: {dup_count}")

    return df


@test
def test_data_quality(df: pd.DataFrame) -> None:
    """All quality gates must pass."""
    assert df.isnull().sum().sum() == 0, "Found null values"
    assert df["target"].nunique() == 3, "Expected exactly 3 classes"

    feature_cols = [c for c in df.columns if c not in ("target", "split")]
    assert (df[feature_cols] < 0).sum().sum() == 0, "Found negative feature values"

