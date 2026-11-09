"""
Mage block: load_iris
Type: data_loader
Pipeline: iris_classifier

Loads the Iris dataset from sklearn and returns it as a DataFrame.
Can be imported outside Mage (decorators are conditionally imported).
"""

try:
    from mage_ai.data_preparation.decorators import data_loader, test
except ImportError:
    # Allow importing outside the Mage runtime (e.g., in pytest)
    def data_loader(fn):
        return fn

    def test(fn):
        return fn


import pandas as pd
from sklearn.datasets import load_iris


@data_loader
def load_data(*args, **kwargs) -> pd.DataFrame:
    """
    Load the Iris dataset and return it as a DataFrame.
    Columns: sepal length (cm), sepal width (cm),
             petal length (cm), petal width (cm), target
    """
    iris = load_iris(as_frame=True)
    df = iris.frame
    df.columns = [c.replace(" ", "_").replace("(cm)", "").strip("_") for c in df.columns]
    print(f"Loaded {len(df)} rows × {len(df.columns)} columns")
    return df


@test
def test_shape(df: pd.DataFrame) -> None:
    """Iris must have exactly 150 rows and 5 columns."""
    assert df.shape == (150, 5), f"Expected (150, 5), got {df.shape}"


@test
def test_no_nulls(df: pd.DataFrame) -> None:
    """No missing values allowed."""
    assert df.isnull().sum().sum() == 0, "Found null values in dataset"


@test
def test_three_classes(df: pd.DataFrame) -> None:
    """Iris must have exactly 3 target classes (0, 1, 2)."""
    assert df["target"].nunique() == 3, "Expected 3 target classes"
