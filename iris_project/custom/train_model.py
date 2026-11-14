"""
Mage block: train_model
Type: custom (transformer)
Pipeline: iris_classifier

Trains a Decision Tree classifier on the training split.
The @test gate prevents save_results from running if accuracy is too low.
"""

try:
    from mage_ai.data_preparation.decorators import custom, test
except ImportError:
    def custom(fn):
        return fn

    def test(fn):
        return fn


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle


FEATURE_COLS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]
MIN_ACCURACY = 0.85  # gate threshold


@custom
def transform_custom(df: pd.DataFrame, *args, **kwargs) -> dict:
    """
    Train a DecisionTreeClassifier and return a results dict.

    Runtime variable supported:
        max_depth (int): tree max depth, default 3
    """
    max_depth = int(kwargs.get("max_depth", 3))

    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    X_train = train_df[FEATURE_COLS]
    y_train = train_df["target"]
    X_test = test_df[FEATURE_COLS]
    y_test = test_df["target"]

    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"max_depth={max_depth}  test_accuracy={accuracy:.4f}")

    return {
        "model": model,
        "accuracy": accuracy,
        "max_depth": max_depth,
        "y_test": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        "feature_names": FEATURE_COLS,
    }


@test
def test_accuracy_gate(result: dict) -> None:
    """
    Gate: accuracy must be >= MIN_ACCURACY.
    If this @test fails, Mage stops the pipeline and save_results is never called.
    """
    acc = result["accuracy"]
    assert acc >= MIN_ACCURACY, (
        f"Accuracy {acc:.4f} < threshold {MIN_ACCURACY}. "
        "Pipeline stopped — model not saved."
    )

