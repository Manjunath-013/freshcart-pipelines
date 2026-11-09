"""
Mage block: evaluate_model
Type: custom
Pipeline: iris_classifier

Downstream branch from train_model.
Prints a confusion matrix and a per-class metrics table.
"""

try:
    from mage_ai.data_preparation.decorators import custom, test
except ImportError:
    def custom(fn):
        return fn

    def test(fn):
        return fn


import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


CLASS_NAMES = ["setosa", "versicolor", "virginica"]


@custom
def transform_custom(result: dict, *args, **kwargs) -> pd.DataFrame:
    """
    Compute confusion matrix and classification report from train_model output.
    Returns a metrics DataFrame (also saved as iris_metrics.csv by save_results).
    """
    y_test = result["y_test"]
    y_pred = result["y_pred"]
    accuracy = result["accuracy"]

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_string())

    # Per-class metrics
    report = classification_report(
        y_test, y_pred, target_names=CLASS_NAMES, output_dict=True
    )
    metrics_df = pd.DataFrame(report).T
    print("\nClassification Report:")
    print(metrics_df.round(3).to_string())

    # Overall row
    print(f"\nOverall accuracy: {accuracy:.4f}")
    return metrics_df


@test
def test_metrics_shape(df: pd.DataFrame) -> None:
    """Metrics table must have at least 3 class rows."""
    assert len(df) >= 3, "Expected at least 3 rows in metrics table"
