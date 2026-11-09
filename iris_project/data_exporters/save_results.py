"""
Mage block: save_results
Type: data_exporter
Pipeline: iris_classifier

Saves iris_predictions.csv.
Only runs if train_model's @test gate passes (accuracy >= threshold).
Also saves the metrics DataFrame from evaluate_model as iris_metrics.csv.
"""

try:
    from mage_ai.data_preparation.decorators import data_exporter, test
except ImportError:
    def data_exporter(fn):
        return fn

    def test(fn):
        return fn


import os
import pandas as pd


PREDICTIONS_FILE = "iris_predictions.csv"
METRICS_FILE = "iris_metrics.csv"


@data_exporter
def export_data(result: dict, *args, **kwargs) -> None:
    """
    Write predictions and model metadata to CSV files.

    result dict keys expected (from train_model):
        model, accuracy, max_depth, y_test, y_pred, feature_names
    """
    y_test = result["y_test"]
    y_pred = result["y_pred"]
    accuracy = result["accuracy"]
    max_depth = result["max_depth"]

    predictions_df = pd.DataFrame({
        "y_true": y_test,
        "y_pred": y_pred,
        "correct": [t == p for t, p in zip(y_test, y_pred)],
    })
    predictions_df.to_csv(PREDICTIONS_FILE, index=False)
    print(f"Saved {len(predictions_df)} predictions → {PREDICTIONS_FILE}")
    print(f"Model: DecisionTree  max_depth={max_depth}  accuracy={accuracy:.4f}")


@test
def test_file_created(*args) -> None:
    """Predictions file must exist after export."""
    assert os.path.exists(PREDICTIONS_FILE), (
        f"{PREDICTIONS_FILE} was not created"
    )
