"""
prefect/heatwave_pipeline.py
============================
Prefect 3 heatwave early-warning pipeline.
Based on: Latest/Copy_of_Prefect_Heatwave_Pipeline_Student_version.ipynb

Usage:
    # Install dependencies first:
    pip install -r prefect/requirements_prefect.txt

    # Set Prefect Cloud credentials (or use local server):
    export PREFECT_API_URL="https://api.prefect.cloud/api/accounts/.../workspaces/..."
    export PREFECT_API_KEY="your_key"

    python prefect/heatwave_pipeline.py
"""

import os
import random
import joblib
import numpy as np
import pandas as pd
import requests
from datetime import date, timedelta

from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact, create_table_artifact
from prefect.cache_policies import INPUTS

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CITIES = {
    "Delhi":     (28.61, 77.21),
    "Mumbai":    (19.08, 72.88),
    "Chennai":   (13.08, 80.27),
    "Kolkata":   (22.57, 88.36),
    "Bengaluru": (12.97, 77.59),
    "Nagpur":    (21.15, 79.09),
}
START = "2023-01-01"
END   = "2025-12-31"
HEATWAVE_LIMIT = 40.0   # degrees Celsius

FEATURES = [
    "temperature_2m_max", "temperature_2m_min",
    "relative_humidity_2m_mean", "precipitation_sum",
    "month", "tmax_lag1", "tmax_avg7",
]


# ---------------------------------------------------------------------------
# Helper: simulate weather when API is unavailable
# ---------------------------------------------------------------------------
def simulate_city(city: str, lat: float) -> pd.DataFrame:
    """Seasonal fake weather: hotter in May, hotter for northern cities."""
    rng = np.random.default_rng(abs(hash(city)) % 1000)
    dates = pd.date_range(START, END, freq="D")
    n = len(dates)
    month = dates.month.values
    base_temp = 25 + (lat - 13) * 0.3            # northern cities hotter
    seasonal  = 8 * np.sin((month - 5) * np.pi / 6)
    tmax = base_temp + seasonal + rng.normal(0, 2, n)
    tmin = tmax - rng.uniform(5, 10, n)
    return pd.DataFrame({
        "time":                      dates.strftime("%Y-%m-%d"),
        "temperature_2m_max":        tmax.round(1),
        "temperature_2m_min":        tmin.round(1),
        "relative_humidity_2m_mean": rng.uniform(40, 90, n).round(1),
        "precipitation_sum":         np.maximum(0, rng.normal(2, 5, n)).round(1),
    })


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

@task(retries=3, retry_delay_seconds=2, cache_policy=INPUTS)
def extract_city(city: str, lat: float, lon: float) -> pd.DataFrame:
    """Download daily weather for one city from the free Open-Meteo archive API."""
    logger = get_run_logger()
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": START, "end_date": END,
        "daily": "temperature_2m_max,temperature_2m_min,"
                 "relative_humidity_2m_mean,precipitation_sum",
        "timezone": "Asia/Kolkata",
    }
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        df = pd.DataFrame(r.json()["daily"])
        logger.info(f"{city}: downloaded {len(df)} days from Open-Meteo")
    except Exception as e:
        logger.warning(f"{city}: API failed ({type(e).__name__}) -> using simulated data")
        df = simulate_city(city, lat)
    df["city"] = city
    return df


@task
def combine_cities(city_dfs: list) -> pd.DataFrame:
    """Stack all city DataFrames and engineer lag/rolling features."""
    data = pd.concat(city_dfs, ignore_index=True)
    data["date"] = pd.to_datetime(data["time"])
    data = data.sort_values(["city", "date"]).reset_index(drop=True)
    data["month"] = data["date"].dt.month
    data["tmax_lag1"] = data.groupby("city")["temperature_2m_max"].shift(1)
    data["tmax_avg7"] = (
        data.groupby("city")["temperature_2m_max"]
        .transform(lambda s: s.rolling(7, min_periods=1).mean())
    )
    # Target: tomorrow's max temperature
    data["target_tomorrow"] = data.groupby("city")["temperature_2m_max"].shift(-1)
    data = data.dropna(subset=["target_tomorrow"] + FEATURES)
    return data


@task
def split_data(data: pd.DataFrame):
    """Time-based split: train on 2023-2024, test on 2025 (no peeking)."""
    train = data[data["date"] < "2025-01-01"]
    test  = data[data["date"] >= "2025-01-01"]
    print(f"Train: {len(train)} rows | Test: {len(test)} rows")
    return train, test


@task
def train_model(name: str, train: pd.DataFrame):
    """Train one model and return (name, fitted_model)."""
    if name == "linear_regression":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    model.fit(train[FEATURES], train["target_tomorrow"])
    return name, model


@task
def evaluate_model(name: str, model, test: pd.DataFrame) -> dict:
    """Evaluate model on test set and return name + MAE."""
    pred = model.predict(test[FEATURES])
    mae  = mean_absolute_error(test["target_tomorrow"], pred)
    print(f"{name}: MAE = {mae:.2f} °C")
    return {"model": name, "mae": round(mae, 3)}


@task
def select_and_register(results: list, models: dict) -> dict:
    """Pick the model with the lowest MAE and save it to disk."""
    best = min(results, key=lambda r: r["mae"])
    joblib.dump(models[best["model"]], "best_model.joblib")
    print(f"Best model: {best['model']} (MAE {best['mae']})")
    return best


@task
def heatwave_alerts(data: pd.DataFrame, model) -> pd.DataFrame:
    """Use the latest day of each city to forecast tomorrow and raise alerts."""
    latest = data.sort_values("date").groupby("city").tail(1).copy()
    latest["forecast_tmax"] = model.predict(latest[FEATURES]).round(1)
    latest["date"]  = latest["date"].dt.date
    latest["alert"] = np.where(
        latest["forecast_tmax"] >= HEATWAVE_LIMIT, "HEATWAVE", "normal"
    )
    alerts = latest[["city", "date", "forecast_tmax", "alert"]]
    print(alerts.to_string(index=False))
    return alerts


@task
def save_warehouse(data: pd.DataFrame, path: str = "weather_warehouse.csv"):
    data.to_csv(path, index=False)
    print(f"Saved {len(data)} rows to {path}")
    return path


# ---------------------------------------------------------------------------
# Sub-flows
# ---------------------------------------------------------------------------

@flow(name="data-pipeline", log_prints=True)
def data_pipeline() -> pd.DataFrame:
    """Extract → Combine → Feature engineer."""
    city_dfs = [extract_city(city, lat, lon) for city, (lat, lon) in CITIES.items()]
    data = combine_cities(city_dfs)
    save_warehouse(data)
    return data


@flow(name="ml-pipeline", log_prints=True)
def ml_pipeline(data: pd.DataFrame):
    """Split → Train two models → Evaluate → Select best → Forecast."""
    train, test = split_data(data)

    model_names = ["linear_regression", "random_forest"]
    trained = {name: mdl for name, mdl in [train_model(n, train) for n in model_names]}
    results  = [evaluate_model(name, mdl, test) for name, mdl in trained.items()]
    best     = select_and_register(results, trained)

    best_model = trained[best["model"]]
    alerts = heatwave_alerts(data, best_model)

    # Prefect artifact: ML report
    report_md = "## Heatwave Pipeline — ML Report\n\n"
    report_md += "| Model | MAE (°C) |\n|---|---|\n"
    for r in results:
        report_md += f"| {r['model']} | {r['mae']} |\n"
    report_md += f"\n**Winner:** `{best['model']}` with MAE {best['mae']} °C\n"
    create_markdown_artifact(key="ml-report", markdown=report_md)

    # Prefect artifact: heatwave table
    create_table_artifact(
        key="heatwave-alerts",
        table=alerts.to_dict(orient="records"),
        description="City heatwave forecasts",
    )
    return best, alerts


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

@flow(name="heatwave-early-warning", log_prints=True)
def heatwave_system():
    """Full ETL + ML heatwave pipeline."""
    data  = data_pipeline()
    best, alerts = ml_pipeline(data)
    heatwave_count = (alerts["alert"] == "HEATWAVE").sum()
    print(f"\nHeatwave alerts: {heatwave_count} / {len(alerts)} cities")
    return alerts


# ---------------------------------------------------------------------------
# Retry demo (Part D3-style)
# ---------------------------------------------------------------------------

@task(retries=4, retry_delay_seconds=1)
def flaky_sensor_api():
    """Simulate a sensor API that fails twice before succeeding."""
    if not hasattr(flaky_sensor_api, '_n'):
        flaky_sensor_api._n = 0
    flaky_sensor_api._n += 1
    print(f"Attempt {flaky_sensor_api._n} ...")
    if flaky_sensor_api._n < 3:
        raise ConnectionError("Sensor gateway timed out!")
    return "data received"


@flow(log_prints=True)
def retry_demo():
    return flaky_sensor_api()


if __name__ == "__main__":
    heatwave_system()
