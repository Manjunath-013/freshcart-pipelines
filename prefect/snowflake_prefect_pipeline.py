"""
prefect/snowflake_prefect_pipeline.py
======================================
Snowflake + Prefect 3 orders pipeline.
Based on: Latest/snowflake_prefect_next_lab.ipynb

Demonstrates:
  - @task with retries for each pipeline stage
  - @flow orchestrating extract → load → silver → gold → validate → report
  - MERGE-based idempotent gold table (re-running gives same totals)
  - Prefect artifact for quality report

Usage:
    # Set env vars:
    export SNOWFLAKE_ACCOUNT="your_account"
    export SNOWFLAKE_USER="your_user"
    export SNOWFLAKE_PASSWORD="your_password"
    export PREFECT_API_URL="..."
    export PREFECT_API_KEY="..."

    python prefect/snowflake_prefect_pipeline.py
"""

import os
from decimal import Decimal
from getpass import getpass

import pandas as pd
import snowflake.connector
from prefect import flow, task
from prefect.artifacts import create_markdown_artifact

# ---------------------------------------------------------------------------
# Input data (same 3-order demo as the lab)
# ---------------------------------------------------------------------------
ORDERS = [(1, "Priya",  100),
          (2, "Arjun",  200),
          (3, "Meera",  150)]

# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------
SQL_SILVER = """
-- Standardize names and add calculated columns.
CREATE OR REPLACE TABLE silver.orders AS
SELECT order_id,
       UPPER(TRIM(customer))   AS customer,
       amount,
       ROUND(amount * 1.18, 2) AS amount_with_tax,
       CURRENT_TIMESTAMP()     AS processed_at
FROM bronze.orders;
"""

SQL_GOLD = """
-- Aggregate customer spend (MERGE so reruns don't double-count).
MERGE INTO gold.customer_summary t
USING (
    SELECT customer,
           SUM(amount)          AS total_spent,
           SUM(amount_with_tax) AS total_with_tax,
           COUNT(*)             AS order_count
    FROM silver.orders
    GROUP BY customer
) s
ON t.customer = s.customer
WHEN MATCHED THEN UPDATE SET
    total_spent    = s.total_spent,
    total_with_tax = s.total_with_tax,
    order_count    = s.order_count
WHEN NOT MATCHED THEN INSERT
    (customer, total_spent, total_with_tax, order_count)
    VALUES (s.customer, s.total_spent, s.total_with_tax, s.order_count);
"""


# ---------------------------------------------------------------------------
# Snowflake connection
# ---------------------------------------------------------------------------
def get_sf_connection():
    return snowflake.connector.connect(
        account   = os.environ.get("SNOWFLAKE_ACCOUNT",   "YOUR_ACCOUNT"),
        user      = os.environ.get("SNOWFLAKE_USER",      "YOUR_USER"),
        password  = os.environ.get("SNOWFLAKE_PASSWORD",  "YOUR_PASSWORD"),
        warehouse = os.environ.get("SNOWFLAKE_WAREHOUSE", "NEXT_LAB_WH"),
        database  = os.environ.get("SNOWFLAKE_DATABASE",  "NEXT_LAB_DB"),
        role      = os.environ.get("SNOWFLAKE_ROLE",      "SYSADMIN"),
    )


def execute(conn, sql: str):
    with conn.cursor() as cur:
        cur.execute(sql)


def read_rows(conn, sql: str):
    with conn.cursor() as cur:
        cur.execute(sql)
        return cur.fetchall()


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

@task(name="Extract and check input")
def extract_orders():
    """EXTRACT: read the small source; check values before loading."""
    rows = list(ORDERS)
    assert all(r[2] > 0 for r in rows), "Negative amounts found in source"
    print(f"Extracted {len(rows)} orders")
    return rows


@task(name="Load to bronze", retries=2)
def load_orders(rows: list):
    """LOAD: insert raw rows into bronze.orders."""
    conn = get_sf_connection()
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS bronze.orders (
                order_id INT, customer STRING, amount NUMBER(10,2)
            )
        """)
        cur.executemany(
            "INSERT INTO bronze.orders VALUES (%s, %s, %s)", rows
        )
    conn.close()
    print(f"Loaded {len(rows)} rows into bronze.orders")


@task(name="Transform to silver", retries=2)
def transform_table(sql: str):
    """TRANSFORM: run a SQL transformation."""
    conn = get_sf_connection()
    execute(conn, sql)
    conn.close()
    print("Transform complete")


@task(name="Validate gold")
def validate(expected_total: float):
    """CHECK: verify gold totals match expectation."""
    conn = get_sf_connection()
    rows = read_rows(conn, "SELECT SUM(total_spent) FROM gold.customer_summary")
    conn.close()
    actual = float(rows[0][0] or 0)
    assert abs(actual - expected_total) < 0.01, \
        f"Expected total {expected_total}, got {actual}"
    print(f"Validation passed: total = {actual}")
    return actual


@task(name="Publish report")
def publish_report(actual_total: float):
    """Create a Prefect markdown artifact with pipeline results."""
    md = f"""## Snowflake Orders Pipeline Report

| Metric | Value |
|---|---|
| Orders processed | {len(ORDERS)} |
| Total amount | ₹{actual_total:,.2f} |
| Status | ✅ OK |
"""
    create_markdown_artifact(key="orders-report", markdown=md)
    print("Report published to Prefect")


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

@flow(name="snowflake-orders-next-lab", log_prints=True)
def orders_pipeline():
    """Extract → Load → Silver → Gold → Validate → Report."""
    expected = sum(r[2] for r in ORDERS)

    rows = extract_orders()
    load_orders(rows)
    transform_table(SQL_SILVER)
    transform_table(SQL_GOLD)
    total = validate(expected)
    publish_report(total)
    return total


if __name__ == "__main__":
    result = orders_pipeline()
    print("Pipeline result:", result)

    # Idempotency demo: run again, totals should not change
    second_result = orders_pipeline()
    assert second_result == result, "Re-run changed the totals!"
    print("Same totals after second run:", second_result)

