"""
stream/stream_consumer.py  — Bonus: Stream into the lakehouse (+10 marks)
==========================================================================
Receives FreshCart-style order messages from RabbitMQ for 60 seconds,
writes them as a raw landing file, then triggers the full ELT pipeline
so streamed orders reach gold.fct_orders — idempotently.

Usage:
    python stream/stream_consumer.py [--url AMQP_URL] [--duration 60]

Environment variables (keep out of Git):
    AMQP_URL          RabbitMQ connection URL
    SNOWFLAKE_ACCOUNT / _USER / _PASSWORD / _DATABASE / _WAREHOUSE
"""

import argparse
import json
import os
import signal
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pika
import snowflake.connector

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXCHANGE = "freshcart-orders"
EXCHANGE_TYPE = "fanout"
LANDING_DIR = Path("elt/landing")
COLLECTION_SECONDS = 60   # collect for one minute then flush

# ---------------------------------------------------------------------------
# Snowflake helpers (mirrors the ELT notebook)
# ---------------------------------------------------------------------------

def get_sf_connection():
    return snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        database=os.environ.get("SNOWFLAKE_DATABASE", "FRESHCART"),
        warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
        schema="BRONZE",
    )


def write_landing(orders: list[dict]) -> Path:
    """Serialize orders list to a JSON-lines file in elt/landing/."""
    LANDING_DIR.mkdir(parents=True, exist_ok=True)
    fname = LANDING_DIR / f"stream_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:6]}.jsonl"
    with open(fname, "w") as f:
        for order in orders:
            f.write(json.dumps(order) + "\n")
    print(f"[consumer] Written {len(orders)} orders → {fname}")
    return fname


def load_file(conn, local_path: Path):
    """PUT file into Snowflake stage then COPY INTO bronze.orders_raw."""
    cur = conn.cursor()
    stage = "@FRESHCART.BRONZE.%orders_raw"
    cur.execute(f"PUT file://{local_path.resolve()} {stage} AUTO_COMPRESS=FALSE OVERWRITE=TRUE")
    cur.execute(f"""
        COPY INTO FRESHCART.BRONZE.ORDERS_RAW
        FROM {stage}
        FILE_FORMAT = (TYPE = 'JSON' STRIP_OUTER_ARRAY = FALSE)
        PURGE = TRUE
        ON_ERROR = 'CONTINUE'
    """)
    print(f"[consumer] Loaded file into bronze.orders_raw")


def run_transforms(conn):
    """Run the same SQL transform chain as the ELT notebook."""
    cur = conn.cursor()
    sqls = [
        # Silver: clean
        """
        CREATE OR REPLACE TABLE FRESHCART.SILVER.ORDERS_CLEAN AS
        SELECT
            COALESCE(raw:order_id::STRING, raw:id::STRING)         AS order_id,
            COALESCE(raw:customer_id::STRING, raw:cust::STRING)    AS customer_id,
            raw:order_date::DATE                                    AS order_date,
            COALESCE(raw:amount::FLOAT, raw:total::FLOAT)          AS amount,
            COALESCE(raw:currency::STRING, 'INR')                  AS currency,
            raw:channel::STRING                                     AS channel,
            raw:updated_at::TIMESTAMP_NTZ                          AS updated_at
        FROM FRESHCART.BRONZE.ORDERS_RAW
        WHERE raw:order_id IS NOT NULL OR raw:id IS NOT NULL
        """,
        # Silver: dedup
        """
        CREATE OR REPLACE TABLE FRESHCART.SILVER.ORDERS_DEDUP AS
        SELECT * FROM (
            SELECT *,
                ROW_NUMBER() OVER (
                    PARTITION BY order_id ORDER BY updated_at DESC NULLS LAST
                ) AS rn
            FROM FRESHCART.SILVER.ORDERS_CLEAN
            WHERE amount IS NOT NULL AND currency IN ('INR', 'USD')
        )
        WHERE rn = 1
        """,
        # Gold: fact orders (MERGE for idempotency)
        """
        MERGE INTO FRESHCART.GOLD.FCT_ORDERS AS tgt
        USING (
            SELECT
                d.order_id,
                d.customer_id,
                d.order_date,
                CASE WHEN d.currency = 'USD' THEN d.amount * 83 ELSE d.amount END AS amount_inr,
                d.channel,
                CURRENT_TIMESTAMP() AS published_at,
                MD5(d.order_id || d.amount::STRING) AS row_hash
            FROM FRESHCART.SILVER.ORDERS_DEDUP d
        ) AS src
        ON tgt.order_id = src.order_id
        WHEN MATCHED AND tgt.row_hash <> src.row_hash THEN
            UPDATE SET
                amount_inr   = src.amount_inr,
                published_at = src.published_at,
                row_hash     = src.row_hash
        WHEN NOT MATCHED THEN
            INSERT (order_id, customer_id, order_date, amount_inr, channel, published_at, row_hash)
            VALUES (src.order_id, src.customer_id, src.order_date, src.amount_inr,
                    src.channel, src.published_at, src.row_hash)
        """,
    ]
    for sql in sqls:
        cur.execute(sql.strip())
    print("[consumer] Transforms complete — gold.fct_orders updated (idempotent MERGE).")


# ---------------------------------------------------------------------------
# Main consumer loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FreshCart stream consumer → Snowflake")
    parser.add_argument("--url", default=os.getenv("AMQP_URL", "amqp://guest:guest@localhost/"))
    parser.add_argument("--duration", type=int, default=COLLECTION_SECONDS)
    args = parser.parse_args()

    params = pika.URLParameters(args.url)
    connection = pika.BlockingConnection(params)
    channel = connection.channel()

    channel.exchange_declare(exchange=EXCHANGE, exchange_type=EXCHANGE_TYPE, durable=True)
    result = channel.queue_declare(queue="", exclusive=True)
    queue_name = result.method.queue
    channel.queue_bind(exchange=EXCHANGE, queue=queue_name, routing_key="")
    channel.basic_qos(prefetch_count=1)

    collected: list[dict] = []
    deadline = time.time() + args.duration

    def callback(ch, method, properties, body):
        try:
            payload = json.loads(body)
            collected.append(payload)
            print(f"[consumer] Received order: {payload.get('order_id', payload.get('id'))}")
            ch.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as e:
            print(f"[consumer] Error: {e}")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    channel.basic_consume(queue=queue_name, on_message_callback=callback)
    print(f"[consumer] Collecting for {args.duration}s from exchange='{EXCHANGE}'...")

    # Poll until deadline
    while time.time() < deadline:
        connection.process_data_events(time_limit=1)

    connection.close()
    print(f"\n[consumer] Collected {len(collected)} orders. Flushing to Snowflake...")

    if not collected:
        print("[consumer] No orders received. Nothing to load.")
        return

    # Write landing file
    landing_file = write_landing(collected)

    # Load into Snowflake and transform
    sf_conn = get_sf_connection()
    load_file(sf_conn, landing_file)
    run_transforms(sf_conn)
    sf_conn.close()
    print("[consumer] Done. Run again — gold totals will NOT double (MERGE is idempotent).")


if __name__ == "__main__":
    main()

