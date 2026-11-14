"""
stream/dispatcher.py  — Part D2 & D3
=========================================
Rapido-style food-order dispatcher using RabbitMQ topic exchange.

Usage:
    python stream/dispatcher.py [--url AMQP_URL] [--interval SECONDS]

Environment variable (keep out of Git!):
    AMQP_URL  e.g. amqp://guest:guest@localhost/
               or   amqps://user:pass@xyz.cloudamqp.com/vhost

Design:
  - Exchange : rapido-dispatch  (topic)
  - Routing key: <city>.<zone>.<type>   e.g. bangalore.koramangala.food
  - Every 10th message is intentionally "bad" (missing 'zone' field) for D3.
"""

import argparse
import json
import os
import random
import time
import uuid
from datetime import datetime, timezone

import pika

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EXCHANGE = "rapido-dispatch"
EXCHANGE_TYPE = "topic"

CITIES = ["bangalore", "chennai"]
ZONES = {
    "bangalore": ["koramangala", "indiranagar", "whitefield", "jayanagar", "hebbal"],
    "chennai": ["tnagar", "adyar", "velachery", "mylapore", "anna_nagar"],
}
ORDER_TYPES = ["ride", "food"]


def build_message(seq: int) -> tuple[str, dict]:
    """
    Returns (routing_key, payload).
    Every 10th message is 'bad' — the zone field is omitted (for D3 NACK demo).
    """
    city = random.choice(CITIES)
    order_type = random.choice(ORDER_TYPES)
    zone = random.choice(ZONES[city])

    is_bad = (seq % 10 == 0)  # one bad message every 10

    routing_key = f"{city}.{zone}.{order_type}" if not is_bad else f"{city}.{order_type}"
    payload: dict = {
        "id": str(uuid.uuid4())[:8],
        "city": city,
        "type": order_type,
        "ts": datetime.now(timezone.utc).isoformat(),
        "seq": seq,
    }
    if not is_bad:
        payload["zone"] = zone
    else:
        payload["_bad"] = True  # marker so we can identify bad messages in logs

    return routing_key, payload


def main():
    parser = argparse.ArgumentParser(description="Rapido order dispatcher")
    parser.add_argument("--url", default=os.getenv("AMQP_URL", "amqp://guest:guest@localhost/"),
                        help="AMQP broker URL")
    parser.add_argument("--interval", type=float, default=2.0,
                        help="Seconds between messages (default: 2)")
    args = parser.parse_args()

    params = pika.URLParameters(args.url)
    connection = pika.BlockingConnection(params)
    channel = connection.channel()

    channel.exchange_declare(exchange=EXCHANGE, exchange_type=EXCHANGE_TYPE, durable=True)
    print(f"[dispatcher] Connected. Publishing to exchange '{EXCHANGE}' every {args.interval}s")
    print(f"[dispatcher] Every 10th message is intentionally BAD (missing 'zone') for D3.\n")

    seq = 1
    try:
        while True:
            routing_key, payload = build_message(seq)
            body = json.dumps(payload)

            channel.basic_publish(
                exchange=EXCHANGE,
                routing_key=routing_key,
                body=body,
                properties=pika.BasicProperties(
                    delivery_mode=2,          # persistent
                    content_type="application/json",
                ),
            )

            tag = "BAD " if payload.get("_bad") else "    "
            print(f"[dispatcher] {tag}#{seq:04d}  key={routing_key:<35}  id={payload['id']}")
            seq += 1
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n[dispatcher] Stopped.")
    finally:
        connection.close()


if __name__ == "__main__":
    main()

