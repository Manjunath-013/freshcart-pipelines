"""
stream/driver.py  — Part D2 & D3
=========================================
Rapido-style driver that listens on a private RabbitMQ queue.

Usage:
    python stream/driver.py <NAME> <BINDING_PATTERN> [--url AMQP_URL]

Examples:
    python stream/driver.py Ravi  "bangalore.koramangala.*"
    python stream/driver.py Priya "bangalore.indiranagar.*"
    python stream/driver.py Arjun "bangalore.*.ride"
    python stream/driver.py Meena "bangalore.*.food"
    python stream/driver.py Karthik "chennai.tnagar.food"
    python stream/driver.py Control "bangalore.#"

Part D3:
    A message missing the 'zone' field is rejected with
    basic_nack(requeue=False) — it goes to the dead-letter queue or is dropped.
    Every good message gets basic_ack after printing.

Environment variable (keep out of Git!):
    AMQP_URL  e.g. amqp://guest:guest@localhost/
"""

import argparse
import json
import os
import sys

import pika

EXCHANGE = "rapido-dispatch"
EXCHANGE_TYPE = "topic"


def on_message(channel, method, properties, body, driver_name: str):
    """
    Callback invoked for each message.

    D3 logic:
      - If 'zone' key is missing → NACK with requeue=False (message is discarded).
      - Otherwise → print details and ACK.
    """
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        print(f"[{driver_name}] ✗ Invalid JSON — NACK")
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        return

    if "zone" not in payload:
        # D3: bad message — reject without requeue
        print(
            f"[{driver_name}] ✗ BAD MSG (missing 'zone')  "
            f"id={payload.get('id', '?')}  key={method.routing_key}  — NACK requeue=False"
        )
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        return

    # Good message — process and ACK
    print(
        f"[{driver_name}] ✓ ORDER  "
        f"id={payload['id']}  "
        f"city={payload['city']}  "
        f"zone={payload['zone']}  "
        f"type={payload['type']}  "
        f"ts={payload['ts']}  "
        f"key={method.routing_key}"
    )
    channel.basic_ack(delivery_tag=method.delivery_tag)


def main():
    parser = argparse.ArgumentParser(description="Rapido driver consumer")
    parser.add_argument("name", help="Driver name, e.g. Ravi")
    parser.add_argument("pattern", help="Binding pattern, e.g. bangalore.koramangala.*")
    parser.add_argument(
        "--url",
        default=os.getenv("AMQP_URL", "amqp://guest:guest@localhost/"),
        help="AMQP broker URL",
    )
    args = parser.parse_args()

    params = pika.URLParameters(args.url)
    connection = pika.BlockingConnection(params)
    channel = connection.channel()

    # Declare exchange (idempotent)
    channel.exchange_declare(exchange=EXCHANGE, exchange_type=EXCHANGE_TYPE, durable=True)

    # Declare a private, exclusive queue for this driver
    result = channel.queue_declare(queue="", exclusive=True)
    queue_name = result.method.queue

    # Bind with the driver's pattern
    channel.queue_bind(exchange=EXCHANGE, queue=queue_name, routing_key=args.pattern)

    # Only prefetch 1 so an unACKed message blocks further delivery (fair dispatch)
    channel.basic_qos(prefetch_count=1)

    callback = lambda ch, method, props, body: on_message(
        ch, method, props, body, args.name
    )
    channel.basic_consume(queue=queue_name, on_message_callback=callback)

    print(f"[{args.name}] Listening on exchange='{EXCHANGE}'  pattern='{args.pattern}'")
    print(f"[{args.name}] Queue: {queue_name}")
    print(f"[{args.name}] Press Ctrl+C to stop.\n")

    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        print(f"\n[{args.name}] Stopped.")
    finally:
        connection.close()


if __name__ == "__main__":
    main()
