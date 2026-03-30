from __future__ import annotations

import argparse
import random
import time
import sys
from pathlib import Path

import httpx
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_training import train_and_persist_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate mixed traffic for the model API.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Base URL for the model API.")
    parser.add_argument("--requests", type=int, default=60, help="Number of valid requests to send.")
    parser.add_argument(
        "--invalid-requests",
        type=int,
        default=6,
        help="Number of intentionally invalid requests to send for error/validation metrics.",
    )
    parser.add_argument("--sleep-ms", type=int, default=120, help="Sleep between requests in milliseconds.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifacts = train_and_persist_model()
    sample_batch = pd.read_csv(artifacts.sample_batch_path)
    grouped = {
        target: df.drop(columns=["target"]).to_dict(orient="records")
        for target, df in sample_batch.groupby("target")
    }

    valid_mix = [0] * max(args.requests // 3, 1) + [1] * max(args.requests // 3, 1) + [2] * max(args.requests // 3, 1)
    while len(valid_mix) < args.requests:
        valid_mix.append(random.choice([0, 1, 2]))
    random.shuffle(valid_mix)

    success = 0
    invalid = 0
    with httpx.Client(timeout=10) as client:
        for target in valid_mix:
            payload = random.choice(grouped[target])
            response = client.post(f"{args.base_url}/predict", json=payload)
            if response.status_code == 200:
                success += 1
            time.sleep(args.sleep_ms / 1000)

        for _ in range(args.invalid_requests):
            bad_payload = {"alcohol": -1, "malic_acid": "oops"}
            response = client.post(f"{args.base_url}/predict", json=bad_payload)
            if response.status_code >= 400:
                invalid += 1
            time.sleep(args.sleep_ms / 1000)

    print("Demo traffic generation finished")
    print(f"successful_requests: {success}")
    print(f"invalid_requests: {invalid}")


if __name__ == "__main__":
    main()
