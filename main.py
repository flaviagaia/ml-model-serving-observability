from __future__ import annotations

from src.model_training import train_and_persist_model


def main() -> None:
    artifacts = train_and_persist_model()
    print("ML Model Serving Observability")
    print("-" * 37)
    for key, value in artifacts.metrics.items():
        print(f"{key}: {value}")
    print(f"model_path: {artifacts.model_path}")


if __name__ == "__main__":
    main()
