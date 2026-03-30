from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


@dataclass
class TrainingArtifacts:
    model_path: Path
    metrics_path: Path
    sample_path: Path
    metrics: dict


def train_and_persist_model() -> TrainingArtifacts:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_wine(as_frame=True)
    X = dataset.data
    y = dataset.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=3000)),
        ]
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    metrics = {
        "dataset_name": "wine",
        "features": int(X.shape[1]),
        "classes": int(y.nunique()),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "accuracy": round(float(accuracy), 4),
        "macro_f1": round(float(macro_f1), 4),
    }

    model_path = ARTIFACTS_DIR / "wine_classifier.joblib"
    metrics_path = ARTIFACTS_DIR / "training_metrics.csv"
    sample_path = ARTIFACTS_DIR / "sample_payload.csv"
    report_path = ARTIFACTS_DIR / "classification_report.txt"

    joblib.dump(model, model_path)
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    X.head(5).assign(target=y.head(5).values).to_csv(sample_path, index=False)
    report_path.write_text(classification_report(y_test, y_pred), encoding="utf-8")

    return TrainingArtifacts(
        model_path=model_path,
        metrics_path=metrics_path,
        sample_path=sample_path,
        metrics=metrics,
    )
