#!/usr/bin/env python3
"""Train an XGBoost-based classifier for HASE if the library is available."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:  # pragma: no cover - depends on environment
    from xgboost import XGBClassifier  # type: ignore

    XGB_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on environment
    XGB_AVAILABLE = False

try:  # pragma: no cover - depends on environment
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on environment
    SKLEARN_AVAILABLE = False

try:  # pragma: no cover - depends on environment
    import joblib

    JOBLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    JOBLIB_AVAILABLE = False

DEFAULT_DATASET = Path("data/hase/hase_training_dataset_full.csv.gz")
DEFAULT_MODEL_OUTPUT = Path("models/hase/hase_xgboost_model.joblib")
DEFAULT_METRICS_OUTPUT = Path("models/hase/hase_xgboost_metrics.json")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGBoost model for HASE")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_OUTPUT)
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS_OUTPUT)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--subsample", type=float, default=0.8)
    return parser.parse_args(argv)


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    if "default_flag" not in df.columns:
        raise ValueError("Dataset must contain 'default_flag'")
    for col in ["placa", "fecha_dia", "label_date"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df


def build_pipeline(n_estimators: int, learning_rate: float, max_depth: int, subsample: float) -> Pipeline:
    if not SKLEARN_AVAILABLE:
        raise ModuleNotFoundError("scikit-learn is required. Install 'scikit-learn' to use train_xgboost.py")
    categorical_cols = ["plaza_limpia", "label_reason"]

    if not XGB_AVAILABLE:
        raise ModuleNotFoundError(
            "XGBoost is not installed. Install 'xgboost' or use train_baseline.py as fallback."
        )

    model = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=0,
    )

    def build_column_transformer(sample: pd.DataFrame) -> ColumnTransformer:
        numeric_cols = sample.select_dtypes(include=["number"]).columns.tolist()
        transformers = []
        cat_cols_present = [col for col in categorical_cols if col in sample.columns]
        if cat_cols_present:
            transformers.append(
                (
                    "cat",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    cat_cols_present,
                )
            )
        if numeric_cols:
            transformers.append(("num", SimpleImputer(strategy="median"), numeric_cols))
        return ColumnTransformer(transformers=transformers)

    preprocessing = build_column_transformer(pd.DataFrame(columns=categorical_cols))

    pipeline = Pipeline(steps=[("preprocess", preprocessing), ("model", model)])
    pipeline._build_column_transformer = build_column_transformer  # type: ignore[attr-defined]
    return pipeline


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if not SKLEARN_AVAILABLE:
        raise ModuleNotFoundError(
            "scikit-learn is required. Install 'scikit-learn' to use train_xgboost.py"
        )
    df = load_dataset(args.dataset)

    X = df.drop(columns=["default_flag"])
    y = df["default_flag"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y if y.nunique() > 1 else None,
    )

    pipeline = build_pipeline(
        args.n_estimators,
        args.learning_rate,
        args.max_depth,
        args.subsample,
    )

    if hasattr(pipeline, "_build_column_transformer"):
        preprocessing = pipeline._build_column_transformer(X_train)  # type: ignore[attr-defined]
        pipeline.steps[0] = ("preprocess", preprocessing)
        delattr(pipeline, "_build_column_transformer")

    pipeline.fit(X_train, y_train)
    pred_labels = pipeline.predict(X_test)
    if hasattr(pipeline.named_steps["model"], "predict_proba"):
        pred_proba = pipeline.predict_proba(X_test)[:, 1]
    else:  # pragma: no cover - xgboost always exposes predict_proba
        pred_proba = pipeline.decision_function(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred_labels)),
        "roc_auc": float(roc_auc_score(y_test, pred_proba)) if y_test.nunique() > 1 else None,
        "confusion_matrix": confusion_matrix(y_test, pred_labels).tolist(),
        "classification_report": classification_report(y_test, pred_labels, output_dict=True),
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "subsample": args.subsample,
    }

    if not JOBLIB_AVAILABLE:
        raise ModuleNotFoundError("joblib is required to persist the model. Install 'joblib' or adjust script.")

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.model_output)
    with args.metrics_output.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    print("Training completed. Metrics:")
    print(json.dumps(metrics, indent=2))
    print(f"Model saved to {args.model_output}")
    print(f"Metrics saved to {args.metrics_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
