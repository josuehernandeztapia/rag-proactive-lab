#!/usr/bin/env python3
"""Train baseline HASE model (Logistic Regression)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    import joblib

    SKLEARN_AVAILABLE = True
except ImportError:  # noqa: SIM105 - explicit flag clearer
    SKLEARN_AVAILABLE = False


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline logistic model for HASE")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/hase/hase_training_dataset.csv.gz"),
        help="Path to merged training dataset",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("models/hase/hase_logistic_baseline.json" if not SKLEARN_AVAILABLE else "models/hase/hase_logistic_baseline.joblib"),
        help="Where to persist trained model",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("models/hase/hase_logistic_metrics.json"),
        help="Path to store evaluation metrics",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.3,
        help="Test split proportion",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Learning rate for fallback gradient descent",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=4000,
        help="Epochs for fallback gradient descent",
    )
    return parser.parse_args(argv)


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    if "default_flag" not in df.columns:
        raise ValueError("Dataset must contain 'default_flag'")
    return df


def train_with_sklearn(df: pd.DataFrame, args: argparse.Namespace) -> tuple[dict, dict]:
    X = df.drop(columns=["default_flag"])
    y = df["default_flag"].astype(int)

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical_cols,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers)
    pipeline = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=1000)),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y if y.nunique() > 1 else None,
    )

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, probs)) if y_test.nunique() > 1 else None,
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "classification_report": classification_report(y_test, preds, output_dict=True),
        "sklearn": True,
    }

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.model_output)
    return metrics, {
        "model_path": str(args.model_output),
        "backend": "sklearn",
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
    }


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def auc_score(y_true: np.ndarray, y_scores: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    order = np.argsort(y_scores)
    y_true = y_true[order]
    y_scores = y_scores[order]
    cum_pos = np.cumsum(y_true[::-1])[::-1]
    cum_neg = np.cumsum(1 - y_true)
    auc = (cum_pos * (1 - y_true)).sum() / (cum_pos[0] * cum_neg[-1]) if cum_pos[0] and cum_neg[-1] else None
    return float(auc) if auc is not None else None


def train_fallback(df: pd.DataFrame, args: argparse.Namespace) -> tuple[dict, dict]:
    X = df.drop(columns=["default_flag"])
    y = df["default_flag"].astype(int).to_numpy()

    X_encoded = pd.get_dummies(X, drop_first=False)
    X_values = X_encoded.to_numpy(dtype=float)

    # Standardize
    mean = X_values.mean(axis=0, keepdims=True)
    std = X_values.std(axis=0, keepdims=True)
    std[std == 0] = 1
    X_std = (X_values - mean) / std

    # Train/test split (manual)
    rng = np.random.default_rng(args.random_state)
    indices = np.arange(len(X_std))
    rng.shuffle(indices)
    split = int(len(indices) * (1 - args.test_size))
    train_idx, test_idx = indices[:split], indices[split:]
    X_train, X_test = X_std[train_idx], X_std[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Initialize weights
    weights = np.zeros(X_train.shape[1])
    bias = 0.0
    lr = args.learning_rate
    m = len(X_train)

    for _ in range(args.epochs):
        linear = X_train @ weights + bias
        preds = sigmoid(linear)
        error = preds - y_train
        grad_w = (X_train.T @ error) / m
        grad_b = error.mean()
        weights -= lr * grad_w
        bias -= lr * grad_b

    test_scores = sigmoid(X_test @ weights + bias)
    test_preds = (test_scores >= 0.5).astype(int)

    accuracy = float((test_preds == y_test).mean()) if len(y_test) > 0 else math.nan
    auc = auc_score(y_test, test_scores)

    metrics = {
        "accuracy": accuracy,
        "roc_auc": auc,
        "confusion_matrix": [
            [int(((test_preds == 0) & (y_test == 0)).sum()), int(((test_preds == 1) & (y_test == 0)).sum())],
            [int(((test_preds == 0) & (y_test == 1)).sum()), int(((test_preds == 1) & (y_test == 1)).sum())],
        ],
        "sklearn": False,
    }

    model_info = {
        "backend": "custom_gd",
        "weights": weights.tolist(),
        "bias": bias,
        "feature_columns": X_encoded.columns.tolist(),
        "mean": mean.flatten().tolist(),
        "std": std.flatten().tolist(),
    }

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    with args.model_output.open("w", encoding="utf-8") as f:
        json.dump(model_info, f)

    return metrics, model_info


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    df = load_dataset(args.dataset)

    if df["default_flag"].nunique() < 2:
        raise SystemExit("Need at least two classes in default_flag to train a model.")

    if SKLEARN_AVAILABLE:
        metrics, _ = train_with_sklearn(df, args)
    else:
        metrics, _ = train_fallback(df, args)

    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics_output.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
