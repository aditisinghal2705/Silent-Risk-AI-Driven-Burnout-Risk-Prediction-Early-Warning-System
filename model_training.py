from __future__ import annotations

import csv
import json
from typing import Dict, List, Tuple

from utils import (
    FEATURE_COLUMNS,
    METRICS_PATH,
    MODEL_PATH,
    TARGET_COLUMN,
    FallbackBurnoutRegressor,
    save_model,
)


DATA_PATH = "processed_burnout_data.csv"


def load_rows() -> Tuple[List[Dict[str, float]], List[float]]:
    rows: List[Dict[str, float]] = []
    targets: List[float] = []

    with open(DATA_PATH, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            try:
                feature_row = {column: float(raw_row[column]) for column in FEATURE_COLUMNS}
                target = float(raw_row[TARGET_COLUMN])
            except (KeyError, TypeError, ValueError):
                continue

            rows.append(feature_row)
            targets.append(target)

    return rows, targets


def simple_train_test_split(
    rows: List[Dict[str, float]], targets: List[float], test_fraction: float = 0.2
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]], List[float], List[float]]:
    split_index = int(len(rows) * (1 - test_fraction))
    return (
        rows[:split_index],
        rows[split_index:],
        targets[:split_index],
        targets[split_index:],
    )


def mae(y_true: List[float], y_pred: List[float]) -> float:
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)


def rmse(y_true: List[float], y_pred: List[float]) -> float:
    squared = [(a - b) ** 2 for a, b in zip(y_true, y_pred)]
    return (sum(squared) / len(squared)) ** 0.5


def r2(y_true: List[float], y_pred: List[float]) -> float:
    mean_y = sum(y_true) / len(y_true)
    ss_res = sum((a - b) ** 2 for a, b in zip(y_true, y_pred))
    ss_tot = sum((a - mean_y) ** 2 for a in y_true)
    if ss_tot == 0:
        return 0.0
    return 1 - (ss_res / ss_tot)


def train_with_sklearn():
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).copy()
    for column in FEATURE_COLUMNS + [TARGET_COLUMN]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).copy()

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        max_depth=12,
        min_samples_leaf=2,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "training_backend": "scikit-learn",
        "rmse": round(float(mean_squared_error(y_test, predictions) ** 0.5), 4),
        "mae": round(float(mean_absolute_error(y_test, predictions)), 4),
        "r2": round(float(r2_score(y_test, predictions)), 4),
        "feature_columns": FEATURE_COLUMNS,
        "feature_importance": {
            column: round(float(importance), 6)
            for column, importance in zip(FEATURE_COLUMNS, model.feature_importances_)
        },
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }
    return model, metrics


def train_with_fallback():
    rows, targets = load_rows()
    train_rows, test_rows, train_targets, test_targets = simple_train_test_split(rows, targets)

    model = FallbackBurnoutRegressor()
    model.fit(train_rows, train_targets)
    predictions = model.predict(test_rows)

    metrics = {
        "training_backend": "python-fallback",
        "rmse": round(float(rmse(test_targets, predictions)), 4),
        "mae": round(float(mae(test_targets, predictions)), 4),
        "r2": round(float(r2(test_targets, predictions)), 4),
        "feature_columns": FEATURE_COLUMNS,
        "feature_importance": {
            column: round(float(importance), 6)
            for column, importance in zip(FEATURE_COLUMNS, model.feature_importances_)
        },
        "training_rows": int(len(train_rows)),
        "test_rows": int(len(test_rows)),
    }
    return model, metrics


def main() -> None:
    try:
        model, metrics = train_with_sklearn()
    except Exception as exc:  # pragma: no cover
        print(f"scikit-learn training unavailable, using fallback model: {exc}")
        model, metrics = train_with_fallback()

    save_model(model, MODEL_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print("Model training complete.")
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
