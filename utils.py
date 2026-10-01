from __future__ import annotations

import math
import pickle
from typing import Dict, List

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None


FEATURE_COLUMNS = [
    "Designation",
    "Resource Allocation",
    "Mental Fatigue Score",
    "Tenure_Days",
    "Workload_Intensity",
]

TARGET_COLUMN = "Burn Rate"
MODEL_PATH = "burnout_model.pkl"
METRICS_PATH = "model_metrics.json"


def sieve_risk_category(score: float) -> str:
    if score < 0.30:
        return "Low Risk"
    if score < 0.70:
        return "Medium Risk"
    return "High Risk"


def engineer_features(df: "pd.DataFrame") -> "pd.DataFrame":
    if pd is None:
        raise ImportError("pandas is required for dataframe-based feature engineering.")
    engineered = df.copy()
    for column in FEATURE_COLUMNS:
        engineered[column] = pd.to_numeric(engineered[column], errors="coerce")
    return engineered


def prepare_training_data(df: "pd.DataFrame") -> "pd.DataFrame":
    prepared = engineer_features(df)
    prepared = prepared.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).copy()
    return prepared


def build_feature_frame(
    designation: float,
    resource_allocation: float,
    mental_fatigue_score: float,
    tenure_days: float,
    workload_intensity: float,
) -> "pd.DataFrame":
    if pd is None:
        raise ImportError("pandas is required to build a dataframe feature frame.")
    return pd.DataFrame(
        [
            {
                "Designation": designation,
                "Resource Allocation": resource_allocation,
                "Mental Fatigue Score": mental_fatigue_score,
                "Tenure_Days": tenure_days,
                "Workload_Intensity": workload_intensity,
            }
        ]
    )


def save_model(model, path: str) -> None:
    try:
        import joblib

        joblib.dump(model, path)
    except ImportError:
        with open(path, "wb") as handle:
            pickle.dump(model, handle)


def load_model(path: str):
    try:
        import joblib

        return joblib.load(path)
    except ImportError:
        with open(path, "rb") as handle:
            return pickle.load(handle)


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: List[float], mean_value: float) -> float:
    if not values:
        return 0.0
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def _correlation(x_values: List[float], y_values: List[float]) -> float:
    mean_x = _mean(x_values)
    mean_y = _mean(y_values)
    std_x = _std(x_values, mean_x)
    std_y = _std(y_values, mean_y)
    if std_x == 0 or std_y == 0:
        return 0.0
    covariance = sum(
        (x_value - mean_x) * (y_value - mean_y)
        for x_value, y_value in zip(x_values, y_values)
    ) / len(x_values)
    return covariance / (std_x * std_y)


class FallbackBurnoutRegressor:
    def __init__(self) -> None:
        self.feature_importances_: List[float] = []
        self.feature_mins: Dict[str, float] = {}
        self.feature_maxs: Dict[str, float] = {}
        self.mean_target = 0.0

    def fit(self, rows: List[Dict[str, float]], y_values: List[float]) -> None:
        self.mean_target = _mean(y_values)
        raw_weights: List[float] = []

        for feature in FEATURE_COLUMNS:
            values = [float(row[feature]) for row in rows]
            self.feature_mins[feature] = min(values)
            self.feature_maxs[feature] = max(values)
            raw_weights.append(abs(_correlation(values, y_values)))

        weight_total = sum(raw_weights) or float(len(FEATURE_COLUMNS))
        self.feature_importances_ = [weight / weight_total for weight in raw_weights]

    def _normalize(self, feature: str, value: float) -> float:
        minimum = self.feature_mins[feature]
        maximum = self.feature_maxs[feature]
        if maximum == minimum:
            return 0.0
        return (value - minimum) / (maximum - minimum)

    def predict(self, rows):
        if pd is not None and isinstance(rows, pd.DataFrame):
            rows = rows.to_dict(orient="records")
            
        predictions: List[float] = []
        for row in rows:
            weighted_sum = 0.0
            for feature, importance in zip(FEATURE_COLUMNS, self.feature_importances_):
                weighted_sum += self._normalize(feature, float(row[feature])) * importance

            prediction = (self.mean_target * 0.35) + (weighted_sum * 0.65)
            predictions.append(max(0.0, min(1.0, prediction)))
        return predictions


def relief_plan(score: float, features: Dict[str, float]) -> str:
    risk = sieve_risk_category(score)
    if risk != "High Risk":
        return "Monitor workload weekly, encourage breaks, and maintain manager check-ins."

    actions: List[str] = [
        "Reduce active workload for the next 2 weeks by reassigning non-critical tasks.",
        "Schedule an HR or manager wellness check-in within 48 hours.",
        "Block one recovery day and protect after-hours boundaries.",
    ]

    if features["Mental Fatigue Score"] >= 7:
        actions.append("Prioritize mental health support and shorten meeting load this week.")
    if features["Resource Allocation"] >= 7:
        actions.append("Redistribute project ownership across teammates to ease capacity pressure.")
    if features["Workload_Intensity"] >= 1.75:
        actions.append("Rebalance deadlines and pause stretch goals until burnout indicators improve.")
    if features["Tenure_Days"] <= 90:
        actions.append("Add onboarding support and a mentor check-in to reduce early-tenure stress.")

    return " ".join(actions)
