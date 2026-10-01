from __future__ import annotations

import json

from flask import Flask, jsonify, request

from utils import (
    METRICS_PATH,
    MODEL_PATH,
    build_feature_frame,
    load_model,
    relief_plan,
    sieve_risk_category,
)


app = Flask(__name__)
model = load_model(MODEL_PATH)
with open(METRICS_PATH, "r", encoding="utf-8") as handle:
    metrics = json.load(handle)


@app.get("/health")
def health() -> tuple:
    return jsonify({"status": "ok", "model_loaded": True}), 200


@app.get("/feature-importance")
def feature_importance() -> tuple:
    return jsonify(metrics["feature_importance"]), 200


@app.post("/predict")
def predict() -> tuple:
    payload = request.get_json(force=True)

    features = {
        "Designation": float(payload["Designation"]),
        "Resource Allocation": float(payload["Resource Allocation"]),
        "Mental Fatigue Score": float(payload["Mental Fatigue Score"]),
        "Tenure_Days": float(payload["Tenure_Days"]),
        "Workload_Intensity": float(payload["Workload_Intensity"]),
    }

    feature_frame = build_feature_frame(
        designation=features["Designation"],
        resource_allocation=features["Resource Allocation"],
        mental_fatigue_score=features["Mental Fatigue Score"],
        tenure_days=features["Tenure_Days"],
        workload_intensity=features["Workload_Intensity"],
    )

    prediction = float(model.predict(feature_frame)[0])
    prediction = max(0.0, min(1.0, prediction))
    risk = sieve_risk_category(prediction)

    return (
        jsonify(
            {
                "predicted_burn_rate": round(prediction, 4),
                "risk_category": risk,
                "relief_plan": relief_plan(prediction, features),
            }
        ),
        200,
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
