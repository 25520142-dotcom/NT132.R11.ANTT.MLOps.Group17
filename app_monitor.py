from fastapi import FastAPI, HTTPException
from fastapi.responses import Response

from prometheus_client import (Gauge, generate_latest, CONTENT_TYPE_LATEST)
from rollback_model import load_production, current_production
from drift_detector import detect_data_drift, detect_prediction_drift, detect_performance_drift

performance_metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
def reload_production_artifacts():
    global artifacts, baseline
    artifacts = load_production()
    baseline  = artifacts["baseline"]

reload_production_artifacts()

def sync_if_needed():
    if current_production().version != artifacts["version"]:
         reload_production_artifacts()

app = FastAPI(title="Customer Churn Drift Monitor", version=str(artifacts["version"]))

feature_gauges = {}
for feature in baseline.keys():
        feature_gauges[feature] = Gauge(f"psi_{feature.lower().replace(' ', '_')}", f"PSI for {feature}")

overall_data_drift = Gauge("overall_data_drift", "Overall Data Drift Score")

prediction_drift_psi = Gauge("prediction_drift_psi", "PSI of predicted probability vs training pipeline")
prediction_positive_rate = Gauge("prediction_positive_rate", "Predicted churn rate in production")
prediction_positive_rate_shift = Gauge("prediction_positive_rate_shift", "Predicted churn rate minus baseline")

model_performance      = Gauge("model_performance", "Live metric computed from feedback labels", ["metric"])
model_performance_drop = Gauge("model_performance_drop", "Baseline metric minus live metric", ["metric"])
model_f1_drop          = Gauge("model_f1_drop", "Baseline F1 minus live F1")
model_labeled_samples  = Gauge("model_labeled_samples", "Labeled samples used for performance drift")

def calculate_data_drift():
    report = detect_data_drift(baseline)
    if report["status"] != "ok": return report

    for feature, item in report["features_psi"].items():
        feature_gauges[feature].set(item["psi"])
    overall_data_drift.set(report["overall_data_drift"])
    return report

def calculate_prediction_drift():
    report = detect_prediction_drift(artifacts["prediction_baseline"], artifacts["version"])
    if(report["status"] != "ok"): return report

    prediction_drift_psi.set(report["psi"])
    prediction_positive_rate.set(report["positive_rate"])
    prediction_positive_rate_shift.set(report["positive_rate_shift"])
    return report

def calculate_performance_drift():
    report = detect_performance_drift(artifacts["metadata"]["metrics"], artifacts["version"])
    if(report["status"] != "ok"): return report

    for name in performance_metrics:
        model_performance.labels(name).set(report["current_metrics"].get(name))
        model_performance_drop.labels(name).set(report["metric_drops"].get(name))
    model_f1_drop.set(report["f1_drop"])
    model_labeled_samples.set(report["labeled_samples"])
    return report

@app.get("/health")
def health():
    sync_if_needed()
    if artifacts is None:
        raise HTTPException(503, "Model is not loaded")
    return {
        "status": "healthy", "version": artifacts["version"]}

@app.get("/metrics")
def metrics():
    sync_if_needed()

    calculate_data_drift()
    calculate_prediction_drift()
    calculate_performance_drift()

    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)     

@app.post("/admin/reload-production")
def reload_production():
    try:
        reload_production_artifacts()
        return {"status": "success", "version": artifacts["version"]}
    except Exception as e:
        raise HTTPException(503, str(e))