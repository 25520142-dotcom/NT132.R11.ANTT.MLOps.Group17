import json
import os
import joblib
import numpy as np


import mlflow

from datetime import datetime, timezone
from pathlib import Path
from retraining.config import artifact_dir

def build_psi(column):
    unique, count = np.unique(column, return_counts=True)
    if len(unique) <= 10:
        return {
            "type": "categorical",
            "values": unique.tolist(),
            "expected": (count / count.sum()).tolist() + [0.0]
        }
    
    edges = np.unique(np.quantile(column, np.linspace(0, 1, 11)))
    inner = edges[1:-1]
    idx = np.searchsorted(inner, column, side="right")
    expected = np.bincount(idx, minlength=len(inner) + 1) / len(column)
    return{
        "type": "numeric",
        "values": inner.tolist(),
        "expected": (expected / expected.sum()).tolist() 
    }

def save_artifacts(model, scaler, feature_names, X_train, model_name, metrics, version, run_id, y_prob):

    os.makedirs(artifact_dir, exist_ok=True)

    version_dir = artifact_dir / "versions" / f"v{version}"
    version_dir.mkdir(parents=True, exist_ok=True)

    model_path  = version_dir / "churn_model.pkl"
    scaler_path = version_dir / "scaler.pkl"
    baseline_path = version_dir / "baseline.json"
    prediction_baseline_path = version_dir /"prediction_baseline.json"
    metadata_path = version_dir / "metadata.json"

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    baseline = {}
    for column in X_train.columns:
        baseline[column] = {
            "mean": float(X_train[column].mean()),
            "std" : float(X_train[column].std()),
            "min" : float(X_train[column].min()),
            "max" : float(X_train[column].max()),
            "psi" : build_psi(X_train[column])
        }
    with open(baseline_path, "w") as f:
        json.dump(baseline, f, indent=4)

    prediction_baseline = {
        "samples": len(y_prob),
        "positive_rate": float((y_prob >= 0.5).mean()),
        "mean_prob": float(y_prob.mean()),
        "psi": build_psi(y_prob)
    }
    with open(prediction_baseline_path, "w") as f:
        json.dump(prediction_baseline, f, indent=4)

    metadata = {
        "model_name"      : model_name,
        "version"         : version,
        "created_at"      : datetime.now(timezone.utc).isoformat(),
        "training_samples": len(X_train),
        "feature_count"   : len(feature_names),
        "feature_names"   : feature_names,
        "metrics"         : {
            "accuracy" : round(metrics["Accuracy"], 4),
            "precision": round(metrics["Precision"], 4),
            "recall"   : round(metrics["Recall"], 4),
            "f1_score" : round(metrics["F1 Score"], 4),
            "roc_auc"  : round(metrics["ROC AUC"], 4)
        }
    }
    with open( metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    mlflow.log_artifact(scaler_path, "Scaler", run_id)
    mlflow.log_artifact(baseline_path, "Baseline", run_id)
    mlflow.log_artifact(prediction_baseline_path, "Prediction Baseline", run_id)
    mlflow.log_artifact(metadata_path, "Metadata", run_id)

    print("\nProduction artifacts created successfully.")