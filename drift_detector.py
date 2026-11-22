import os
import numpy as np
import pandas as pd

from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score)

from retraining.config import (production_logs, prediction_logs, feedback_logs)

psi_eps = 1e-4

def actual_distribution(column, psi_spec):
    if psi_spec["type"] == "numeric":
        idx = np.searchsorted(psi_spec["values"], column, side="right")
        counts = np.bincount(idx, minlength=len(psi_spec["expected"]))
    else:
        counts = np.array([(column == v).sum() for v in psi_spec["values"]])
        counts = np.append(counts, len(column) - counts.sum())
    return counts / len(column)

def calculate_psi(expected, actual):
    e = np.clip(expected, psi_eps, None)
    a = np.clip(actual, psi_eps, None)
    e /= e.sum()
    a /= a.sum()
    return float(np.sum((a - e) * np.log(a / e)))

def read_csv(path):
    if not os.path.isfile(path): return None
    return pd.read_csv(path)

def detect_data_drift(baseline):
    logs = read_csv(production_logs)
    if logs is None: return {"status": "no_data", "message": "No production logs "}
    logs = logs.tail(500)
    if len(logs) < 100: return {"status": "insufficient_data", "message": "Not enough sample", "sample": len(logs)}

    features_psi, psi_values = {}, []
    for feature, stats in baseline.items():
        actual = actual_distribution(logs[feature], stats["psi"])
        psi    = calculate_psi(stats["psi"]["expected"], actual)
        psi_values.append(psi)
        features_psi[feature] = {"psi": round(psi, 4)}
    overall_data_drift = float(np.mean(psi_values))
    return {
        "status": "ok",
        "sample": len(logs),
        "features_psi": features_psi,
        "overall_data_drift": round(overall_data_drift, 4)
    }

def detect_prediction_drift(prediction_baseline, version):
    logs = read_csv(prediction_logs)
    if logs is None: return {"status": "no_data", "message": "No prediction logs "}
    logs = logs[logs["version"].astype(str) == str(version)].tail(500)
    if len(logs) < 100: return {"status": "insufficient_data", "message": "Not enough sample", "sample": len(logs)}

    psi = calculate_psi(prediction_baseline["psi"]["expected"],actual_distribution(logs["probability"], prediction_baseline["psi"]))
    positive_rate = float(logs["prediction"].mean())
    positive_rate_shift = positive_rate - prediction_baseline["positive_rate"]

    return {
        "status": "ok",
        "sample": len(logs),
        "psi": psi,
        "positive_rate": positive_rate,
        "baseline_positive_rate": prediction_baseline["positive_rate"],
        "positive_rate_shift": positive_rate_shift,
        "mean_probability": logs["probability"].mean(),
        "baseline_mean_probability": prediction_baseline["mean_prob"]
    }

def detect_performance_drift(baseline_metrics, version):
    preds = read_csv(prediction_logs)
    fbs   = read_csv(feedback_logs)
    if preds is None or fbs is None or fbs.empty: return {"status": "no_data", "message": "No feedback"}

    fbs = fbs.drop_duplicates("prediction_id", keep="last")
    merged = preds.merge(fbs[["prediction_id", "actual"]], on="prediction_id")
    merged = merged[merged["version"].astype(str) == str(version)].tail(500)
    if len(merged) < 100: return {"status": "insufficient_data", "message": "Not enough sample", "sample": len(merged)}

    metrics = {
        "accuracy":  round(accuracy_score(merged["actual"], merged["prediction"]), 4),
        "precision": round(precision_score(merged["actual"], merged["prediction"]), 4),
        "recall":  round(recall_score(merged["actual"], merged["prediction"]), 4),
        "f1_score":  round(f1_score(merged["actual"], merged["prediction"]), 4),
        "roc_auc":  round(roc_auc_score(merged["actual"], merged["probability"]), 4)
    }

    drops = {}
    for metric, value in metrics.items():
        drops[metric] = round(baseline_metrics[metric] - value, 4)
    f1_drop = max(drops.get("f1_score", 0.0), 0.0)

    return{
        "status": "ok",
        "labeled_samples": len(merged),
        "current_metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "metric_drops": drops,
        "f1_drop": f1_drop
    }