
import joblib
import pandas as pd

from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score)

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

from retraining.feature_engineering import feature_engineering
from retraining.config import (model_name, uri_path, artifact_dir)

MIN_F1_IMPROVEMENT = 0.005

def evaluate(model, X, y):
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]

    return {
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred),
        "Recall": recall_score(y, pred),
        "F1": f1_score(y, pred),
        "ROC_AUC": roc_auc_score(y, prob)
    }

def results(deploy, reason, old_metrics, new_metrics):
    return {
        "deploy": deploy,
        "reason": reason,
        "old_metrics": old_metrics,
        "new_metrics": new_metrics
    }

def compare_models(training_results, test_data):
    test_encoded = feature_engineering(pd.read_csv(test_data).dropna(subset=["Churn"]))
    X_compare = test_encoded.drop(columns=["CustomerID", "Churn"])
    y_compare = test_encoded["Churn"].astype(int)

    new_scaler = training_results["scaler"]
    new_X       = new_scaler.transform(X_compare.loc[:, new_scaler.feature_names_in_])
    new_metrics = evaluate(training_results["best_model"], new_X, y_compare) 

    mlflow.set_tracking_uri(f"sqlite:///{uri_path}")
    try:
        old_version = MlflowClient().get_model_version_by_alias(model_name, "production").version
    except MlflowException:
        return results(True, "First deployment", None, new_metrics)
    
    old_dir = artifact_dir / "versions" / f"v{old_version}"
    old_model = joblib.load(old_dir / "churn_model.pkl")
    old_scaler = joblib.load(old_dir / "scaler.pkl")
    old_X = old_scaler.transform(X_compare.loc[:, old_scaler.feature_names_in_])
    old_metrics = evaluate(old_model, old_X, y_compare)

    print("\nProduction Model")
    for metric, value in old_metrics.items():
        print(f"{metric:12}: {value:.4f}")

    print("\nRetrained Model")
    for metric, value in new_metrics.items():
        print(f"{metric:12}: {value:.4f}")


    print("\nDecision")
    print(f"Old F1: {old_metrics['F1']:.4f}")
    print(f"New F1: {new_metrics['F1']:.4f}")

    improvement = new_metrics["F1"] - old_metrics["F1"]
    if improvement >= MIN_F1_IMPROVEMENT:
        return results(
            True,
            f"F1 improved by {improvement:.4f}",
            old_metrics,
            new_metrics
        )
    
    return results(
        False,
        f"F1 improvement {improvement:.4f} below threshold",
        old_metrics,
        new_metrics
    )
