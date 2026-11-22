from retraining.config import (model_name, uri_path, artifact_dir)
import mlflow
import joblib
import json
from mlflow.tracking import MlflowClient

def current_production():
    mlflow.set_tracking_uri(f"sqlite:///{uri_path}")
    return MlflowClient().get_model_version_by_alias(model_name, "production")


def rollback():

    mlflow.set_tracking_uri(f"sqlite:///{uri_path}")

    try:
        previous_ver = MlflowClient().get_model_version_by_alias(model_name, "archived").version
    except:
        raise RuntimeError("Error")
    
    MlflowClient().set_registered_model_alias(model_name, "archived", current_production().version)
    MlflowClient().set_registered_model_alias(model_name, "production", previous_ver)
    return previous_ver;

def load_production():

    mlflow.set_tracking_uri(f"sqlite:///{uri_path}")

    version_dir = artifact_dir / "versions" / f"v{current_production().version}"
    model = joblib.load(version_dir / "churn_model.pkl")
    scaler = joblib.load(version_dir / "scaler.pkl")

    with open(version_dir / "baseline.json", "r") as f:
        baseline = json.load(f)

    with open(version_dir / "prediction_baseline.json", "r") as f:
        prediction_baseline = json.load(f)

    with open(version_dir / "metadata.json", "r") as f:
        metadata = json.load(f)

    return{
        "version_dir"  : version_dir,
        "version"      : current_production().version,
        "model"        : model,
        "scaler"       : scaler,
        "feature_names": metadata["feature_names"],
        "baseline"     : baseline,
        "prediction_baseline": prediction_baseline,
        "metadata"     : metadata        
    }



