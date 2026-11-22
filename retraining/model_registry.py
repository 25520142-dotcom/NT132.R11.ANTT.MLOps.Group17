from retraining.config import (model_name, uri_path)

import mlflow
from mlflow.tracking import MlflowClient


def model_registry(uri, metrics):
    mlflow.set_tracking_uri(f"sqlite:///{uri_path}")
    registered = mlflow.register_model(model_uri=uri, name = model_name)

    for key, value in metrics.items():
        MlflowClient().set_model_version_tag(model_name, registered.version, key, value)
    MlflowClient().set_registered_model_alias(model_name, "candidate", registered.version)
    
    return registered.version