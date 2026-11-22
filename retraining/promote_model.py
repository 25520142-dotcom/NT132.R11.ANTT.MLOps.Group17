from retraining.config import (model_name, uri_path)

import mlflow
from mlflow.tracking import MlflowClient

def promote_model(new_version):
    mlflow.set_tracking_uri(f"sqlite:///{uri_path}")
    client = MlflowClient()

    try:
        old_production = client.get_model_version_by_alias(model_name, "production")
        old_version = old_production.version

    except:
        old_version = None

    if old_version is not None:
        client.delete_registered_model_alias(model_name, "production")
        client.set_registered_model_alias(model_name, "archived", old_version)

    client.delete_registered_model_alias(model_name, "candidate")
    client.set_registered_model_alias(model_name, "production", new_version)
