import pandas as pd

import mlflow
import mlflow.sklearn
import mlflow.xgboost

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier)
from xgboost import XGBClassifier

from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score)

from retraining.config import (random_state, uri_path, experiment_name)

def train_models(preprocessed_data):
    X_train_scaled = preprocessed_data["X_train_scaled"]
    X_test_scaled =  preprocessed_data["X_test_scaled"]

    y_train = preprocessed_data["y_train"]
    y_test =  preprocessed_data["y_test"]

    mlflow.set_tracking_uri(f"sqlite:///{uri_path}")
    mlflow.set_experiment(experiment_name)

    models = {
        "Logistic Regression": LogisticRegression(random_state=random_state, max_iter=200),
        "Random Forest"      : RandomForestClassifier(random_state=random_state, max_depth=16, min_samples_leaf=2),
        "Gradient Boosting"  : GradientBoostingClassifier(random_state=random_state),
        "XGBoost"            : XGBClassifier(random_state=random_state, eval_metric = "logloss")
    }

    results ={}
    best_run_id = None
    best_model = None
    best_model_name = None
    best_model_uri = None
    best_y_prob = None
    best_f1 = -1

    for name, model in models.items():
        with mlflow.start_run(run_name=name) as run:
            model.fit(X_train_scaled, y_train)

            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]

            accuracy = accuracy_score(y_test, y_pred)
            precision= precision_score(y_test, y_pred)
            recall   = recall_score(y_test, y_pred)
            f1       = f1_score(y_test, y_pred)
            roc_auc  = roc_auc_score(y_test, y_prob)

            results[name] = {
                "Accuracy" : accuracy,
                "Precision": precision,
                "Recall"   : recall,
                "F1 Score" : f1,
                "ROC AUC"  : roc_auc
            }

            params = model.get_params()
            mlflow.log_params({
                "model_name": name,
                "random_state": random_state,
                **params
            })

            match name:
                case "XGBoost": model_info = mlflow.xgboost.log_model(xgb_model=model, name="model")
                case         _: model_info = mlflow.sklearn.log_model(sk_model =model, name="model", skops_trusted_types=["sklearn.tree._tree.Tree"])

            mlflow.log_metrics(results[name])

            if f1 > best_f1:
                best_run_id = run.info.run_id
                best_f1 = f1
                best_model = model
                best_model_name = name
                best_y_prob = y_prob
                best_model_uri = f"models:/{model_info.model_id}"

    results_df = pd.DataFrame(results).T

    print("\n==============================")
    print("MODEL COMPARISON")
    print("==============================")
    print(results_df.round(4))

    print("\nBest Model :", best_model_name)
    print("Best F1    :", round(best_f1, 4))


    return {
        "best_model": best_model,
        "best_model_run_id": best_run_id,
        "best_model_name": best_model_name,
        "best_y_prob": best_y_prob,
        "best_model_uri": best_model_uri,

        "best_f1": best_f1,

        "results_df": results_df,

        "models": models,
        
        "X_test": preprocessed_data["X_test"],
        "y_test": y_test,

        "scaler": preprocessed_data["scaler"],
        "X_test_scaled": X_test_scaled

    }
