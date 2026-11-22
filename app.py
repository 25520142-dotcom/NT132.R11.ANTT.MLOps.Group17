from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import Response
from prometheus_client import (Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST)
from rollback_model import (rollback, load_production)
from retraining.config import production_logs, prediction_logs, feedback_logs
from retraining.feature_engineering import feature_engineering

import time
import uuid
import pandas as pd
from typing import Literal

def log(df: pd.DataFrame, path):
    if path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)

def reload_production_artifacts():
    global artifacts
    artifacts = load_production()

reload_production_artifacts()

prediction_requests = Counter("prediction_requests_total", "Total number of prediction requests")
prediction_errors   = Counter("prediction_errors_total", "Total failed prediction requests")
prediction_latency  = Histogram("prediction_latency_seconds", "Prediction latency")   

feedback_received   = Counter("feedback_received_total", "Total number of feedbacks")
feedback_errors   = Counter("feedback_errors_total", "Total failed feedback requests")

app = FastAPI(title= "Customer Churn Prediction API", version=str(artifacts["version"]))

class Customer(BaseModel):
    Age: float
    Gender: Literal["Female", "Male"]
    Tenure: float
    Usage_Frequency: float
    Support_Calls: float
    Payment_Delay: float
    Subscription_Type: Literal["Basic", "Standard", "Premium"]
    Contract_Length: Literal["Monthly", "Quarterly", "Annual"]
    Total_Spend: float
    Last_Interaction: float

class FeedBack(BaseModel):
    prediction_id: str
    actual: Literal[0,1]

@app.get("/health")
def health():
    if artifacts is None:
        raise HTTPException(503, "Model is not loaded")
    return {
        "status": "healthy", "version": artifacts["version"]}

@app.post("/predict")
def predict(customer: Customer):
    start_time = time.time()

    prediction_requests.inc()

    try:
        data          = customer.model_dump()
        model         = artifacts["model"]
        scaler        = artifacts["scaler"]
        feature_names = artifacts["feature_names"]

        df_features = pd.DataFrame([{
            "Age": data["Age"],
            "Gender": data["Gender"],
            "Tenure": data["Tenure"],
            "Usage Frequency": data["Usage_Frequency"],
            "Support Calls": data["Support_Calls"],
            "Payment Delay": data["Payment_Delay"],
            "Subscription Type": data["Subscription_Type"],
            "Contract Length": data["Contract_Length"],
            "Total Spend": data["Total_Spend"],
            "Last Interaction": data["Last_Interaction"],
        }])

        df_features = feature_engineering(df_features)
        df_features = df_features.loc[:, feature_names]
        log(df_features, production_logs)

        scaled      = scaler.transform(df_features)

        prediction_id = str(uuid.uuid4())
        prediction  = int(model.predict(scaled)[0])
        probability = float(model.predict_proba(scaled)[0][1])

        df_results = pd.DataFrame([{
            "prediction_id": prediction_id,
            "prediction"   : prediction,
            "probability"  : probability,
            "version"      : str(artifacts["version"])
        }])

        log(df_results, prediction_logs)

        prediction_latency.observe(
            time.time() - start_time
        )

        return {
            "prediction_id": prediction_id,
            "prediction": prediction,
            "probability": round(probability, 4)
        }

    except Exception as e:
        prediction_errors.inc()
        raise HTTPException(500, str(e))
    
@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

@app.post("/feedback")
def feedback (fb : FeedBack):
    if not prediction_logs.exists(): raise HTTPException(404, "No production logs.")
    found = pd.read_csv(prediction_logs, usecols=["prediction_id"])["prediction_id"]  
    if fb.prediction_id not in set(found): raise HTTPException(404, "Unknown prediction id.")

    df = pd.DataFrame([{
        "prediction_id": fb.prediction_id,
        "actual"       : fb.actual
    }])

    log(df, feedback_logs)
    feedback_received.inc()


@app.post("/admin/rollback_and_load")
def rollback_and_load():
    after_rollback_ver = rollback()
    reload_production_artifacts()
    return{
        "status" : "ok",
        "current version": after_rollback_ver
    }

@app.post("/admin/reload-production")
def reload_production():
    try:
        reload_production_artifacts()
        return {"status": "success", "version": artifacts["version"]}
    except Exception as e:
        raise HTTPException(503, str(e))