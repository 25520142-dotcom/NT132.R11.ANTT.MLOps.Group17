from pathlib import Path

project_root = Path(__file__).resolve().parent.parent


data_dir = project_root / "data"
artifact_dir = project_root / "artifacts"

# train_data = data_dir / "train_data_degraded.csv"
train_data = data_dir / "train_data.csv"
# train_data = data_dir / "train_data_new.csv"
test_data = data_dir / "test_data.csv"
compare_data = data_dir / "test_data_new.csv"

production_logs = artifact_dir / "production_logs.csv"
prediction_logs = artifact_dir / "prediction_logs.csv"
feedback_logs   = artifact_dir / "feedback_logs.csv"

experiment_name = "Customer Churn"
model_name = "Churn_Prediction_Model"
uri_path = project_root / 'mlflow.db'

random_state = 42
test_size = 0.2

