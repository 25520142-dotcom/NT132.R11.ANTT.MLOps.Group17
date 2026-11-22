from retraining.config              import (train_data, test_data, compare_data)
from retraining.load_data           import load_data
from retraining.feature_engineering import feature_engineering
from retraining.preprocessing       import preprocessing
from retraining.retrain             import train_models
from retraining.compare_models      import compare_models
from retraining.model_registry      import model_registry
from retraining.promote_model       import promote_model
from retraining.save_artifacts      import save_artifacts

def main():
    print("="*60)
    print("CUSTOMER CHURN RETRAINING PIPELINE")
    print("="*60)

    print("\nLoading datasets...")
    df = load_data(train_data, test_data)
    print(f"Dataset Shape: {df.shape}")

    print("\nFeature Engineering...")
    df_encoded = feature_engineering(df)
    print(f"Total Features: {len(df_encoded.columns)}")

    print("\nPreprocessing...")
    preprocessed = preprocessing(df_encoded)
    print("Preprocessing completed.")

    print("\nTraining models...")
    training_results = train_models(preprocessed)
    print("Training completed.")

    best_metrics = training_results["results_df"].loc[training_results["best_model_name"]].to_dict()
    print("\nRegistering model...")
    model_version = model_registry(training_results["best_model_uri"], best_metrics)
    print("Register model completed.")

    print("\nComparing with production model...")
    compare_result = compare_models(training_results, compare_data)
    print("Comparing completed.")

    if compare_result["deploy"]:   
        print("\nSaving new production artifacts...")
        save_artifacts(training_results["best_model"], preprocessed["scaler"], preprocessed["feature_names"], preprocessed["X_train"], training_results["best_model_name"], best_metrics, model_version, training_results["best_model_run_id"], training_results["best_y_prob"])
        print("Saving completed.")

        print("\nPromoting model...")
        promote_model(model_version)
        print("Promoted model.")

    else:
        print("\nCurrent production model performs better.")

    print("="*60)
    print("CUSTOMER CHURN RETRAINING PIPELINE COMPLETED")
    print("="*60)   

    print(f"Best Retrained Model : {training_results['best_model_name']}")
    print(f"Best F1 Score        : {training_results['best_f1']:.4f}")

    if compare_result["reason"]:
        print(f"Reason: {compare_result['reason']}")

if __name__ == "__main__":
    main()
