import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from retraining.config import(random_state, test_size)

def preprocessing(df_encoded):
    missing_labels = df_encoded["Churn"].isna().sum()

    if missing_labels:
        print(f"\nRemoving {missing_labels} rows with missing Churn labels...")
        df_encoded = df_encoded.dropna(subset=["Churn"], ignore_index=True)

    X = df_encoded.drop(columns=["CustomerID", "Churn"])
    y = df_encoded["Churn"].astype(int)

    print(f"\nTraining sample : {len(X)}")
    print(f"Features          : {X.shape[1]}")
    print(f"Churn Distribution:\n{y.value_counts()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"\nTraining set: {len(X_train)}")
    print(f"\nTesting set : {len(X_test)}")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns= X_train.columns)
    X_test_scaled_df  = pd.DataFrame(X_test_scaled, columns= X_test.columns)

    return {
        "X_train": X_train,
        "X_test" : X_test,

        "y_train": y_train,
        "y_test" : y_test,

        "scaler": scaler,

        "X_train_scaled": X_train_scaled,
        "X_test_scaled" : X_test_scaled,

        "X_train_scaled_df": X_train_scaled_df,
        "X_test_scaled_df" : X_test_scaled_df,

        "feature_names"    : X.columns.tolist(),
    }