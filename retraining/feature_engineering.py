import pandas as pd

def feature_engineering(df):
    categorical_col = df.select_dtypes(include=["object"]).columns.tolist()

    df["Subscription_Type_Num"] = (
        df["Subscription Type"].map({
            "Basic": 0,
            "Standard": 1,
            "Premium": 2
        })
    )

    df["Contract_Length_Num"] = (
        df["Contract Length"].map({
            "Monthly": 0,
            "Quarterly": 1,
            "Annual": 2
        })
    )

    df["Gender"] = (df["Gender"] == "Female").astype(int)

    df_encoded = df.drop(columns=categorical_col)

    df_encoded["Avg_Spend_Per_Month"] = (df_encoded["Total Spend"] / (df_encoded["Tenure"] + 1))
    df_encoded["Support_Calls_Per_Month"] = (df_encoded["Support Calls"] / (df_encoded["Tenure"] + 1))
    df_encoded["Engagement_Score"] = (
    df_encoded["Usage Frequency"] * 0.4
    + 1 / (df_encoded["Support Calls"] + 1) * 0.3
    + 1 / (df_encoded["Payment Delay"] + 1) * 0.3
)

    df_encoded["Tenure_Group_Num"] = (
        pd.cut(
        df_encoded["Tenure"].clip(lower=0, upper=100),
        bins=[0, 6, 12, 24, 48, 100],
        labels = ["0-6", "6-12", "12-24", "24-48", "48+"],
        include_lowest=True
        ).map({
            "0-6": 0,
            "6-12": 1,
            "12-24": 2,
            "24-48": 3,
            "48+": 4
        }).astype(float)
    )

    df_encoded["High_Support_Risk"] = (df_encoded["Support Calls"] > 5).astype(int)
    df_encoded["High_Payment_Risk"] = (df_encoded["Payment Delay"] > 10).astype(int)

    return df_encoded

