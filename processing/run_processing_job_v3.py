# SageMaker SDK v3 equivalent of Processing Job
# Uses @remote decorator — converts local function to managed SageMaker compute
# Compare with v2 approach in run_processing_job.py

import os
os.environ["SAGEMAKER_SDK_VERSION"] = "3"

from sagemaker.core.remote_function import remote
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import boto3

ROLE   = "arn:aws:iam::011528270076:role/sagemaker-churn-execution-role"
BUCKET = "churn-sagemaker-artifacts"

@remote(
    role=ROLE,
    instance_type="ml.m5.large",
    job_name_prefix="churn-preprocess-v3",
)
def preprocess_and_split():
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    import boto3, io

    # Read from S3 directly inside remote function
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket="churn-sagemaker-artifacts", Key="data/raw/churn.csv")
    df = pd.read_csv(io.BytesIO(obj["Body"].read()))

    # Preprocessing
    df = df.drop(columns=["customerID"])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    le = LabelEncoder()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = le.fit_transform(df[col])

    # Split
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Save back to S3
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df  = pd.concat([X_test, y_test], axis=1)

    s3.put_object(
        Bucket="churn-sagemaker-artifacts",
        Key="data/processed/v3/train/train.csv",
        Body=train_df.to_csv(index=False)
    )
    s3.put_object(
        Bucket="churn-sagemaker-artifacts",
        Key="data/processed/v3/test/test.csv",
        Body=test_df.to_csv(index=False)
    )

    return {
        "train_shape": train_df.shape,
        "test_shape": test_df.shape,
        "train_s3": f"s3://churn-sagemaker-artifacts/data/processed/v3/train/train.csv",
        "test_s3":  f"s3://churn-sagemaker-artifacts/data/processed/v3/test/test.csv",
    }

if __name__ == "__main__":
    result = preprocess_and_split()
    print(f"Done: {result}")
