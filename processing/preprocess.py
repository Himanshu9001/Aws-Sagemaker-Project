# processing/preprocess.py
# SageMaker Processing Job entry point for churn data preprocessing.
# Core logic unchanged from EKS version — only paths adapted for
# SageMaker's container mount conventions (/opt/ml/processing/*)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import logging

# Setup logging — SageMaker streams these to CloudWatch automatically
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(filepath):
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Data shape: {df.shape}")
    return df


def preprocess(df):
    logger.info("Starting preprocessing...")

    # Drop customerID — not a predictive feature
    df = df.drop(columns=['customerID'])

    # TotalCharges has whitespace strings — coerce to float
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Fill NaN TotalCharges with median
    median_val = df['TotalCharges'].median()
    df['TotalCharges'] = df['TotalCharges'].fillna(median_val)
    logger.info(f"Filled missing TotalCharges with median: {median_val}")

    # Binary encode target
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Label encode all remaining categoricals
    categorical_cols = df.select_dtypes(include=['object', 'str']).columns.tolist()
    logger.info(f"Encoding categorical columns: {categorical_cols}")
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    logger.info(f"Preprocessed data shape: {df.shape}")
    return df


def split_data(df, test_size=0.2, random_state=42):
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # preserve churn ratio in both splits
    )
    logger.info(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def save_data(X_train, X_test, y_train, y_test, train_dir, test_dir):
    # SageMaker expects separate output channels — train and test
    # Each channel maps to a distinct S3 prefix after job completes
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_path = os.path.join(train_dir, 'train.csv')
    test_path = os.path.join(test_dir, 'test.csv')

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(f"Train data saved: {train_path} — shape: {train_df.shape}")
    logger.info(f"Test data saved:  {test_path} — shape: {test_df.shape}")


if __name__ == "__main__":
    # SageMaker Processing Job mounts S3 inputs to /opt/ml/processing/input
    # and uploads everything under /opt/ml/processing/output/* back to S3
    # These paths are fixed by SageMaker — do not change them
    INPUT_PATH  = "/opt/ml/processing/input/churn.csv"
    TRAIN_DIR   = "/opt/ml/processing/output/train"
    TEST_DIR    = "/opt/ml/processing/output/test"

    df = load_data(INPUT_PATH)
    df = preprocess(df)
    X_train, X_test, y_train, y_test = split_data(df)
    save_data(X_train, X_test, y_train, y_test, TRAIN_DIR, TEST_DIR)

    logger.info("Preprocessing complete!")