import pytest
import pandas as pd
import numpy as np
import sys, os
from processing.preprocess import preprocess, split_data

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "customerID":      ["1","2","3","4","5"],
        "gender":          ["Male","Female","Male","Female","Male"],
        "SeniorCitizen":   [0,1,0,0,1],
        "Partner":         ["Yes","No","Yes","No","Yes"],
        "Dependents":      ["No","No","Yes","No","No"],
        "tenure":          [12,1,72,5,24],
        "PhoneService":    ["Yes","No","Yes","Yes","No"],
        "MultipleLines":   ["No","No phone service","Yes","No","No phone service"],
        "InternetService": ["DSL","Fiber optic","DSL","No","Fiber optic"],
        "OnlineSecurity":  ["No","No","Yes","No internet service","No"],
        "OnlineBackup":    ["Yes","No","No","No internet service","Yes"],
        "DeviceProtection":["No","Yes","Yes","No internet service","No"],
        "TechSupport":     ["No","No","Yes","No internet service","No"],
        "StreamingTV":     ["No","No","Yes","No internet service","No"],
        "StreamingMovies": ["No","No","No","No internet service","Yes"],
        "Contract":        ["Month-to-month","Month-to-month","Two year","Month-to-month","One year"],
        "PaperlessBilling":["Yes","Yes","No","No","Yes"],
        "PaymentMethod":   ["Electronic check","Mailed check","Bank transfer","Credit card","Electronic check"],
        "MonthlyCharges":  [65.5,29.85,115.5,20.25,45.5],
        "TotalCharges":    ["786.0","29.85","8684.8","100.0"," "],
        "Churn":           ["Yes","No","No","No","Yes"],
    })

def test_drops_customer_id(sample_df):
    result = preprocess(sample_df)
    assert "customerID" not in result.columns

def test_churn_encoded_binary(sample_df):
    result = preprocess(sample_df)
    assert set(result["Churn"].unique()).issubset({0, 1})

def test_total_charges_numeric(sample_df):
    result = preprocess(sample_df)
    assert result["TotalCharges"].dtype in [np.float64, np.float32]

def test_total_charges_no_nulls(sample_df):
    result = preprocess(sample_df)
    assert result["TotalCharges"].isna().sum() == 0

def test_categorical_columns_encoded(sample_df):
    result = preprocess(sample_df)
    object_cols = result.select_dtypes(include=["object"]).columns
    assert len(object_cols) == 0

def test_row_count_preserved(sample_df):
    result = preprocess(sample_df)
    assert len(result) == len(sample_df)

def test_split_sizes(sample_df):
    df = preprocess(sample_df)
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.4, random_state=42)
    assert len(X_train) + len(X_test) == len(df)

def test_split_no_churn_in_features(sample_df):
    # Use larger test_size to ensure both classes represented in small sample
    df = preprocess(sample_df)
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.4, random_state=42)
    assert "Churn" not in X_train.columns
    assert "Churn" not in X_test.columns
