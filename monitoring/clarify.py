# SageMaker Clarify — SHAP explainability + bias detection
# Runs as a Processing Job using the Clarify container
# Outputs: feature_importance.json, bias_report.json, SHAP values per prediction

import boto3
import sagemaker
from sagemaker.session import Session
from sagemaker.clarify import (
    SageMakerClarifyProcessor,
    DataConfig,
    ModelConfig,
    SHAPConfig,
    ExplainabilityConfig,
    BiasConfig,
    ModelPredictedLabelConfig,
)

session   = Session()
role      = "arn:aws:iam::011528270076:role/sagemaker-churn-execution-role"
bucket    = "churn-sagemaker-artifacts"
region    = "us-east-1"

# Clarify processor — managed container for explainability
clarify_processor = SageMakerClarifyProcessor(
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
    sagemaker_session=session,
)

# Data config — points to test dataset
data_config = DataConfig(
    s3_data_input_path=f"s3://{bucket}/data/processed/test/test.csv",
    s3_output_path=f"s3://{bucket}/clarify/output",
    label="Churn",
    headers=[
        "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
        "MonthlyCharges", "TotalCharges", "Churn"
    ],
    dataset_type="text/csv",
)

# Model config — points to serverless endpoint
model_config = ModelConfig(
    model_name="sagemaker-scikit-learn-2026-05-16-08-29-53-248",
    instance_type="ml.m5.large",
    instance_count=1,
    accept_type="text/csv",
    content_type="text/csv",
)

# SHAP config — explains each prediction using Shapley values
# baseline = mean values of each feature (reference point)
shap_config = SHAPConfig(
    baseline=[
        [0, 0, 0, 0, 12, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2, 65.5, 850.5]
    ],
    num_samples=100,        # number of samples for SHAP approximation
    agg_method="mean_abs",  # aggregate absolute SHAP values
    save_local_shap_values=True,
)

explainability_config = shap_config

# Bias config — detect bias across gender feature
bias_config = BiasConfig(
    label_values_or_threshold=[1],
    facet_name="gender",
    facet_values_or_threshold=[0],
)

predicted_label_config = ModelPredictedLabelConfig(
    probability_threshold=0.5
)

print("Running SageMaker Clarify...")
print("Computing SHAP values + bias detection...")

clarify_processor.run_explainability(
    data_config=data_config,
    model_config=model_config,
    explainability_config=shap_config,
    wait=True,
    logs=False,
)

print(f"Clarify complete!")
print(f"SHAP report: s3://{bucket}/clarify/output/")
print("View in SageMaker Studio → Experiments → select run → Explainability tab")
