import boto3
import sagemaker
from sagemaker.session import Session
from sagemaker.sklearn.model import SKLearnModel

session   = Session()
sm_client = boto3.client("sagemaker", region_name="us-east-1")
role      = "arn:aws:iam::011528270076:role/sagemaker-churn-execution-role"
bucket    = "churn-sagemaker-artifacts"

# Best model from HPO Phase 4
best_job       = "churn-hpo-260516-0610-013-7f95b1dd"
model_artifacts = f"s3://{bucket}/models/hpo/{best_job}/output/model.tar.gz"

print(f"Registering model from: {model_artifacts}")

# Create model group if not exists
model_group_name = "churn-prediction-models"
try:
    sm_client.create_model_package_group(
        ModelPackageGroupName=model_group_name,
        ModelPackageGroupDescription="Customer Churn Prediction — RandomForest models"
    )
    print(f"Created model group: {model_group_name}")
except sm_client.exceptions.ConflictException:
    print(f"Model group already exists: {model_group_name}")

# Register model package
response = sm_client.create_model_package(
    ModelPackageGroupName=model_group_name,
    ModelPackageDescription="RandomForest ROC AUC 0.8445 — Bayesian HPO best trial",
    InferenceSpecification={
        "Containers": [
            {
                "Image": "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",
                "ModelDataUrl": model_artifacts,
                "Framework": "SKLEARN",
                "FrameworkVersion": "1.2-1",
            }
        ],
        "SupportedContentTypes": ["text/csv"],
        "SupportedResponseMIMETypes": ["text/csv"],
    },
    ModelApprovalStatus="PendingManualApproval",
    ModelMetrics={
        "ModelQuality": {
            "Statistics": {
                "ContentType": "application/json",
                "S3Uri": f"s3://{bucket}/models/metrics/model_quality.json"
            }
        }
    },
    CustomerMetadataProperties={
        "roc_auc":           "0.8445",
        "accuracy":          "0.7963",
        "f1_score":          "0.5710",
        "model_type":        "RandomForestClassifier",
        "n_estimators":      "291",
        "max_depth":         "7",
        "min_samples_split": "2",
        "training_job":      best_job,
        "dataset":           "telco-churn",
        "developer":         "heman"
    }
)

model_package_arn = response["ModelPackageArn"]
print(f"Model registered: {model_package_arn}")
print(f"Status: PendingManualApproval")
print(f"Next step: approve model to enable deployment")
