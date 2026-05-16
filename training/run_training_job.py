import sagemaker
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.session import Session

session = Session()
role    = "arn:aws:iam::011528270076:role/sagemaker-churn-execution-role"
bucket  = "churn-sagemaker-artifacts"

metric_definitions = [
    {"Name": "roc_auc",   "Regex": "roc_auc=([0-9\\.]+)"},
    {"Name": "accuracy",  "Regex": "accuracy=([0-9\\.]+)"},
    {"Name": "f1_score",  "Regex": "f1_score=([0-9\\.]+)"},
    {"Name": "precision", "Regex": "precision=([0-9\\.]+)"},
    {"Name": "recall",    "Regex": "recall=([0-9\\.]+)"},
]

# Custom ECR image — pre-installed dependencies, no runtime pip install
ECR_IMAGE = "011528270076.dkr.ecr.us-east-1.amazonaws.com/churn-mlops:latest"

estimator = SKLearn(
    entry_point="train.py",
    source_dir="training/",
    framework_version="1.2-1",
    image_uri=ECR_IMAGE,
    instance_type="ml.m5.large",
    instance_count=1,
    role=role,
    sagemaker_session=session,
    base_job_name="churn-training",
    use_spot_instances=True,
    max_run=3600,
    max_wait=7200,
    metric_definitions=metric_definitions,
    hyperparameters={
        "n_estimators":      100,
        "max_depth":         10,
        "min_samples_split": 2,
        "experiment_name":   "churn-prediction"
    },
    output_path=f"s3://{bucket}/models/",
    checkpoint_s3_uri=f"s3://{bucket}/checkpoints/",
    checkpoint_local_path="/opt/ml/checkpoints",
)

inputs = {
    "train": f"s3://{bucket}/data/processed/train/",
    "test":  f"s3://{bucket}/data/processed/test/",
}

print("Submitting Training Job...")
estimator.fit(inputs, wait=True, logs=True)
print(f"Training complete!")
print(f"Model artifacts: {estimator.model_data}")
