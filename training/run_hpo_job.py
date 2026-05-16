import sagemaker
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.tuner import (
    HyperparameterTuner,
    IntegerParameter,
    CategoricalParameter
)
from sagemaker.session import Session

session = Session()
role    = "arn:aws:iam::011528270076:role/sagemaker-churn-execution-role"
bucket  = "churn-sagemaker-artifacts"

# Base estimator — same as training job
estimator = SKLearn(
    entry_point="train.py",
    source_dir="training/",
    framework_version="1.2-1",
    instance_type="ml.m5.large",
    instance_count=1,
    role=role,
    sagemaker_session=session,
    base_job_name="churn-hpo",
    use_spot_instances=True,
    max_run=3600,
    max_wait=7200,
    metric_definitions=[
        {"Name": "roc_auc",   "Regex": "roc_auc=([0-9.]+)"},
        {"Name": "accuracy",  "Regex": "accuracy=([0-9.]+)"},
        {"Name": "f1_score",  "Regex": "f1_score=([0-9.]+)"},
    ],
    hyperparameters={"experiment_name": "churn-hpo"},
    output_path=f"s3://{bucket}/models/hpo/",
)

# Hyperparameter search space — from roadmap Phase 4
hyperparameter_ranges = {
    "n_estimators":      IntegerParameter(50, 300),
    "max_depth":         IntegerParameter(5, 25),
    "min_samples_split": IntegerParameter(2, 10),
}

# Bayesian tuner — learns from previous trials to find optimum faster
tuner = HyperparameterTuner(
    estimator=estimator,
    objective_metric_name="roc_auc",
    objective_type="Maximize",
    hyperparameter_ranges=hyperparameter_ranges,
    max_jobs=20,
    max_parallel_jobs=5,
    strategy="Bayesian",
    base_tuning_job_name="churn-hpo",
    metric_definitions=[
        {"Name": "roc_auc",   "Regex": "roc_auc=([0-9.]+)"},
        {"Name": "accuracy",  "Regex": "accuracy=([0-9.]+)"},
        {"Name": "f1_score",  "Regex": "f1_score=([0-9.]+)"},
    ],
)

# Input channels
inputs = {
    "train": f"s3://{bucket}/data/processed/train/",
    "test":  f"s3://{bucket}/data/processed/test/",
}

print("Starting Hyperparameter Tuning Job...")
print("Strategy: Bayesian | Max jobs: 20 | Parallel: 5")
print("Objective: maximize roc_auc")
tuner.fit(inputs, wait=True, logs="None")

# Get best training job results
best_job = tuner.best_training_job()
print(f"\nBest training job: {best_job}")

tuning_job = tuner.latest_tuning_job
description = tuning_job.describe()
best_metrics = description["BestCandidate"] if "BestCandidate" in description else {}
print(f"Best ROC AUC: {description.get('BestObjectiveValue', 'N/A')}")
print(f"Best model: s3://{bucket}/models/hpo/{best_job}/output/model.tar.gz")
