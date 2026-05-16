# SageMaker Pipeline — Customer Churn Prediction
# DAG: preprocess → train → evaluate → conditional register → deploy
# Replaces Airflow DAG from EKS project with fully managed SageMaker orchestration

import boto3
import sagemaker
from sagemaker.session import Session
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.sklearn.estimator import SKLearnModel
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.inputs import TrainingInput
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, CacheConfig
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.parameters import ParameterFloat, ParameterString, ParameterInteger
from sagemaker.workflow.properties import PropertyFile

session = Session()
ECR_IMAGE = "011528270076.dkr.ecr.us-east-1.amazonaws.com/churn-mlops:latest"

role    = "arn:aws:iam::011528270076:role/sagemaker-churn-execution-role"
bucket  = "churn-sagemaker-artifacts"
region  = "us-east-1"

# Pipeline parameters — configurable at runtime without code changes
roc_auc_threshold   = ParameterFloat(name="RocAucThreshold",   default_value=0.83)
instance_type       = ParameterString(name="InstanceType",      default_value="ml.m5.large")
model_approval      = ParameterString(name="ModelApprovalStatus", default_value="PendingManualApproval")

# Cache config — reuse step outputs if inputs unchanged (saves cost)
cache_config = CacheConfig(enable_caching=True, expire_after="PT24H")

# ── Step 1: Processing ──────────────────────────────────────────────────────
processor = SKLearnProcessor(
    framework_version="1.2-1",
    role=role,
    instance_type=instance_type,
    instance_count=1,
    base_job_name="churn-pipeline-preprocess",
    sagemaker_session=session,
)

step_process = ProcessingStep(
    name="ChurnPreprocess",
    processor=processor,
    code="processing/preprocess.py",
    inputs=[
        ProcessingInput(
            source=f"s3://{bucket}/data/raw/churn.csv",
            destination="/opt/ml/processing/input"
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name="train",
            source="/opt/ml/processing/output/train",
            destination=f"s3://{bucket}/pipeline/data/train"
        ),
        ProcessingOutput(
            output_name="test",
            source="/opt/ml/processing/output/test",
            destination=f"s3://{bucket}/pipeline/data/test"
        ),
    ],
    cache_config=cache_config,
)

# ── Step 2: Training ────────────────────────────────────────────────────────
estimator = SKLearn(
    entry_point="train.py",
    source_dir="training/",
    framework_version="1.2-1",
    instance_type=instance_type,
    instance_count=1,
    role=role,
    sagemaker_session=session,
    base_job_name="churn-pipeline-train",
    disable_profiler=True,
    image_uri=ECR_IMAGE,
    use_spot_instances=True,
    max_run=3600,
    max_wait=7200,
    metric_definitions=[
        {"Name": "roc_auc",   "Regex": "roc_auc=([0-9.]+)"},
        {"Name": "accuracy",  "Regex": "accuracy=([0-9.]+)"},
        {"Name": "f1_score",  "Regex": "f1_score=([0-9.]+)"},
    ],
    hyperparameters={
        "n_estimators":      291,
        "max_depth":         7,
        "min_samples_split": 2,
        "experiment_name":   "churn-pipeline"
    },
    output_path=f"s3://{bucket}/pipeline/models/",
)

step_train = TrainingStep(
    name="ChurnTrain",
    estimator=estimator,
    inputs={
        "train": TrainingInput(
            s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
            content_type="text/csv"
        ),
        "test": TrainingInput(
            s3_data=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
            content_type="text/csv"
        ),
    },
    cache_config=cache_config,
    depends_on=[step_process],
)

# ── Step 3: Evaluate ────────────────────────────────────────────────────────
evaluation_processor = SKLearnProcessor(
    framework_version="1.2-1",
    role=role,
    instance_type=instance_type,
    instance_count=1,
    base_job_name="churn-pipeline-evaluate",
    sagemaker_session=session,
)

# Property file — stores evaluation metrics as JSON for conditional step
evaluation_report = PropertyFile(
    name="EvaluationReport",
    output_name="evaluation",
    path="evaluation.json"
)

step_evaluate = ProcessingStep(
    name="ChurnEvaluate",
    processor=evaluation_processor,
    code="processing/evaluate.py",
    inputs=[
        ProcessingInput(
            source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/model"
        ),
        ProcessingInput(
            source=f"s3://{bucket}/pipeline/data/test",
            destination="/opt/ml/processing/test"
        ),
    ],
    outputs=[
        ProcessingOutput(
            output_name="evaluation",
            source="/opt/ml/processing/evaluation",
            destination=f"s3://{bucket}/pipeline/evaluation"
        )
    ],
    property_files=[evaluation_report],
    cache_config=cache_config,
    depends_on=[step_train],
)

# ── Step 4: Register Model ──────────────────────────────────────────────────
step_register = RegisterModel(
    name="ChurnRegisterModel",
    estimator=estimator,
    model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
    content_types=["text/csv"],
    response_types=["text/csv", "application/json"],
    inference_instances=["ml.m5.large"],
    transform_instances=["ml.m5.large"],
    model_package_group_name="churn-prediction-models",
    approval_status=model_approval,
    model_metrics=sagemaker.model_metrics.ModelMetrics(
        model_statistics=sagemaker.model_metrics.MetricsSource(
            s3_uri=f"s3://{bucket}/pipeline/evaluation/evaluation.json",
            content_type="application/json"
        )
    ),
)

# ── Step 5: Conditional — only register if roc_auc >= threshold ─────────────
condition_gte = ConditionGreaterThanOrEqualTo(
    left=JsonGet(
        step_name=step_evaluate.name,
        property_file=evaluation_report,
        json_path="binary_classification_metrics.auc.value"
    ),
    right=roc_auc_threshold,
)

step_condition = ConditionStep(
    name="CheckRocAuc",
    conditions=[condition_gte],
    if_steps=[step_register],   # register if roc_auc >= 0.83
    else_steps=[],              # stop pipeline if below threshold
)

# ── Pipeline Definition ─────────────────────────────────────────────────────
pipeline = Pipeline(
    name="churn-prediction-pipeline",
    parameters=[roc_auc_threshold, instance_type, model_approval],
    steps=[step_process, step_train, step_evaluate, step_condition],
    sagemaker_session=session,
)

if __name__ == "__main__":
    # Upsert pipeline — creates or updates existing definition
    pipeline.upsert(role_arn=role)
    print("Pipeline upserted: churn-prediction-pipeline")

    # Start execution with default parameters
    execution = pipeline.start(
        parameters={
            "RocAucThreshold":    0.83,
            "InstanceType":       "ml.m5.large",
            "ModelApprovalStatus": "PendingManualApproval",
        }
    )
    print(f"Pipeline execution started: {execution.arn}")
    print("Monitor: SageMaker Studio → Pipelines → churn-prediction-pipeline")
