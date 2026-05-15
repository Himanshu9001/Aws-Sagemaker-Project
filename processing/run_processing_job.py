# Submits preprocess.py as a SageMaker Processing Job
# Compatible with sagemaker SDK v2.257.3
# Reads churn.csv from S3, writes train.csv + test.csv back to S3

import sagemaker
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput

# Session + role
session = sagemaker.Session()
role    = "arn:aws:iam::011528270076:role/sagemaker-churn-execution-role"
bucket  = "churn-sagemaker-artifacts"

# SKLearnProcessor — managed scikit-learn container, no custom Docker needed
processor = SKLearnProcessor(
    framework_version="1.2-1",
    role=role,
    instance_type="ml.m5.large",
    instance_count=1,
    base_job_name="churn-preprocess",
    sagemaker_session=session,
)

# Submit Processing Job
processor.run(
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
            destination=f"s3://{bucket}/data/processed/train"
        ),
        ProcessingOutput(
            output_name="test",
            source="/opt/ml/processing/output/test",
            destination=f"s3://{bucket}/data/processed/test"
        ),
    ],
    wait=True,
    logs=True,
)

print("Processing job complete!")
print(f"Train: s3://{bucket}/data/processed/train/train.csv")
print(f"Test:  s3://{bucket}/data/processed/test/test.csv")
