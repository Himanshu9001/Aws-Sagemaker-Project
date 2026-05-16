import boto3
import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.session import Session
from sagemaker.inputs import TrainingInput

session    = Session()
role       = "arn:aws:iam::011528270076:role/sagemaker-churn-execution-role"
bucket     = "churn-sagemaker-artifacts"
ecr        = "011528270076.dkr.ecr.us-east-1.amazonaws.com/churn-mlops:latest"
region     = "us-east-1"

# Upload finetune.py directly to S3 as the entry point
s3 = boto3.client("s3", region_name=region)
s3.upload_file(
    "/home/sagemaker-user/Aws-Sagemaker-Project/llm/finetuning/finetune.py",
    bucket,
    "llm/code/finetune.py"
)
print("Uploaded finetune.py to S3")

estimator = Estimator(
    image_uri=ecr,
    role=role,
    instance_type="ml.m5.2xlarge",
    instance_count=1,
    sagemaker_session=session,
    base_job_name="flan-t5-finetune",
    use_spot_instances=True,
    max_run=7200,
    max_wait=14400,
    hyperparameters={
        "epochs":        5,
        "batch_size":    4,
        "learning_rate": "3e-4",
    },
    output_path=f"s3://{bucket}/llm/finetuned-model/",
    environment={
        "SAGEMAKER_PROGRAM":     "finetune.py",
        "SAGEMAKER_SUBMIT_DIRECTORY": f"s3://{bucket}/llm/code/",
    },
)

inputs = {
    "train": TrainingInput(
        s3_data=f"s3://{bucket}/llm/finetuning/",
        content_type="application/json"
    )
}

print("Submitting Flan-T5 fine-tuning job...")
print("Instance: ml.m5.2xlarge (CPU) | Epochs: 5 | Dataset: 40 examples")
estimator.fit(inputs, wait=True, logs=True)
print(f"\nFine-tuning complete!")
print(f"Model: {estimator.model_data}")
