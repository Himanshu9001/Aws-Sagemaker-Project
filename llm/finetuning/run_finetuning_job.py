# Fine-tuning job using custom ECR image (has transformers installed)
# Bypasses HuggingFace estimator GPU requirement
# Uses our existing churn-mlops ECR image which has all dependencies

import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.session import Session

session = Session()
role    = "arn:aws:iam::011528270076:role/sagemaker-churn-execution-role"
bucket  = "churn-sagemaker-artifacts"
ecr     = "011528270270076.dkr.ecr.us-east-1.amazonaws.com/churn-mlops:latest"

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
    environment={"SAGEMAKER_PROGRAM": "finetune.py"},
)

inputs = {"train": f"s3://{bucket}/llm/finetuning/"}

print("Submitting Flan-T5 fine-tuning job via custom ECR image...")
print("Instance: ml.m5.2xlarge (CPU) | Epochs: 5 | Dataset: 40 examples")
estimator.fit(inputs, wait=True, logs=True)

print(f"Fine-tuning complete!")
print(f"Model: {estimator.model_data}")
