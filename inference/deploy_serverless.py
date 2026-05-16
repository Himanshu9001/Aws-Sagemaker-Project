# Serverless Inference endpoint — cost-efficient for low-traffic portfolio project
# Charges per request instead of per hour
# Cold start: ~2-3 seconds | Warm: ~50ms | Cost: ~$0.000016/GB-second

import boto3
import sagemaker
from sagemaker.session import Session
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.serverless import ServerlessInferenceConfig
import time

session   = Session()
role      = "arn:aws:iam::011528270076:role/sagemaker-churn-execution-role"
bucket    = "churn-sagemaker-artifacts"
best_job  = "churn-hpo-260516-0610-013-7f95b1dd"
model_artifacts = f"s3://{bucket}/models/hpo/{best_job}/output/model.tar.gz"
endpoint_name   = "churn-prediction-serverless"

print(f"Deploying serverless endpoint...")
print(f"Model: {model_artifacts}")

# SKLearnModel with inference.py
model = SKLearnModel(
    model_data=model_artifacts,
    role=role,
    entry_point="inference.py",
    source_dir="inference/",
    framework_version="1.2-1",
    sagemaker_session=session,
)

# Serverless config — 2GB memory, max 5 concurrent requests
serverless_config = ServerlessInferenceConfig(
    memory_size_in_mb=2048,   # 2GB — sufficient for RandomForest with 291 trees
    max_concurrency=5,         # max simultaneous requests
)

# Deploy serverless endpoint
predictor = model.deploy(
    serverless_inference_config=serverless_config,
    endpoint_name=endpoint_name,
)

print(f"Serverless endpoint deployed: {endpoint_name}")
print(f"Memory: 2048 MB | Max concurrency: 5")
print(f"Cost: ~$0.000016/GB-second (only charged when invoked)")
print(f"Cold start: ~2-3 seconds | Warm latency: ~50ms")
