# A/B Testing Endpoint — two model variants with weighted traffic split
# Variant A: champion (80% traffic) — current production model
# Variant B: challenger (20% traffic) — newly approved model
# Compare latency, error rate, and business metrics in CloudWatch

import boto3
import sagemaker
from sagemaker.session import Session
from sagemaker.serverless import ServerlessInferenceConfig
from sagemaker.sklearn.model import SKLearnModel
import time

session   = Session()
sm_client = boto3.client("sagemaker", region_name="us-east-1")
role      = "arn:aws:iam::011528270076:role/sagemaker-churn-execution-role"
bucket    = "churn-sagemaker-artifacts"
endpoint_name = "churn-ab-test-endpoint"

# Champion — best HPO model (Phase 4)
champion_artifacts  = f"s3://{bucket}/models/hpo/churn-hpo-260516-0610-013-7f95b1dd/output/model.tar.gz"

# Challenger — latest pipeline model (Phase 7)
challenger_artifacts = f"s3://{bucket}/pipeline/models/pipelines-g0z5cgxbleki-ChurnTrain-OMrHwGlkox/output/model.tar.gz"

print("Creating A/B test models...")

# Create champion model
champion_model = SKLearnModel(
    model_data=champion_artifacts,
    role=role,
    entry_point="inference.py",
    source_dir="inference/",
    framework_version="1.2-1",
    sagemaker_session=session,
    name=f"churn-champion-{int(time.time())}",
)
champion_container = champion_model.prepare_container_def(
    instance_type="ml.m5.large"
)

# Create challenger model
challenger_model = SKLearnModel(
    model_data=challenger_artifacts,
    role=role,
    entry_point="inference.py",
    source_dir="inference/",
    framework_version="1.2-1",
    sagemaker_session=session,
    name=f"churn-challenger-{int(time.time())}",
)
challenger_container = challenger_model.prepare_container_def(
    instance_type="ml.m5.large"
)

# Register both models
champion_name   = f"churn-champion-{int(time.time())}"
challenger_name = f"churn-challenger-{int(time.time()) + 1}"

sm_client.create_model(
    ModelName=champion_name,
    ExecutionRoleArn=role,
    Containers=[champion_container]
)
print(f"Champion model: {champion_name}")

sm_client.create_model(
    ModelName=challenger_name,
    ExecutionRoleArn=role,
    Containers=[challenger_container]
)
print(f"Challenger model: {challenger_name}")

# Create endpoint config with two variants
config_name = f"churn-ab-config-{int(time.time())}"
sm_client.create_endpoint_config(
    EndpointConfigName=config_name,
    ProductionVariants=[
        {
            "VariantName":          "champion",
            "ModelName":            champion_name,
            "InstanceType":         "ml.m5.large",
            "InitialInstanceCount": 1,
            "InitialVariantWeight": 0.8,  # 80% traffic
        },
        {
            "VariantName":          "challenger",
            "ModelName":            challenger_name,
            "InstanceType":         "ml.m5.large",
            "InitialInstanceCount": 1,
            "InitialVariantWeight": 0.2,  # 20% traffic
        }
    ]
)
print(f"Endpoint config: {config_name}")
print(f"Traffic split: champion=80%, challenger=20%")

# Create endpoint
sm_client.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=config_name
)
print(f"Creating A/B endpoint: {endpoint_name}")
print("Waiting for InService (~5 minutes)...")

waiter = sm_client.get_waiter("endpoint_in_service")
waiter.wait(
    EndpointName=endpoint_name,
    WaiterConfig={"Delay": 30, "MaxAttempts": 20}
)
print(f"A/B endpoint InService: {endpoint_name}")
print(f"Champion variant:   80% traffic — {champion_name}")
print(f"Challenger variant: 20% traffic — {challenger_name}")
