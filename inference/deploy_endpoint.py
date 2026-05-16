import boto3
import sagemaker
from sagemaker.session import Session
from sagemaker.sklearn.model import SKLearnModel
import time

session   = Session()
sm_client = boto3.client("sagemaker", region_name="us-east-1")
role      = "arn:aws:iam::011528270076:role/sagemaker-churn-execution-role"
bucket    = "churn-sagemaker-artifacts"

# Best model artifacts from HPO Phase 4
best_job        = "churn-hpo-260516-0610-013-7f95b1dd"
model_artifacts = f"s3://{bucket}/models/hpo/{best_job}/output/model.tar.gz"
endpoint_name   = "churn-prediction-endpoint"

print(f"Deploying model: {model_artifacts}")

# SKLearnModel with inference.py — handles model_fn, predict_fn, output_fn
model = SKLearnModel(
    model_data=model_artifacts,
    role=role,
    entry_point="inference.py",       # serves predictions
    source_dir="inference/",          # uploads inference.py to container
    framework_version="1.2-1",
    sagemaker_session=session,
)

# Deploy to real-time endpoint
predictor = model.deploy(
    instance_type="ml.m5.large",
    initial_instance_count=1,
    endpoint_name=endpoint_name,
    wait=True,
)

print(f"Endpoint InService: {endpoint_name}")

# Configure auto-scaling
asg_client  = boto3.client("application-autoscaling", region_name="us-east-1")
resource_id = f"endpoint/{endpoint_name}/variant/AllTraffic"

asg_client.register_scalable_target(
    ServiceNamespace="sagemaker",
    ResourceId=resource_id,
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    MinCapacity=1,
    MaxCapacity=5,
)

asg_client.put_scaling_policy(
    PolicyName="churn-endpoint-scaling",
    ServiceNamespace="sagemaker",
    ResourceId=resource_id,
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    PolicyType="TargetTrackingScaling",
    TargetTrackingScalingPolicyConfiguration={
        "TargetValue": 100.0,
        "PredefinedMetricSpecification": {
            "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance"
        },
        "ScaleInCooldown":  300,
        "ScaleOutCooldown": 60,
    }
)

print(f"Auto-scaling configured: min=1, max=5, target=100 invocations/instance")
print(f"Endpoint ready: {endpoint_name}")
