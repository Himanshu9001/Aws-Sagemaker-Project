# Lambda function — auto-deploys serverless endpoint when model is approved
# Triggered by EventBridge: SageMaker Model Package State Change → Approved
# Completes the full CD loop: code push → approved model → live endpoint

import boto3
import json
import time
import os

sm_client  = boto3.client("sagemaker", region_name="us-east-1")
ROLE       = "arn:aws:iam::011528270076:role/sagemaker-churn-execution-role"
BUCKET     = "churn-sagemaker-artifacts"
ENDPOINT   = "churn-prediction-serverless"

def lambda_handler(event, context):
    print(f"Event: {json.dumps(event)}")

    detail               = event.get("detail", {})
    model_package_arn    = detail.get("ModelPackageArn")
    approval_status      = detail.get("ModelApprovalStatus")

    # Only deploy on approval
    if approval_status != "Approved":
        print(f"Status is {approval_status} — skipping deployment")
        return {"statusCode": 200, "body": "Not approved — skipped"}

    print(f"Deploying approved model: {model_package_arn}")

    # Create new model from approved package
    model_name = f"churn-auto-deploy-{int(time.time())}"
    sm_client.create_model(
        ModelName=model_name,
        ExecutionRoleArn=ROLE,
        Containers=[{"ModelPackageName": model_package_arn}]
    )
    print(f"Model created: {model_name}")

    # Create serverless endpoint config
    config_name = f"churn-serverless-config-{int(time.time())}"
    sm_client.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[{
            "VariantName": "AllTraffic",
            "ModelName":   model_name,
            "ServerlessConfig": {
                "MemorySizeInMB": 2048,
                "MaxConcurrency": 5,
            }
        }]
    )
    print(f"Endpoint config created: {config_name}")

    # Update or create endpoint
    try:
        sm_client.update_endpoint(
            EndpointName=ENDPOINT,
            EndpointConfigName=config_name
        )
        print(f"Endpoint updated: {ENDPOINT}")
    except sm_client.exceptions.ResourceNotFound:
        sm_client.create_endpoint(
            EndpointName=ENDPOINT,
            EndpointConfigName=config_name
        )
        print(f"Endpoint created: {ENDPOINT}")

    return {
        "statusCode": 200,
        "body": f"Deployed {model_package_arn} to {ENDPOINT}"
    }
