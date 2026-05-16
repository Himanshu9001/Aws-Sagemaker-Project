# Lambda function — auto-approves SageMaker model if ROC AUC >= threshold
# Triggered by EventBridge when a new model package is created
# Replaces manual CLI approval from Phase 5

import boto3
import json
import os

sm_client = boto3.client("sagemaker", region_name="us-east-1")

ROC_AUC_THRESHOLD = float(os.environ.get("ROC_AUC_THRESHOLD", "0.83"))

def lambda_handler(event, context):
    print(f"Event received: {json.dumps(event)}")

    # Extract model package ARN from EventBridge event
    detail = event.get("detail", {})
    model_package_arn = detail.get("ModelPackageArn")

    if not model_package_arn:
        print("No ModelPackageArn found in event")
        return {"statusCode": 400, "body": "No ModelPackageArn"}

    print(f"Checking model: {model_package_arn}")

    # Get model package details
    response = sm_client.describe_model_package(
        ModelPackageName=model_package_arn
    )

    # Extract ROC AUC from customer metadata
    metadata = response.get("CustomerMetadataProperties", {})
    roc_auc = float(metadata.get("roc_auc", 0))
    approval_status = response.get("ModelApprovalStatus")

    print(f"ROC AUC: {roc_auc}, Threshold: {ROC_AUC_THRESHOLD}")
    print(f"Current status: {approval_status}")

    # Only process pending models
    if approval_status != "PendingManualApproval":
        print(f"Model already processed: {approval_status}")
        return {"statusCode": 200, "body": "Already processed"}

    # Auto-approve if ROC AUC meets threshold
    if roc_auc >= ROC_AUC_THRESHOLD:
        sm_client.update_model_package(
            ModelPackageName=model_package_arn,
            ModelApprovalStatus="Approved",
            ApprovalDescription=f"Auto-approved: ROC AUC {roc_auc:.4f} >= threshold {ROC_AUC_THRESHOLD}"
        )
        print(f"Model APPROVED: ROC AUC {roc_auc:.4f}")
        return {
            "statusCode": 200,
            "body": f"Approved — ROC AUC {roc_auc:.4f}"
        }
    else:
        sm_client.update_model_package(
            ModelPackageName=model_package_arn,
            ModelApprovalStatus="Rejected",
            ApprovalDescription=f"Auto-rejected: ROC AUC {roc_auc:.4f} < threshold {ROC_AUC_THRESHOLD}"
        )
        print(f"Model REJECTED: ROC AUC {roc_auc:.4f}")
        return {
            "statusCode": 200,
            "body": f"Rejected — ROC AUC {roc_auc:.4f}"
        }
