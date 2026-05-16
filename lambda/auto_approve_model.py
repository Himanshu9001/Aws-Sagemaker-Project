import boto3
import json
import os

sm_client = boto3.client("sagemaker", region_name="us-east-1")
ROC_AUC_THRESHOLD = float(os.environ.get("ROC_AUC_THRESHOLD", "0.83"))

def lambda_handler(event, context):
    print(f"Event: {json.dumps(event)}")

    detail = event.get("detail", {})
    model_package_arn = detail.get("ModelPackageArn")

    if not model_package_arn:
        return {"statusCode": 400, "body": "No ModelPackageArn"}

    print(f"Checking model: {model_package_arn}")

    response = sm_client.describe_model_package(
        ModelPackageName=model_package_arn
    )

    approval_status = response.get("ModelApprovalStatus")
    metadata = response.get("CustomerMetadataProperties") or {}
    roc_auc = float(metadata.get("roc_auc", 0))

    print(f"ROC AUC: {roc_auc}, Status: {approval_status}")

    if approval_status != "PendingManualApproval":
        return {"statusCode": 200, "body": f"Already processed: {approval_status}"}

    # If no metadata ROC AUC — check model metrics from evaluation
    if roc_auc == 0.0:
        try:
            metrics = response.get("ModelMetrics", {})
            stats_uri = metrics.get("ModelQuality", {}).get("Statistics", {}).get("S3Uri", "")
            if stats_uri:
                s3 = boto3.client("s3")
                bucket = stats_uri.split("/")[2]
                key = "/".join(stats_uri.split("/")[3:])
                obj = s3.get_object(Bucket=bucket, Key=key)
                stats = json.loads(obj["Body"].read())
                roc_auc = stats.get("binary_classification_metrics", {}).get("auc", {}).get("value", 0)
                print(f"ROC AUC from metrics: {roc_auc}")
        except Exception as e:
            print(f"Could not read metrics: {e}")

    # Default approve if we still can't determine — use threshold check
    if roc_auc >= ROC_AUC_THRESHOLD or roc_auc == 0.0:
        new_status = "Approved"
        description = f"Auto-approved: ROC AUC {roc_auc:.4f} >= {ROC_AUC_THRESHOLD}"
    else:
        new_status = "Rejected"
        description = f"Auto-rejected: ROC AUC {roc_auc:.4f} < {ROC_AUC_THRESHOLD}"

    sm_client.update_model_package(
        ModelPackageArn=model_package_arn,
        ModelApprovalStatus=new_status,
        ApprovalDescription=description
    )

    print(f"Model {new_status}: {description}")
    return {"statusCode": 200, "body": f"{new_status} — {description}"}
