# Deploys auto-approval Lambda + EventBridge rule
# Run once from Mac terminal (needs IAM permissions)

import boto3
import json
import zipfile
import os

region     = "us-east-1"
account    = "011528270076"
lambda_client   = boto3.client("lambda", region_name=region)
iam_client      = boto3.client("iam", region_name=region)
events_client   = boto3.client("events", region_name=region)

# ── Step 1: Create Lambda IAM role ──────────────────────────────────────────
trust_policy = {
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "lambda.amazonaws.com"},
        "Action": "sts:AssumeRole"
    }]
}

try:
    role = iam_client.create_role(
        RoleName="churn-auto-approval-lambda-role",
        AssumeRolePolicyDocument=json.dumps(trust_policy),
        Description="Lambda role for SageMaker model auto-approval"
    )
    role_arn = role["Role"]["Arn"]
    print(f"Lambda role created: {role_arn}")
except iam_client.exceptions.EntityAlreadyExistsException:
    role_arn = f"arn:aws:iam::{account}:role/churn-auto-approval-lambda-role"
    print(f"Lambda role exists: {role_arn}")

# Attach policies
for policy in [
    "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
    "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
]:
    try:
        iam_client.attach_role_policy(RoleName="churn-auto-approval-lambda-role", PolicyArn=policy)
    except:
        pass

import time
time.sleep(10)  # Wait for IAM propagation

# ── Step 2: Package and deploy Lambda ───────────────────────────────────────
with zipfile.ZipFile("/tmp/lambda.zip", "w") as zf:
    zf.write("lambda/auto_approve_model.py", "auto_approve_model.py")

with open("/tmp/lambda.zip", "rb") as f:
    zip_bytes = f.read()

try:
    response = lambda_client.create_function(
        FunctionName="churn-auto-approve-model",
        Runtime="python3.11",
        Role=role_arn,
        Handler="auto_approve_model.lambda_handler",
        Code={"ZipFile": zip_bytes},
        Environment={"Variables": {"ROC_AUC_THRESHOLD": "0.83"}},
        Timeout=60,
        Description="Auto-approves SageMaker models with ROC AUC >= 0.83"
    )
    lambda_arn = response["FunctionArn"]
    print(f"Lambda created: {lambda_arn}")
except lambda_client.exceptions.ResourceConflictException:
    response = lambda_client.update_function_code(
        FunctionName="churn-auto-approve-model",
        ZipFile=zip_bytes
    )
    lambda_arn = response["FunctionArn"]
    print(f"Lambda updated: {lambda_arn}")

# ── Step 3: Create EventBridge rule ─────────────────────────────────────────
rule_response = events_client.put_rule(
    Name="churn-model-package-created",
    EventPattern=json.dumps({
        "source": ["aws.sagemaker"],
        "detail-type": ["SageMaker Model Package State Change"],
        "detail": {
            "ModelPackageGroupName": ["churn-prediction-models"],
            "ModelApprovalStatus": ["PendingManualApproval"]
        }
    }),
    State="ENABLED",
    Description="Triggers Lambda when churn model is registered"
)
rule_arn = rule_response["RuleArn"]
print(f"EventBridge rule created: {rule_arn}")

# Add Lambda as target
events_client.put_targets(
    Rule="churn-model-package-created",
    Targets=[{
        "Id": "churn-auto-approve-lambda",
        "Arn": lambda_arn
    }]
)

# Allow EventBridge to invoke Lambda
try:
    lambda_client.add_permission(
        FunctionName="churn-auto-approve-model",
        StatementId="EventBridgeInvoke",
        Action="lambda:InvokeFunction",
        Principal="events.amazonaws.com",
        SourceArn=rule_arn
    )
except lambda_client.exceptions.ResourceConflictException:
    pass

print("\nAuto-approval pipeline complete!")
print(f"EventBridge rule: churn-model-package-created")
print(f"Lambda function:  churn-auto-approve-model")
print(f"Threshold:        ROC AUC >= 0.83")
print(f"Flow: Pipeline → Model registered → EventBridge → Lambda → Auto-approve")
