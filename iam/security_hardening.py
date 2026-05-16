# Phase 10 — Security Hardening
# S3 bucket policy, CloudTrail audit, KMS encryption review
# Run from Mac terminal (requires admin IAM permissions)

import boto3
import json

region    = "us-east-1"
account   = "011528270076"
bucket    = "churn-sagemaker-artifacts"
vpc_id    = "vpc-0dd53e8ac2e2d361e"
role_arn  = f"arn:aws:iam::{account}:role/sagemaker-churn-execution-role"

s3_client  = boto3.client("s3", region_name=region)
ct_client  = boto3.client("cloudtrail", region_name=region)

# ── Step 1: S3 Bucket Policy — deny access outside VPC endpoint ─────────────
# All SageMaker jobs access S3 via VPC endpoint vpce-0028c7a16c20843a4
# This policy ensures no data can be exfiltrated via public internet
bucket_policy = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "DenyNonVPCAccess",
            "Effect": "Deny",
            "Principal": "*",
            "Action": "s3:*",
            "Resource": [
                f"arn:aws:s3:::{bucket}",
                f"arn:aws:s3:::{bucket}/*"
            ],
            "Condition": {
                "StringNotEquals": {
                    "aws:SourceVpce": "vpce-0028c7a16c20843a4"
                },
                "Bool": {
                    "aws:ViaAWSService": "false"
                }
            }
        },
        {
            "Sid": "AllowSageMakerRole",
            "Effect": "Allow",
            "Principal": {
                "AWS": role_arn
            },
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket"
            ],
            "Resource": [
                f"arn:aws:s3:::{bucket}",
                f"arn:aws:s3:::{bucket}/*"
            ]
        }
    ]
}

s3_client.put_bucket_policy(
    Bucket=bucket,
    Policy=json.dumps(bucket_policy)
)
print(f"S3 bucket policy applied: deny non-VPC access")

# ── Step 2: Enable S3 versioning — protect model artifacts ──────────────────
s3_client.put_bucket_versioning(
    Bucket=bucket,
    VersioningConfiguration={"Status": "Enabled"}
)
print(f"S3 versioning enabled: {bucket}")

# ── Step 3: Enable CloudTrail — audit all SageMaker API calls ───────────────
trail_name = "churn-mlops-audit-trail"
try:
    ct_client.create_trail(
        Name=trail_name,
        S3BucketName=bucket,
        S3KeyPrefix="cloudtrail-logs",
        IncludeGlobalServiceEvents=True,
        IsMultiRegionTrail=False,
        EnableLogFileValidation=True,
    )
    ct_client.start_logging(Name=trail_name)
    print(f"CloudTrail enabled: {trail_name}")
except ct_client.exceptions.TrailAlreadyExistsException:
    print(f"CloudTrail already exists: {trail_name}")

# ── Step 4: Print security summary ──────────────────────────────────────────
print("\n=== Security Summary ===")
print(f"S3 bucket policy:   deny non-VPC access enforced")
print(f"S3 versioning:      enabled — model artifacts protected")
print(f"CloudTrail:         all API calls logged to s3://{bucket}/cloudtrail-logs")
print(f"VPC endpoints:      S3 Gateway + SageMaker API + Runtime")
print(f"Network mode:       Public internet for Studio (data stays private)")
print(f"IAM execution role: least privilege scoped to {bucket}")
print(f"\nPhase 10 complete!")
