# SageMaker Studio Setup Guide

Complete step-by-step guide to recreate the AWS infrastructure for this project from scratch.

---

## Prerequisites

- AWS account with admin access
- AWS CLI installed and configured
- Python 3.9+
- Docker (for ECR image builds)
- Mac or Linux terminal

---

## Step 1 — AWS CLI Configuration

```bash
# Configure AWS CLI with your credentials
aws configure
# AWS Access Key ID: <your-access-key>
# AWS Secret Access Key: <your-secret-key>
# Default region: us-east-1
# Default output format: json

# Verify
aws sts get-caller-identity
```

Expected output:
```json
{
    "UserId": "AIDAXXXXXXXXXXXXXXXXX",
    "Account": "011528270076",
    "Arn": "arn:aws:iam::011528270076:user/Himanshu_demo_01"
}
```

---

## Step 2 — Create Default VPC (if missing)

```bash
# Check if default VPC exists
aws ec2 describe-vpcs \
  --filters "Name=isDefault,Values=true" \
  --query "Vpcs[0].VpcId" \
  --output text \
  --region us-east-1

# If output is "None" — create default VPC
aws ec2 create-default-vpc --region us-east-1

# Get subnet IDs (needed for Studio domain)
aws ec2 describe-subnets \
  --filters "Name=defaultForAz,Values=true" \
  --query "Subnets[*].{ID:SubnetId,AZ:AvailabilityZone}" \
  --output table \
  --region us-east-1
```

---

## Step 3 — Create S3 Bucket

```bash
# Create bucket
aws s3api create-bucket \
  --bucket churn-sagemaker-artifacts \
  --region us-east-1

# Enable versioning (protects model artifacts)
aws s3api put-bucket-versioning \
  --bucket churn-sagemaker-artifacts \
  --versioning-configuration Status=Enabled

# Block public access
aws s3api put-public-access-block \
  --bucket churn-sagemaker-artifacts \
  --public-access-block-configuration \
    BlockPublicAcls=true,IgnorePublicAcls=true,\
BlockPublicPolicy=true,RestrictPublicBuckets=true

echo "S3 bucket created: churn-sagemaker-artifacts"
```

---

## Step 4 — Create IAM Execution Role

```bash
# Create trust policy
cat > /tmp/trust-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": [
          "sagemaker.amazonaws.com",
          "lambda.amazonaws.com"
        ]
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

# Create role
aws iam create-role \
  --role-name sagemaker-churn-execution-role \
  --assume-role-policy-document file:///tmp/trust-policy.json

# Attach managed policies
aws iam attach-role-policy \
  --role-name sagemaker-churn-execution-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

aws iam attach-role-policy \
  --role-name sagemaker-churn-execution-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

aws iam attach-role-policy \
  --role-name sagemaker-churn-execution-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryFullAccess

aws iam attach-role-policy \
  --role-name sagemaker-churn-execution-role \
  --policy-arn arn:aws:iam::aws:policy/CloudWatchFullAccess

aws iam attach-role-policy \
  --role-name sagemaker-churn-execution-role \
  --policy-arn arn:aws:iam::aws:policy/AWSLambda_FullAccess

# Get role ARN
aws iam get-role \
  --role-name sagemaker-churn-execution-role \
  --query 'Role.Arn' \
  --output text

echo "IAM role created: sagemaker-churn-execution-role"
```

---

## Step 5 — Create VPC Endpoints

```bash
# Get VPC ID
VPC_ID=$(aws ec2 describe-vpcs \
  --filters "Name=isDefault,Values=true" \
  --query "Vpcs[0].VpcId" \
  --output text \
  --region us-east-1)

# Get route table ID
RT_ID=$(aws ec2 describe-route-tables \
  --filters "Name=vpc-id,Values=$VPC_ID" "Name=association.main,Values=true" \
  --query "RouteTables[0].RouteTableId" \
  --output text \
  --region us-east-1)

# Get subnet IDs
SUBNET_IDS=$(aws ec2 describe-subnets \
  --filters "Name=vpc-id,Values=$VPC_ID" \
  --query "Subnets[*].SubnetId" \
  --output text \
  --region us-east-1)

echo "VPC: $VPC_ID"
echo "Route Table: $RT_ID"
echo "Subnets: $SUBNET_IDS"

# 1. S3 Gateway endpoint (free)
aws ec2 create-vpc-endpoint \
  --vpc-id $VPC_ID \
  --service-name com.amazonaws.us-east-1.s3 \
  --vpc-endpoint-type Gateway \
  --route-table-ids $RT_ID \
  --region us-east-1

# Get default security group
SG_ID=$(aws ec2 describe-security-groups \
  --filters "Name=vpc-id,Values=$VPC_ID" "Name=group-name,Values=default" \
  --query "SecurityGroups[0].GroupId" \
  --output text \
  --region us-east-1)

# 2. SageMaker API Interface endpoint
aws ec2 create-vpc-endpoint \
  --vpc-id $VPC_ID \
  --service-name com.amazonaws.us-east-1.sagemaker.api \
  --vpc-endpoint-type Interface \
  --subnet-ids $(echo $SUBNET_IDS | tr ' ' '\n' | head -1) \
  --security-group-ids $SG_ID \
  --private-dns-enabled \
  --region us-east-1

# 3. SageMaker Runtime Interface endpoint
aws ec2 create-vpc-endpoint \
  --vpc-id $VPC_ID \
  --service-name com.amazonaws.us-east-1.sagemaker.runtime \
  --vpc-endpoint-type Interface \
  --subnet-ids $(echo $SUBNET_IDS | tr ' ' '\n' | head -1) \
  --security-group-ids $SG_ID \
  --private-dns-enabled \
  --region us-east-1

echo "VPC endpoints created"
```

---

## Step 6 — Create ECR Repository

```bash
# Create repository
aws ecr create-repository \
  --repository-name churn-mlops \
  --region us-east-1

# Get repository URI
ECR_URI=$(aws ecr describe-repositories \
  --repository-names churn-mlops \
  --region us-east-1 \
  --query 'repositories[0].repositoryUri' \
  --output text)

echo "ECR repository: $ECR_URI"

# Build and push Docker image
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  683313688378.dkr.ecr.us-east-1.amazonaws.com

aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  $ECR_URI

docker build \
  --platform linux/amd64 \
  -t churn-mlops:latest \
  -t $ECR_URI:latest \
  -f Dockerfile .

docker push $ECR_URI:latest

echo "ECR image pushed: $ECR_URI:latest"
```

---

## Step 7 — Create KMS Key

```bash
# Create customer-managed key
KEY_ID=$(aws kms create-key \
  --description "churn-mlops-encryption-key" \
  --region us-east-1 \
  --query 'KeyMetadata.KeyId' \
  --output text)

# Create alias
aws kms create-alias \
  --alias-name alias/churn-mlops-key \
  --target-key-id $KEY_ID \
  --region us-east-1

# Get full ARN
KEY_ARN=$(aws kms describe-key \
  --key-id $KEY_ID \
  --region us-east-1 \
  --query 'KeyMetadata.Arn' \
  --output text)

echo "KMS Key ARN: $KEY_ARN"

# Add key policy for SageMaker role
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

aws kms put-key-policy \
  --key-id $KEY_ID \
  --policy-name default \
  --region us-east-1 \
  --policy "{
    \"Version\": \"2012-10-17\",
    \"Statement\": [
      {
        \"Sid\": \"Enable IAM User Permissions\",
        \"Effect\": \"Allow\",
        \"Principal\": {\"AWS\": \"arn:aws:iam::${ACCOUNT_ID}:root\"},
        \"Action\": \"kms:*\",
        \"Resource\": \"*\"
      },
      {
        \"Sid\": \"Allow SageMaker Execution Role\",
        \"Effect\": \"Allow\",
        \"Principal\": {\"AWS\": \"arn:aws:iam::${ACCOUNT_ID}:role/sagemaker-churn-execution-role\"},
        \"Action\": [\"kms:Encrypt\",\"kms:Decrypt\",\"kms:ReEncrypt*\",\"kms:GenerateDataKey*\",\"kms:DescribeKey\"],
        \"Resource\": \"*\"
      }
    ]
  }"

# Enable S3 bucket encryption
aws s3api put-bucket-encryption \
  --bucket churn-sagemaker-artifacts \
  --server-side-encryption-configuration "{
    \"Rules\": [{
      \"ApplyServerSideEncryptionByDefault\": {
        \"SSEAlgorithm\": \"aws:kms\",
        \"KMSMasterKeyID\": \"$KEY_ARN\"
      },
      \"BucketKeyEnabled\": true
    }]
  }"

echo "KMS encryption enabled"
```

---

## Step 8 — Create SageMaker Studio Domain

```bash
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/sagemaker-churn-execution-role"

VPC_ID=$(aws ec2 describe-vpcs \
  --filters "Name=isDefault,Values=true" \
  --query "Vpcs[0].VpcId" \
  --output text --region us-east-1)

SUBNET_ID=$(aws ec2 describe-subnets \
  --filters "Name=vpc-id,Values=$VPC_ID" \
  --query "Subnets[0].SubnetId" \
  --output text --region us-east-1)

# Create domain
DOMAIN_ID=$(aws sagemaker create-domain \
  --domain-name churn-mlops-domain \
  --auth-mode IAM \
  --default-user-settings "{
    \"ExecutionRole\": \"$ROLE_ARN\"
  }" \
  --vpc-id $VPC_ID \
  --subnet-ids $SUBNET_ID \
  --app-network-access-type PublicInternetOnly \
  --region us-east-1 \
  --query 'DomainArn' \
  --output text | cut -d'/' -f2)

echo "Domain creating: $DOMAIN_ID"
echo "Wait 5-10 minutes for domain to be InService..."

# Wait for domain to be ready
aws sagemaker wait endpoint-in-service \
  --endpoint-name dummy 2>/dev/null || true

# Check status
aws sagemaker describe-domain \
  --domain-id $DOMAIN_ID \
  --region us-east-1 \
  --query '{Status:Status,DomainId:DomainId}'
```

---

## Step 9 — Create User Profile

```bash
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/sagemaker-churn-execution-role"

# Replace with your domain ID
DOMAIN_ID="d-7ag46xxgf2lj"

aws sagemaker create-user-profile \
  --domain-id $DOMAIN_ID \
  --user-profile-name heman-dev \
  --user-settings "{
    \"ExecutionRole\": \"$ROLE_ARN\"
  }" \
  --region us-east-1

echo "User profile created: heman-dev"
```

---

## Step 10 — Configure Studio Environment

Open Studio:
1. Go to **AWS Console → SageMaker → Studio**
2. Click **Open Studio** next to your domain
3. Click **Code Editor** → **Create space** → name it `churn-dev`
4. Select `ml.t3.medium` instance
5. Click **Run space**

Once inside Studio Code Editor:

```bash
# Clone the repository
git clone https://github.com/Himanshu9001/Aws-Sagemaker-Project.git

# Create virtualenv with SageMaker SDK v2
python -m venv ~/sagemaker-env
source ~/sagemaker-env/bin/activate
pip install "sagemaker>=2.200,<3.0" boto3 pytz --quiet

# Configure git
git config --global user.email "your-email@gmail.com"
git config --global user.name "YourUsername"
git config --global --unset core.askpass

# Verify setup
python3 -c "import sagemaker; print(sagemaker.__version__)"
aws sts get-caller-identity
```

---

## Step 11 — Upload Dataset

```bash
# Download Telco Customer Churn dataset
# Source: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

# Upload to S3
aws s3 cp WA_Fn-UseC_-Telco-Customer-Churn.csv \
  s3://churn-sagemaker-artifacts/data/raw/churn.csv

echo "Dataset uploaded"
```

---

## Step 12 — Set Up GitHub Actions Secrets

Go to **GitHub → your repo → Settings → Secrets → Actions → New secret**

| Secret | Value |
|--------|-------|
| `AWS_ACCESS_KEY_ID` | Your IAM user access key |
| `AWS_SECRET_ACCESS_KEY` | Your IAM user secret key |
| `AWS_REGION` | `us-east-1` |

---

## Step 13 — CloudTrail Setup

```bash
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Apply CloudTrail bucket policy first
aws s3api put-bucket-policy \
  --bucket churn-sagemaker-artifacts \
  --policy "{
    \"Version\": \"2012-10-17\",
    \"Statement\": [
      {
        \"Sid\": \"AllowCloudTrailACLCheck\",
        \"Effect\": \"Allow\",
        \"Principal\": {\"Service\": \"cloudtrail.amazonaws.com\"},
        \"Action\": \"s3:GetBucketAcl\",
        \"Resource\": \"arn:aws:s3:::churn-sagemaker-artifacts\"
      },
      {
        \"Sid\": \"AllowCloudTrailWrite\",
        \"Effect\": \"Allow\",
        \"Principal\": {\"Service\": \"cloudtrail.amazonaws.com\"},
        \"Action\": \"s3:PutObject\",
        \"Resource\": \"arn:aws:s3:::churn-sagemaker-artifacts/cloudtrail-logs/AWSLogs/${ACCOUNT_ID}/*\",
        \"Condition\": {
          \"StringEquals\": {\"s3:x-amz-acl\": \"bucket-owner-full-control\"}
        }
      }
    ]
  }"

# Create trail
aws cloudtrail create-trail \
  --name churn-mlops-audit-trail \
  --s3-bucket-name churn-sagemaker-artifacts \
  --s3-key-prefix cloudtrail-logs \
  --region us-east-1

# Start logging
aws cloudtrail start-logging \
  --name churn-mlops-audit-trail \
  --region us-east-1

echo "CloudTrail configured"
```

---

## Step 14 — Run the Full Pipeline

```bash
# From Studio terminal
source ~/sagemaker-env/bin/activate
cd ~/Aws-Sagemaker-Project

# Option A — Run full SageMaker Pipeline (recommended)
python pipelines/churn_pipeline.py

# Option B — Run individual steps
make process    # Processing Job
make train      # Training Job
make hpo        # Hyperparameter Tuning
make deploy     # Deploy endpoint
make monitor    # Set up monitoring
make clarify    # Run SHAP analysis
```

---

## Infrastructure Reference

| Resource | Name | Notes |
|----------|------|-------|
| S3 Bucket | `churn-sagemaker-artifacts` | KMS encrypted |
| IAM Role | `sagemaker-churn-execution-role` | Used by all jobs |
| ECR Repo | `churn-mlops` | Custom training image |
| KMS Key | `alias/churn-mlops-key` | S3 + EBS encryption |
| Studio Domain | `churn-mlops-domain` | Public internet access |
| User Profile | `heman-dev` | Code Editor space |
| VPC | Default VPC | us-east-1 |
| VPC Endpoints | S3 Gateway + SM API + SM Runtime | Data stays in VPC |
| CloudTrail | `churn-mlops-audit-trail` | All API calls logged |

---

## Environment Variables Reference

Copy `.env.example` and fill in your values:

```bash
cp .env.example .env
```

```env
# AWS
AWS_REGION=us-east-1
AWS_ACCOUNT_ID=<your-account-id>

# SageMaker
SAGEMAKER_ROLE_ARN=arn:aws:iam::<account>:role/sagemaker-churn-execution-role
S3_BUCKET=churn-sagemaker-artifacts
STUDIO_DOMAIN_ID=<your-domain-id>

# ECR
ECR_IMAGE=<account>.dkr.ecr.us-east-1.amazonaws.com/churn-mlops:latest

# KMS
KMS_KEY_ARN=arn:aws:kms:us-east-1:<account>:key/<key-id>
KMS_ALIAS=alias/churn-mlops-key

# Endpoints
SERVERLESS_ENDPOINT=churn-prediction-serverless

# LLMOps
MLFLOW_TRACKING_URI=https://<mlflow-app>.mlflow.sagemaker.us-east-1.app.aws/
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

---

## Estimated Setup Time

| Step | Time |
|------|------|
| AWS CLI + IAM setup | 10 min |
| VPC + S3 + ECR | 10 min |
| KMS + CloudTrail | 10 min |
| Studio domain creation | 10 min |
| Docker build + ECR push | 15 min |
| Studio environment setup | 10 min |
| **Total** | **~65 minutes** |

---

## Cost to Recreate

| Phase | Cost |
|-------|------|
| Infrastructure setup | $0 |
| Processing Job | ~$0.001 |
| Training Job (Spot) | ~$0.001 |
| HPO (20 trials, Spot) | ~$0.02 |
| Pipeline run | ~$0.005 |
| Clarify job | ~$0.006 |
| Serverless endpoint (1 month) | ~$0.01 |
| KMS key (1 month) | $1.00 |
| **Total to fully recreate** | **~$1.05** |

---

*Setup guide for [Aws-Sagemaker-Project](https://github.com/Himanshu9001/Aws-Sagemaker-Project)*
*Last updated: May 2026*