# Troubleshooting Guide — SageMaker MLOps Project

This document captures every real issue encountered during the construction of this project, with root causes, fixes, and prevention strategies. These are not hypothetical — every issue below was hit in production during development.

---

## Table of Contents

1. [Git & GitHub Issues](#1-git--github-issues)
2. [SageMaker SDK Version Issues](#2-sagemaker-sdk-version-issues)
3. [Training Job Failures](#3-training-job-failures)
4. [Docker & ECR Issues](#4-docker--ecr-issues)
5. [IAM & Permissions Issues](#5-iam--permissions-issues)
6. [S3 Bucket Policy Lockout](#6-s3-bucket-policy-lockout)
7. [Endpoint Deployment Issues](#7-endpoint-deployment-issues)
8. [SageMaker Pipeline Issues](#8-sagemaker-pipeline-issues)
9. [Networking & VPC Issues](#9-networking--vpc-issues)
10. [MLflow Integration Issues](#10-mlflow-integration-issues)
11. [Dependency Conflicts](#11-dependency-conflicts)
12. [Monitoring Issues](#12-monitoring-issues)
13. [Git Merge Conflicts](#13-git-merge-conflicts)

---

## 1. Git & GitHub Issues

---

### Issue 1.1 — VS Code askpass.sh broken

**Error:**
```
fatal: cannot exec '/private/var/folders/nm/.../Visual Studio Code.app/Contents/Resources/app/extensions/git/dist/askpass.sh': No such file or directory
```

**Root cause:** VS Code was running from a macOS translocated (quarantine) path, not from `/Applications`. The credential helper path stored in git config pointed to a non-existent file.

**Fix:**
```bash
git config --global --unset core.askpass
git config --global credential.helper osxkeychain
```

**Permanent fix for Studio Code Editor** — add to Lifecycle Configuration:
```bash
git config --global --unset core.askpass
git config --global core.askpass ""
git config --global credential.helper store
```

**Prevention:** Always move VS Code to `/Applications` folder. Never run from Downloads or a quarantine path.

---

### Issue 1.2 — GitHub PAT exposed in chat

**What happened:** A Personal Access Token (`github_pat_11BKWM62Q0...`) was accidentally pasted into a chat message while debugging git push failures.

**Immediate action:**
1. Go to GitHub → Settings → Developer Settings → Personal Access Tokens
2. Find the exposed token → **Delete/Revoke immediately**
3. Generate a new PAT

**Prevention:**
- Never paste git remote URLs (they contain tokens) into any chat or log
- After successful push, always clean the token from remote URL:
```bash
git remote set-url origin https://github.com/USERNAME/REPO.git
```
- Use SSH keys instead of HTTPS tokens to eliminate this risk permanently

---

### Issue 1.3 — Fine-grained PAT returns 403

**Error:**
```
remote: Permission to Himanshu9001/Aws-Sagemaker-Project.git denied to Himanshu9001.
fatal: unable to access '...': The requested URL returned error: 403
```

**Root cause:** Fine-grained PATs require explicit `Contents: Read and write` permission to be added under the Repositories tab. Selecting "All repositories" with 0 permissions configured gives a valid token with no actual access.

**Fix:**
- GitHub → Settings → Developer Settings → Fine-grained tokens → Edit token
- Repository access: `Only select repositories` → select your repo
- Permissions → Contents: **Read and write**
- Permissions → Metadata: **Read-only** (auto-selected)

---

### Issue 1.4 — Diverged branches between Studio and Mac

**Error:**
```
hint: You have divergent branches and need to specify how to reconcile them.
fatal: Need to specify how to reconcile divergent branches.
```

**Root cause:** Both Mac terminal and Studio Code Editor were committing to `main` independently, creating diverged histories.

**Fix:**
```bash
git pull --rebase origin main
# If conflicts:
git checkout --theirs <conflicted-file>
git add <conflicted-file>
git rebase --continue
```

**Prevention:** Adopt a strict workflow — Mac for code commits, Studio only for running jobs. Never commit from Studio unless absolutely necessary. Use feature branches.

---

### Issue 1.5 — git stash conflict on pull

**Error:**
```
error: cannot pull with rebase: You have unstaged changes.
error: Please commit or stash them.
```

**Fix:**
```bash
git stash
git pull origin main
git stash pop
# If pop causes conflicts:
git checkout --theirs .
git add .
git stash drop
```

---

## 2. SageMaker SDK Version Issues

---

### Issue 2.1 — SageMaker SDK v3 has no Processing Job support

**Error:**
```
ModuleNotFoundError: No module named 'sagemaker.processing'
ModuleNotFoundError: No module named 'sagemaker.sklearn'
```

**Root cause:** SageMaker SDK v3 (installed in Studio as system package) completely restructured modules. Processing Jobs, SKLearnProcessor, and most classical MLOps classes were removed in favor of GenAI/LLM focused modules.

**v3 module structure:**
```
sagemaker/
├── train/      # LLM fine-tuning (SFT, DPO, RLHF)
├── serve/      # Model serving
├── mlops/      # Feature store, pipelines
└── core/       # Remote functions
```

**Fix:** Install v2 in a virtualenv:
```bash
python -m venv ~/sagemaker-env
source ~/sagemaker-env/bin/activate
pip install "sagemaker>=2.200,<3.0"
```

**Prevention:** Always check SDK version before starting:
```bash
python -c "import sagemaker; print(sagemaker.__version__)"
```

---

### Issue 2.2 — `pytz` missing in v2 virtualenv

**Error:**
```
ModuleNotFoundError: No module named 'pytz'
```

**Root cause:** `pytz` is a transitive dependency of sagemaker v2 that wasn't installed when using `pip install sagemaker` with some pip versions.

**Fix:**
```bash
pip install pytz --quiet
```

---

### Issue 2.3 — `sagemaker.experiments.run` not in container

**Error:**
```
ModuleNotFoundError: No module named 'sagemaker'
```

**Root cause:** The SKLearn Training container (Python 3.9, miniconda) does not have the SageMaker Python SDK installed. Only `sagemaker_containers` (the toolkit) is present, not the full SDK.

**Fix:** Remove all SageMaker SDK imports from `train.py`. Use stdout metric logging instead:
```python
# Instead of run.log_metric("roc_auc", value)
print(f"roc_auc={value:.4f}")  # captured by CloudWatch regex
```

---

## 3. Training Job Failures

---

### Issue 3.1 — `No module named 'matplotlib'`

**Error:**
```
ModuleNotFoundError: No module named 'matplotlib'
```

**Root cause:** The `requirements.txt` file was not included in the `sourcedir.tar.gz` uploaded to S3. This happened because the first attempt used `entry_point="training/train.py"` without `source_dir`, so only the entry point file was packaged.

**Fix:** Use `source_dir="training/"` to package the entire directory:
```python
estimator = SKLearn(
    entry_point="train.py",
    source_dir="training/",  # packages ALL files including requirements.txt
    ...
)
```

**Verify the tar contains requirements.txt:**
```bash
aws s3 cp s3://bucket/job-name/source/sourcedir.tar.gz /tmp/
tar -tzf /tmp/sourcedir.tar.gz
# Should show: train.py, requirements.txt, run_training_job.py
```

---

### Issue 3.2 — `seaborn barplot` legacy keyword argument

**Error:**
```
AttributeError: Rectangle.set() got an unexpected keyword argument 'legend'
```

**Root cause:** The `hue` + `legend=False` pattern in `sns.barplot()` was introduced in seaborn 0.12+ but the syntax changed in later versions. The container's seaborn version didn't support the `legend` parameter directly.

**Fix:**
```python
# Wrong
sns.barplot(data=df, x="importance", y="feature", hue="feature", legend=False)

# Correct
sns.barplot(data=df, x="importance", y="feature", palette="viridis")
```

---

### Issue 3.3 — Training Job fails immediately with `ExitCode 1`

**Diagnosis approach:**
```bash
# Get log stream name
aws logs describe-log-streams \
  --log-group-name /aws/sagemaker/TrainingJobs \
  --log-stream-name-prefix "JOB-NAME" \
  --region us-east-1 \
  --query 'logStreams[*].logStreamName' \
  --output text

# Get actual error
aws logs get-log-events \
  --log-group-name /aws/sagemaker/TrainingJobs \
  --log-stream-name "JOB-NAME/algo-1-TIMESTAMP" \
  --region us-east-1 \
  --query 'events[*].message' \
  --output text | tail -30
```

**Common causes and fixes:**

| Error | Cause | Fix |
|-------|-------|-----|
| `No module named 'X'` | Missing from requirements.txt or ECR image | Add to requirements.txt or Dockerfile |
| `ExitCode 1, no message` | Script syntax error | Test locally first |
| `AttributeError` | API version mismatch | Pin library versions |
| `FileNotFoundError` | Wrong input path | Check `/opt/ml/input/data/<channel>/` |

---

### Issue 3.4 — Spot instance savings vary significantly

**Observed:** Spot savings ranged from 48.9% to 63.3% across runs.

**Root cause:** AWS Spot pricing fluctuates based on EC2 capacity in the region and AZ. `us-east-1` has high capacity but savings are not guaranteed.

**Mitigation:**
```python
estimator = SKLearn(
    use_spot_instances=True,
    max_run=3600,    # 1 hour max
    max_wait=7200,   # 2 hour wait for spot capacity
    ...
)
```

---

## 4. Docker & ECR Issues

---

### Issue 4.1 — Platform mismatch warning (ARM vs AMD64)

**Warning:**
```
InvalidBaseImagePlatform: Base image was pulled with platform "linux/amd64", 
expected "linux/arm64" for current build
```

**Root cause:** Mac M1/M2/M3 chips use ARM64 architecture but SageMaker runs on AMD64 (x86_64). Building without `--platform` flag creates an ARM image that fails on SageMaker.

**Fix:**
```bash
docker build --platform linux/amd64 -t your-image:latest .
```

**Prevention:** Always specify `--platform linux/amd64` when building SageMaker images on Apple Silicon Macs.

---

### Issue 4.2 — Docker build blocked in Studio

**Error:**
```
Error response from daemon: {"message":"Forbidden. Reason: [ImageBuild] 
'sagemaker' is the only user allowed network input"}
```

**Root cause:** SageMaker Studio with Docker enabled still restricts network access during builds to the `sagemaker` system user. Regular user (`sagemaker-user`) cannot pull base images from ECR during build.

**Fix:** Build locally on Mac instead of in Studio:
```bash
# On Mac terminal
docker build --platform linux/amd64 -t image:latest .
docker push account.dkr.ecr.us-east-1.amazonaws.com/repo:latest
```

**Alternative:** Use AWS CodeBuild for automated image builds in the cloud.

---

### Issue 4.3 — protobuf version conflict with mlflow

**Error:**
```
TypeError: Descriptors cannot be created directly.
If this call came from a _pb2.py file, your generated code is out of date 
and must be regenerated with protoc >= 3.19.0.
```

**Root cause:** `mlflow>=2.0.0` pulls `protobuf>=4.x` but `sagemaker_containers` requires `protobuf<=3.20.x`. The newer protobuf breaks the older sagemaker_containers.

**Fix:** Pin protobuf in Dockerfile:
```dockerfile
RUN pip install --no-cache-dir \
    mlflow>=2.0.0 \
    protobuf==3.20.3 \   # Last compatible version
    ...
```

---

### Issue 4.4 — ECR authentication expires

**Error:**
```
no basic auth credentials
```

**Root cause:** ECR login tokens expire after 12 hours.

**Fix:** Re-authenticate before pushing:
```bash
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  ACCOUNT.dkr.ecr.us-east-1.amazonaws.com
```

---

## 5. IAM & Permissions Issues

---

### Issue 5.1 — SageMaker execution role cannot perform IAM operations

**Error:**
```
AccessDenied: User arn:aws:sts::011528270076:assumed-role/sagemaker-churn-execution-role/SageMaker 
is not authorized to perform: iam:CreateRole
```

**Root cause:** SageMaker execution role (used by Studio) doesn't have IAM admin permissions — by design. Running IAM operations from Studio always fails.

**Fix:** Run all IAM operations from Mac terminal using your personal IAM user:
```bash
# On Mac terminal (not Studio)
python3 lambda/deploy_lambda.py
aws iam put-role-policy ...
```

**Rule:** IAM management = Mac terminal. ML workloads = Studio.

---

### Issue 5.2 — CloudWatch PutDashboard denied

**Error:**
```
AccessDenied: User is not authorized to perform: cloudwatch:PutDashboard
```

**Root cause:** SageMaker execution role didn't have CloudWatch dashboard permissions.

**Fix:** Add inline policy from Mac terminal (not Studio):
```bash
aws iam put-role-policy \
  --role-name sagemaker-churn-execution-role \
  --policy-name cloudwatch-dashboard-policy \
  --policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Action": ["cloudwatch:PutDashboard", "cloudwatch:GetDashboard", ...],
      "Resource": "*"
    }]
  }'
```

---

### Issue 5.3 — Lambda cannot update model package

**Error:**
```
ParamValidationError: Missing required parameter in input: "ModelPackageArn"
Unknown parameter in input: "ModelPackageName"
```

**Root cause:** `update_model_package` API requires `ModelPackageArn` not `ModelPackageName`. The parameter name changed between SDK versions.

**Fix:**
```python
# Wrong
sm_client.update_model_package(
    ModelPackageName=model_package_arn,  # Wrong parameter
    ModelApprovalStatus="Approved"
)

# Correct
sm_client.update_model_package(
    ModelPackageArn=model_package_arn,   # Correct parameter
    ModelApprovalStatus="Approved"
)
```

---

## 6. S3 Bucket Policy Lockout

---

### Issue 6.1 — Locked out of S3 bucket via overly aggressive Deny policy

**What happened:** Applied a bucket policy with `Effect: Deny` for all non-VPC traffic. The policy used `aws:SourceVpce` condition which blocked access from the AWS Console and Mac terminal (not going through VPC endpoint).

**Symptoms:**
- Console shows: `Access denied - s3:GetBucketPublicAccessBlock`
- CLI shows: `AccessDenied calling PutBucketPolicy`
- Even `PutBucketPolicy` to fix it was blocked

**Recovery procedure:**
1. Log in as **AWS root account** (not IAM user)
2. S3 console → bucket → Permissions → Bucket policy → **Delete entire policy**
3. Log out of root
4. Apply a corrected policy from Mac terminal

**Corrected safe policy:**
```json
{
  "Statement": [
    {
      "Sid": "AllowIAMUser",
      "Effect": "Allow",
      "Principal": {"AWS": "arn:aws:iam::ACCOUNT:user/YOUR_USER"},
      "Action": "s3:*",
      "Resource": ["arn:aws:s3:::BUCKET", "arn:aws:s3:::BUCKET/*"]
    },
    {
      "Sid": "AllowSageMakerRole",
      "Effect": "Allow",
      "Principal": {"AWS": "arn:aws:iam::ACCOUNT:role/sagemaker-role"},
      "Action": "s3:*",
      "Resource": ["arn:aws:s3:::BUCKET", "arn:aws:s3:::BUCKET/*"]
    }
  ]
}
```

**Prevention:**
- Never use `Effect: Deny` with `aws:SourceVpce` without first explicitly allowing your IAM user
- Always test with `aws s3 ls s3://bucket/` immediately after applying any bucket policy
- Use `aws:PrincipalArn` conditions in Deny statements rather than network-based conditions

---

### Issue 6.2 — CloudTrail `InsufficientS3BucketPolicyException`

**Error:**
```
InsufficientS3BucketPolicyException: Incorrect S3 bucket policy is detected for bucket
```

**Root cause:** CloudTrail requires specific S3 permissions — `s3:GetBucketAcl` and `s3:PutObject` with `bucket-owner-full-control` ACL condition — that are not part of a standard bucket policy.

**Fix:** Add CloudTrail-specific statements to bucket policy:
```json
{
  "Sid": "AllowCloudTrailACLCheck",
  "Effect": "Allow",
  "Principal": {"Service": "cloudtrail.amazonaws.com"},
  "Action": "s3:GetBucketAcl",
  "Resource": "arn:aws:s3:::BUCKET"
},
{
  "Sid": "AllowCloudTrailWrite",
  "Effect": "Allow",
  "Principal": {"Service": "cloudtrail.amazonaws.com"},
  "Action": "s3:PutObject",
  "Resource": "arn:aws:s3:::BUCKET/cloudtrail-logs/AWSLogs/ACCOUNT/*",
  "Condition": {
    "StringEquals": {"s3:x-amz-acl": "bucket-owner-full-control"}
  }
}
```

---

## 7. Endpoint Deployment Issues

---

### Issue 7.1 — `ml.t3.medium` not valid for endpoints

**Error:**
```
ValidationException: Value 'ml.t3.medium' at 'productionVariants.1.member.instanceType' 
failed to satisfy constraint: Member must satisfy enum value set
```

**Root cause:** SageMaker maintains separate instance type allowlists for Training Jobs vs Endpoints. `ml.t3.medium` is valid for Processing/Training but **not** for real-time endpoints.

**Fix:** Use `ml.m5.large` as minimum for real-time endpoints:
```python
"InstanceType": "ml.m5.large"  # minimum for real-time endpoints
```

**Valid endpoint instance families:** `ml.m5`, `ml.m6i`, `ml.c5`, `ml.c6i`, `ml.r5`, `ml.r6i`, GPU families

---

### Issue 7.2 — Endpoint stays in `Creating` state for >10 minutes

**Diagnosis:**
```bash
aws sagemaker describe-endpoint \
  --endpoint-name ENDPOINT-NAME \
  --region us-east-1 \
  --query '{Status:EndpointStatus,FailureReason:FailureReason}'

# Check container logs
aws logs describe-log-streams \
  --log-group-name "/aws/sagemaker/Endpoints/ENDPOINT-NAME" \
  --query 'logStreams[*].logStreamName'

aws logs get-log-events \
  --log-group-name "/aws/sagemaker/Endpoints/ENDPOINT-NAME" \
  --log-stream-name "STREAM-NAME" \
  --query 'events[*].message' | tail -20
```

**Common causes:**
- Image pull taking long (ECR VPC endpoint missing)
- Model loading error (wrong model format)
- Port binding failure (inference.py error)

---

### Issue 7.3 — Inference 500 error: `NoneType has no attribute 'startswith'`

**Error:**
```
AttributeError: 'NoneType' object has no attribute 'startswith'
sagemaker_sklearn_container.serving - Encountered an unexpected error
```

**Root cause:** The SKLearn serving container expects an `inference.py` file with `model_fn`, `input_fn`, `predict_fn`, `output_fn` handlers. Without it, `serving_env.module_name` is `None`, causing the error.

**Fix:** Create `inference/inference.py` with all required handlers:
```python
def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, "model.joblib"))

def input_fn(request_body, content_type="text/csv"):
    df = pd.read_csv(StringIO(request_body), header=None)
    return df.values

def predict_fn(input_data, model):
    return model.predict_proba(input_data)[:, 1]

def output_fn(prediction, accept="text/csv"):
    # handle both text/csv and application/json
```

---

### Issue 7.4 — Inference 500 error: `Unsupported accept type: application/json`

**Error:**
```
ValueError: Unsupported accept type: application/json
```

**Root cause:** boto3's `invoke_endpoint` sends `Accept: application/json` by default. Our `output_fn` only handled `text/csv`.

**Fix:** Add `application/json` support to `output_fn`:
```python
def output_fn(prediction, accept="text/csv"):
    if accept == "text/csv":
        return f"{int(pred)},{prob:.4f}", "text/csv"
    elif accept == "application/json":
        return json.dumps([{"prediction": int(pred), "probability": prob}]), "application/json"
    raise ValueError(f"Unsupported accept type: {accept}")
```

Or explicitly set Accept header in client:
```python
runtime.invoke_endpoint(
    EndpointName=endpoint,
    ContentType="text/csv",
    Accept="text/csv",  # explicit
    Body=sample
)
```

---

### Issue 7.5 — Cannot delete endpoint: MonitoringSchedule attached

**Error:**
```
ValidationException: The Endpoint currently has one or more MonitoringSchedules. 
Please delete the MonitoringSchedules before deleting the Endpoint.
```

**Fix:** Delete monitoring schedule first, then endpoint:
```bash
aws sagemaker delete-monitoring-schedule \
  --monitoring-schedule-name churn-monitor-schedule \
  --region us-east-1

sleep 10

aws sagemaker delete-endpoint \
  --endpoint-name churn-prediction-endpoint \
  --region us-east-1
```

**Teardown order for full cleanup:**
```
1. Delete monitoring schedule
2. Delete endpoint
3. Delete endpoint config
4. Delete model
5. Delete model package versions (if needed)
```

---

### Issue 7.6 — Cannot create endpoint: already exists

**Error:**
```
ValidationException: Cannot create already existing endpoint
```

**Fix:**
```bash
# Delete existing endpoint first
aws sagemaker delete-endpoint --endpoint-name NAME --region us-east-1
sleep 60  # wait for deletion

# Then redeploy
python inference/deploy_endpoint.py
```

**Idempotent deployment pattern:**
```python
try:
    sm_client.create_endpoint(EndpointName=name, ...)
except sm_client.exceptions.ResourceInUse:
    sm_client.update_endpoint(EndpointName=name, ...)
```

---

## 8. SageMaker Pipeline Issues

---

### Issue 8.1 — HPO tuning job: metric required

**Error:**
```
ValidationException: A metric is required for this hyperparameter tuning job objective. 
Provide a metric in the metric definitions.
```

**Root cause:** Metric definitions must be passed to **both** the `SKLearn` estimator AND the `HyperparameterTuner`. They are not inherited from the estimator.

**Fix:**
```python
tuner = HyperparameterTuner(
    estimator=estimator,
    objective_metric_name="roc_auc",
    metric_definitions=[          # Must be specified here too
        {"Name": "roc_auc", "Regex": "roc_auc=([0-9.]+)"},
        {"Name": "accuracy", "Regex": "accuracy=([0-9.]+)"},
    ],
    ...
)
```

---

### Issue 8.2 — Regex escaping in metric definitions

**Wrong:**
```python
{"Name": "roc_auc", "Regex": "roc_auc=([0-9\\\\.]+)"}  # over-escaped
```

**Correct:**
```python
{"Name": "roc_auc", "Regex": "roc_auc=([0-9.]+)"}  # simple regex
```

**Rule:** Use plain regex patterns in metric definitions. The SageMaker API handles escaping internally.

---

### Issue 8.3 — Pipeline cache miss despite unchanged inputs

**Warning:**
```
UserWarning: Profiling is enabled on the provided estimator. The default profiler rule 
includes a timestamp which will change each time the pipeline is upserted, causing cache misses.
```

**Fix:** Disable profiler on the estimator:
```python
estimator = SKLearn(
    ...
    disable_profiler=True,  # prevents timestamp-based cache invalidation
)
```

---

### Issue 8.4 — `ExplainabilityConfig() takes no arguments`

**Error:**
```
TypeError: ExplainabilityConfig() takes no arguments
```

**Root cause:** `ExplainabilityConfig` API changed between SageMaker SDK minor versions. In v2.257.3, `shap_config` is passed directly to `run_explainability()`, not wrapped in `ExplainabilityConfig`.

**Fix:**
```python
# Wrong
explainability_config = ExplainabilityConfig(shap_config=shap_config)
clarify_processor.run_explainability(explainability_config=explainability_config, ...)

# Correct
clarify_processor.run_explainability(explainability_config=shap_config, ...)
```

---

### Issue 8.5 — `FacetConfig` import error in Clarify

**Error:**
```
ImportError: cannot import name 'FacetConfig' from 'sagemaker.clarify'
```

**Root cause:** `FacetConfig` was removed from the `sagemaker.clarify` module in SDK v2.257.3. Bias config uses `facet_name` directly.

**Fix:**
```python
# Remove FacetConfig from imports
from sagemaker.clarify import (
    SageMakerClarifyProcessor, DataConfig, ModelConfig,
    SHAPConfig, BiasConfig, ModelPredictedLabelConfig
    # No FacetConfig
)

# Use facet_name directly in BiasConfig
bias_config = BiasConfig(
    label_values_or_threshold=[1],
    facet_name="gender",           # Direct string, no FacetConfig wrapper
    facet_values_or_threshold=[0],
)
```

---

## 9. Networking & VPC Issues

---

### Issue 9.1 — No default VPC in us-east-1

**Error:**
```
# VPC dropdown in SageMaker Studio setup was empty
```

**Root cause:** Default VPC was deleted at some point in account history.

**Fix:**
```bash
aws ec2 create-default-vpc --region us-east-1

# Verify
aws ec2 describe-vpcs \
  --filters "Name=isDefault,Values=true" \
  --query "Vpcs[0].VpcId" --output text
```

---

### Issue 9.2 — Studio cannot reach GitHub (VPC Only mode)

**Error:**
```
fatal: unable to access 'https://github.com/...': 
Failed to connect to github.com port 443 after 131960 ms
```

**Root cause:** Studio set to `VPC Only` mode — no internet access. This is by design for security but blocks GitHub access.

**Options:**
1. **Switch to Public internet mode** (recommended for portfolio)
   - SageMaker console → Domains → Edit → Network → Public internet access
   - Note: Stop all running apps before changing (ValidationException otherwise)
2. **Use S3 as code bridge** (workaround for VPC Only)
   ```bash
   # Mac: upload to S3
   aws s3 sync . s3://bucket/code/ --exclude ".git/*"
   # Studio: download from S3
   aws s3 sync s3://bucket/code/ ~/project/
   ```
3. **Set up CodeCommit** (private git inside VPC)

**Switching network mode requires stopping all apps first:**
```
ValidationException: Unable to update AppNetworkAccessType for Domain with InService/Pending Apps.
Delete all InService/Pending apps and try again.
```

---

### Issue 9.3 — S3 Gateway endpoint has no route table associated

**Symptom:** S3 endpoint created but `RouteTableIds: []` — traffic still routes via internet.

**Fix:**
```bash
# Get main route table
aws ec2 describe-route-tables \
  --filters "Name=vpc-id,Values=VPC_ID" \
  --query "RouteTables[?Associations[?Main==\`true\`]].RouteTableId" \
  --output text

# Associate endpoint with route table
aws ec2 modify-vpc-endpoint \
  --vpc-endpoint-id vpce-XXXXX \
  --add-route-table-ids rtb-XXXXX
```

---

## 10. MLflow Integration Issues

---

### Issue 10.1 — SageMaker managed MLflow returns 403

**Error:**
```
WARNING - MLflow logging failed: API request to endpoint /api/2.0/mlflow/experiments/get-by-name 
failed with error code 403 != 200.
<title>Error - SageMaker MLflow App access</title>
Issue with session - encountering a permission issue
```

**Root cause:** SageMaker managed MLflow uses IAM-based authentication. Simply providing the HTTPS URL is insufficient — requests must be signed with AWS credentials using the `sagemaker-mlflow` SDK.

**Architecture of SageMaker managed MLflow auth:**
```
Training Job container
    → needs: sagemaker-mlflow SDK
    → calls: STS to get temporary credentials
    → signs: MLflow API requests with SigV4
    → sends: to MLflow tracking server
```

**Fix:** Install `sagemaker-mlflow` in the ECR image and use it:
```dockerfile
RUN pip install sagemaker-mlflow sagemaker==2.257.3
```

**Remaining challenge:** `sagemaker==2.257.3` inside the container conflicts with the container's built-in `sagemaker_containers` package. This is a known dependency conflict.

**Alternative approach:** Use MLflow REST API directly with boto3 SigV4 signing instead of the Python SDK.

---

### Issue 10.2 — `No module named 'sagemaker'` in Training container

**Error:**
```
WARNING - MLflow logging failed (non-fatal): No module named 'sagemaker'
```

**Root cause:** `sagemaker-mlflow` depends on the `sagemaker` Python package, which is not installed in the SKLearn Training container by default.

**Status:** Non-fatal due to `try/except` wrapper. Training Jobs still succeed and log metrics via stdout.

**Long-term fix:** Use MLflow REST API with boto3 auth:
```python
import boto3
import requests
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

session = boto3.Session()
credentials = session.get_credentials()
# Sign requests manually using SigV4
```

---

## 11. Dependency Conflicts

---

### Issue 11.1 — pandas version conflict in Studio

**Warning:**
```
sparkmagic 0.21.0 requires pandas<2.0.0,>=0.17.1, 
but you have pandas 2.3.3 which is incompatible.
```

**Impact:** Non-fatal warning. SageMaker SDK still works correctly. `sparkmagic` is a Jupyter extension for Spark — not used in this project.

**Prevention:** Use `pip install --quiet` to suppress non-fatal warnings. Only investigate conflicts that cause `ImportError` or `AttributeError`.

---

### Issue 11.2 — snowflake-connector-python cffi conflict

**Warning:**
```
snowflake-connector-python 3.17.4 requires cffi<2.0.0,>=1.9, 
but you have cffi 2.0.0 which is incompatible.
```

**Impact:** Non-fatal. Snowflake connector is pre-installed in Studio Distribution image but not used in this project.

---

### Issue 11.3 — Mac pandas broken import

**Error:**
```
File "pandas/_libs/interval.pyx", line 1, in init pandas._libs.interval
KeyboardInterrupt (during import)
```

**Root cause:** Mac system Python had a broken pandas installation due to version conflicts between homebrew Python packages.

**Fix:** Run project scripts from Studio terminal or a clean virtualenv instead of Mac system Python:
```bash
# Create clean environment on Mac
python3 -m venv ~/clean-env
source ~/clean-env/bin/activate
pip install boto3 sagemaker==2.257.3
```

---

## 12. Monitoring Issues

---

### Issue 12.1 — `update_endpoint` with DataCaptureConfig fails

**Error:**
```
ParamValidationError: Unknown parameter in DeploymentConfig: "DataCaptureConfig", 
must be one of: BlueGreenUpdatePolicy, RollingUpdatePolicy, AutoRollbackConfiguration
```

**Root cause:** Data capture cannot be enabled via `update_endpoint` + `DeploymentConfig`. It must be set in the endpoint configuration at creation time.

**Fix:** Create new endpoint config with data capture, then update endpoint to use it:
```python
# Create new config with DataCaptureConfig
sm_client.create_endpoint_config(
    EndpointConfigName=new_config_name,
    ProductionVariants=existing_config["ProductionVariants"],
    DataCaptureConfig={
        "EnableCapture": True,
        "InitialSamplingPercentage": 100,
        "DestinationS3Uri": f"s3://bucket/monitoring/data-capture",
        "CaptureOptions": [{"CaptureMode": "Input"}, {"CaptureMode": "Output"}],
    }
)

# Update endpoint to new config
sm_client.update_endpoint(
    EndpointName=endpoint,
    EndpointConfigName=new_config_name
)
```

---

### Issue 12.2 — Monitoring schedule already exists

**Error:**
```
ResourceInUse: Monitoring schedule already exists
```

**Fix:** Use try/except to handle idempotent creation:
```python
try:
    monitor.create_monitoring_schedule(...)
except Exception as e:
    if "already exists" in str(e):
        print("Schedule exists — skipping")
    else:
        raise
```

---

## 13. Git Merge Conflicts

---

### Issue 13.1 — `git checkout --theirs` vs `--ours` during rebase

**Confusion:** During a `git rebase`, the semantics of `--theirs` and `--ours` are **reversed** compared to a merge:

| Operation | `--ours` | `--theirs` |
|-----------|---------|-----------|
| `git merge` | Current branch | Incoming branch |
| `git rebase` | Upstream (remote) | Your commits being replayed |

**Safe approach during rebase:**
```bash
# See both versions
git diff HEAD

# Accept remote version (most common during rebase)
git checkout --theirs filename
git add filename
git rebase --continue

# Accept local version
git checkout --ours filename
git add filename
git rebase --continue
```

---

### Issue 13.2 — Heredoc truncation in terminal

**Symptom:** File created by `cat > file.py << 'EOF'` is truncated, missing last sections.

**Root cause:** When pasting long heredocs into a remote terminal (Studio), the terminal emulator sometimes drops characters, causing early EOF detection.

**Fix:** Use Python to write files instead of bash heredoc:
```python
python3 << 'PYEOF'
content = """
# your file content here
"""
with open("filename.py", "w") as f:
    f.write(content)
print("SUCCESS")
PYEOF
```

Or use the Code Editor file explorer to create/edit files directly.

---

## Quick Reference — Common Fixes

| Problem | Quick Fix |
|---------|----------|
| Git askpass error | `git config --global --unset core.askpass` |
| Import error in Training container | Remove SDK imports, use stdout logging |
| Endpoint stays Creating | Check CloudWatch `/aws/sagemaker/Endpoints/` logs |
| Spot savings low | Increase `max_wait`, try different AZ |
| Pipeline cache miss | Add `disable_profiler=True` to estimator |
| S3 permission denied | Check bucket policy allows IAM user + SageMaker role |
| Docker build fails in Studio | Build on Mac with `--platform linux/amd64` |
| Protobuf conflict | Pin `protobuf==3.20.3` in Dockerfile |
| MLflow 403 | SageMaker managed MLflow requires IAM auth via `sagemaker-mlflow` SDK |
| Endpoint already exists | Delete endpoint + config + model, then redeploy |
| VPC subnet not showing | Run `aws ec2 create-default-vpc --region us-east-1` |

---

## Debugging Toolkit

### CloudWatch Log Commands
```bash
# Get Training Job logs
aws logs describe-log-streams \
  --log-group-name /aws/sagemaker/TrainingJobs \
  --log-stream-name-prefix "JOB-NAME"

aws logs get-log-events \
  --log-group-name /aws/sagemaker/TrainingJobs \
  --log-stream-name "JOB-NAME/algo-1-TIMESTAMP" \
  --query 'events[*].message' --output text | tail -30

# Get Endpoint logs
aws logs get-log-events \
  --log-group-name /aws/sagemaker/Endpoints/ENDPOINT-NAME \
  --log-stream-name "primary/INSTANCE-ID" \
  --query 'events[*].message' --output text | tail -20

# Get Lambda logs
aws logs tail /aws/lambda/FUNCTION-NAME --follow --region us-east-1
```

### SageMaker Status Commands
```bash
# Training Job status
aws sagemaker describe-training-job \
  --training-job-name JOB-NAME \
  --query '{Status:TrainingJobStatus,Reason:FailureReason}'

# Endpoint status
aws sagemaker describe-endpoint \
  --endpoint-name ENDPOINT-NAME \
  --query '{Status:EndpointStatus,Reason:FailureReason}'

# Pipeline execution status
aws sagemaker list-pipeline-execution-steps \
  --pipeline-execution-arn ARN \
  --query 'PipelineExecutionSteps[*].{Step:StepName,Status:StepStatus}' \
  --output table

# Model package status
aws sagemaker list-model-packages \
  --model-package-group-name GROUP-NAME \
  --query 'ModelPackageSummaryList[*].{Version:ModelPackageVersion,Status:ModelApprovalStatus}'
```

### S3 Debugging
```bash
# Check what's in sourcedir.tar.gz
aws s3 cp s3://bucket/job-name/source/sourcedir.tar.gz /tmp/
tar -tzf /tmp/sourcedir.tar.gz

# Check checkpoint files
aws s3 ls s3://bucket/checkpoints/ --recursive

# Check model artifacts
aws s3 ls s3://bucket/models/ --recursive | grep model.tar.gz
```

---
---

### Issue — Flan-T5 Fine-tuning: Custom ECR + Entry Point Conflict

**Problem:** SKLearn-based custom ECR image bakes `SAGEMAKER_PROGRAM=train.py` — overriding `finetune.py` entry point requires full image rebuild including `sentencepiece`, `torch`, `transformers` (~3GB).

**Root cause:** Two conflicting design decisions:
1. Custom ECR image bakes `train.py` for fast RandomForest training
2. Fine-tuning needs a different entry point with different dependencies

**Correct solution:** Two separate ECR images:
- `churn-mlops:inference` — sklearn, joblib, matplotlib (current image)
- `churn-mlops:finetune` — transformers, torch, sentencepiece, accelerate

**Status:** Fine-tuning code complete (`finetune.py`, `create_dataset.py`, `run_finetuning_job.py`). Pending GPU quota approval + separate fine-tuning ECR image.

---

*Document maintained alongside [Aws-Sagemaker-Project](https://github.com/Himanshu9001/Aws-Sagemaker-Project)*
*Last updated: May 2026*