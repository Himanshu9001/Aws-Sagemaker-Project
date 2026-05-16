# SageMaker MLOps — Customer Churn Prediction

![AWS](https://img.shields.io/badge/AWS-SageMaker-orange?logo=amazonaws)
![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2.1-green?logo=scikitlearn)
![MLflow](https://img.shields.io/badge/MLflow-3.10.1-blue?logo=mlflow)
![Docker](https://img.shields.io/badge/Docker-Custom%20ECR-blue?logo=docker)
![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-black?logo=githubactions)

Production-grade end-to-end MLOps pipeline built on AWS SageMaker for telecom customer churn prediction. Implements the complete ML lifecycle — from raw data to monitored, auto-scaling production deployment — with full CI/CD automation, explainability, and security hardening.

Built as a deliberate architectural comparison against a self-built EKS/Kubernetes MLOps stack (see [EKS MLOps Project](https://github.com/Himanshu9001/MLOps-Projects)) to document managed vs self-managed trade-offs from hands-on experience.

---

## Table of Contents

- [Architecture](#architecture)
- [Key Results](#key-results)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Phases](#phases)
- [Improvements](#improvements)
- [Key Design Decisions](#key-design-decisions)
- [SHAP Feature Importance](#shap-feature-importance)
- [CI/CD Pipeline](#cicd-pipeline)
- [Cost Analysis](#cost-analysis)
- [Security](#security)
- [Setup](#setup)
- [Lessons Learned](#lessons-learned)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     CI/CD Trigger                               │
│              git push → GitHub Actions                          │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                 SageMaker Pipeline (DAG)                        │
│                                                                 │
│  S3 (churn.csv)                                                 │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │  Preprocess │───▶│    Train    │───▶│  Evaluate   │         │
│  │  (SKLearn)  │    │ (RF+Spot)   │    │  (ROC AUC)  │         │
│  └─────────────┘    └─────────────┘    └──────┬──────┘         │
│                                               │                 │
│                                    AUC >= 0.83?                 │
│                                        │                        │
│                              ┌─────────┴─────────┐             │
│                              │                   │             │
│                              ▼                   ▼             │
│                       ┌────────────┐      ┌──────────┐         │
│                       │  Register  │      │   Stop   │         │
│                       │   Model    │      │ Pipeline │         │
│                       └─────┬──────┘      └──────────┘         │
│                             │                                   │
└─────────────────────────────┼───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Auto-Approval (EventBridge + Lambda)               │
│         ROC AUC >= 0.83 → Approved | else → Rejected           │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Serverless Endpoint                            │
│         churn-prediction-serverless (ml.m5.large)              │
│         199ms warm latency | $0.000016/GB-second               │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Observability                               │
│  Model Monitor (hourly) │ Clarify (SHAP) │ CloudWatch Dashboard │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Results

| Metric | Value |
|--------|-------|
| ROC AUC — baseline (default params) | 0.8357 |
| ROC AUC — Bayesian HPO best trial | **0.8445** |
| ROC AUC — Ray Tune baseline (EKS) | 0.8427 |
| **SageMaker HPO beats Ray Tune by** | **+0.0018** |
| Spot instance savings | **63.3%** |
| Cold start latency (serverless) | 1,114ms |
| Warm inference latency | **199ms** |
| Pipeline execution time | ~20 minutes |
| Monthly idle cost | **~$1.61** |
| Top churn driver | Contract type (SHAP 0.115) |
| Gender bias detected | **None** ✅ |

---

## Tech Stack

| Category | Technology |
|----------|-----------|
| **Cloud** | AWS SageMaker, S3, ECR, Lambda, EventBridge |
| **ML Framework** | scikit-learn 1.2.1, RandomForestClassifier |
| **MLOps** | SageMaker Pipelines, Experiments, Model Registry, Model Monitor |
| **Explainability** | SageMaker Clarify (SHAP values, bias detection) |
| **Experiment Tracking** | SageMaker Experiments + SageMaker managed MLflow v3.10.1 |
| **Serving** | SageMaker Serverless Inference + Real-time Endpoint |
| **CI/CD** | GitHub Actions → SageMaker Pipeline |
| **Containers** | Custom ECR image (extends SKLearn base) |
| **Security** | KMS encryption, CloudTrail, VPC endpoints, IAM least privilege |
| **IaC** | AWS CLI + SageMaker Python SDK v2.257.3 |
| **Monitoring** | CloudWatch Dashboard, SageMaker Model Monitor |

---

## Project Structure

```
sagemaker-mlops/
├── .github/
│   └── workflows/
│       └── sagemaker-pipeline.yml    # CI/CD — triggers pipeline on code push
│
├── processing/
│   ├── preprocess.py                 # SageMaker Processing Job entry point
│   ├── evaluate.py                   # Pipeline evaluation step (ROC AUC → JSON)
│   ├── run_processing_job.py         # v2 SKLearnProcessor submission
│   └── run_processing_job_v3.py      # v3 @remote comparison (documented)
│
├── training/
│   ├── train.py                      # Training script — metrics, checkpointing, MLflow
│   ├── run_training_job.py           # Training Job submission (Spot + KMS + ECR)
│   ├── run_hpo_job.py                # Bayesian HPO — 20 trials, maximize ROC AUC
│   └── requirements.txt             # Container dependencies
│
├── inference/
│   ├── inference.py                  # model_fn, input_fn, predict_fn, output_fn
│   ├── deploy_endpoint.py            # Real-time endpoint deployment
│   ├── deploy_serverless.py          # Serverless inference deployment
│   ├── deploy_ab_test.py             # A/B testing — 80/20 traffic split
│   └── test_endpoint.py             # Endpoint invocation test
│
├── pipelines/
│   ├── churn_pipeline.py             # 5-step SageMaker Pipeline DAG
│   └── register_model.py            # Model Registry + metadata + approval
│
├── monitoring/
│   ├── monitor.py                    # Data capture + baseline + hourly schedule
│   ├── clarify.py                    # SHAP explainability + bias detection
│   └── cloudwatch_dashboard.py      # CloudWatch dashboard creation
│
├── lambda/
│   ├── auto_approve_model.py         # Auto-approval Lambda function
│   └── deploy_lambda.py             # EventBridge rule + Lambda deployment
│
├── iam/
│   ├── sagemaker_role.json           # Execution role policy
│   ├── security_hardening.py        # S3 policy + versioning + CloudTrail
│   └── kms_config.json              # KMS key configuration
│
├── Dockerfile                        # Custom ECR image — pre-installed deps
├── Makefile                          # make train | make deploy | make clarify
├── requirements.txt                  # Project-level dependencies
└── .env.example                      # Environment variable template
```

---

## Phases

### Phase 1 — Setup & Infrastructure

**What:** IAM execution role, S3 bucket, VPC endpoints, SageMaker Studio Domain.

**Key decisions:**
- VPC Only mode for Studio → data never leaves AWS network
- Created VPC endpoints: S3 Gateway + SageMaker API Interface + SageMaker Runtime Interface
- Used `Public internet` for Studio IDE (code access) while keeping data plane private
- IAM execution role scoped to `churn-sagemaker-artifacts` bucket only

**Production insight:** The architectural boundary between "public internet for code" and "private VPC for data" is the correct production pattern. Most SageMaker deployments follow this — developers access Studio normally while training data and model artifacts stay private via VPC endpoints.

---

### Phase 2 — Processing Job

**What:** Converted `preprocess.py` to SageMaker Processing Job using `SKLearnProcessor`.

**SageMaker path conventions:**
```python
INPUT_PATH = "/opt/ml/processing/input/churn.csv"   # S3 → container
TRAIN_DIR  = "/opt/ml/processing/output/train"       # container → S3
TEST_DIR   = "/opt/ml/processing/output/test"        # container → S3
```

**SDK v2 vs v3 comparison (hands-on):**

| Aspect | v2 SKLearnProcessor | v3 @remote decorator |
|--------|--------------------|--------------------|
| Job type | ProcessingJob | TrainingJob |
| Code transport | Script file → S3 | cloudpickle serialization |
| S3 I/O | Explicit channels | Manual boto3 calls |
| Billable seconds | ~120s | **240s (2x more expensive)** |
| SDK version warning | None | Container v3.0.0.dev0 vs client 3.11.0 |
| S3 bucket | Controlled | Auto-created default bucket |

**Verdict:** v3 costs 2x more for identical work and has an uncontrolled S3 bucket. v2 used for this project. v3 is purpose-built for GenAI/LLM fine-tuning, not classical MLOps.

**Output:**
- `s3://churn-sagemaker-artifacts/data/processed/train/train.csv` — 5,634 rows
- `s3://churn-sagemaker-artifacts/data/processed/test/test.csv` — 1,409 rows

---

### Phase 3 — Training Job

**What:** `RandomForestClassifier` training with Spot instances, metric logging, and model artifact upload.

**Key implementation:**
```python
# Metrics logged to stdout — captured by CloudWatch regex filters
print(f"roc_auc={metrics['roc_auc']:.4f}")

# Model saved to SageMaker path — auto-uploaded to S3 as model.tar.gz
joblib.dump(model, "/opt/ml/model/model.joblib")
```

**Results:**

| Metric | Value |
|--------|-------|
| ROC AUC | 0.8357 |
| Accuracy | 0.7963 |
| F1 Score | 0.5710 |
| Billable seconds | 55s |
| Spot savings | **63.3%** |

**Custom ECR image advantage:** Eliminated 60-90 second `pip install` at runtime by baking dependencies into the image at build time. Every Training Job now starts from a pre-configured container.

---

### Phase 4 — Hyperparameter Tuning

**What:** Bayesian optimization across 20 trials to find optimal RandomForest hyperparameters.

**Search space:**
```python
hyperparameter_ranges = {
    "n_estimators":      IntegerParameter(50, 300),
    "max_depth":         IntegerParameter(5, 25),
    "min_samples_split": IntegerParameter(2, 10),
}
```

**Best trial:**

| Hyperparameter | Value |
|---------------|-------|
| n_estimators | 291 |
| max_depth | **7** (shallow — dataset insight) |
| min_samples_split | 2 |
| **ROC AUC** | **0.8445** |

**Comparison:**

| Method | ROC AUC |
|--------|---------|
| Default params | 0.8357 |
| EKS Ray Tune | 0.8427 |
| **SageMaker Bayesian HPO** | **0.8445 ✅** |

**Key insight:** `max_depth=7` was optimal despite searching up to 25. Shallow trees generalizing better indicates the dataset has limited feature interaction depth — overfitting risk with deep trees.

---

### Phase 5 — Model Registry

**What:** Registered best HPO model in SageMaker Model Registry with versioning, metadata, and approval workflow.

**Model package group:** `churn-prediction-models`

**Metadata stored per version:**
```python
CustomerMetadataProperties={
    "roc_auc": "0.8445",
    "model_type": "RandomForestClassifier",
    "n_estimators": "291",
    "max_depth": "7",
    "developer": "heman"
}
```

**Approval workflow:** `PendingManualApproval` → automated Lambda approval → `Approved`

**Model Card created:** `churn-prediction-rf-v1` with ethical considerations, SHAP insights, bias detection results, and regulatory scope documentation.

---

### Phase 6 — Endpoint Deployment

**What:** Deployed approved model to real-time endpoint with custom inference script, then migrated to serverless inference.

**Custom inference handlers:**
```python
def model_fn(model_dir):     # loads model.joblib on container start
def input_fn(body, ctype):   # parses CSV input → numpy array
def predict_fn(data, model): # runs RandomForest prediction
def output_fn(pred, accept): # returns CSV or JSON response
```

**Sample inference:**
```json
{
  "prediction": 0,
  "churn_probability": 0.1842,
  "label": "No Churn"
}
```

**A/B Testing:** Deployed two model variants simultaneously:
- Champion: 80% traffic
- Challenger: 20% traffic
- `InvokedProductionVariant` field in response identifies which variant served each request

---

### Phase 7 — SageMaker Pipeline

**What:** 5-step automated DAG replacing Airflow from the EKS project.

```
ChurnPreprocess → ChurnTrain → ChurnEvaluate → CheckRocAuc → ChurnRegisterModel
```

**Conditional step:** Uses `PropertyFile` + `JsonGet` to read ROC AUC from evaluation JSON and conditionally register — entirely server-side, no Python polling required.

```python
condition_gte = ConditionGreaterThanOrEqualTo(
    left=JsonGet(step_name="ChurnEvaluate",
                 property_file=evaluation_report,
                 json_path="binary_classification_metrics.auc.value"),
    right=roc_auc_threshold,  # 0.83
)
```

**Pipeline parameters:** `RocAucThreshold`, `InstanceType`, `ModelApprovalStatus` — all configurable at runtime without code changes.

**Cache config:** `CacheConfig(enable_caching=True, expire_after="PT24H")` — reuses Processing and Training outputs if inputs unchanged within 24 hours.

**vs Airflow (EKS):**

| Aspect | Airflow (EKS) | SageMaker Pipelines |
|--------|--------------|---------------------|
| Orchestrator | Separate EC2/pod | Managed by AWS |
| DAG definition | Python + operators | Python SDK steps |
| State persistence | Postgres | SageMaker API |
| Crash recovery | Manual restart | Automatic |
| Cost | Always-on EC2 | Pay per execution |

---

### Phase 8 — Model Monitor

**What:** Hourly data drift detection with baseline statistics from training data.

**Components:**
- Data capture: 100% sampling of all endpoint requests/responses
- Baseline: computed from `train.csv` — mean, std, completeness per feature
- Monitoring schedule: `cron(0 * ? * * *)` — hourly
- Violation reports: `s3://churn-sagemaker-artifacts/monitoring/reports/`

**vs Evidently AI (EKS):**

| Aspect | Evidently AI | SageMaker Model Monitor |
|--------|-------------|------------------------|
| Deployment | Docker container on EKS | Fully managed |
| Statistical test | Kolmogorov-Smirnov | Custom constraints |
| Integration | Custom webhooks | Native CloudWatch |
| Cost | EC2 always-on | Per monitoring job |

---

### Phase 9 — Observability

**What:** CloudWatch dashboard covering endpoint health, latency, errors, and cost.

**Dashboard metrics:**
- Endpoint invocations (sum per 5 min)
- Model latency + overhead latency (average)
- 4XX/5XX error rates
- CPU/Memory utilization
- Auto-scaling instance count
- Cost summary table

**CloudWatch URL:** `https://us-east-1.console.aws.amazon.com/cloudwatch/home?region=us-east-1#dashboards:name=ChurnMLOps-Dashboard`

---

### Phase 10 — Security

**What:** Defense-in-depth security hardening across all layers.

| Control | Implementation |
|---------|---------------|
| Network | VPC endpoints for S3, SageMaker API, SageMaker Runtime |
| Encryption at rest | KMS customer-managed key (`alias/churn-mlops-key`) for S3 + EBS |
| S3 access control | Bucket policy — allow only IAM user + SageMaker execution role |
| Audit trail | CloudTrail — all API calls logged with log file validation |
| Data protection | S3 versioning — model artifacts protected against accidental deletion |
| IAM | Least privilege execution role — scoped to single S3 bucket |

---

## Improvements

Beyond the 10 core phases, the following production improvements were implemented:

### Improvement 1 — GitHub Actions CI/CD

Automated pipeline trigger on every code push to `training/`, `processing/`, or `pipelines/`:

```yaml
on:
  push:
    branches: [main]
    paths:
      - 'training/**'
      - 'processing/**'
      - 'pipelines/**'
```

Workflow: authenticate AWS → install SDK → start pipeline → poll completion → print step table.

### Improvement 2 — Serverless Inference

Replaced `ml.m5.large` real-time endpoint ($0.13/hr = $95/month) with serverless inference:

```python
serverless_config = ServerlessInferenceConfig(
    memory_size_in_mb=2048,
    max_concurrency=5,
)
```

**Latency benchmark:**
- Cold start: 1,114ms
- Warm: **199ms**
- Cold/Warm ratio: 5.6x

**Cost: ~$0.01/month for portfolio traffic vs $95/month always-on.**

### Improvement 3 — SageMaker Clarify

SHAP values computed for all 1,409 test samples. See [SHAP Feature Importance](#shap-feature-importance) section.

Outputs:
- `explanations_shap/out.csv` — per-sample SHAP values
- `report.html` — interactive explainability report
- `report.pdf` — shareable governance artifact

### Improvement 4 — Model Card

Published `churn-prediction-rf-v1` model card with:
- Intended uses + explicitly excluded uses
- Ethical considerations (fairness, transparency, human oversight)
- Risk rating: **Low** (marketing decisions only, not credit/service termination)
- SHAP feature importance documented
- Bias detection results: gender not in top 10 features ✅

### Improvement 5 — Custom ECR Image

Eliminated runtime `pip install` by baking dependencies into a custom image:

```dockerfile
FROM 683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3

RUN pip install --no-cache-dir \
    mlflow>=2.0.0 \
    protobuf==3.20.3 \
    sagemaker-mlflow \
    matplotlib==3.7.1 \
    seaborn==0.12.2 \
    joblib==1.3.2
```

**Time saved:** 60-90 seconds per Training Job. Significant for HPO with 20 parallel trials.

### Improvement 6 — Auto Model Approval

EventBridge + Lambda pipeline that automatically approves models meeting quality threshold:

```
Model registered (Pending) 
→ EventBridge: SageMaker Model Package State Change
→ Lambda: reads ROC AUC from CustomerMetadataProperties
→ ROC AUC >= 0.83 → update_model_package(Approved)
→ ROC AUC < 0.83  → update_model_package(Rejected)
```

**Result:** Zero human intervention from code push to approved model.

### Improvement 7 — A/B Testing

Two model variants serving production traffic simultaneously:

```python
ProductionVariants=[
    {"VariantName": "champion",   "InitialVariantWeight": 0.8},  # 80%
    {"VariantName": "challenger", "InitialVariantWeight": 0.2},  # 20%
]
```

`InvokedProductionVariant` field in response identifies which variant served each request for metric comparison.

### Improvement 8 — Spot Checkpointing

Checkpoint saved after training — SageMaker syncs to S3 on Spot interruption:

```python
# Saves to /opt/ml/checkpoints/ → synced to s3://churn-sagemaker-artifacts/checkpoints/
save_checkpoint(model, metrics, epoch=1, checkpoint_dir=CHECKPOINT_DIR)

# On restart — loads from S3-synced checkpoint
checkpoint = load_latest_checkpoint(CHECKPOINT_DIR)
if checkpoint:
    model = checkpoint["model"]  # skip retraining
```

**Verified:** `checkpoint_1.pkl` (5.5MB) confirmed in S3 after Training Job completion.

### Improvement 9 — KMS Encryption

Customer-managed KMS key (`alias/churn-mlops-key`) applied to:
- S3 bucket: server-side encryption with `BucketKeyEnabled=True` (reduces API calls)
- Training Job EBS volumes: `volume_kms_key` parameter on SKLearn estimator
- SageMaker Pipeline training step: same KMS key

### Improvement 10 — MLflow Parallel Tracking

SageMaker managed MLflow v3.10.1 deployed (`churn-mlflow` app). Training script logs to both SageMaker Experiments (stdout regex) and MLflow simultaneously using a non-fatal try/except pattern.

**Auth challenge documented:** SageMaker managed MLflow requires IAM-signed requests via `sagemaker-mlflow` SDK. Conflict between `sagemaker==2.257.3` inside the Training container and the container's own `sagemaker_containers` package. Resolution: use MLflow's direct REST API with boto3-signed requests as an alternative to the Python SDK.

---

## Key Design Decisions

### Why SageMaker over Self-Built EKS Stack?

| Concern | EKS (Self-Built) | SageMaker (Managed) |
|---------|-----------------|---------------------|
| Infrastructure | Terraform + Helm + K8s | Zero infrastructure |
| Experiment tracking | MLflow on RDS PostgreSQL | SageMaker Experiments |
| HPO | Ray Tune | Automatic Model Tuning |
| Orchestration | Apache Airflow | SageMaker Pipelines |
| Monitoring | Evidently AI | Model Monitor |
| Serving | FastAPI + Docker | Managed Endpoint |
| Registry | MLflow Model Registry | SageMaker Model Registry |
| Time to production | Days | Hours |
| Cost at scale | **Lower** | Higher |
| Multi-cloud portability | **Yes** | AWS only |
| Infrastructure control | **Full** | None |
| Debugging depth | **Full** | Limited |
| Governance/audit trail | Custom | **Built-in** |

**Verdict:** SageMaker wins on speed-to-production, governance, and operational overhead. EKS wins on cost at scale, portability, and control. For a team of 1-5 ML engineers moving fast — SageMaker. For a platform team supporting 50+ data scientists at scale — self-built on EKS.

### Why SDK v2 over v3?

SageMaker SDK v3 was evaluated and deliberately rejected for this project:

- **Processing Jobs removed** in v3 — uses Training Job infrastructure (2x cost)
- **v3 @remote costs 2x more** — 240 billable seconds vs 120 for identical work
- **Version mismatch** — container runs v3.0.0.dev0, client expects 3.11.0
- **Model Monitor unavailable** in v3
- **HPO API changed** significantly in v3

v3 is purpose-built for GenAI/LLM fine-tuning (`sft_trainer`, `dpo_trainer`, `rlvr_trainer`). For classical MLOps pipelines, v2 is the correct choice as of 2026.

### Serverless vs Real-Time Inference

| | Serverless | Real-Time |
|--|-----------|-----------|
| Cost (portfolio) | ~$0.01/month | ~$95/month |
| Cold start | 1,114ms | None |
| Warm latency | 199ms | ~50ms |
| Auto-scaling | Automatic to zero | Manual policy |
| Max concurrency | 5 (configurable) | Unlimited (with scaling) |
| Best for | <1,000 req/day | >1,000 req/day |

**Decision:** Serverless for portfolio/development. Switch to real-time `ml.m5.large` for production with sustained traffic.

### Why Bayesian HPO over Random Search?

With 20 trials and 3 hyperparameters, Bayesian optimization uses a Gaussian Process surrogate model to predict which combinations will perform best based on completed trials. Each wave of 5 parallel trials informs the next wave's suggestions.

**Result:** Beat Ray Tune (random search equivalent) by 0.0018 ROC AUC — a meaningful improvement in a domain where 0.01 AUC can represent thousands of correctly identified churners.

---

## SHAP Feature Importance

SageMaker Clarify computed SHAP values across all 1,409 test samples:

| Rank | Feature | SHAP Value | Business Interpretation |
|------|---------|-----------|------------------------|
| 1 | **Contract** | 0.1150 | Month-to-month → highest churn risk. 85% more impactful than monthly charges |
| 2 | **MonthlyCharges** | 0.0803 | Higher charges → higher churn probability |
| 3 | **tenure** | 0.0624 | Longer-tenured customers → lower churn risk |
| 4 | **TotalCharges** | 0.0419 | Correlated with tenure |
| 5 | **InternetService** | 0.0249 | Fiber optic customers churn more than DSL |
| 6 | **OnlineSecurity** | 0.0234 | No security addon → higher churn |
| 7 | **PaperlessBilling** | 0.0215 | Paperless billing customers churn more |
| 8 | **TechSupport** | 0.0209 | No tech support → higher churn |
| 9 | **PaymentMethod** | 0.0208 | Electronic check payment → highest churn |
| 10 | **MultipleLines** | 0.0206 | Multiple lines customers slightly less likely to churn |

**Bias check:** Gender does not appear in top 10 features — no gender-based discrimination detected ✅

**Business recommendation:** Retention campaigns should prioritize converting month-to-month customers to annual contracts (SHAP 0.115) over offering price discounts (MonthlyCharges SHAP 0.080). Contract type is 44% more impactful than price.

---

## CI/CD Pipeline

```
Developer pushes code to training/, processing/, or pipelines/
         │
         ▼
GitHub Actions workflow triggered
         │
         ├── Configure AWS credentials (OIDC or access key)
         ├── Install sagemaker==2.257.3 + dependencies
         ├── Verify AWS connection (sts get-caller-identity)
         │
         ▼
SageMaker Pipeline execution started
         │
         ├── ChurnPreprocess  (~3 min)  ← SKLearnProcessor
         ├── ChurnTrain       (~5 min)  ← Spot Training Job
         ├── ChurnEvaluate    (~3 min)  ← ROC AUC computation
         ├── CheckRocAuc      (<1 sec)  ← Conditional: AUC >= 0.83?
         └── ChurnRegisterModel         ← Model Registry
         │
         ▼ (on successful registration)
EventBridge fires: SageMaker Model Package State Change
         │
         ▼
Lambda: churn-auto-approve-model
         │
         ├── Read ROC AUC from CustomerMetadataProperties
         ├── ROC AUC >= 0.83 → update_model_package(Approved)
         └── ROC AUC < 0.83  → update_model_package(Rejected)
         │
         ▼
Model approved and ready for deployment
Total time: ~20 minutes from git push to approved model
```

**Zero human intervention required.**

---

## Cost Analysis

### Monthly Cost Breakdown (Portfolio/Development)

| Component | Resource | Monthly Cost |
|-----------|---------|-------------|
| Serverless endpoint | Pay per request | ~$0.01 |
| KMS key | 1 CMK | $1.00 |
| S3 storage | ~5GB artifacts | ~$0.12 |
| CloudTrail | Management events | ~$0.10 |
| MLflow app | Managed tracking server | ~$0.50 |
| VPC endpoints | 2 Interface endpoints | ~$0.72 |
| **Total idle** | | **~$2.45/month** |

### Per-Run Costs

| Job | Instance | Duration | Spot Savings | Cost |
|-----|---------|---------|-------------|------|
| Processing Job | ml.m5.large | ~120s | N/A | ~$0.001 |
| Training Job | ml.m5.large | 55s billable | **63.3%** | ~$0.001 |
| HPO (20 trials) | ml.m5.large | ~1,100s total | **~60%** | ~$0.02 |
| Pipeline run | ml.m5.large | ~300s | 63% | ~$0.005 |
| Clarify | ml.m5.large | ~600s | N/A | ~$0.006 |

### Cost Comparison: SageMaker vs EKS

| Component | EKS (self-built) | SageMaker |
|-----------|-----------------|-----------|
| Always-on infrastructure | ~$150/month (EKS + EC2) | $0 |
| Training compute | On-demand EC2 | Spot (63% cheaper) |
| Monitoring | EC2 always-on | Pay per job |
| **Total at low traffic** | **~$200/month** | **~$2.45/month** |
| **Total at high traffic** | **Lower** | **Higher** |

**Crossover point:** ~50 Training Jobs/day where SageMaker per-job costs exceed EKS always-on infrastructure costs.

---

## Security

### Defense-in-Depth Architecture

```
Internet
    │
    │ (Studio access only — code, not data)
    ▼
Public Internet Gateway
    │
    ▼
VPC (172.31.0.0/16)
    │
    ├── VPC Endpoint: S3 Gateway (free)
    │       └── churn-sagemaker-artifacts (KMS encrypted)
    │
    ├── VPC Endpoint: SageMaker API Interface
    │       └── Training/Processing Job control plane
    │
    └── VPC Endpoint: SageMaker Runtime Interface
            └── Endpoint invocation
```

### Security Controls

| Layer | Control | Implementation |
|-------|---------|---------------|
| Network | VPC endpoints | S3 Gateway + SageMaker API + Runtime |
| Network | No direct S3 internet | All S3 traffic via VPC endpoint |
| Encryption | KMS CMK | S3 objects + EBS training volumes |
| Encryption | TLS | All API calls (HTTPS) |
| Access control | S3 bucket policy | Allow only IAM user + SageMaker role |
| Access control | IAM least privilege | Scoped execution role |
| Audit | CloudTrail | All API calls logged + integrity validation |
| Data protection | S3 versioning | Model artifacts protected |
| Governance | Model Card | Documented risk rating + ethical considerations |

### IAM Execution Role Permissions

```json
{
  "Actions": [
    "s3:GetObject", "s3:PutObject", "s3:ListBucket",
    "logs:CreateLogGroup", "logs:PutLogEvents",
    "ecr:GetAuthorizationToken", "ecr:BatchGetImage",
    "cloudwatch:PutMetricData", "cloudwatch:PutDashboard"
  ],
  "Resources": ["arn:aws:s3:::churn-sagemaker-artifacts/*"]
}
```

---

## Setup

### Prerequisites

```bash
# AWS CLI configured
aws configure

# Python 3.9+
python --version

# Docker (for ECR image builds)
docker --version

# SageMaker SDK v2
pip install "sagemaker>=2.200,<3.0"
```

### Environment Variables

Copy `.env.example` and fill in values:

```bash
cp .env.example .env
```

```env
AWS_REGION=us-east-1
AWS_ACCOUNT_ID=<your-account-id>
SAGEMAKER_ROLE_ARN=arn:aws:iam::<account>:role/sagemaker-churn-execution-role
S3_BUCKET=churn-sagemaker-artifacts
ECR_IMAGE=<account>.dkr.ecr.us-east-1.amazonaws.com/churn-mlops:latest
KMS_KEY_ARN=arn:aws:kms:us-east-1:<account>:key/<key-id>
SERVERLESS_ENDPOINT=churn-prediction-serverless
MLFLOW_TRACKING_URI=https://<mlflow-app>.mlflow.sagemaker.us-east-1.app.aws/
```

### Running the Pipeline

```bash
# Full pipeline (all phases)
make pipeline

# Individual components
make process    # Phase 2: Processing Job
make train      # Phase 3: Training Job
make hpo        # Phase 4: Hyperparameter Tuning
make deploy     # Phase 6: Serverless Endpoint
make monitor    # Phase 8: Model Monitor
make clarify    # Phase 3 Improvement: SHAP Explainability

# Build and push custom ECR image
make ecr-build
make ecr-push

# Cleanup
make clean
```

### GitHub Actions Setup

Add these secrets to your GitHub repository:

| Secret | Description |
|--------|-------------|
| `AWS_ACCESS_KEY_ID` | IAM user access key |
| `AWS_SECRET_ACCESS_KEY` | IAM user secret key |
| `AWS_REGION` | `us-east-1` |

Pipeline triggers automatically on push to `training/`, `processing/`, or `pipelines/`.

---

## Optimizations

Beyond the 10 core phases and 9 improvements, the following production optimizations were implemented:

---

### 1. ECR Auto-Build CI/CD

GitHub Actions workflow automatically rebuilds and pushes the ECR image when `Dockerfile`, `train.py`, or `requirements.txt` changes. Images tagged with git commit SHA — eliminates the `latest` anti-pattern.

```yaml
# .github/workflows/ecr-build.yml
on:
  push:
    paths:
      - 'Dockerfile'
      - 'training/train.py'
      - 'training/requirements.txt'
```

**Impact:** No more manual `docker build && docker push`. Every code change automatically produces a versioned, traceable image.

---

### 2. Auto-Deploy After Approval

Second Lambda (`churn-auto-deploy-endpoint`) triggered by EventBridge on model approval. Automatically updates the serverless endpoint — completing the full CD loop with zero human intervention:

```
git push
  → GitHub Actions (unit tests)
  → SageMaker Pipeline (preprocess → train → evaluate → register)
  → EventBridge → Lambda (auto-approve if ROC AUC >= 0.83)
  → EventBridge → Lambda (auto-deploy to serverless endpoint)
```

**Impact:** Complete CI/CD loop — code push to live endpoint with no manual steps.

---

### 3. Unit Tests (15/15 Passing)

```
tests/
├── test_inference.py    # 7 tests — input_fn, output_fn, CSV/JSON formats
└── test_preprocess.py   # 8 tests — customerID drop, encoding, split, nulls
```

**Test coverage:**

| File | Tests | What's Covered |
|------|-------|---------------|
| `test_inference.py` | 7 | CSV parsing, JSON output, error handling, both response formats |
| `test_preprocess.py` | 8 | customerID dropped, Churn encoded, TotalCharges numeric, no nulls, categorical encoding, row count, split sizes |

**GitHub Actions output:**
```
15 passed in 0.87s
```

Tests run as first step in CI/CD — failed tests block pipeline execution before any AWS cost is incurred.

---

### 4. Multi-Stage Dockerfile

Separated dependency installation from code copying for maximum Docker layer cache hits:

```dockerfile
# Stage 1: dependencies — only rebuilds when requirements change
FROM sagemaker-scikit-learn:1.2-1-cpu-py3 AS deps
RUN pip install --no-cache-dir \
    mlflow>=2.0.0 \
    protobuf==3.20.3 \
    matplotlib==3.7.1 \
    seaborn==0.12.2 \
    joblib==1.3.2

# Stage 2: code — only rebuilds when train.py changes
FROM deps AS final
COPY training/train.py /opt/ml/code/train.py
ENV SAGEMAKER_PROGRAM=train.py
```

**Impact:** Changing `train.py` no longer triggers a full pip reinstall. Build time reduced from ~8 minutes to ~30 seconds on code-only changes.

---

### 5. Pinned Dependencies

All packages pinned across all environments — eliminates version drift:

```
# requirements.txt
sagemaker==2.257.3
boto3==1.34.0
scikit-learn==1.2.1
pandas==1.5.3
numpy==1.24.3
mlflow==2.14.0
protobuf==3.20.3
matplotlib==3.7.1
seaborn==0.12.2
joblib==1.3.2
```

**Why protobuf==3.20.3:** `mlflow>=2.0.0` pulls protobuf 4.x which breaks `sagemaker_containers`. This is a known transitive dependency conflict — pinning to 3.20.3 is the last compatible version.

---

### 6. Model Lifecycle Cleanup

`pipelines/cleanup_model_versions.py` — keeps only last N approved versions, auto-deletes older ones:

```python
def cleanup_old_versions(group_name, keep_n=5):
    packages = sm_client.list_model_packages(
        ModelPackageGroupName=group_name,
        SortBy="CreationTime",
        SortOrder="Descending"
    )["ModelPackageSummaryList"]

    for pkg in packages[keep_n:]:
        sm_client.delete_model_package(
            ModelPackageName=pkg["ModelPackageArn"]
        )
```

**Impact:** During development, 6+ model versions accumulated. This script keeps the registry clean and reduces storage costs.

---

### 7. Cost Alerting

CloudWatch billing alarm triggers SNS email notification when SageMaker monthly spend exceeds $50:

```bash
aws cloudwatch put-metric-alarm \
  --alarm-name "SageMaker-Monthly-Cost-50USD" \
  --metric-name EstimatedCharges \
  --namespace AWS/Billing \
  --threshold 50 \
  --alarm-actions $SNS_TOPIC_ARN
```

**Impact:** Prevents surprise bills during active development. Particularly important when running HPO (20 trials) or leaving endpoints running.

---

### 8. Lifecycle Configuration (Git Credentials)

SageMaker Studio Lifecycle Configuration script runs on every Code Editor space start — permanently fixes the VS Code askpass credential issue that caused authentication failures every session:

```bash
#!/bin/bash
# Runs automatically on Code Editor space start
git config --global --unset core.askpass || true
git config --global core.askpass ""
git config --global user.email "himanshu9001@gmail.com"
git config --global user.name "Himanshu9001"

# Fetch PAT securely from Secrets Manager
PAT=$(aws secretsmanager get-secret-value \
  --secret-id github-pat \
  --region us-east-1 \
  --query SecretString \
  --output text)

git config --global credential.helper store
echo "https://Himanshu9001:${PAT}@github.com" > ~/.git-credentials
chmod 600 ~/.git-credentials
```

**Impact:** Eliminates the recurring `askpass.sh: No such file or directory` error that required manual PAT-in-URL workaround every session.

---

### 9. Branch Protection Rules

GitHub branch protection enforces quality gates on `main`:

- GitHub Actions must pass before merging
- No direct pushes to `main`
- All changes via pull request

**Impact:** Prevents the diverged branch problem that occurred multiple times during development when both Mac and Studio committed directly to `main`.

---

### 10. pytest in GitHub Actions

Unit tests added as first step in CI/CD workflow — fast feedback before expensive pipeline runs:

```yaml
- name: Run unit tests
  run: |
    pip install pytest scikit-learn joblib pandas numpy matplotlib seaborn
    pytest tests/ -v
```

**Execution time comparison:**

| Step | Time | Cost |
|------|------|------|
| Unit tests | **0.87s** | $0.00 |
| SageMaker Pipeline | ~20 minutes | ~$0.03 |

Failed unit tests block pipeline execution — saves $0.03 per prevented bad run and ~20 minutes of waiting.

---

### Optimization Results Summary

| Optimization | Before | After | Impact |
|---|---|---|---|
| ECR image rebuild | Manual (~10 min) | Automatic on push | Zero manual steps |
| Endpoint update | Manual CLI | Auto on model approval | Complete CD loop |
| Dependency conflicts | Runtime failures in container | Caught at build time | 5 failed jobs → 0 |
| Test coverage | 0% | 15 tests, all passing | Bugs caught in 0.87s |
| Image tagging | `latest` only | SHA + `latest` | Full traceability |
| Model versions | Accumulating unbounded | Auto-cleanup (keep 5) | Clean registry |
| Git credentials | Manual PAT per session | Auto from Secrets Manager | No session friction |
| Cost visibility | None | $50 CloudWatch alarm | No surprise bills |
| Branch safety | Direct push to main | PR + CI required | No accidental breaks |
| CI feedback | Pipeline runs always | Tests gate pipeline | Fast failure detection |

---

### Full Automated Pipeline Flow (Post-Optimizations)

```
Developer pushes code to training/, processing/, or pipelines/
                    │
                    ▼
         GitHub Actions triggered
                    │
         ┌──────────┴──────────┐
         │                     │
         ▼                     ▼
   Unit Tests (0.87s)    ECR Image Build
   15/15 passing         (if Dockerfile changed)
         │                     │
         └──────────┬──────────┘
                    │
                    ▼
         SageMaker Pipeline (~20 min)
                    │
         ┌──────────▼──────────┐
         │  ChurnPreprocess    │
         │  ChurnTrain         │
         │  ChurnEvaluate      │
         │  CheckRocAuc        │
         │  ChurnRegisterModel │
         └──────────┬──────────┘
                    │
                    ▼
         EventBridge fires
                    │
                    ▼
         Lambda: Auto-Approve
         (ROC AUC >= 0.83)
                    │
                    ▼
         EventBridge fires
                    │
                    ▼
         Lambda: Auto-Deploy
         (update serverless endpoint)
                    │
                    ▼
         Model live in production
         Zero human intervention
```

---

## Lessons Learned

### Technical Lessons

**1. SageMaker SDK v3 is not ready for classical MLOps**
Evaluated v3 `@remote` decorator — costs 2x more than v2 Processing Jobs and lacks Processing Job support entirely. v3 is purpose-built for GenAI fine-tuning. For classical ML pipelines, v2 is the correct choice.

**2. Container dependency management is the hardest problem**
Five failed Training Jobs due to dependency conflicts (matplotlib, sagemaker, protobuf). Custom ECR images solve this permanently — bake dependencies at build time, not runtime.

**3. S3 bucket policies with explicit Deny are dangerous**
Applied a `DenyNonVPCAccess` policy that locked out both the IAM user and the console. Required root account access to recover. Lesson: always test bucket policies with `PutBucketPolicy` before applying `Deny` statements. Use `aws:PrincipalArn` conditions rather than IP-based restrictions.

**4. Spot checkpointing only triggers on interruption**
Checkpoints don't appear in S3 on successful job completion — only on Spot interruption. The 2-minute warning window is when SageMaker syncs `/opt/ml/checkpoints/` to S3. Implementation is correct; testing requires simulating interruption.

**5. SageMaker managed MLflow requires IAM-signed requests**
The `sagemaker-mlflow` SDK handles auth but conflicts with the Training container's `sagemaker_containers` package. Resolution path: use MLflow REST API directly with boto3-signed requests, bypassing the Python SDK layer.

### Architectural Lessons

**6. The correct boundary between public and private**
Data plane (S3, training, endpoints) stays private via VPC endpoints. Control plane (Studio IDE, GitHub) can use public internet. This is the industry standard pattern — not all traffic needs to be private, just the sensitive data paths.

**7. EventBridge + Lambda is the right auto-approval pattern**
Polling-based approval (checking metric in a loop) would require an always-running process. Event-driven approval via EventBridge fires instantly when a model is registered, costs ~$0.00001 per invocation, and scales to any volume.

**8. Managed services trade control for speed — know when each matters**
SageMaker eliminated ~80% of infrastructure code compared to the EKS stack. The remaining 20% (IAM, VPC, KMS, CloudTrail) still requires deep AWS knowledge. "Managed" doesn't mean "zero ops" — it means different ops.

---

## Comparison: SageMaker vs EKS MLOps Stack

This project was built as a deliberate comparison against the [self-built EKS MLOps stack](https://github.com/Himanshu9001/MLOps-Projects).

### Component Mapping

| EKS Component | SageMaker Equivalent | Winner |
|--------------|---------------------|--------|
| Airflow DAG | SageMaker Pipelines | SageMaker (zero ops) |
| MLflow tracking | SageMaker Experiments | Tie (MLflow more flexible) |
| Ray Tune HPO | Automatic Model Tuning | SageMaker (0.8445 vs 0.8427) |
| Evidently AI | Model Monitor | SageMaker (native integration) |
| FastAPI endpoint | Managed Endpoint | SageMaker (no container mgmt) |
| MLflow Registry | Model Registry | SageMaker (approval workflow) |
| Kubernetes HPA | Auto-scaling | SageMaker (simpler config) |
| Terraform | None needed | SageMaker (zero IaC) |
| Helm charts | None needed | SageMaker (zero K8s) |

### When to Choose Each

**Choose SageMaker when:**
- Small team (1-5 ML engineers)
- Speed to production is priority
- AWS-only deployment
- Regulatory compliance required (built-in audit trail)
- Classical ML with standard frameworks

**Choose self-built EKS when:**
- Large team (20+ engineers)
- Multi-cloud strategy
- Cost optimization at scale (>50 jobs/day)
- Custom infrastructure requirements
- Full debugging and observability control needed

---

---

## LLMOps — Hybrid ML Pipeline

Beyond classical MLOps, this project implements a hybrid ML + LLM pipeline that combines RandomForest for churn prediction with Flan-T5-Large for natural language retention recommendations.

---

### Architecture

```
Customer features (19 columns)
        │
        ▼
RandomForest Serverless Endpoint
        │
        ├── churn_probability: 0.68
        ├── label: "Churn"
        └── top_features: [Contract, MonthlyCharges, tenure]
        │
        ▼
Prompt Manager (versioned templates)
        │
        ▼
Flan-T5-Large Endpoint (ml.m5.xlarge)
        │
        ▼
┌─────────────────────────────────────────────┐
│  Combined Response                          │
│  prediction:         1                      │
│  churn_probability:  0.68                   │
│  label:              "Churn"                │
│  risk_level:         "Medium"               │
│  explanation:        "Customer has high..." │
│  llm_recommendation: "a free upgrade"       │
│  model_pipeline:     "RF + Flan-T5-Large"  │
└─────────────────────────────────────────────┘
        │
        ▼
Langfuse Observability
(every call traced — input, output, latency)
```

---

### Components

#### 1. Hybrid Inference

Two endpoints work together per request:

| Step | Component | Output |
|------|-----------|--------|
| 1 | RandomForest (serverless) | churn_probability, top_features |
| 2 | Prompt Manager | compiled prompt from versioned template |
| 3 | Flan-T5-Large (ml.m5.xlarge) | natural language recommendation |
| 4 | Combined response | structured JSON with all fields |

**Sample output:**
```json
{
  "prediction": 1,
  "churn_probability": 0.6812,
  "label": "Churn",
  "risk_level": "Medium",
  "top_features": [
    {"feature": "Contract", "value": "month-to-month"},
    {"feature": "MonthlyCharges", "value": "$86"},
    {"feature": "tenure", "value": "3 months"}
  ],
  "explanation": "Customer has high churn risk (68%). Key factors: Contract=month-to-month, MonthlyCharges=$86, tenure=3 months.",
  "llm_recommendation": "a free upgrade to a higher plan",
  "model_pipeline": "RandomForest + Flan-T5-Large"
}
```

**Why hybrid?** RandomForest gives accurate, explainable predictions (ROC AUC 0.8445). Flan-T5 converts SHAP-derived insights into business-readable retention actions — giving both ML accuracy and business interpretability.

---

#### 2. Langfuse Observability

Every inference call traced with two observation types:

| Observation | Type | Captures |
|-------------|------|----------|
| `randomforest-prediction` | SPAN | features input, prediction output, latency |
| `flan-t5-recommendation` | GENERATION | prompt, model output, model parameters, latency |

**Dashboard:** https://cloud.langfuse.com → project `churn-prediction`

**What Langfuse provides that CloudWatch doesn't:**
- Full prompt + response logging per call
- LLM-specific cost estimation
- Token usage tracking
- Model parameter versioning
- Latency breakdown by component

**Implementation:** Non-fatal tracing — if Langfuse is unavailable, inference continues normally. Traces are flushed asynchronously after each request.

---

#### 3. Prompt Versioning

Version-controlled prompt templates with Langfuse as deployment registry:

```
llm/prompts/
├── v1/
│   ├── retention.txt    ← production (simple Q&A format)
│   └── engagement.txt   ← production
├── v2/
│   ├── retention.txt    ← staging (role-play format)
│   └── engagement.txt   ← staging
└── registry.json        ← maps versions to Langfuse names + labels
```

**Deployment workflow:**
```
Edit prompts/v2/retention.txt
    → python prompt_manager.py register_version("v2")
    → Langfuse: churn-retention-v2 (staging label)
    → Evaluate with evaluate_llm.py
    → If score improves: promote_to_production("v2", "retention")
    → Langfuse: churn-retention-v2 (production label)
    → Update registry.json: active_version = "v2"
```

**Current state:**

| Version | Type | Label | Score |
|---------|------|-------|-------|
| v1 | retention | **production** | 0.36 |
| v1 | engagement | **production** | 0.36 |
| v2 | retention | staging | 0.21 |
| v2 | engagement | staging | 0.21 |

v1 remains in production — evaluation confirmed it outperforms v2 for Flan-T5-Large.

---

#### 4. LLM Evaluation (DeepEval)

Automated quality scoring — no human review needed:

```
llm/evaluate_llm.py
├── 3 test cases (high/medium/low churn risk)
├── Rule-based relevancy scoring (keyword matching)
├── Actionability scoring (business-specific)
└── v1 vs v2 comparison with winner recommendation
```

**Evaluation metrics:**

| Metric | What it measures |
|--------|-----------------|
| Relevancy | Does output contain business-relevant keywords? |
| Actionability | Does output suggest a specific action? |
| Pass rate | % of outputs scoring above threshold (0.3) |
| Overall score | Average of relevancy + actionability |

**Results:**

| Metric | v1 (production) | v2 (staging) | Winner |
|--------|----------------|--------------|--------|
| Pass rate | **100%** | 100% | tie |
| Avg relevancy | **0.51** | 0.41 | v1 ✅ |
| Avg actionability | **0.20** | 0.00 | v1 ✅ |
| Overall score | **0.36** | 0.21 | v1 ✅ |

**Key finding:** v2 role-play format performs worse with Flan-T5-Large — the model echoes structured prompt elements instead of generating new recommendations. Larger models (Llama-70B, Claude) would likely reverse this finding.

---

#### 5. Fine-Tuning Pipeline (Implemented — Pending GPU Quota)

Dataset, training script, and job submission code complete:

```
llm/finetuning/
├── create_dataset.py      # generates 40 telecom Q&A pairs
├── train_data.json        # saved to S3
├── finetune.py            # HuggingFace Trainer, 5 epochs, lr=3e-4
└── run_finetuning_job.py  # SageMaker Training Job submission
```

**Dataset:** 40 telecom retention Q&A pairs covering high/medium/low churn scenarios with price and tenure variations.

**Training config:**
```python
TrainingArguments(
    num_train_epochs=5,
    per_device_train_batch_size=4,
    learning_rate=3e-4,
    eval_strategy="epoch",
    load_best_model_at_end=True,
)
```

**Status:** Code complete. Infrastructure blocker documented:
- Current ECR image (`churn-mlops:inference`) designed for RandomForest training
- Fine-tuning requires separate image with `transformers`, `torch`, `sentencepiece`
- Solution: split into `churn-mlops:inference` and `churn-mlops:finetune` images
- GPU quota request submitted for `ml.g4dn.xlarge`

---

### LLMOps File Structure

```
llm/
├── deploy_flan_t5.py              # JumpStart deployment script
├── hybrid_inference.py            # RF + Flan-T5 without tracing
├── hybrid_inference_langfuse.py   # RF + Flan-T5 with full Langfuse tracing
├── prompt_manager.py              # Prompt versioning + Langfuse registry
├── evaluate_llm.py                # Automated evaluation suite
├── prompts/
│   ├── registry.json              # Version → Langfuse name mapping
│   ├── v1/
│   │   ├── retention.txt          # Production prompt
│   │   └── engagement.txt         # Production prompt
│   └── v2/
│       ├── retention.txt          # Staging prompt
│       └── engagement.txt         # Staging prompt
└── finetuning/
    ├── create_dataset.py          # Dataset generation
    ├── train_data.json            # 40 Q&A pairs
    ├── finetune.py                # HuggingFace fine-tuning script
    └── run_finetuning_job.py      # SageMaker Training Job
```

---

### Model Comparison: Base vs Fine-tuned (Expected)

| Customer Type | Base Flan-T5-Large | Fine-tuned (Expected) |
|--------------|-------------------|-----------------------|
| High risk, month-to-month | `a free upgrade` | `Offer 20% discount on annual contract` |
| Low risk, 72 months | `a customer retention program` | `Award platinum loyalty status` |
| Medium risk, fiber optic | `a customer retention program` | `Add free tech support bundle` |

Fine-tuning on domain-specific telecom data expected to improve specificity and actionability scores from 0.36 → 0.65+.

---

### LLMOps vs Classical MLOps: Key Differences

| Concern | Classical MLOps | LLMOps |
|---------|----------------|--------|
| Model evaluation | ROC AUC, F1, accuracy | Relevancy, faithfulness, actionability |
| Versioning | Model artifacts in S3 | Prompt templates + model weights |
| Monitoring | Data drift, prediction drift | Hallucination rate, output quality |
| Observability | CloudWatch metrics | LLM traces (Langfuse) |
| Deployment | Endpoint update | Prompt promotion (zero downtime) |
| Testing | Unit tests, integration tests | LLM evaluation suites |
| Cost unit | $/Training Job | $/1K tokens |

---

### Why Flan-T5 Instead of GPT-4 / Claude API

| Concern | Flan-T5 (self-hosted) | GPT-4 / Claude (API) |
|---------|----------------------|----------------------|
| Data privacy | Customer data stays in VPC | Sent to third-party servers |
| Cost at scale | Fixed endpoint cost | $0.03-0.12 per 1K tokens |
| Latency | ~2s warm | ~1-3s (network dependent) |
| Quality | Limited (780M params) | Excellent |
| GDPR compliance | Full control | Requires DPA agreement |
| Customization | Fine-tunable | Prompt engineering only |

**For a real telecom with GDPR obligations and millions of customers:** self-hosted fine-tuned model wins on privacy and cost. For a startup with <1M predictions/month: Claude API wins on quality and time-to-production.

---

### Running the LLMOps Stack

```bash
# Deploy Flan-T5 endpoint (~8 minutes)
python llm/deploy_flan_t5.py

# Run hybrid inference (RF + Flan-T5)
python llm/hybrid_inference.py

# Run with Langfuse tracing
python llm/hybrid_inference_langfuse.py

# Manage prompts
python llm/prompt_manager.py

# Evaluate LLM quality
python llm/evaluate_llm.py

# Delete endpoint when done (~$0.46/hr)
aws sagemaker delete-endpoint \
  --endpoint-name churn-flan-t5-endpoint \
  --region us-east-1
```

---

## Author

**Himanshu Singh (Heman)**
Cloud DevOps & AI Engineer | 3+ years experience

- GitHub: [@Himanshu9001](https://github.com/Himanshu9001)
- Stack: AWS, Azure, Kubernetes, Terraform, Docker, MLflow, SageMaker, LangChain

**Related Projects:**
- [Self-Built EKS MLOps Stack](https://github.com/Himanshu9001/MLOps-Projects) — same use case, self-managed infrastructure
- [Azure Databricks AI Platform](https://github.com/Himanshu9001) — multimodal product catalog enrichment with GPT-4

---

*Built May 2026 | AWS SageMaker SDK v2.257.3 | Python 3.9 | scikit-learn 1.2.1*