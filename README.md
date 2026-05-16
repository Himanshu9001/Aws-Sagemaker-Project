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