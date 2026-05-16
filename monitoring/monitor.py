# SageMaker Model Monitor setup
# Enables data capture, computes baseline, creates monitoring schedule

import boto3
import sagemaker
from sagemaker.session import Session
from sagemaker.model_monitor import DefaultModelMonitor
from sagemaker.model_monitor.dataset_format import DatasetFormat

session   = Session()
sm_client = boto3.client("sagemaker", region_name="us-east-1")
role      = "arn:aws:iam::011528270076:role/sagemaker-churn-execution-role"
bucket    = "churn-sagemaker-artifacts"
endpoint  = "churn-prediction-endpoint"

# ── Step 1: Enable Data Capture via new endpoint config ──────────────────────
import time
existing = sm_client.describe_endpoint(EndpointName=endpoint)
existing_config = sm_client.describe_endpoint_config(
    EndpointConfigName=existing["EndpointConfigName"]
)

new_config_name = f"churn-endpoint-config-monitor-{int(time.time())}"
sm_client.create_endpoint_config(
    EndpointConfigName=new_config_name,
    ProductionVariants=existing_config["ProductionVariants"],
    DataCaptureConfig={
        "EnableCapture": True,
        "InitialSamplingPercentage": 100,
        "DestinationS3Uri": f"s3://{bucket}/monitoring/data-capture",
        "CaptureOptions": [
            {"CaptureMode": "Input"},
            {"CaptureMode": "Output"}
        ],
        "CaptureContentTypeHeader": {
            "CsvContentTypes": ["text/csv"],
            "JsonContentTypes": ["application/json"]
        }
    }
)
print(f"New endpoint config created: {new_config_name}")

# Update endpoint to use new config with data capture
sm_client.update_endpoint(
    EndpointName=endpoint,
    EndpointConfigName=new_config_name
)
print(f"Endpoint updating with data capture enabled...")

# Wait for endpoint update
waiter = sm_client.get_waiter("endpoint_in_service")
waiter.wait(EndpointName=endpoint, WaiterConfig={"Delay": 30, "MaxAttempts": 20})
print(f"Data capture enabled on: {endpoint}")

# ── Step 2: Compute baseline statistics ─────────────────────────────────────
monitor = DefaultModelMonitor(
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
    volume_size_in_gb=20,
    max_runtime_in_seconds=3600,
    sagemaker_session=session,
)

print("Computing baseline statistics from training data...")
monitor.suggest_baseline(
    baseline_dataset=f"s3://{bucket}/data/processed/train/train.csv",
    dataset_format=DatasetFormat.csv(header=True),
    output_s3_uri=f"s3://{bucket}/monitoring/baseline",
    wait=True,
    logs=False,
)
print(f"Baseline computed: s3://{bucket}/monitoring/baseline")

# ── Step 3: Create hourly monitoring schedule ────────────────────────────────
try:
    monitor.create_monitoring_schedule(
        monitor_schedule_name="churn-monitor-schedule",
        endpoint_input=endpoint,
        output_s3_uri=f"s3://{bucket}/monitoring/reports",
        statistics=monitor.baseline_statistics(),
        constraints=monitor.suggested_constraints(),
        schedule_cron_expression="cron(0 * ? * * *)",
    )
    print("Monitoring schedule created: churn-monitor-schedule")
except Exception as e:
    if "already exists" in str(e):
        print("Monitoring schedule already exists — skipping")
    else:
        raise e

print("\nModel Monitor setup complete!")
print(f"Data capture: s3://{bucket}/monitoring/data-capture")
print(f"Baseline:     s3://{bucket}/monitoring/baseline")
print(f"Reports:      s3://{bucket}/monitoring/reports")
