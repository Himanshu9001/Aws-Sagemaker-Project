# CloudWatch Dashboard — Churn MLOps Observability
# Covers endpoint metrics, training costs, and Spot savings
# Run once to create dashboard — updates automatically

import boto3
import json

cw_client = boto3.client("cloudwatch", region_name="us-east-1")
endpoint  = "churn-prediction-endpoint"
dashboard_name = "ChurnMLOps-Dashboard"

dashboard_body = {
    "widgets": [
        # Row 1 — Endpoint Health
        {
            "type": "metric",
            "x": 0, "y": 0, "width": 8, "height": 6,
            "properties": {
                "title": "Endpoint Invocations",
                "metrics": [[
                    "AWS/SageMaker",
                    "Invocations",
                    "EndpointName", endpoint,
                    "VariantName", "AllTraffic"
                ]],
                "period": 300,
                "stat": "Sum",
                "view": "timeSeries",
                "region": "us-east-1"
            }
        },
        {
            "type": "metric",
            "x": 8, "y": 0, "width": 8, "height": 6,
            "properties": {
                "title": "Endpoint Latency (ms)",
                "metrics": [
                    ["AWS/SageMaker", "ModelLatency",
                     "EndpointName", endpoint, "VariantName", "AllTraffic"],
                    ["AWS/SageMaker", "OverheadLatency",
                     "EndpointName", endpoint, "VariantName", "AllTraffic"],
                ],
                "period": 300,
                "stat": "Average",
                "view": "timeSeries",
                "region": "us-east-1"
            }
        },
        {
            "type": "metric",
            "x": 16, "y": 0, "width": 8, "height": 6,
            "properties": {
                "title": "Endpoint Errors",
                "metrics": [
                    ["AWS/SageMaker", "Invocation4XXErrors",
                     "EndpointName", endpoint, "VariantName", "AllTraffic"],
                    ["AWS/SageMaker", "Invocation5XXErrors",
                     "EndpointName", endpoint, "VariantName", "AllTraffic"],
                ],
                "period": 300,
                "stat": "Sum",
                "view": "timeSeries",
                "region": "us-east-1"
            }
        },
        # Row 2 — Instance Metrics
        {
            "type": "metric",
            "x": 0, "y": 6, "width": 8, "height": 6,
            "properties": {
                "title": "CPU Utilization %",
                "metrics": [[
                    "/aws/sagemaker/Endpoints",
                    "CPUUtilization",
                    "EndpointName", endpoint,
                    "VariantName", "AllTraffic"
                ]],
                "period": 300,
                "stat": "Average",
                "view": "timeSeries",
                "region": "us-east-1"
            }
        },
        {
            "type": "metric",
            "x": 8, "y": 6, "width": 8, "height": 6,
            "properties": {
                "title": "Memory Utilization %",
                "metrics": [[
                    "/aws/sagemaker/Endpoints",
                    "MemoryUtilization",
                    "EndpointName", endpoint,
                    "VariantName", "AllTraffic"
                ]],
                "period": 300,
                "stat": "Average",
                "view": "timeSeries",
                "region": "us-east-1"
            }
        },
        {
            "type": "metric",
            "x": 16, "y": 6, "width": 8, "height": 6,
            "properties": {
                "title": "Instance Count (Auto-scaling)",
                "metrics": [[
                    "AWS/SageMaker",
                    "InvocationsPerInstance",
                    "EndpointName", endpoint,
                    "VariantName", "AllTraffic"
                ]],
                "period": 300,
                "stat": "Average",
                "view": "timeSeries",
                "region": "us-east-1"
            }
        },
        # Row 3 — Cost Summary (text widget)
        {
            "type": "text",
            "x": 0, "y": 12, "width": 24, "height": 4,
            "properties": {
                "markdown": """## Cost Summary — Churn MLOps Project
| Component | Instance | Billable Time | Spot Savings | Cost |
|---|---|---|---|---|
| Processing Job (Phase 2) | ml.m5.large | ~120s | N/A | ~$0.001 |
| Training Job (Phase 3) | ml.m5.large | 55s | 63.3% | ~$0.001 |
| HPO — 20 trials (Phase 4) | ml.m5.large | ~1100s total | ~63% | ~$0.02 |
| Pipeline execution (Phase 7) | ml.m5.large | ~300s | 63% | ~$0.005 |
| Endpoint (Phase 6+) | ml.m5.large | ongoing | N/A | ~$0.13/hr |
| Model Monitor baseline | ml.m5.large | ~180s | N/A | ~$0.002 |

**Key insight:** Spot instances saved 63%+ on all training workloads. Endpoint is the primary ongoing cost driver."""
            }
        }
    ]
}

cw_client.put_dashboard(
    DashboardName=dashboard_name,
    DashboardBody=json.dumps(dashboard_body)
)

print(f"Dashboard created: {dashboard_name}")
print(f"View at: https://us-east-1.console.aws.amazon.com/cloudwatch/home?region=us-east-1#dashboards:name={dashboard_name}")
