import boto3
import json

runtime       = boto3.client("sagemaker-runtime", region_name="us-east-1")
endpoint_name = "churn-prediction-endpoint"

# Sample: gender=0, tenure=12, Partner=1, Dependents=0...
sample = "0,12,1,0,1,1,2,1,1,1,1,1,0,1,1,3,1,65.5,850.5"

print(f"Invoking endpoint: {endpoint_name}")
print(f"Input: {sample}")

# Test 1 — CSV response
response = runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType="text/csv",
    Accept="text/csv",
    Body=sample
)
result = response["Body"].read().decode("utf-8")
print(f"\nCSV response: {result}")

# Test 2 — JSON response
response = runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType="text/csv",
    Accept="application/json",
    Body=sample
)
result = json.loads(response["Body"].read().decode("utf-8"))
print(f"JSON response: {json.dumps(result, indent=2)}")
print(f"\nPrediction: {result[0]['label']}")
print(f"Churn probability: {result[0]['churn_probability']}")
