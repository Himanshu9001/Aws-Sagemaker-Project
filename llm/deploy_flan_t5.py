# Deploy Flan-T5-Base from SageMaker JumpStart
# Runs on ml.m5.xlarge — no GPU needed
# Used for generating natural language churn explanations

import boto3
import sagemaker
from sagemaker.jumpstart.model import JumpStartModel
from sagemaker.session import Session

session  = Session()
role     = "arn:aws:iam::011528270076:role/sagemaker-churn-execution-role"
region   = "us-east-1"

print("Deploying Flan-T5-Base from JumpStart...")
print("Instance: ml.m5.xlarge (CPU — no GPU needed)")

# Deploy Flan-T5-Base
model = JumpStartModel(
    model_id="huggingface-text2text-flan-t5-base",
    role=role,
    sagemaker_session=session,
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.xlarge",
    endpoint_name="churn-flan-t5-endpoint",
)

print(f"Flan-T5 endpoint deployed: churn-flan-t5-endpoint")

# Test with a sample churn explanation prompt
test_payload = {
    "text_inputs": """Given the following customer information, explain in one sentence 
why this customer is at high risk of churning:
- Contract type: Month-to-month
- Monthly charges: $85.50 (high)
- Tenure: 3 months (new customer)
- Tech support: No
- Online security: No
- Churn probability: 78%

Explanation:""",
    "max_length": 100,
    "temperature": 0.7,
}

import json
response = predictor.predict(test_payload)
print(f"\nTest explanation: {response}")
print(f"\nEndpoint ready: churn-flan-t5-endpoint")
