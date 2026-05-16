import boto3
import json
import time
import os
from langfuse import Langfuse

# Set credentials before creating client
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-d298883b-9e0a-46f3-b519-d7b1f1297af1"
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-8238ff79-2ba7-46f7-97d8-f6e11cdc46df"
os.environ["LANGFUSE_HOST"]       = "https://cloud.langfuse.com"

runtime  = boto3.client("sagemaker-runtime", region_name="us-east-1")

# Import prompt manager for versioned prompts
import sys
sys.path.insert(0, "/home/sagemaker-user/Aws-Sagemaker-Project/llm")
from prompt_manager import compile_prompt
langfuse = Langfuse(
    public_key="pk-lf-d298883b-9e0a-46f3-b519-d7b1f1297af1",
    secret_key="sk-lf-8238ff79-2ba7-46f7-97d8-f6e11cdc46df",
    host="https://cloud.langfuse.com"
)

RF_ENDPOINT   = "churn-prediction-serverless"
FLAN_ENDPOINT = "churn-flan-t5-endpoint"

FEATURE_NAMES = [
    "gender","SeniorCitizen","Partner","Dependents","tenure",
    "PhoneService","MultipleLines","InternetService","OnlineSecurity",
    "OnlineBackup","DeviceProtection","TechSupport","StreamingTV",
    "StreamingMovies","Contract","PaperlessBilling","PaymentMethod",
    "MonthlyCharges","TotalCharges"
]
CONTRACT_MAP = {0:"month-to-month",1:"one-year",2:"two-year"}
INTERNET_MAP = {0:"DSL",1:"Fiber optic",2:"No internet"}
PAYMENT_MAP  = {0:"Bank transfer",1:"Credit card",
                2:"Electronic check",3:"Mailed check"}


def get_top_features(features_csv, n=3):
    values = features_csv.split(",")
    importance_order = [14,17,4,11,8,15,16]
    top = []
    for idx in importance_order[:n]:
        name  = FEATURE_NAMES[idx]
        value = float(values[idx])
        if name == "Contract":
            readable = CONTRACT_MAP.get(int(value), str(value))
        elif name == "InternetService":
            readable = INTERNET_MAP.get(int(value), str(value))
        elif name == "PaymentMethod":
            readable = PAYMENT_MAP.get(int(value), str(value))
        elif name in ["MonthlyCharges","TotalCharges"]:
            readable = f"${value:.0f}"
        elif name == "tenure":
            readable = f"{int(value)} months"
        else:
            readable = "Yes" if value == 1 else "No"
        top.append({"feature": name, "value": readable})
    return top


def hybrid_predict_traced(features_csv, customer_id=None):
    # Step 1 — RandomForest prediction
    with langfuse.start_as_current_observation(
        name="randomforest-prediction",
        as_type="span",
        input={"features": features_csv},
    ):
        start    = time.time()
        response = runtime.invoke_endpoint(
            EndpointName=RF_ENDPOINT, ContentType="text/csv",
            Accept="application/json", Body=features_csv
        )
        rf_result   = json.loads(response["Body"].read())
        prediction  = rf_result[0]["prediction"]
        probability = rf_result[0]["churn_probability"]
        langfuse.update_current_span(
            output={"prediction": prediction, "probability": probability},
            metadata={"latency_ms": round((time.time()-start)*1000)}
        )

    # Top features
    top_features = get_top_features(features_csv)
    contract = next((f["value"] for f in top_features if f["feature"]=="Contract"), "unknown")
    charges  = next((f["value"] for f in top_features if f["feature"]=="MonthlyCharges"), "unknown")
    tenure   = next((f["value"] for f in top_features if f["feature"]=="tenure"), "unknown")

    if prediction == 1:
        prompt, _ = compile_prompt("churn-retention-v1", {
            "contract": contract, "charges": charges,
            "tenure": tenure, "probability": f"{probability:.0%}"
        })
        prompt = prompt or f"What is the best offer to prevent churn for a telecom customer on {contract} contract paying {charges} monthly with {tenure} tenure and {probability:.0%} churn risk?"
    else:
        prompt, _ = compile_prompt("churn-engagement-v1", {
            "tenure": tenure, "probability": f"{probability:.0%}"
        })
        prompt = prompt or f"What engagement strategy should a telecom company use for a loyal customer with {tenure} tenure and only {probability:.0%} churn risk?"

    # Step 2 — Flan-T5 LLM generation
    with langfuse.start_as_current_observation(
        name="flan-t5-recommendation",
        as_type="generation",
        input=prompt,
        model="flan-t5-large",
    ):
        start   = time.time()
        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens":50,"temperature":0.1,
                           "do_sample":False,"repetition_penalty":1.5}
        }
        response = runtime.invoke_endpoint(
            EndpointName=FLAN_ENDPOINT, ContentType="application/json",
            Accept="application/json", Body=json.dumps(payload)
        )
        llm_result     = json.loads(response["Body"].read())
        recommendation = llm_result[0]["generated_text"]
        langfuse.update_current_generation(
            output=recommendation,
            metadata={"latency_ms": round((time.time()-start)*1000)}
        )

    # Final result
    factors     = ", ".join([f"{f['feature']}={f['value']}" for f in top_features])
    risk_word   = "high" if probability > 0.5 else "low"
    explanation = f"Customer has {risk_word} churn risk ({probability:.0%}). Key factors: {factors}."

    result = {
        "prediction":         int(prediction),
        "churn_probability":  round(float(probability), 4),
        "label":              "Churn" if prediction == 1 else "No Churn",
        "risk_level":         "High" if probability > 0.7 else "Medium" if probability > 0.4 else "Low",
        "top_features":       top_features,
        "explanation":        explanation,
        "llm_recommendation": recommendation,
        "model_pipeline":     "RandomForest + Flan-T5-Large",
    }

    langfuse.flush()
    return result


if __name__ == "__main__":
    samples = [
        ("customer-001", "0,0,1,0,3,1,0,1,0,0,0,0,0,0,0,1,2,85.5,256.5"),
        ("customer-002", "1,0,1,1,72,1,1,0,1,1,1,1,1,1,2,0,0,45.5,3276.0"),
        ("customer-003", "0,1,0,0,12,1,0,1,0,1,0,0,1,0,0,1,2,65.5,786.0"),
    ]

    for customer_id, features in samples:
        print(f"\n{'='*50}")
        print(f"Customer: {customer_id}")
        result = hybrid_predict_traced(features, customer_id=customer_id)
        print(f"Label:          {result['label']} ({result['churn_probability']:.1%})")
        print(f"Risk:           {result['risk_level']}")
        print(f"Explanation:    {result['explanation']}")
        print(f"Recommendation: {result['llm_recommendation']}")

    print(f"\nView traces: https://cloud.langfuse.com")
