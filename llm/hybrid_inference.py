# Hybrid ML Pipeline — RandomForest + Flan-T5-Large
# RandomForest: churn probability prediction (fast, accurate)
# Flan-T5-Large: natural language retention recommendation
# Combined response: probability + label + explanation + action

import boto3
import json

runtime = boto3.client("sagemaker-runtime", region_name="us-east-1")

RF_ENDPOINT   = "churn-prediction-serverless"
FLAN_ENDPOINT = "churn-flan-t5-endpoint"

FEATURE_NAMES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges"
]

CONTRACT_MAP = {0: "month-to-month", 1: "one-year", 2: "two-year"}
INTERNET_MAP = {0: "DSL", 1: "Fiber optic", 2: "No internet"}
PAYMENT_MAP  = {0: "Bank transfer", 1: "Credit card",
                2: "Electronic check", 3: "Mailed check"}


def get_churn_prediction(features_csv):
    response = runtime.invoke_endpoint(
        EndpointName=RF_ENDPOINT,
        ContentType="text/csv",
        Accept="application/json",
        Body=features_csv
    )
    result = json.loads(response["Body"].read())
    return result[0]["prediction"], result[0]["churn_probability"]


def get_top_features(features_csv, n=3):
    values = features_csv.split(",")
    importance_order = [14, 17, 4, 11, 8, 15, 16]

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
        elif name in ["MonthlyCharges", "TotalCharges"]:
            readable = f"${value:.0f}"
        elif name == "tenure":
            readable = f"{int(value)} months"
        else:
            readable = "Yes" if value == 1 else "No"
        top.append({"feature": name, "value": readable})
    return top


def get_llm_recommendation(probability, top_features, prediction):
    contract = next((f["value"] for f in top_features if f["feature"] == "Contract"), "unknown")
    charges  = next((f["value"] for f in top_features if f["feature"] == "MonthlyCharges"), "unknown")
    tenure   = next((f["value"] for f in top_features if f["feature"] == "tenure"), "unknown")

    if prediction == 1:
        prompt = f"What is the best offer to prevent churn for a telecom customer on {contract} contract paying {charges} monthly with {tenure} tenure and {probability:.0%} churn risk?"
    else:
        prompt = f"What engagement strategy should a telecom company use for a loyal customer with {tenure} tenure and only {probability:.0%} churn risk?"

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 50,
            "temperature": 0.1,
            "do_sample": False,
            "repetition_penalty": 1.5,
        }
    }

    response = runtime.invoke_endpoint(
        EndpointName=FLAN_ENDPOINT,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload)
    )
    result = json.loads(response["Body"].read())
    return result[0]["generated_text"]


def hybrid_predict(features_csv):
    prediction, probability = get_churn_prediction(features_csv)
    top_features = get_top_features(features_csv, n=3)
    recommendation = get_llm_recommendation(probability, top_features, prediction)

    top_names = [f["feature"] for f in top_features]
    factors = ", ".join([f"{f['feature']}={f['value']}" for f in top_features])
    risk_word = "high" if probability > 0.5 else "low"
    explanation = f"Customer has {risk_word} churn risk ({probability:.0%}). Key factors: {factors}."

    return {
        "prediction":        int(prediction),
        "churn_probability": round(float(probability), 4),
        "label":             "Churn" if prediction == 1 else "No Churn",
        "risk_level":        "High" if probability > 0.7 else "Medium" if probability > 0.4 else "Low",
        "top_features":      top_features,
        "explanation":       explanation,
        "llm_recommendation": recommendation,
        "model_pipeline":    "RandomForest (prediction) + Flan-T5-Large (recommendation)"
    }


if __name__ == "__main__":
    samples = [
        ("High risk — month-to-month, high charges, new customer",
         "0,0,1,0,3,1,0,1,0,0,0,0,0,0,0,1,2,85.5,256.5"),
        ("Low risk — two year contract, long tenure",
         "1,0,1,1,72,1,1,0,1,1,1,1,1,1,2,0,0,45.5,3276.0"),
        ("Medium risk — 12 months, fiber optic",
         "0,1,0,0,12,1,0,1,0,1,0,0,1,0,0,1,2,65.5,786.0"),
    ]

    for name, features in samples:
        print(f"\n{'='*60}")
        print(f"Customer: {name}")
        print(f"{'='*60}")
        result = hybrid_predict(features)
        print(f"Prediction:       {result['label']} ({result['churn_probability']:.1%})")
        print(f"Risk Level:       {result['risk_level']}")
        print(f"Top Features:     {result['top_features']}")
        print(f"Explanation:      {result['explanation']}")
        print(f"LLM Recommendation: {result['llm_recommendation']}")
        print(f"Pipeline:         {result['model_pipeline']}")
