# Hybrid ML inference — RandomForest + Flan-T5
# RandomForest: fast, accurate churn probability prediction
# Flan-T5: natural language explanation of prediction
# Combined response gives both accuracy and interpretability

import boto3
import json
import numpy as np

runtime  = boto3.client("sagemaker-runtime", region_name="us-east-1")

RF_ENDPOINT   = "churn-prediction-serverless"
FLAN_ENDPOINT = "churn-flan-t5-endpoint"

# Feature names matching training order
FEATURE_NAMES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges"
]

# Contract type mapping for human-readable output
CONTRACT_MAP   = {0: "Month-to-month", 1: "One year", 2: "Two year"}
INTERNET_MAP   = {0: "DSL", 1: "Fiber optic", 2: "No"}
PAYMENT_MAP    = {0: "Bank transfer", 1: "Credit card", 2: "Electronic check", 3: "Mailed check"}


def get_churn_prediction(features_csv):
    """Get churn probability from RandomForest serverless endpoint."""
    response = runtime.invoke_endpoint(
        EndpointName=RF_ENDPOINT,
        ContentType="text/csv",
        Accept="application/json",
        Body=features_csv
    )
    result = json.loads(response["Body"].read())
    return result[0]["prediction"], result[0]["churn_probability"]


def get_top_features(features, n=3):
    """Extract top N most important features for explanation."""
    # Simplified SHAP-like importance based on known feature importances
    # In production — call Clarify endpoint for real SHAP values
    feature_values = features.split(",")
    importance_order = [14, 17, 4, 18, 8, 11, 15, 16, 7]  # Contract, MonthlyCharges, tenure...

    top_features = []
    for idx in importance_order[:n]:
        name  = FEATURE_NAMES[idx]
        value = float(feature_values[idx])

        # Human-readable values
        if name == "Contract":
            readable = CONTRACT_MAP.get(int(value), str(value))
        elif name == "InternetService":
            readable = INTERNET_MAP.get(int(value), str(value))
        elif name == "PaymentMethod":
            readable = PAYMENT_MAP.get(int(value), str(value))
        elif name in ["MonthlyCharges", "TotalCharges"]:
            readable = f"${value:.2f}"
        elif name == "tenure":
            readable = f"{int(value)} months"
        else:
            readable = "Yes" if value == 1 else "No"

        top_features.append({"feature": name, "value": readable})

    return top_features


def generate_explanation(churn_probability, top_features, prediction):
    """Generate natural language explanation using Flan-T5."""
    label     = "high" if churn_probability > 0.5 else "low"
    features_text = ", ".join([f"{f['feature']}={f['value']}" for f in top_features])

    prompt = f"""Customer churn analysis: probability={churn_probability:.0%}, risk={label}.
Key factors: {features_text}.
In one sentence, explain why this customer {"will" if prediction == 1 else "will not"} churn:"""

    payload = {
        "text_inputs": prompt,
        "max_length": 80,
        "temperature": 0.3,
        "num_return_sequences": 1,
    }

    response = runtime.invoke_endpoint(
        EndpointName=FLAN_ENDPOINT,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload)
    )

    result = json.loads(response["Body"].read())
    # JumpStart Flan-T5 returns list of generated texts
    if isinstance(result, list):
        return result[0].get("generated_text", "Explanation unavailable")
    return result.get("generated_text", "Explanation unavailable")


def hybrid_predict(features_csv):
    """
    Full hybrid inference pipeline:
    1. RandomForest → churn probability (fast, accurate)
    2. Extract top features for context
    3. Flan-T5 → natural language explanation
    4. Return combined structured response
    """
    # Step 1 — RandomForest prediction
    prediction, probability = get_churn_prediction(features_csv)

    # Step 2 — Extract top contributing features
    top_features = get_top_features(features_csv, n=3)

    # Step 3 — Generate explanation
    explanation = generate_explanation(probability, top_features, prediction)

    # Step 4 — Combined response
    return {
        "prediction":        prediction,
        "churn_probability": probability,
        "label":             "Churn" if prediction == 1 else "No Churn",
        "risk_level":        "High" if probability > 0.7 else "Medium" if probability > 0.4 else "Low",
        "top_features":      top_features,
        "explanation":       explanation,
        "model_pipeline":    "RandomForest (prediction) + Flan-T5-Base (explanation)"
    }


if __name__ == "__main__":
    # Sample customers
    samples = [
        # High churn risk — month-to-month, high charges, short tenure
        ("High risk customer", "0,0,1,0,3,1,0,1,0,0,0,0,0,0,0,1,2,85.5,256.5"),
        # Low churn risk — two year contract, long tenure
        ("Low risk customer",  "1,0,1,1,72,1,1,0,1,1,1,1,1,1,2,0,0,45.5,3276.0"),
        # Medium risk
        ("Medium risk customer", "0,1,0,0,12,1,0,1,0,1,0,0,1,0,0,1,2,65.5,786.0"),
    ]

    for name, features in samples:
        print(f"\n{'='*60}")
        print(f"Customer: {name}")
        print(f"{'='*60}")

        result = hybrid_predict(features)

        print(f"Prediction:    {result['label']}")
        print(f"Probability:   {result['churn_probability']:.1%}")
        print(f"Risk Level:    {result['risk_level']}")
        print(f"Top Features:")
        for f in result['top_features']:
            print(f"  - {f['feature']}: {f['value']}")
        print(f"Explanation:   {result['explanation']}")
        print(f"Pipeline:      {result['model_pipeline']}")
