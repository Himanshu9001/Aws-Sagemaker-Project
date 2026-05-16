import joblib
import os
import json
import numpy as np
import pandas as pd
from io import StringIO

def model_fn(model_dir):
    """Load model from /opt/ml/model/ — called once at container startup."""
    model_path = os.path.join(model_dir, "model.joblib")
    return joblib.load(model_path)

def input_fn(request_body, content_type="text/csv"):
    """Parse incoming request — convert CSV string to numpy array."""
    if content_type == "text/csv":
        df = pd.read_csv(StringIO(request_body), header=None)
        return df.values
    raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    """Run inference — return both class prediction and probability."""
    prediction  = model.predict(input_data)
    probability = model.predict_proba(input_data)[:, 1]
    return np.column_stack([prediction, probability])

def output_fn(prediction, accept="text/csv"):
    """Format response — supports both text/csv and application/json."""
    if accept == "text/csv":
        output = []
        for pred, prob in prediction:
            output.append(f"{int(pred)},{prob:.4f}")
        return "\n".join(output), "text/csv"
    elif accept == "application/json":
        output = []
        for pred, prob in prediction:
            output.append({
                "prediction": int(pred),
                "churn_probability": round(float(prob), 4),
                "label": "Churn" if int(pred) == 1 else "No Churn"
            })
        return json.dumps(output), "application/json"
    raise ValueError(f"Unsupported accept type: {accept}")
