# Evaluation script for SageMaker Pipeline evaluate step
# Loads model + test data, computes metrics, saves evaluation.json
# PropertyFile reads this JSON to make conditional register decision

import joblib
import json
import os
import tarfile
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

# Paths mounted by SageMaker Processing Job
MODEL_DIR = "/opt/ml/processing/model"
TEST_DIR  = "/opt/ml/processing/test"
EVAL_DIR  = "/opt/ml/processing/evaluation"

os.makedirs(EVAL_DIR, exist_ok=True)

# Untar model artifact — SageMaker stores model as model.tar.gz
model_tar = os.path.join(MODEL_DIR, "model.tar.gz")
with tarfile.open(model_tar) as tar:
    tar.extractall(MODEL_DIR)

# Load model
model = joblib.load(os.path.join(MODEL_DIR, "model.joblib"))

# Load test data
test_df = pd.read_csv(os.path.join(TEST_DIR, "test.csv"))
X_test  = test_df.drop(columns=["Churn"])
y_test  = test_df["Churn"]

# Evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

auc      = roc_auc_score(y_test, y_prob)
accuracy = accuracy_score(y_test, y_pred)
f1       = f1_score(y_test, y_pred)

print(f"roc_auc={auc:.4f}")
print(f"accuracy={accuracy:.4f}")
print(f"f1_score={f1:.4f}")

# Save evaluation report — PropertyFile reads this for conditional step
evaluation = {
    "binary_classification_metrics": {
        "auc":      {"value": auc,      "standard_deviation": "NaN"},
        "accuracy": {"value": accuracy, "standard_deviation": "NaN"},
        "f1":       {"value": f1,       "standard_deviation": "NaN"},
    }
}

eval_path = os.path.join(EVAL_DIR, "evaluation.json")
with open(eval_path, "w") as f:
    json.dump(evaluation, f, indent=2)

print(f"Evaluation report saved: {eval_path}")
