import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
import os
import json
import argparse


import pickle

# SageMaker Spot checkpoint paths
CHECKPOINT_DIR = "/opt/ml/checkpoints"

def save_checkpoint(model, metrics, epoch, checkpoint_dir):
    """Save model checkpoint to local dir — SageMaker syncs to S3 automatically."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = {
        "model": model,
        "metrics": metrics,
        "epoch": epoch
    }
    path = os.path.join(checkpoint_dir, f"checkpoint_{epoch}.pkl")
    with open(path, "wb") as f:
        pickle.dump(checkpoint, f)
    logger.info(f"Checkpoint saved: {path}")
    return path

def load_latest_checkpoint(checkpoint_dir):
    """Resume from latest checkpoint if exists — handles Spot interruption recovery."""
    if not os.path.exists(checkpoint_dir):
        return None
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pkl")]
    if not checkpoints:
        return None
    latest = sorted(checkpoints)[-1]
    path = os.path.join(checkpoint_dir, latest)
    with open(path, "rb") as f:
        checkpoint = pickle.load(f)
    logger.info(f"Resumed from checkpoint: {path}")
    return checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

INPUT_TRAIN = "/opt/ml/input/data/train/train.csv"
INPUT_TEST  = "/opt/ml/input/data/test/test.csv"
MODEL_DIR   = "/opt/ml/model"
OUTPUT_DIR  = "/opt/ml/output/data/plots"

def load_data():
    logger.info("Loading data from SageMaker input channels...")
    train_df = pd.read_csv(INPUT_TRAIN)
    test_df  = pd.read_csv(INPUT_TEST)
    X_train  = train_df.drop(columns=["Churn"])
    y_train  = train_df["Churn"]
    X_test   = test_df.drop(columns=["Churn"])
    y_test   = test_df["Churn"]
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "f1_score":  f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "roc_auc":   roc_auc_score(y_test, y_prob)
    }
    return metrics, y_pred, y_prob

def plot_confusion_matrix(y_test, y_pred, output_dir):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"])
    plt.title("Confusion Matrix"); plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path); plt.close()

def plot_roc_curve(y_test, y_prob, output_dir):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.title("ROC Curve"); plt.legend(); plt.tight_layout()
    path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(path); plt.close()

def plot_feature_importance(model, feature_names, output_dir):
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    plt.figure(figsize=(10, 8))
    sns.barplot(data=importance_df, x="importance", y="feature", palette="viridis")
    plt.title("Feature Importance"); plt.tight_layout()
    path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(path); plt.close()

def train(args):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    X_train, X_test, y_train, y_test = load_data()

    logger.info(f"Training RandomForest: n_estimators={args.n_estimators}, max_depth={args.max_depth}")
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        random_state=42,
        n_jobs=-1
    )
    # Check for existing checkpoint — resume if Spot was interrupted
    checkpoint = load_latest_checkpoint(CHECKPOINT_DIR)
    if checkpoint:
        logger.info(f"Resuming from checkpoint — skipping retraining")
        model = checkpoint["model"]
    else:
        model.fit(X_train, y_train)
        # Save checkpoint after training
        save_checkpoint(model, {}, 1, CHECKPOINT_DIR)

    metrics, y_pred, y_prob = evaluate_model(model, X_test, y_test)

    # Log metrics to stdout — captured by CloudWatch metric regex filters
    for name, value in metrics.items():
        logger.info(f"{name}={value:.4f}")
        print(f"{name}={value:.4f}")

    plot_confusion_matrix(y_test, y_pred, OUTPUT_DIR)
    plot_roc_curve(y_test, y_prob, OUTPUT_DIR)
    plot_feature_importance(model, X_train.columns.tolist(), OUTPUT_DIR)

    fi = dict(zip(X_train.columns.tolist(), model.feature_importances_.tolist()))
    with open(os.path.join(OUTPUT_DIR, "feature_importance.json"), "w") as f:
        json.dump(fi, f, indent=2)

    model_path = os.path.join(MODEL_DIR, "model.joblib")
    joblib.dump(model, model_path)
    logger.info(f"Model saved: {model_path}")
    logger.info(f"Training complete! ROC AUC: {metrics['roc_auc']:.4f}")
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators",      type=int, default=100)
    parser.add_argument("--max_depth",         type=int, default=10)
    parser.add_argument("--min_samples_split", type=int, default=2)
    parser.add_argument("--experiment_name",   type=str, default="churn-prediction")
    args = parser.parse_args()
    train(args)
