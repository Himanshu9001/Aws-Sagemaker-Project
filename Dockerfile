# Multi-stage Dockerfile — maximizes layer cache hits
# Stage 1: deps — only rebuilds when requirements change
# Stage 2: code — only rebuilds when train.py changes

# ── Stage 1: Install dependencies ────────────────────────────────────────────
FROM 683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3 AS deps

# Install at build time — eliminates 60-90s runtime pip install
# protobuf==3.20.3 required — mlflow>=2.0 pulls 4.x which breaks sagemaker_containers
RUN pip install --no-cache-dir \
    mlflow>=2.0.0 \
    protobuf==3.20.3 \
    sagemaker-mlflow \
    transformers==4.36.0 \
    torch \
    accelerate \
    sentencepiece \
    datasets \
    matplotlib==3.7.1 \
    seaborn==0.12.2 \
    joblib==1.3.2 \
    pandas==1.5.3 \
    numpy==1.24.3

# ── Stage 2: Copy code ────────────────────────────────────────────────────────
FROM deps AS final

# Copy training scripts — invalidates only this layer when code changes
COPY training/train.py /opt/ml/code/train.py
COPY training/finetune.py /opt/ml/code/finetune.py
COPY training/requirements.txt /opt/ml/code/requirements.txt

WORKDIR /opt/ml/code

# SageMaker entry point
ENV SAGEMAKER_PROGRAM=train.py
