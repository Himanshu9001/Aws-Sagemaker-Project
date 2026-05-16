# Custom SageMaker Training Image — Churn MLOps
# Extends official SKLearn base image with pre-installed dependencies
# Eliminates runtime pip install — saves 60-90 seconds per Training Job

FROM 683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3

# Install dependencies at build time — not at runtime
RUN pip install --no-cache-dir \
    mlflow>=2.0.0 \
    protobuf==3.20.3 \
    matplotlib==3.7.1 \
    seaborn==0.12.2 \
    joblib==1.3.2 \
    pandas==1.5.3 \
    numpy==1.24.3

# Copy training scripts into image
COPY training/train.py /opt/ml/code/train.py
COPY training/requirements.txt /opt/ml/code/requirements.txt

# Set working directory
WORKDIR /opt/ml/code

# SageMaker expects this environment variable
ENV SAGEMAKER_PROGRAM train.py
