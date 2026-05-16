.PHONY: process train hpo pipeline deploy monitor clarify

process:
	python processing/run_processing_job.py

train:
	python training/run_training_job.py

hpo:
	python training/run_hpo_job.py

pipeline:
	python pipelines/churn_pipeline.py

deploy:
	python inference/deploy_serverless.py

monitor:
	python monitoring/monitor.py

clarify:
	python monitoring/clarify.py

ecr-build:
	docker build --platform linux/amd64 \
		-t churn-mlops:latest \
		-t $(ECR_IMAGE) \
		-f Dockerfile .

ecr-push:
	docker push $(ECR_IMAGE)

clean:
	aws sagemaker delete-endpoint --endpoint-name churn-prediction-serverless --region us-east-1 || true
	aws sagemaker delete-monitoring-schedule --monitoring-schedule-name churn-monitor-schedule --region us-east-1 || true
