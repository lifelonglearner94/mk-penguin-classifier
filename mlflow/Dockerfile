# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install MLflow and boto3 for S3 support
# We pin versions for reproducibility, but you might want to use latest or specific versions
RUN pip install --no-cache-dir mlflow==2.14.1 boto3==1.34.128

# Make port 5000 available to the world outside this container
EXPOSE 5000

# The command to run MLflow server is specified in docker-compose.yml
# CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "sqlite:///mlflow_backend_db/mlflow.db", "--default-artifact-root", "s3://mlflow-artifacts/"]
