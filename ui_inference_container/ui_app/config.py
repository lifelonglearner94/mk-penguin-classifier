import os
from pathlib import Path  # Add Path import
from dotenv import load_dotenv  # Add dotenv import

# Load environment variables from .env file
load_dotenv()

# --- S3 Data Lake Configuration (for UI to find data) ---
# Name for the MLflow run that performs data preprocessing
PREPROCESSING_RUN_NAME = os.getenv("PREPROCESSING_RUN_NAME", "data_preprocessing_pipeline")
# Name of the artifact in MLflow that stores the JSON pointer to the S3 database
PREPROCESSED_DB_POINTER_ARTIFACT_NAME = os.getenv("PREPROCESSED_DB_POINTER_ARTIFACT_NAME", "preprocessed_db_uri.json")
# Local temporary path for downloading the database
LOCAL_DOWNLOADED_DB_PATH = os.getenv("LOCAL_DOWNLOADED_DB_PATH", "/app/data/penguins_processed.db")

MODEL_NAME = os.getenv("MODEL_NAME", "penguin-classifier")

# Features used by the model and for UI inputs
SELECTED_FEATURES = ["flipper_length_mm", "bill_length_mm", "bill_depth_mm"]

# Data validation ranges (calculated from statistical analysis)
FLIPPER_LENGTH_MM_MIN_ADJUSTED = round(129.0, 1)
FLIPPER_LENGTH_MM_MAX_ADJUSTED = round(288.75, 1)
BILL_LENGTH_MM_MIN_ADJUSTED = round(24.075, 1)
BILL_LENGTH_MM_MAX_ADJUSTED = round(74.5, 1)
BILL_DEPTH_MM_MIN_ADJUSTED = round(9.825, 1)
BILL_DEPTH_MM_MAX_ADJUSTED = round(26.875, 1)

# API Key Configuration
API_KEY = os.getenv("API_KEY", "Your_API_Key")

# API Base URL for the inference service
# The UI container will call its own backend API
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8080")

# S3 and AWS Configuration
MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
AWS_ACCESS_KEY_ID_UI = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY_UI = os.getenv("AWS_SECRET_ACCESS_KEY", "")
