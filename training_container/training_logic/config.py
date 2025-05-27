from sklearn.linear_model import LogisticRegression
from palmerpenguins import load_penguins
import os
from dotenv import load_dotenv
from pathlib import Path
import logging
# --- Logger Configuration ---
# Create a logger instance
logger = logging.getLogger(__name__) # Use the module name for the logger

# Load environment variables from .env file
load_dotenv()

DATA_MODE = os.getenv("DATA_MODE", "overwrite") # or "append"

RAW_DATA_SOURCE_PATH = os.getenv("RAW_DATA_SOURCE", "palmerpenguins")  # Default: load_penguins package
if RAW_DATA_SOURCE_PATH == "palmerpenguins":
    RAW_DATA_SOURCE = load_penguins()
else:
    RAW_DATA_SOURCE = RAW_DATA_SOURCE_PATH

# --- New S3 Data Lake Configuration ---
# S3 bucket for processed data (not MLflow artifacts)
DATA_LAKE_BUCKET = os.getenv("DATA_LAKE_BUCKET", "penguin-data-lake")
# Key for processed DB in S3 bucket
PREPROCESSED_DB_S3_KEY = "processed_db/penguins_processed.db"
# MLflow artifact name for JSON pointer to S3 DB
PREPROCESSED_DB_POINTER_ARTIFACT_NAME = "preprocessed_db_uri.json"
# MLflow run name for data preprocessing
PREPROCESSING_RUN_NAME = "data_preprocessing_pipeline"

# MLflow's default artifact bucket (must match docker-compose.yml for mlflow service)
MLFLOW_S3_DEFAULT_ARTIFACT_BUCKET_NAME = "mlflow-artifacts"

# Local DB: built here before S3 upload.
LOCAL_PROCESSED_DB_PARENT_DIR = Path(__file__).resolve().parent.parent / "data" / "preprocessed_db"
LOCAL_PROCESSED_DB_FILENAME = "penguins_processed.db"
LOCAL_PROCESSED_DB_PATH = LOCAL_PROCESSED_DB_PARENT_DIR / LOCAL_PROCESSED_DB_FILENAME
# --- End New S3 Data Lake Configuration ---

SELECTED_FEATURES = ["flipper_length_mm", "bill_length_mm", "bill_depth_mm"]

SELECTED_ML_MODEL = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)

# --- Min/Max Feature Values from statistical analysis ---
FLIPPER_LENGTH_MM_MIN_ADJUSTED=129.0
FLIPPER_LENGTH_MM_MAX_ADJUSTED=288.75
BILL_LENGTH_MM_MIN_ADJUSTED=24.075000000000003
BILL_LENGTH_MM_MAX_ADJUSTED=74.5
BILL_DEPTH_MM_MIN_ADJUSTED=9.825
BILL_DEPTH_MM_MAX_ADJUSTED=26.875
