"""Data I/O: raw data loading, processed data persistence (local SQLite/S3), MLflow logging."""

from pathlib import Path
import logging
import pandas as pd
import sqlite3
import os
import boto3
import mlflow
import io
from urllib.parse import urlparse
from .config import DATA_LAKE_BUCKET, PREPROCESSED_DB_S3_KEY, PREPROCESSED_DB_POINTER_ARTIFACT_NAME

logger = logging.getLogger(__name__)

def _get_s3_client():
    """Initializes S3 client, configured for MinIO if endpoint URL is set."""
    s3_endpoint_url = os.getenv("MLFLOW_S3_ENDPOINT_URL")
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")

    if s3_endpoint_url:
        logger.info(f"Using S3 endpoint URL for DataManager: {s3_endpoint_url}")
        return boto3.client(
            's3',
            endpoint_url=s3_endpoint_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )
    else:
        logger.info("MLFLOW_S3_ENDPOINT_URL not set in DataManager. Using default S3 client config.")
        return boto3.client('s3',
                            aws_access_key_id=aws_access_key_id,
                            aws_secret_access_key=aws_secret_access_key)

class DataManager:
    """Loads data from source, saves to local/S3 backed database."""

    def __init__(self, raw_data_source: pd.DataFrame | str | Path, processed_db_path: str | Path):
        """
        Initializes DataManager.

        Args:
            raw_data_source: DataFrame or path to CSV for raw data.
            processed_db_path: Path for the local processed SQLite database.
        """
        self.raw_data_source = raw_data_source
        self.processed_db_path = Path(processed_db_path)
        self.processed_db_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"DataManager initialized. Processed DB: {self.processed_db_path}")

    def load_raw_data(self) -> pd.DataFrame:
        """Loads raw data. Handles local files, S3 URIs, or DataFrame instances.

        Returns:
            pd.DataFrame: Raw data.
        Raises:
            ValueError: If raw_data_source type is invalid or S3 URI is malformed.
        """
        logger.info("Loading raw data...")
        try:
            if isinstance(self.raw_data_source, pd.DataFrame):
                df = self.raw_data_source.copy()
                logger.debug("Raw data loaded from DataFrame instance.")
            elif isinstance(self.raw_data_source, (str, Path)):
                source_str = str(self.raw_data_source)
                if source_str.startswith("s3://"):
                    logger.info(f"Raw data source is an S3 URI: {source_str}")
                    parsed_uri = urlparse(source_str)
                    bucket_name = parsed_uri.netloc
                    s3_key = parsed_uri.path.lstrip('/')

                    if not bucket_name or not s3_key:
                        raise ValueError(f"Malformed S3 URI: {source_str}")

                    s3_client = _get_s3_client() # Uses existing S3 client logic

                    csv_obj = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
                    csv_content = csv_obj['Body'].read().decode('utf-8')
                    df = pd.read_csv(io.StringIO(csv_content))
                    logger.debug(f"Raw data loaded from S3 URI: {source_str} ({df.shape[0]} rows).")
                else: # Local file path
                    df = pd.read_csv(self.raw_data_source)
                    logger.debug(f"Raw data loaded from file: {self.raw_data_source}")
            else:
                raise ValueError("raw_data_source must be a DataFrame, a file path, or an S3 URI.")
            logger.info(f"Raw data loaded successfully ({df.shape[0]} rows).")
            return df
        except Exception as e:
            logger.error(f"Error loading raw data: {e}")
            raise

    def save_processed_data(self, df: pd.DataFrame, if_exists: str = "replace"):
        """Saves data to local SQLite, uploads to S3, logs S3 URI to MLflow.

        Args:
            df: DataFrame to save.
            if_exists: SQLite behavior if table exists.
        Raises:
            ValueError: If if_exists is invalid.
        """
        logger.info(f"Saving processed data (if_exists='{if_exists}').")
        if if_exists not in {"replace", "append", "fail"}:
            raise ValueError("if_exists must be 'replace', 'append', or 'fail'.")

        try:
            with sqlite3.connect(self.processed_db_path) as conn:
                df.to_sql("penguins_processed", conn, if_exists=if_exists, index=False)
            logger.info(f"Saved {len(df)} records to local DB: {self.processed_db_path}")

            if mlflow.active_run():
                logger.info("Active MLflow run. Uploading to S3 and logging pointer.")
                try:
                    s3_client = _get_s3_client()
                    s3_client.upload_file(
                        str(self.processed_db_path),
                        DATA_LAKE_BUCKET,
                        PREPROCESSED_DB_S3_KEY
                    )
                    s3_uri = f"s3://{DATA_LAKE_BUCKET}/{PREPROCESSED_DB_S3_KEY}"
                    logger.info(f"Uploaded processed database to {s3_uri}")

                    pointer_content = {"database_s3_uri": s3_uri}
                    mlflow.log_dict(pointer_content, PREPROCESSED_DB_POINTER_ARTIFACT_NAME)
                    logger.info(f"Logged S3 URI pointer: {PREPROCESSED_DB_POINTER_ARTIFACT_NAME}")
                except Exception as e:
                    logger.error(f"S3 upload or MLflow logging failed: {e}")
            else:
                logger.info("No active MLflow run. Skipping S3 upload and artifact logging.")
        except Exception as e:
            logger.error(f"Error saving processed data locally: {e}")
            raise

    def load_processed_data(self) -> pd.DataFrame:
        """Loads processed data from local SQLite.

        Returns:
            pd.DataFrame: Processed data.
        Raises:
            FileNotFoundError: If DB file not found.
            ValueError: If DB empty or read error.
        """
        logger.info(f"Loading processed data from {self.processed_db_path}...")
        if not self.processed_db_path.exists():
            raise FileNotFoundError(f"DB not found: {self.processed_db_path}. Run preprocessing.")

        try:
            with sqlite3.connect(self.processed_db_path) as conn:
                df = pd.read_sql("SELECT * FROM penguins_processed", conn)
            if df.empty:
                raise ValueError(f"No data in processed DB: {self.processed_db_path}")
            logger.info(f"Loaded {len(df)} records from {self.processed_db_path}")
            return df
        except sqlite3.OperationalError as e:
            raise ValueError(f"Error reading DB {self.processed_db_path}: {e}")
