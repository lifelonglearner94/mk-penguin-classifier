"""
Orchestrates ML model training: preprocessing, training, MLflow tracking.
"""
from pathlib import Path
import logging
import boto3
from botocore.exceptions import ClientError
import os
import mlflow
import pandas as pd
import tempfile
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
import sqlite3
from .data_management import DataManager
from .preprocess import Preprocessor
from .config import (
    SELECTED_ML_MODEL,
    RAW_DATA_SOURCE,
    LOCAL_PROCESSED_DB_PATH,
    DATA_MODE,
    PREPROCESSING_RUN_NAME,
    PREPROCESSED_DB_POINTER_ARTIFACT_NAME,
    DATA_LAKE_BUCKET,
    MLFLOW_S3_DEFAULT_ARTIFACT_BUCKET_NAME,
    SELECTED_FEATURES
)

logger = logging.getLogger(__name__)

def _get_s3_client():
    """Initializes S3 client, configured for MinIO if endpoint URL is set."""
    s3_endpoint_url = os.getenv("MLFLOW_S3_ENDPOINT_URL")
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")

    if s3_endpoint_url:
        logger.info(f"Using S3 endpoint URL for ml_controller: {s3_endpoint_url}")
        return boto3.client(
            's3',
            endpoint_url=s3_endpoint_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )
    else:
        logger.info("MLFLOW_S3_ENDPOINT_URL not set in ml_controller. Using default S3 client config.")
        return boto3.client('s3',
                            aws_access_key_id=aws_access_key_id,
                            aws_secret_access_key=aws_secret_access_key)

def _ensure_s3_bucket_exists(bucket_name: str):
    """Ensures S3 bucket exists, creates if needed."""
    s3_client = _get_s3_client()
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        logger.info(f"S3 bucket '{bucket_name}' already exists.")
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        if error_code == '404' or error_code == 'NoSuchBucket':
            try:
                s3_client.create_bucket(Bucket=bucket_name)
                logger.info(f"S3 bucket '{bucket_name}' created successfully.")
            except ClientError as ce:
                logger.error(f"Failed to create S3 bucket '{bucket_name}'. Error: {ce}")
                raise
        else:
            logger.error(f"Unexpected S3 ClientError when checking/creating bucket '{bucket_name}': {e}")
            raise

def run_preprocessing_pipeline():
    """Executes data preprocessing pipeline without MLflow run.

    Handles S3 bucket creation, data loading, preprocessing, saves to S3.
    To be used within a larger training run.

    Returns:
        tuple: (preprocessed_data_df, database_s3_uri)
    """
    logger.info("Starting data preprocessing pipeline...")

    try:
        _ensure_s3_bucket_exists(DATA_LAKE_BUCKET)
        _ensure_s3_bucket_exists(MLFLOW_S3_DEFAULT_ARTIFACT_BUCKET_NAME)
    except Exception as e:
        logger.error(f"Failed to ensure S3 bucket existence. Preprocessing aborted. Error: {e}")
        raise

    data_manager = DataManager(raw_data_source=RAW_DATA_SOURCE,
                               processed_db_path=LOCAL_PROCESSED_DB_PATH)
    preprocessor = Preprocessor()

    logger.info("Loading and preprocessing new raw data.")
    raw_data = data_manager.load_raw_data()
    preprocessed_data = preprocessor.prepreprocess(raw_data)

    if DATA_MODE == "overwrite":
        logger.info("Overwrite mode: Saving new preprocessed data.")
        data_manager.save_processed_data(preprocessed_data, if_exists="replace")
        logger.info("Overwrite mode: Successfully saved new preprocessed data.")
    elif DATA_MODE == "append":
        logger.info("Append mode: Attempting to append new data to existing preprocessed data.")
        existing_data_df = pd.DataFrame()
        try:
            client = mlflow.tracking.MlflowClient()
            runs = client.search_runs(
                filter_string=f"tags.mlflow.runName = 'penguin_model_training' AND status = 'FINISHED'",
                order_by=["start_time DESC"],
                max_results=1
            )

            if not runs:
                logger.warning("Append mode: No previous successful training run found. Will proceed with new data only.")
            else:
                latest_run = runs[0]
                logger.info(f"Append mode: Found latest successful training run: {latest_run.info.run_id}")

                with tempfile.TemporaryDirectory() as tmpdir:
                    try:
                        pointer_artifact_path = client.download_artifacts(
                            latest_run.info.run_id,
                            PREPROCESSED_DB_POINTER_ARTIFACT_NAME,
                            tmpdir
                        )
                        with open(pointer_artifact_path, 'r') as f:
                            s3_pointer_data = json.load(f)
                        db_s3_uri = s3_pointer_data.get("database_s3_uri")

                        if db_s3_uri:
                            logger.info(f"Append mode: Downloading existing database from {db_s3_uri}")
                            s3_client_for_download = _get_s3_client()
                            bucket_name_from_uri, key_from_uri = db_s3_uri.replace("s3://", "").split("/", 1)

                            s3_client_for_download.download_file(
                                bucket_name_from_uri,
                                key_from_uri,
                                str(LOCAL_PROCESSED_DB_PATH)
                            )

                            temp_dm_for_load = DataManager(raw_data_source=None, processed_db_path=str(LOCAL_PROCESSED_DB_PATH))
                            existing_data_df = temp_dm_for_load.load_processed_data()
                            logger.info(f"Append mode: Loaded {len(existing_data_df)} records from existing database.")
                    except Exception as artifact_error:
                        logger.warning(f"Could not load existing data: {artifact_error}")

        except Exception as e:
            logger.error(f"Append mode: Error loading existing data: {e}. Will proceed with new data only.")

        if not existing_data_df.empty:
            combined_data = pd.concat([existing_data_df, preprocessed_data], ignore_index=True)
            logger.info(f"Append mode: Combined existing data ({len(existing_data_df)} rows) "
                        f"with new data ({len(preprocessed_data)} rows). Total: {len(combined_data)} rows.")
        else:
            combined_data = preprocessed_data
            logger.info("Append mode: No existing data found/loaded. Using new data only.")

        data_manager.save_processed_data(combined_data, if_exists="replace")
        logger.info("Append mode: Saved combined data.")
        preprocessed_data = combined_data

    # Get the S3 URI for the saved database
    # Construct S3 URI manually since we know where the data was saved
    db_filename = Path(LOCAL_PROCESSED_DB_PATH).name
    database_s3_uri = f"s3://{DATA_LAKE_BUCKET}/processed_data/{db_filename}"
    logger.info(f"Preprocessing pipeline completed. Database S3 URI: {database_s3_uri}")

    return preprocessed_data, database_s3_uri

def run_training_pipeline():
    """Executes complete ML pipeline: preprocessing + training in single MLflow run.

    Returns:
        str: MLflow run ID for the complete pipeline.
    """
    logger.info("Starting complete ML training pipeline...")

    with mlflow.start_run(run_name="penguin_model_training") as run:
        training_run_id = run.info.run_id
        logger.info(f"MLflow run started for complete pipeline: {training_run_id}")

        # Step 1: Run preprocessing
        preprocessed_data, database_s3_uri = run_preprocessing_pipeline()

        # Log the database S3 pointer as artifact
        pointer_data = {"database_s3_uri": database_s3_uri}
        mlflow.log_dict(pointer_data, PREPROCESSED_DB_POINTER_ARTIFACT_NAME)
        logger.info(f"Logged database S3 pointer: {database_s3_uri}")

        # Step 2: Prepare for training
        label_encoder = LabelEncoder()
        X = preprocessed_data[SELECTED_FEATURES]
        y_original = preprocessed_data['species']
        y = label_encoder.fit_transform(y_original)
        logger.info(f"Label encoder classes: {list(label_encoder.classes_)}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        logger.info(f"Data split: Train {X_train.shape}, Test {X_test.shape}")

        # Step 3: Model training
        full_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SELECTED_ML_MODEL)
        ])

        param_grid = {}
        if isinstance(SELECTED_ML_MODEL, LogisticRegression):
            param_grid = {
                'classifier__C': [0.01, 0.1, 1, 10, 100],
                'classifier__solver': ['liblinear', 'saga'],
                'classifier__penalty': ['l1', 'l2']
            }
            logger.info(f"Using LogisticRegression param_grid: {param_grid}")

        f1_macro = make_scorer(f1_score, average='macro', zero_division=0)
        grid_search = GridSearchCV(full_pipeline, param_grid, cv=5, scoring=f1_macro, n_jobs=-1, verbose=1)

        logger.info("Fitting GridSearchCV...")
        grid_search.fit(X_train, y_train)
        logger.info("GridSearchCV fitting completed.")

        best_pipeline = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score_cv = grid_search.best_score_

        # Log training metrics and model
        mlflow.log_params(best_params)
        mlflow.log_metric("best_cv_f1_macro", best_score_cv)

        y_pred_test = best_pipeline.predict(X_test)
        test_f1_macro = f1_score(y_test, y_pred_test, average='macro', zero_division=0)
        mlflow.log_metric("test_f1_macro", test_f1_macro)

        # Log the trained model
        mlflow.sklearn.log_model(
            sk_model=best_pipeline,
            artifact_path="penguin_classification_model"
        )

        # Log label encoder classes
        label_encoder_classes_dict = {str(i): cls for i, cls in enumerate(label_encoder.classes_)}
        mlflow.log_dict(label_encoder_classes_dict, "model_utils/label_encoder_classes.json")

        logger.info("Complete ML pipeline finished successfully.")
        return training_run_id
