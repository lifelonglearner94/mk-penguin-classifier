\
import os
import pytest
import pandas as pd
from pathlib import Path
import sqlite3
from unittest.mock import patch, MagicMock

from training_logic.data_management import DataManager
# Assuming config.py defines these, adjust if necessary
from training_logic.config import (
    DATA_LAKE_BUCKET,
    PREPROCESSED_DB_S3_KEY,
    PREPROCESSED_DB_POINTER_ARTIFACT_NAME
)

@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Create a temporary database path for testing."""
    return tmp_path / "test_processed_penguins.db"

@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Return a sample DataFrame for testing."""
    return pd.DataFrame({
        'flipper_length_mm': [200.0, 210.0],
        'bill_length_mm': [40.0, 50.0],
        'bill_depth_mm': [15.0, 18.0],
        'species': ['Adelie', 'Gentoo']
    })

@pytest.fixture
def data_manager_with_df_source(sample_dataframe: pd.DataFrame, temp_db_path: Path) -> DataManager:
    """Initialize DataManager with a DataFrame as raw_data_source."""
    return DataManager(raw_data_source=sample_dataframe.copy(), processed_db_path=temp_db_path)

@pytest.fixture
def data_manager_with_file_source(tmp_path: Path, sample_dataframe: pd.DataFrame, temp_db_path: Path) -> DataManager:
    """Initialize DataManager with a CSV file path as raw_data_source."""
    csv_path = tmp_path / "raw_penguins.csv"
    sample_dataframe.to_csv(csv_path, index=False)
    return DataManager(raw_data_source=csv_path, processed_db_path=temp_db_path)

# --- Test load_raw_data ---
def test_load_raw_data_from_dataframe(data_manager_with_df_source: DataManager, sample_dataframe: pd.DataFrame):
    loaded_df = data_manager_with_df_source.load_raw_data()
    pd.testing.assert_frame_equal(loaded_df, sample_dataframe)

def test_load_raw_data_from_file(data_manager_with_file_source: DataManager, sample_dataframe: pd.DataFrame):
    loaded_df = data_manager_with_file_source.load_raw_data()
    pd.testing.assert_frame_equal(loaded_df, sample_dataframe)

def test_load_raw_data_invalid_source(temp_db_path: Path):
    dm = DataManager(raw_data_source=123, processed_db_path=temp_db_path) # Invalid source type
    with pytest.raises(ValueError, match="raw_data_source must be a DataFrame, a file path, or an S3 URI"):
        dm.load_raw_data()

# --- Test save_processed_data & load_processed_data (Local DB) ---
@pytest.mark.parametrize("if_exists_strategy", ["replace", "append", "fail"])
def test_save_and_load_processed_data_local_db(
    data_manager_with_df_source: DataManager,
    sample_dataframe: pd.DataFrame,
    temp_db_path: Path,
    if_exists_strategy: str
):
    dm = data_manager_with_df_source

    # Mock S3 and MLflow interactions for these local DB tests
    with patch('training_logic.data_management.mlflow') as mock_mlflow, \
         patch('training_logic.data_management._get_s3_client') as mock_get_s3_client:

        mock_mlflow.active_run.return_value = None # Simulate no active MLflow run

        if if_exists_strategy == "fail" and temp_db_path.exists():
            # Pre-create table for 'fail' strategy to test failure
            with sqlite3.connect(temp_db_path) as conn:
                sample_dataframe.to_sql("penguins_processed", conn, index=False)
            with pytest.raises(ValueError): # sqlite3.OperationalError is wrapped in ValueError by pandas
                dm.save_processed_data(sample_dataframe, if_exists=if_exists_strategy)
            return # Test ends here for 'fail' with pre-existing table

        dm.save_processed_data(sample_dataframe, if_exists=if_exists_strategy)
        assert temp_db_path.exists()

        loaded_df = dm.load_processed_data()
        pd.testing.assert_frame_equal(loaded_df, sample_dataframe)

        if if_exists_strategy == "append":
            # Save again to test append
            dm.save_processed_data(sample_dataframe, if_exists="append")
            appended_df = dm.load_processed_data()
            expected_appended_df = pd.concat([sample_dataframe, sample_dataframe], ignore_index=True)
            pd.testing.assert_frame_equal(appended_df, expected_appended_df)

def test_save_processed_data_invalid_if_exists(data_manager_with_df_source: DataManager, sample_dataframe: pd.DataFrame):
    with pytest.raises(ValueError, match="if_exists must be 'replace', 'append', or 'fail'"):
        data_manager_with_df_source.save_processed_data(sample_dataframe, if_exists="invalid_strategy")

def test_load_processed_data_db_not_found(data_manager_with_df_source: DataManager):
    # Ensure DB does not exist
    if data_manager_with_df_source.processed_db_path.exists():
        data_manager_with_df_source.processed_db_path.unlink()

    with pytest.raises(FileNotFoundError, match="DB not found"):
        data_manager_with_df_source.load_processed_data()

def test_load_processed_data_empty_table(data_manager_with_df_source: DataManager, temp_db_path: Path):
    # Create DB with empty table
    with sqlite3.connect(temp_db_path) as conn:
        conn.execute("CREATE TABLE penguins_processed (id INTEGER)") # Create empty table

    with pytest.raises(ValueError, match="No data in processed DB"):
        data_manager_with_df_source.load_processed_data()

# --- Test S3 and MLflow interactions in save_processed_data ---
@patch('training_logic.data_management.mlflow')
@patch('training_logic.data_management._get_s3_client')
def test_save_processed_data_s3_mlflow_integration(
    mock_get_s3_client: MagicMock,
    mock_mlflow: MagicMock,
    data_manager_with_df_source: DataManager,
    sample_dataframe: pd.DataFrame,
    temp_db_path: Path
):
    dm = data_manager_with_df_source
    mock_s3_client_instance = MagicMock()
    mock_get_s3_client.return_value = mock_s3_client_instance
    mock_mlflow.active_run.return_value = True # Simulate an active MLflow run

    dm.save_processed_data(sample_dataframe, if_exists="replace")

    # Verify S3 upload was called
    mock_s3_client_instance.upload_file.assert_called_once_with(
        str(temp_db_path),
        DATA_LAKE_BUCKET,
        PREPROCESSED_DB_S3_KEY
    )

    # Verify MLflow logging was called
    expected_s3_uri = f"s3://{DATA_LAKE_BUCKET}/{PREPROCESSED_DB_S3_KEY}"
    expected_pointer_content = {"database_s3_uri": expected_s3_uri}
    mock_mlflow.log_dict.assert_called_once_with(
        expected_pointer_content,
        PREPROCESSED_DB_POINTER_ARTIFACT_NAME
    )

@patch('training_logic.data_management.mlflow')
@patch('training_logic.data_management._get_s3_client')
def test_save_processed_data_s3_upload_fails(
    mock_get_s3_client: MagicMock,
    mock_mlflow: MagicMock,
    data_manager_with_df_source: DataManager,
    sample_dataframe: pd.DataFrame,
    caplog: pytest.LogCaptureFixture
):
    dm = data_manager_with_df_source
    mock_s3_client_instance = MagicMock()
    mock_get_s3_client.return_value = mock_s3_client_instance
    mock_mlflow.active_run.return_value = True
    mock_s3_client_instance.upload_file.side_effect = Exception("S3 Upload Error")

    dm.save_processed_data(sample_dataframe, if_exists="replace") # Should not raise, but log error

    assert "S3 upload or MLflow logging failed: S3 Upload Error" in caplog.text
    mock_mlflow.log_dict.assert_not_called() # MLflow logging should be skipped if S3 fails

@patch('training_logic.data_management.mlflow')
@patch('training_logic.data_management._get_s3_client')
def test_save_processed_data_mlflow_log_dict_fails(
    mock_get_s3_client: MagicMock,
    mock_mlflow: MagicMock,
    data_manager_with_df_source: DataManager,
    sample_dataframe: pd.DataFrame,
    caplog: pytest.LogCaptureFixture
):
    dm = data_manager_with_df_source
    mock_s3_client_instance = MagicMock()
    mock_get_s3_client.return_value = mock_s3_client_instance
    mock_mlflow.active_run.return_value = True
    mock_mlflow.log_dict.side_effect = Exception("MLflow Logging Error")

    dm.save_processed_data(sample_dataframe, if_exists="replace") # Should not raise, but log error

    mock_s3_client_instance.upload_file.assert_called_once()
    assert "S3 upload or MLflow logging failed: MLflow Logging Error" in caplog.text

# Test _get_s3_client with and without environment variable
@patch.dict(os.environ, {"MLFLOW_S3_ENDPOINT_URL": "http://minio:9000"})
@patch('training_logic.data_management.boto3.client')
def test_get_s3_client_with_endpoint_url(mock_boto_client):
    from training_logic.data_management import _get_s3_client
    _get_s3_client()
    mock_boto_client.assert_called_once_with(
        's3',
        endpoint_url="http://minio:9000",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
    )

@patch.dict(os.environ, {}, clear=True) # Ensure MLFLOW_S3_ENDPOINT_URL is not set
@patch('training_logic.data_management.boto3.client')
def test_get_s3_client_without_endpoint_url(mock_boto_client):
    # Need to reload the module or the function to re-evaluate os.getenv at import time
    # For simplicity, we can re-import or use importlib.reload if it was a module-level var
    # Here, _get_s3_client is a function, so it re-evaluates os.getenv on each call.
    from training_logic.data_management import _get_s3_client
    # Remove the env var if it was set by a previous test or globally
    if "MLFLOW_S3_ENDPOINT_URL" in os.environ:
        del os.environ["MLFLOW_S3_ENDPOINT_URL"]

    _get_s3_client()
    mock_boto_client.assert_called_once_with(
        's3',
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
    )
