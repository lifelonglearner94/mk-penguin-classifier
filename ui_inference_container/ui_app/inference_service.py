"""
Inference service for the penguin classification model.
Handles model loading and prediction functionality using MLflow.
"""
import logging
import os
from typing import Dict, Tuple, Optional, Any, Union
from sklearn.linear_model import LogisticRegression
import mlflow
import pandas as pd
import json  # Add json import
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
from .config import (  # Updated imports from config
    MODEL_NAME,
    FLIPPER_LENGTH_MM_MIN_ADJUSTED, FLIPPER_LENGTH_MM_MAX_ADJUSTED,
    BILL_LENGTH_MM_MIN_ADJUSTED, BILL_LENGTH_MM_MAX_ADJUSTED,
    BILL_DEPTH_MM_MIN_ADJUSTED, BILL_DEPTH_MM_MAX_ADJUSTED
)

# Configure logging
logger = logging.getLogger(__name__)

# MLflow Configuration
MLFLOW_TRACKING_URI_ENV = os.getenv("MLFLOW_TRACKING_URI")
if MLFLOW_TRACKING_URI_ENV:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI_ENV)
    logger.info(f"MLflow tracking URI set to: {MLFLOW_TRACKING_URI_ENV} (from environment variable)")
else:
    logger.warning(f"MLFLOW_TRACKING_URI environment variable not set.")
    raise ValueError("MLFLOW_TRACKING_URI environment variable must be set to use the inference service.")

class InferenceService:
    """
    Service for loading and using ML models for inference.

    Attributes:
        model: Loaded ML model for making predictions
        model_info: Information about the loaded model version
        features: List of feature names expected by the model
        feature_ranges: Dictionary of valid ranges for each feature
        label_mapping: Dictionary mapping encoded labels to original string labels
    """

    def __init__(self):
        """
        Initialize the inference service.
        The MLflow tracking URI is now set globally based on the environment variable.
        """
        logger.info("InferenceService __init__ called.")
        self.model = None
        self.model_info = None
        self.label_mapping = None  # Initialize label_mapping
        self.features = ["flipper_length_mm", "bill_length_mm", "bill_depth_mm"]
        self.feature_ranges = {
            "bill_length_mm": (BILL_LENGTH_MM_MIN_ADJUSTED, BILL_LENGTH_MM_MAX_ADJUSTED),
            "bill_depth_mm": (BILL_DEPTH_MM_MIN_ADJUSTED, BILL_DEPTH_MM_MAX_ADJUSTED),
            "flipper_length_mm": (FLIPPER_LENGTH_MM_MIN_ADJUSTED, FLIPPER_LENGTH_MM_MAX_ADJUSTED),
        }
        # MLflow initialization
        self._load_model()
        if self.model_info:  # Ensure model_info is loaded before trying to load label mapping
            self._load_label_mapping(self.model_info.run_id)

    def _load_model(self) -> None:
        """
        Load the production model from MLflow.
        Attempts to load from 'Production' stage first, falls back to latest version.
        """
        logger.info("Loading model...")

        # First try loading from Production stage
        self.model, self.model_info = self._load_model_by_stage(MODEL_NAME, "Production")

        # If no Production model, try latest version
        if self.model is None:
            logger.info("No Production model found, falling back to latest version")
            self.model, self.model_info = self._load_latest_version(MODEL_NAME)

        if self.model is None:
            raise RuntimeError("Failed to load model")

        logger.info(f"Successfully loaded model version: {self.model_info.version}")

    def _load_label_mapping(self, run_id: str) -> None:
        """
        Load the label mapping from MLflow artifacts.
        Args:
            run_id: The MLflow run ID from which the model was loaded.
        """
        logger.info(f"Attempting to load label mapping for run_id: {run_id}")
        client = mlflow.tracking.MlflowClient()
        try:
            # Define the path to the artifact
            # This assumes it was saved in a subdirectory named 'model_utils' during training
            artifact_path = "model_utils/label_encoder_classes.json"
            local_path = client.download_artifacts(run_id, artifact_path)

            with open(local_path, 'r') as f:
                loaded_mapping_str_keys = json.load(f)
            # Convert string keys from JSON back to integers if necessary
            self.label_mapping = {int(k): v for k, v in loaded_mapping_str_keys.items()}
            logger.info(f"Successfully loaded label mapping: {self.label_mapping}")

        except MlflowException as e:
            logger.error(f"MLflow error loading label mapping artifact for run {run_id} from path {artifact_path}: {e}")
            # Fallback or raise error if mapping is critical
            # For now, we'll allow it to be None and handle it in predict
            self.label_mapping = None
        except Exception as e:
            logger.error(f"Unexpected error loading label mapping for run {run_id}: {e}")
            self.label_mapping = None

    @staticmethod
    def _load_latest_version(model_name: str) -> Tuple[Optional[Any], Optional[Any]]:
        """
        Load the latest version of a model from MLflow.

        Args:
            model_name: Name of the model in MLflow

        Returns:
            Tuple of (loaded_model, version_info) or (None, None) if loading fails
        """
        client = MlflowClient()
        try:
            # Get all versions and sort by version number
            versions = client.search_model_versions(f"name='{model_name}'")
            if not versions:
                logger.error(f"No versions found for model '{model_name}'")
                return None, None

            latest_version = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
            model_uri = f"models:/{model_name}/{latest_version.version}"

            logger.info(f"Loading model '{model_name}' version {latest_version.version}")
            model = mlflow.sklearn.load_model(model_uri)
            return model, latest_version

        except Exception as e:
            logger.error(f"Error loading latest model version: {str(e)}")
            return None, None

    @staticmethod
    def _load_model_by_stage(model_name: str, stage: str) -> Tuple[Optional[Any], Optional[Any]]:
        """
        Load a model version from a specific stage in MLflow.

        Args:
            model_name: Name of the model in MLflow
            stage: Stage to load from (e.g., 'Production', 'Staging')

        Returns:
            Tuple of (loaded_model, version_info) or (None, None) if loading fails
        """
        model_uri = f"models:/{model_name}/{stage}"
        try:
            logger.info(f"Loading model '{model_name}' from {stage} stage")
            model = mlflow.sklearn.load_model(model_uri)

            # Get version details
            client = MlflowClient()
            version_info = client.get_latest_versions(model_name, stages=[stage])[0]
            return model, version_info

        except MlflowException as e:
            logger.error(f"MLflow error loading {stage} model: {str(e)}")
            return None, None
        except Exception as e:
            logger.error(f"Unexpected error loading model: {str(e)}")
            return None, None

    def check_data_integrity(self, df: pd.DataFrame) -> bool:
        """
        Validate data integrity including presence of required columns,
        data types, and value ranges.

        Args:
            df: Input DataFrame to validate

        Returns:
            bool: True if all checks pass

        Raises:
            ValueError: If any validation check fails
        """
        logger.info("Checking data integrity for input DataFrame...")

        if df is None or df.empty:
            msg = "Input DataFrame is empty or None."
            logger.error(msg)
            raise ValueError(msg)

        # Check for expected columns
        missing_cols = set(self.features) - set(df.columns)
        if missing_cols:
            msg = f"Missing required columns: {missing_cols}. Expected: {self.features}"
            logger.error(msg)
            raise ValueError(msg)

        # Check for appropriate data types (all features should be numeric)
        for feature in self.features:
            if not pd.api.types.is_numeric_dtype(df[feature]):
                msg = f"Column '{feature}' must be numeric, but found dtype {df[feature].dtype}."
                logger.error(msg)
                raise ValueError(msg)

        # Check for reasonable value ranges based on config
        for feature, (min_val, max_val) in self.feature_ranges.items():
            feature_series = df[feature].dropna()
            if feature_series.empty and df[feature].isnull().any():
                msg = f"Feature '{feature}' contains only NaN values."
                logger.error(msg)
                raise ValueError(msg)

            mask = feature_series.between(min_val, max_val)
            if not mask.all():
                invalid_values = feature_series[~mask].tolist()
                if invalid_values:
                    msg = (
                        f"Found {len(invalid_values)} measurements outside valid range for {feature}. "
                        f"Valid range: [{min_val:.2f}, {max_val:.2f}]. "
                        f"Invalid values found: {invalid_values}"
                    )
                    logger.error(msg)
                    raise ValueError(msg)

        logger.info("Data integrity checks passed successfully.")
        return True

    def predict(self, features: Dict[str, float]) -> Dict[str, Union[str, int, float]]:
        """
        Make a prediction using the loaded model and provide a confidence score.
        The predicted species will be the original string label if mapping is available.

        Args:
            features: Dictionary containing feature values:
                     - bill_length_mm: float
                     - bill_depth_mm: float
                     - flipper_length_mm: float

        Returns:
            A dictionary containing:
            - 'prediction': Predicted penguin species (string if mapped, else integer)
            - 'confidence': Confidence score for the prediction (float)

        Raises:
            ValueError: If model not loaded or features invalid
            RuntimeError: If the model does not support predict_proba or for other unexpected errors.
        """
        logger.info(f"InferenceService.predict method entered with features: {features}")  # ADDED THIS LINE

        if self.model is None:
            logger.error("Model not loaded at prediction time.")
            raise ValueError("Model not loaded")

        if not hasattr(self.model, 'predict_proba'):
            logger.error(f"Model of type {type(self.model)} does not support predict_proba for confidence scores.")
            raise RuntimeError("Loaded model does not support probability estimates (predict_proba) needed for confidence scores.")

        try:
            # Convert features to DataFrame, ensuring correct column order
            df = pd.DataFrame([features], columns=self.features)

            # Validate input data
            self.check_data_integrity(df)

            # Make prediction for the class (returns an array)
            prediction_array = self.model.predict(df)
            encoded_prediction = prediction_array[0]  # This is the numeric label

            # Predict probabilities
            probabilities = self.model.predict_proba(df)

            # Confidence is the maximum probability for the predicted class
            confidence = float(probabilities[0].max())

            # Map encoded prediction back to string label if mapping exists
            if self.label_mapping:
                predicted_species = self.label_mapping.get(int(encoded_prediction), str(encoded_prediction))
            else:
                predicted_species = int(encoded_prediction)  # Return the numeric prediction if no mapping
                logger.warning("Label mapping not available. Returning encoded label.")

            logger.info(f"Prediction successful for input: {features}. Output: {predicted_species}, Confidence: {confidence:.4f}")

            return {
                "prediction": predicted_species,
                "confidence": confidence
            }

        except ValueError as ve:
            logger.error(f"Validation error during prediction: {str(ve)}")
            raise
        except AttributeError as ae:
            logger.error(f"Model attribute error during prediction (e.g. predict_proba missing): {str(ae)}")
            raise RuntimeError(f"Prediction failed due to model incompatibility: {str(ae)}")
        except Exception as e:
            logger.error(f"Unexpected error during prediction: {str(e)} for input {features}", exc_info=True)
            raise RuntimeError(f"Prediction failed due to an unexpected error: {str(e)}")
