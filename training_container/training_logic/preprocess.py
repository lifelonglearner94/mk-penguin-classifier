"""
Penguin classification: data validation, cleaning, feature preparation.
"""
import pandas as pd
import logging
import numpy as np
from .config import (
    SELECTED_FEATURES,
    FLIPPER_LENGTH_MM_MIN_ADJUSTED,
    FLIPPER_LENGTH_MM_MAX_ADJUSTED,
    BILL_LENGTH_MM_MIN_ADJUSTED,
    BILL_LENGTH_MM_MAX_ADJUSTED,
    BILL_DEPTH_MM_MIN_ADJUSTED,
    BILL_DEPTH_MM_MAX_ADJUSTED,
)

logger = logging.getLogger(__name__)

class Preprocessor:
    """Handles data preprocessing: validation, feature selection.

    Attributes:
        features: List of features to use for classification
        feature_ranges: Dictionary of valid ranges for each feature
    """

    def __init__(self):
        self.features = SELECTED_FEATURES
        self.feature_ranges = {
            'flipper_length_mm': (FLIPPER_LENGTH_MM_MIN_ADJUSTED, FLIPPER_LENGTH_MM_MAX_ADJUSTED),
            'bill_length_mm': (BILL_LENGTH_MM_MIN_ADJUSTED, BILL_LENGTH_MM_MAX_ADJUSTED),
            'bill_depth_mm': (BILL_DEPTH_MM_MIN_ADJUSTED, BILL_DEPTH_MM_MAX_ADJUSTED)
        }
        logger.debug("Initialized Preprocessor with features: %s", self.features)

    def check_data_integrity(self, df: pd.DataFrame) -> bool:
        """Validates data: required columns, types, value ranges.

        Args:
            df: Input DataFrame.

        Returns:
            bool: True if all checks pass.

        Raises:
            ValueError: If validation fails.
        """
        logger.info("Checking data integrity...")

        if df is None or df.empty:
            msg = "DataFrame is empty or None"
            logger.error(msg)
            raise ValueError(msg)

        # Check for expected columns
        missing_cols = set(self.features) - set(df.columns)
        if missing_cols:
            msg = f"Missing required columns: {missing_cols}"
            logger.error(msg)
            raise ValueError(msg)

        # Check for appropriate data types
        numeric_cols = df[self.features].select_dtypes(include=[np.number]).columns
        if len(numeric_cols) != len(self.features):
            msg = "All feature columns must be numeric"
            logger.error(msg)
            raise ValueError(msg)

        # Check for reasonable value ranges based on config
        for feature, (min_val, max_val) in self.feature_ranges.items():
            mask = df[feature].between(min_val, max_val)
            invalid_rows = df[~mask]
            if not invalid_rows.empty:
                msg = (
                    f"Found {len(invalid_rows)} measurements outside valid range for {feature}. "
                    f"Valid range: [{min_val:.2f}, {max_val:.2f}]. "
                    f"Invalid values: {invalid_rows[feature].tolist()}"
                )
                logger.error(msg)
                raise ValueError(msg)

        logger.info("Data integrity checks passed successfully")
        return True

    def prepreprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Initial preprocessing: validation, feature selection.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame: Preprocessed data.
        """
        logger.info("Starting preprocessing...")

        # Extract features and target
        df_selected = df[self.features + ['species']]

        # Handle missing values
        original_len = len(df_selected)
        df_selected = df_selected.dropna()
        dropped_rows = original_len - len(df_selected)
        if dropped_rows > 0:
            logger.warning(f"Dropped {dropped_rows} rows with missing values")

        # Verify data integrity only if we have data left
        if not df_selected.empty:
            self.check_data_integrity(df_selected)
        else:
            logger.warning("No valid data remaining after preprocessing")

        logger.info(f"Preprocessing complete. Output shape: {df_selected.shape}")
        return df_selected
