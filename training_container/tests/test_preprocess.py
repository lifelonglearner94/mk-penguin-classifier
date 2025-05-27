\
import pytest
import pandas as pd
import numpy as np
from training_logic.preprocess import Preprocessor
from training_logic.config import (
    FLIPPER_LENGTH_MM_MIN_ADJUSTED, FLIPPER_LENGTH_MM_MAX_ADJUSTED,
    BILL_LENGTH_MM_MIN_ADJUSTED, BILL_LENGTH_MM_MAX_ADJUSTED,
    BILL_DEPTH_MM_MIN_ADJUSTED, BILL_DEPTH_MM_MAX_ADJUSTED
)

# Use selected features from config for consistency, add species for target
TEST_FEATURES = ["flipper_length_mm", "bill_length_mm", "bill_depth_mm"]
TEST_COLUMNS = TEST_FEATURES + ["species"]

@pytest.fixture
def preprocessor():
    return Preprocessor()

@pytest.fixture
def valid_data():
    return pd.DataFrame({
        'flipper_length_mm': [200.0, 210.0, (FLIPPER_LENGTH_MM_MIN_ADJUSTED + FLIPPER_LENGTH_MM_MAX_ADJUSTED) / 2],
        'bill_length_mm': [40.0, 50.0, (BILL_LENGTH_MM_MIN_ADJUSTED + BILL_LENGTH_MM_MAX_ADJUSTED) / 2],
        'bill_depth_mm': [15.0, 18.0, (BILL_DEPTH_MM_MIN_ADJUSTED + BILL_DEPTH_MM_MAX_ADJUSTED) / 2],
        'species': ['Adelie', 'Gentoo', 'Chinstrap'],
        'island': ['Torgersen', 'Biscoe', 'Dream'] # Extra column to test selection
    })

@pytest.fixture
def data_with_nan():
    return pd.DataFrame({
        'flipper_length_mm': [200.0, np.nan, 210.0],
        'bill_length_mm': [40.0, 50.0, np.nan],
        'bill_depth_mm': [np.nan, 18.0, 19.0],
        'species': ['Adelie', 'Gentoo', 'Chinstrap']
    })

def test_check_data_integrity_valid_data(preprocessor, valid_data):
    assert preprocessor.check_data_integrity(valid_data[TEST_COLUMNS]) is True

def test_check_data_integrity_empty_df(preprocessor):
    with pytest.raises(ValueError, match="DataFrame is empty or None"):
        preprocessor.check_data_integrity(pd.DataFrame())

def test_check_data_integrity_missing_columns(preprocessor, valid_data):
    missing_col_data = valid_data.drop(columns=['flipper_length_mm'])
    with pytest.raises(ValueError, match="Missing required columns: {'flipper_length_mm'}"):
        preprocessor.check_data_integrity(missing_col_data)

def test_check_data_integrity_non_numeric(preprocessor, valid_data):
    non_numeric_data = valid_data.copy()
    non_numeric_data['flipper_length_mm'] = non_numeric_data['flipper_length_mm'].astype(str)
    with pytest.raises(ValueError, match="All feature columns must be numeric"):
        preprocessor.check_data_integrity(non_numeric_data[TEST_COLUMNS])

@pytest.mark.parametrize("feature, bad_value_low, bad_value_high", [
    ("flipper_length_mm", FLIPPER_LENGTH_MM_MIN_ADJUSTED - 1, FLIPPER_LENGTH_MM_MAX_ADJUSTED + 1),
    ("bill_length_mm", BILL_LENGTH_MM_MIN_ADJUSTED - 1, BILL_LENGTH_MM_MAX_ADJUSTED + 1),
    ("bill_depth_mm", BILL_DEPTH_MM_MIN_ADJUSTED - 1, BILL_DEPTH_MM_MAX_ADJUSTED + 1),
])
def test_check_data_integrity_out_of_range(preprocessor, valid_data, feature, bad_value_low, bad_value_high):
    data_low = valid_data[TEST_COLUMNS].copy()
    data_low.loc[0, feature] = bad_value_low
    with pytest.raises(ValueError, match=f"Found 1 measurements outside valid range for {feature}"):
        preprocessor.check_data_integrity(data_low)

    data_high = valid_data[TEST_COLUMNS].copy()
    data_high.loc[0, feature] = bad_value_high
    with pytest.raises(ValueError, match=f"Found 1 measurements outside valid range for {feature}"):
        preprocessor.check_data_integrity(data_high)

def test_prepreprocess_valid_data(preprocessor, valid_data):
    processed_df = preprocessor.prepreprocess(valid_data.copy()) # Pass copy to avoid modifying fixture
    pd.testing.assert_frame_equal(processed_df, valid_data[TEST_COLUMNS])
    assert list(processed_df.columns) == TEST_COLUMNS
    assert len(processed_df) == 3 # No rows should be dropped

def test_prepreprocess_with_nan(preprocessor, data_with_nan):
    processed_df = preprocessor.prepreprocess(data_with_nan.copy())
    assert len(processed_df) == 0 # All rows have at least one NaN in selected features or species
    assert list(processed_df.columns) == TEST_COLUMNS

def test_prepreprocess_drops_nan_selects_cols(preprocessor):
    # More complex case: NaNs and extra columns
    data = pd.DataFrame({
        'flipper_length_mm': [200.0, np.nan, 210.0, 205.0],
        'bill_length_mm': [40.0, 50.0, np.nan, 45.0],
        'bill_depth_mm': [15.0, 18.0, 19.0, 16.0], # No NaN in this row for selected features
        'species': ['Adelie', 'Gentoo', 'Chinstrap', 'Adelie'],
        'extra_col': [1, 2, 3, 4],
        'another_extra': ['a', 'b', 'c', 'd']
    })
    # Expected: row 1 (index 0) and row 4 (index 3) should remain.
    # Row 2 (index 1) has NaN flipper_length_mm. Row 3 (index 2) has NaN bill_length_mm.

    # Create expected DataFrame
    expected_data = pd.DataFrame({
        'flipper_length_mm': [200.0, 205.0],
        'bill_length_mm': [40.0, 45.0],
        'bill_depth_mm': [15.0, 16.0],
        'species': ['Adelie', 'Adelie']
    }).reset_index(drop=True)


    processed_df = preprocessor.prepreprocess(data.copy())
    # Reset index for comparison as dropna can change indices
    processed_df = processed_df.reset_index(drop=True)

    pd.testing.assert_frame_equal(processed_df, expected_data)
    assert list(processed_df.columns) == TEST_COLUMNS
    assert len(processed_df) == 2
