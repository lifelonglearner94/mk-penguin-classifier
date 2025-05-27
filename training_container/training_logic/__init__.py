"""
Training logic package for penguin classification model.
"""

from .data_management import DataManager
from .preprocess import Preprocessor
from .ml_controller import run_preprocessing_pipeline, run_training_pipeline # Expose these instead if needed
import logging
# Setup logging for the training logic package
def setup_logging(level=logging.INFO, format_string=None):
    """
    Setup basic logging configuration for the application.

    Args:
        level: Logging level (default: INFO)
        format_string: Custom format string (optional)
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=[
            logging.StreamHandler(),
        ]
    )
setup_logging()

__all__ = ['DataManager', 'Preprocessor', 'run_preprocessing_pipeline', 'run_training_pipeline']
