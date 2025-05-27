"""
UI and inference service package for penguin classification.
"""
import logging
from .app import app, server
from .inference_service import InferenceService


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

__all__ = ['app', 'server', 'InferenceService', 'setup_logging']
