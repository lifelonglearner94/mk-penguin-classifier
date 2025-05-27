"""
Entry point script for running the penguin classifier training pipeline.
"""
from training_logic.ml_controller import run_preprocessing_pipeline, run_training_pipeline
from training_logic.config import logger

if __name__ == "__main__":
    logger.info("Starting complete pipeline: Preprocessing and Training")
    try:
        logger.info("Executing preprocessing pipeline...")
        preprocessing_run_id = run_preprocessing_pipeline()
        logger.info(f"Preprocessing pipeline completed. Run ID: {preprocessing_run_id}")

        logger.info("Executing model training pipeline...")
        run_training_pipeline(preprocessing_run_id=preprocessing_run_id)
        logger.info("Model training pipeline completed successfully")

    except Exception as e:
        logger.error(f"Complete pipeline failed: {str(e)}", exc_info=True)
        # Depending on the desired behavior, you might want to exit with an error code
        # import sys
        # sys.exit(1)
        raise
