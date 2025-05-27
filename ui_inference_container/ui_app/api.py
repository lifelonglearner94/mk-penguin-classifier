"""
Standalone API service for penguin classifier.
Run this as a separate Flask application on port 8080.
"""
import logging
from flask import Flask, request, jsonify
from .config import API_KEY as DEFAULT_API_KEY

# Create Flask app instead of Blueprint
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize InferenceService with better error handling
inference_service = None
try:
    from .inference_service import InferenceService
    inference_service = InferenceService()
    logger.info("InferenceService initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize InferenceService: {e}", exc_info=True)
    logger.warning("API will start without inference service - predictions will fail")

@app.route('/api/predict', methods=['POST'])
def predict_route():
    """Handle penguin classification predictions via API endpoint."""
    logger.info(f"/api/predict route entered. Request headers: {request.headers}")

    if inference_service is None:
        logger.error("InferenceService not available")
        return jsonify({"error": "Service unavailable", "message": "Inference service not initialized"}), 503

    # API Key Authentication
    expected_api_key = DEFAULT_API_KEY
    submitted_api_key = request.headers.get("X-API-Key")

    if not expected_api_key or not submitted_api_key or submitted_api_key != expected_api_key:
        logger.warning(f"API key validation failed. Submitted: '{submitted_api_key}'. Expected: '{expected_api_key}' (masked if long)")
        return jsonify({"error": "Unauthorized", "message": "Invalid or missing API Key"}), 401

    if not request.is_json:
        logger.warning("Request is not JSON")
        return jsonify({"error": "Invalid input", "message": "Request must be JSON"}), 400

    data = request.get_json()
    logger.info(f"Received JSON data: {data}")

    # Validate presence of required features
    required_fields = inference_service.features
    missing_fields = [field for field in required_fields if field not in data or data[field] is None]

    if missing_fields:
        logger.warning(f"Missing or None fields in request: {missing_fields}")
        return jsonify({"error": "Invalid input", "message": f"Missing or null fields: {', '.join(missing_fields)}"}), 400

    features = {field: data[field] for field in required_fields}

    try:
        logger.info(f"Calling inference_service.predict with features: {features}")
        result = inference_service.predict(features)
        logger.info(f"Inference service returned: {result}")
        return jsonify(result)
    except ValueError as ve:
        logger.warning(f"Validation error for /predict: {str(ve)}")
        return jsonify({"error": "Validation error", "message": str(ve)}), 400
    except RuntimeError as re:
        logger.error(f"Runtime error for /predict: {str(re)}", exc_info=True)
        return jsonify({"error": "Prediction runtime error", "message": str(re)}), 500
    except Exception as e:
        logger.error(f"Unexpected error for /predict: {str(e)}", exc_info=True)
        return jsonify({"error": "Unexpected server error", "message": str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting API service on port 8080...")
    app.run(debug=True, host='0.0.0.0', port=8080)
