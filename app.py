"""
Flask API for Sugarcane Disease Detection
Provides REST endpoints for image upload and prediction
"""

import os
import base64
import shutil
from io import BytesIO
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import torch

from inference import SugarcaneDiseaseDetector
import config

app = Flask(__name__, static_folder='static')
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
MODEL_PATH = 'models/best_model.pth'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs('static', exist_ok=True)

# Initialize detector (lazy loading)
detector = None


def get_detector():
    """Lazy load the detector"""
    global detector
    if detector is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please train the model first.")
        detector = SugarcaneDiseaseDetector(model_path=MODEL_PATH, model_type='resnet50')
    return detector


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def image_to_base64(image_path):
    """Convert image to base64 string"""
    with open(image_path, 'rb') as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('static', 'index.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        model_exists = os.path.exists(MODEL_PATH)
        return jsonify({
            'status': 'healthy',
            'model_loaded': detector is not None,
            'model_exists': model_exists
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict disease from uploaded image

    Request: multipart/form-data with 'file' field
    Response: JSON with prediction results
    """
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg, bmp'}), 400

        # Save uploaded file
        filename = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filename)

        # Get detector
        det = get_detector()

        # Run prediction
        results = det.detect(filename, output_dir=OUTPUT_FOLDER, detailed_report=True)

        # Save classified image to data/classification/{class_name}/
        predicted_class = results['overall_prediction']['class_name']
        classification_dir = os.path.join('data', 'classification', predicted_class)
        os.makedirs(classification_dir, exist_ok=True)

        # Copy the uploaded image to the classification folder
        destination_path = os.path.join(classification_dir, file.filename)
        shutil.copy2(filename, destination_path)

        # Get base name for output files
        base_name = os.path.splitext(os.path.basename(filename))[0]
        annotated_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_annotated.png")
        detailed_report_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_detailed_report.png")

        # Path to confusion matrix (from training)
        confusion_matrix_path = os.path.join('models', 'confusion_matrix.png')

        # Convert images to base64
        response_data = {
            'success': True,
            'prediction': {
                'class': results['overall_prediction']['class_name'],
                'confidence': float(results['overall_prediction']['confidence']),
                'probabilities': {
                    cls: float(prob)
                    for cls, prob in zip(config.CLASSES, results['overall_prediction']['probabilities'])
                }
            },
            'grid_analysis': {
                'total_grids': len(results['grid_predictions']),
                'diseased_grids': len(results['diseased_grids']),
                'details': [
                    {
                        'grid_idx': grid['grid_idx'],
                        'disease': grid['disease'],
                        'confidence': float(grid['confidence'])
                    }
                    for grid in results['diseased_grids']
                ]
            },
            'images': {
                'original': image_to_base64(filename),
                'annotated': image_to_base64(annotated_path) if os.path.exists(annotated_path) else None,
                'detailed_report': image_to_base64(detailed_report_path) if os.path.exists(detailed_report_path) else None,
                'confusion_matrix': image_to_base64(confusion_matrix_path) if os.path.exists(confusion_matrix_path) else None
            }
        }

        return jsonify(response_data)

    except FileNotFoundError as e:
        return jsonify({
            'success': False,
            'error': f'Model not found. Please train the model first using train.py'
        }), 500

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/classes', methods=['GET'])
def get_classes():
    """Get list of disease classes"""
    return jsonify({
        'classes': config.CLASSES
    })


@app.route('/api/confusion-matrix', methods=['GET'])
def get_confusion_matrix():
    """
    Get the confusion matrix from the trained model

    Response: JSON with base64 encoded confusion matrix image
    """
    try:
        confusion_matrix_path = os.path.join('models', 'confusion_matrix.png')

        if not os.path.exists(confusion_matrix_path):
            return jsonify({
                'success': False,
                'error': 'Confusion matrix not found. Please train the model first.'
            }), 404

        # Convert to base64
        cm_base64 = image_to_base64(confusion_matrix_path)

        return jsonify({
            'success': True,
            'confusion_matrix': cm_base64,
            'message': 'Confusion matrix from validation set during training'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    print("="*60)
    print("Sugarcane Disease Detection API")
    print("="*60)
    print(f"\nModel path: {MODEL_PATH}")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Output folder: {OUTPUT_FOLDER}")

    if not os.path.exists(MODEL_PATH):
        print("\n⚠ WARNING: Model file not found!")
        print(f"Please train the model first using: python train.py")
    else:
        print("\n✓ Model file found")

    print("\nStarting server...")
    print("Open http://localhost:5000 in your browser")
    print("="*60 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)
