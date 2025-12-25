# Sugarcane Disease Detection System

A comprehensive deep learning system for detecting and localizing diseases in sugarcane leaves using computer vision and grid-based localization. Features a modern web interface for easy image upload and analysis.

## Features

### Core Detection Capabilities
- **Multi-class Disease Classification**: Detects Yellow, Red Rot, Rust, and Mosaic diseases
- **Healthy Leaf Detection**: Identifies healthy sugarcane leaves
- **Random Image Filtering**: Detects when non-sugarcane images are provided
- **Grid-based Disease Localization**: Divides images into grids and highlights affected regions
- **Comprehensive Visualization**: Generates detailed reports with heatmaps and statistics
- **Confusion Matrix Analysis**: View model performance metrics and per-class accuracy
- **Automatic Image Organization**: Classified images are automatically saved to organized folders

### Deployment Options
- **Web Interface**: User-friendly web application with real-time predictions
- **REST API**: RESTful API endpoints for integration with other applications
- **Command-line Interface**: Scripts for batch processing and automation
- **Python Module**: Import and use as a library in your own code

### Performance & Flexibility
- **Batch Processing**: Process multiple images at once
- **Multiple Model Architectures**: Supports ResNet, EfficientNet, MobileNet, and custom CNN
- **Real-time Inference**: Fast prediction times suitable for production use

## Project Structure

```
.
├── Core Scripts
│   ├── config.py                      # Configuration settings
│   ├── dataset.py                     # Dataset and data preprocessing
│   ├── model.py                       # Model architectures
│   ├── grid_localization.py           # Grid-based disease localization
│   ├── visualization.py               # Visualization utilities
│   ├── utils.py                       # Dataset utilities
│   └── requirements.txt               # Python dependencies
│
├── Training & Evaluation
│   ├── train.py                       # Training pipeline
│   └── generate_confusion_matrix.py   # Generate confusion matrix from trained model
│
├── Inference
│   ├── inference.py                   # Command-line inference script
│   └── app.py                         # Flask web application + REST API
│
├── Documentation
│   ├── README.md                      # Comprehensive documentation (this file)
│   └── QUICKSTART.md                  # Quick start guide
│
├── Data Directories (created automatically or by utils.py)
│   ├── data/
│   │   ├── train/                     # Training data (70%)
│   │   │   ├── Healthy/
│   │   │   ├── Yellow/
│   │   │   ├── RedRot/
│   │   │   ├── Rust/
│   │   │   ├── Mosaic/
│   │   │   └── Other/
│   │   ├── val/                       # Validation data (15%)
│   │   │   └── (same structure as train)
│   │   ├── test/                      # Test data (15%)
│   │   │   └── (same structure as train)
│   │   └── classification/            # Auto-organized classified images (by predicted class)
│   │       ├── Healthy/
│   │       ├── Yellow/
│   │       ├── RedRot/
│   │       ├── Rust/
│   │       ├── Mosaic/
│   │       └── Other/
│   │
│   ├── models/                        # Trained models (auto-created)
│   │   ├── best_model.pth             # Best model checkpoint
│   │   ├── final_model.pth            # Final model after all epochs
│   │   ├── training_history.png       # Training curves
│   │   └── confusion_matrix.png       # Model performance matrix
│   │
│   ├── output/                        # Inference outputs (auto-created)
│   │   ├── *_annotated.png            # Annotated images
│   │   └── *_detailed_report.png      # Detailed analysis reports
│   │
│   ├── uploads/                       # Web app uploads (auto-created)
│   └── static/                        # Web frontend files
│       └── index.html                 # Web interface
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- 8GB+ RAM

### Setup

1. Clone or download this project

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

Get up and running in 5 steps:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare dataset (if you have images organized by class)
python utils.py --split path/to/your_images --data-dir data

# 3. Train the model
python train.py

# 4. Generate confusion matrix
python generate_confusion_matrix.py

# 5. Start the web interface
python app.py
```

Then open `http://localhost:5000` in your browser and start detecting diseases!

For detailed instructions, see [QUICKSTART.md](QUICKSTART.md)

## Dataset Preparation

Organize your dataset in the following structure:

```
data/
├── train/
│   ├── Healthy/       # Healthy leaf images
│   ├── Yellow/        # Yellow disease images
│   ├── RedRot/        # Red Rot disease images
│   ├── Rust/          # Rust disease images
│   ├── Mosaic/        # Mosaic disease images
│   └── Other/         # Random non-sugarcane images
├── val/
│   └── (same structure)
└── test/
    └── (same structure)
```

**Tips for dataset preparation:**
- Use at least 100-200 images per class for good performance
- Include the "Other" class with random images to help the model reject invalid inputs
- Ensure images are clear and properly labeled
- Use diverse lighting conditions and angles

## Training

### Basic Training

```bash
python train.py
```

### Configuration

Edit `config.py` to customize training parameters:

```python
# Image Configuration
IMAGE_SIZE = (224, 224)
GRID_SIZE = (4, 4)  # 4x4 grid for localization

# Training Configuration
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Model Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### Training Output

The training script will:
- Train the model for the specified number of epochs
- Save the best model based on validation accuracy
- Generate training history plots
- Save checkpoints every 10 epochs

Outputs will be saved in the `models/` directory:
- `best_model.pth` - Best model checkpoint
- `final_model.pth` - Final model after all epochs
- `training_history.png` - Loss and accuracy plots

## Model Evaluation

### Generate Confusion Matrix

After training (or anytime with an existing model), generate the confusion matrix:

```bash
python generate_confusion_matrix.py
```

This script will:
- Load the trained model from `models/best_model.pth`
- Evaluate on the validation dataset
- Generate and save confusion matrix to `models/confusion_matrix.png`
- Display detailed classification report
- Show per-class accuracy

**Output Example:**
```
============================================================
Generating Confusion Matrix from Trained Model
============================================================

[OK] Model loaded (trained for 33 epochs)
  Validation Accuracy: 0.9832

[OK] Evaluation complete!
  Overall Accuracy: 0.9832 (98.32%)

Per-Class Accuracy:
  Healthy      : 1.0000 (100.00%) - 72 samples
  Yellow       : 0.9275 (92.75%) - 69 samples
  RedRot       : 0.9710 (97.10%) - 69 samples
  Rust         : 1.0000 (100.00%) - 69 samples
  Mosaic       : 0.9855 (98.55%) - 69 samples
  Other        : 1.0000 (100.00%) - 127 samples
```

**Options:**
```bash
# Use specific model
python generate_confusion_matrix.py --model models/checkpoint_epoch_30.pth

# Use different architecture
python generate_confusion_matrix.py --model_type resnet18

# Save to different directory
python generate_confusion_matrix.py --save_dir evaluation/
```

## Web Application

### Starting the Web Server

Launch the web interface with:

```bash
python app.py
```

Then open your browser to: `http://localhost:5000`

### Web Interface Features

The web application provides:
- **Drag-and-drop or click to upload** leaf images
- **Real-time analysis** with progress indication
- **Visual results** including:
  - Original and annotated images
  - Disease probability chart
  - Grid analysis statistics
  - Diseased region details
  - Detailed report with heatmap
  - **Confusion matrix** showing model performance
- **Responsive design** that works on desktop and mobile

### Web Interface Screenshot Flow

1. Upload image (drag & drop or click)
2. View prediction results with confidence scores
3. See highlighted diseased regions
4. View detailed heatmap analysis
5. Check confusion matrix for model reliability

### Automatic Image Organization

When an image is classified through the web interface:
1. The image is processed and classified
2. The image is **automatically saved** to `data/classification/{predicted_class}/`
3. This creates an organized collection of classified images by disease type

**Example**: An image classified as "Yellow" will be saved to `data/classification/Yellow/`

This feature is useful for:
- Building a database of classified samples
- Reviewing classification history
- Quality control and validation
- Creating training datasets from real-world usage

## REST API

### API Endpoints

#### 1. Health Check
```bash
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_exists": true
}
```

#### 2. Predict Disease
```bash
POST /api/predict
Content-Type: multipart/form-data
```

**Request:**
- `file`: Image file (JPG, PNG, BMP)

**Note:** The uploaded image is automatically saved to `data/classification/{predicted_class}/` for organization and future reference.

**Response:**
```json
{
  "success": true,
  "prediction": {
    "class": "Rust",
    "confidence": 0.9432,
    "probabilities": {
      "Healthy": 0.0321,
      "Yellow": 0.0455,
      "RedRot": 0.0187,
      "Rust": 0.9432,
      "Mosaic": 0.0211,
      "Other": 0.0394
    }
  },
  "grid_analysis": {
    "total_grids": 16,
    "diseased_grids": 4,
    "details": [
      {
        "grid_idx": 5,
        "disease": "Rust",
        "confidence": 0.9245
      }
    ]
  },
  "images": {
    "original": "base64_encoded_image",
    "annotated": "base64_encoded_image",
    "detailed_report": "base64_encoded_image",
    "confusion_matrix": "base64_encoded_image"
  }
}
```

#### 3. Get Confusion Matrix
```bash
GET /api/confusion-matrix
```

**Response:**
```json
{
  "success": true,
  "confusion_matrix": "base64_encoded_image",
  "message": "Confusion matrix from validation set during training"
}
```

#### 4. Get Disease Classes
```bash
GET /api/classes
```

**Response:**
```json
{
  "classes": ["Healthy", "Yellow", "RedRot", "Rust", "Mosaic", "Other"]
}
```

### API Usage Example

**Python:**
```python
import requests

# Upload image for prediction
url = "http://localhost:5000/api/predict"
files = {"file": open("leaf.jpg", "rb")}
response = requests.post(url, files=files)
result = response.json()

print(f"Disease: {result['prediction']['class']}")
print(f"Confidence: {result['prediction']['confidence']:.2%}")
```

**cURL:**
```bash
curl -X POST -F "file=@leaf.jpg" http://localhost:5000/api/predict
```

**JavaScript:**
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('/api/predict', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => {
  console.log('Disease:', data.prediction.class);
  console.log('Confidence:', data.prediction.confidence);
});
```

## Command-line Inference

For automation, batch processing, or integration into scripts, use the command-line interface.

### Single Image Detection

```bash
python inference.py --image path/to/image.jpg --output output/
```

### Batch Processing

```bash
python inference.py --batch path/to/image/directory/ --output output/
```

### Advanced Options

```bash
# Generate detailed report with heatmaps
python inference.py --image test.jpg --detailed

# Use different model architecture
python inference.py --image test.jpg --model_type resnet18

# Use specific model checkpoint
python inference.py --image test.jpg --model models/checkpoint_epoch_30.pth
```

### Command-line Arguments

- `--image, -i`: Path to single input image
- `--batch, -b`: Path to directory containing multiple images
- `--model, -m`: Path to trained model (default: `models/best_model.pth`)
- `--model_type`: Model architecture (resnet50, resnet18, efficientnet_b0, mobilenet_v2, custom)
- `--output, -o`: Output directory (default: `output/`)
- `--detailed`: Generate detailed reports with heatmaps

### Output Files

For each processed image, the system generates:

1. **Annotated Image** (`*_annotated.png`): Original image with highlighted diseased grids
2. **Result Visualization** (`*_result.png`): Side-by-side comparison with predictions
3. **Detailed Report** (`*_detailed_report.png`): Comprehensive analysis including:
   - Original and annotated images
   - Disease probability heatmap
   - Class probability bar chart
   - Detection summary statistics

For batch processing, an additional `batch_summary.txt` file is generated with overall statistics.

## Detection Output Examples

### Console Output

```
==============================================================
DETECTION RESULTS
==============================================================

Overall Classification:
  Prediction: Rust
  Confidence: 94.32%

Class Probabilities:
  Healthy      : ██                             3.21%
  Yellow       : ███                            4.55%
  RedRot       : █                              1.87%
  Rust         : ████████████████████████████   94.32%
  Mosaic       : █                              2.11%
  Other        : ██                             3.94%

Grid Analysis:
  Total Grids: 16
  Diseased Grids: 4

Diseased Grid Details:
  1. Grid 5: Rust (Confidence: 92.45%)
  2. Grid 6: Rust (Confidence: 89.73%)
  3. Grid 9: Rust (Confidence: 91.28%)
  4. Grid 10: Rust (Confidence: 88.91%)

==============================================================
ASSESSMENT:
⚠ Disease Detected: Rust
  Affected regions: 4 grid(s)
  Check the output visualization for affected areas
==============================================================
```

## Grid Localization Details

The system uses a grid-based approach to localize diseases:

1. **Grid Division**: The input image is divided into a configurable grid (default: 4x4)
2. **Grid Classification**: Each grid cell is classified independently
3. **Confidence Filtering**: Only grids with confidence above threshold are marked as diseased
4. **Visualization**: Diseased grids are highlighted with colored borders and semi-transparent overlays

### Adjusting Grid Size

Edit `config.py` to change grid size:

```python
GRID_SIZE = (4, 4)  # 4x4 grid (16 cells)
# or
GRID_SIZE = (8, 8)  # 8x8 grid (64 cells) - finer localization
```

## Model Architectures

The system supports multiple CNN architectures:

### Pre-trained Models (Recommended)

- **ResNet50** (default): Best accuracy, moderate speed
- **ResNet18**: Faster, good accuracy
- **EfficientNet-B0**: Excellent accuracy, efficient
- **MobileNet-V2**: Fast inference, lower accuracy

### Custom CNN

A lightweight custom architecture for resource-constrained environments.

### Selecting Model Type

When training:
```python
# In model.py or train.py
model = create_model(model_type='resnet50', pretrained=True)
```

When running inference:
```bash
python inference.py --image test.jpg --model_type efficientnet_b0
```

## Performance Tips

### Training
- Start with a pre-trained model (ResNet50 or EfficientNet-B0)
- Use data augmentation (enabled by default)
- Monitor validation accuracy to avoid overfitting
- Use a GPU for faster training
- Adjust learning rate if training stagnates

### Inference
- Use batch processing for multiple images
- Consider using lighter models (MobileNet-V2) for real-time applications
- Adjust `CONFIDENCE_THRESHOLD` in config.py to control sensitivity

## Troubleshooting

### Web Application Issues

#### Port Already in Use
```bash
# Change port in app.py or use environment variable
export FLASK_PORT=5001
python app.py
```

#### Model Not Found (Web App)
```bash
# Train the model first
python train.py

# Or check that models/best_model.pth exists
ls models/best_model.pth
```

#### CORS Errors
The app uses `flask-cors` which should handle cross-origin requests. If you encounter issues:
```bash
pip install flask-cors --upgrade
```

#### Confusion Matrix Not Showing
```bash
# Generate the confusion matrix
python generate_confusion_matrix.py

# Verify it exists
ls models/confusion_matrix.png
```

### Training Issues

#### CUDA Out of Memory
```python
# Reduce batch size in config.py
BATCH_SIZE = 16  # or even lower like 8
```

#### Poor Accuracy
- Ensure dataset is properly labeled
- Add more training data (100+ images per class minimum)
- Increase number of epochs
- Try different model architectures
- Check for class imbalance
- Verify data augmentation is working

#### Training Very Slow
- Use a GPU if available (check with `torch.cuda.is_available()`)
- Reduce image size in config.py
- Use a lighter model (ResNet18 or MobileNet-V2)
- Reduce batch size if using CPU

### Inference Issues

#### Model Not Found Error
```bash
# Make sure you've trained the model first
python train.py

# Or specify correct model path
python inference.py --image test.jpg --model path/to/your/model.pth
```

#### Low Confidence Predictions
- Check image quality (clear, well-lit, focused)
- Ensure image contains a sugarcane leaf
- Verify model was trained properly
- Check training accuracy metrics

### Installation Issues

#### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

#### Missing Modules (seaborn, flask, etc.)
```bash
# Install specific packages
pip install seaborn flask flask-cors scikit-learn

# Or reinstall all
pip install -r requirements.txt
```

#### Albumentations Warnings
These are usually harmless warnings about deprecated parameters. You can ignore them or update:
```bash
pip install albumentations --upgrade
```

## Advanced Usage

### Using as Python Module

#### Detection Module
```python
from inference import SugarcaneDiseaseDetector

# Initialize detector
detector = SugarcaneDiseaseDetector(
    model_path='models/best_model.pth',
    model_type='resnet50'
)

# Detect disease
results = detector.detect('path/to/image.jpg', output_dir='output/')

# Access results
print(f"Disease: {results['overall_prediction']['class_name']}")
print(f"Confidence: {results['overall_prediction']['confidence']:.2%}")
print(f"Diseased grids: {len(results['diseased_grids'])}")

# Access detailed predictions
for grid in results['diseased_grids']:
    print(f"Grid {grid['grid_idx']}: {grid['disease']} ({grid['confidence']:.2%})")
```

#### Visualization Module
```python
from visualization import Visualizer

# Create visualizer
viz = Visualizer()

# Create visualizations from results
viz.create_result_visualization(results, save_path='output.png')
viz.create_detailed_report(results, save_path='report.png')
viz.save_annotated_image(results, save_path='annotated.png')

# View confusion matrix
viz.create_performance_report(
    confusion_matrix_path='models/confusion_matrix.png',
    save_path='performance.png'
)
```

#### Web Application Integration
```python
from flask import Flask, request, jsonify
from inference import SugarcaneDiseaseDetector
import base64

app = Flask(__name__)
detector = SugarcaneDiseaseDetector('models/best_model.pth')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    file.save('temp.jpg')

    results = detector.detect('temp.jpg')

    return jsonify({
        'disease': results['overall_prediction']['class_name'],
        'confidence': float(results['overall_prediction']['confidence'])
    })

if __name__ == '__main__':
    app.run(port=5000)
```

### Custom Training Loop

```python
from train import Trainer
from model import create_model
from dataset import get_dataloaders
import torch.nn as nn
import torch.optim as optim

# Setup
model = create_model('resnet50')
train_loader, val_loader = get_dataloaders()
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

# Train
trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, scheduler)
history = trainer.train(num_epochs=50)
```

## Citation

If you use this system in your research, please cite:

```
@software{sugarcane_disease_detection,
  title = {Sugarcane Disease Detection System},
  author = {Your Name},
  year = {2025},
  description = {Deep learning system for sugarcane leaf disease detection and localization}
}
```

## License

This project is open-source and available for educational and research purposes.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Contact

For questions or support, please open an issue in the repository.

---

**Note**: This system is designed for research and educational purposes. For production deployment, additional validation and testing is recommended.
