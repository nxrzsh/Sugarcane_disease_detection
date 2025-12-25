SUGARCANE DISEASE DETECTION SYSTEM
PROJECT EXECUTION INSTRUCTIONS

Course: [Your Course Name]
Submitted By: [Your Name]
Roll Number: [Your Roll Number]
Date: December 2, 2025

================================================================================

TABLE OF CONTENTS

1. Project Overview
2. System Requirements
3. Installation Instructions
4. Dataset Preparation
5. Model Training
6. Model Evaluation
7. Running the Application
8. Testing the System
9. Expected Results
10. Troubleshooting

================================================================================

1. PROJECT OVERVIEW

This project implements a comprehensive deep learning system for detecting and localizing diseases in sugarcane leaves using computer vision. The system uses Convolutional Neural Networks (CNN) to classify leaf images into different disease categories and provides a user-friendly web interface for real-time disease detection.

Key Features:
• Multi-class Disease Classification: Detects Yellow, Red Rot, Rust, and Mosaic diseases
• Healthy Leaf Detection: Identifies healthy sugarcane leaves
• Grid-based Disease Localization: Divides images into grids and highlights affected regions
• Web Interface: User-friendly interface for image upload and analysis
• REST API: RESTful endpoints for integration with other applications
• Automatic Image Organization: Classified images are automatically saved to organized folders

Disease Classes Supported:
1. Healthy - Normal, disease-free sugarcane leaves
2. Yellow - Yellow leaf disease
3. Red Rot - Red rot fungal disease
4. Rust - Rust disease caused by Puccinia species
5. Mosaic - Mosaic virus disease
6. Other - Non-sugarcane or random images (for filtering invalid inputs)

================================================================================

2. SYSTEM REQUIREMENTS

Hardware Requirements:
• Processor: Intel Core i5 or equivalent (i7 recommended)
• RAM: Minimum 8GB (16GB recommended)
• Storage: At least 5GB free disk space
• GPU: CUDA-capable GPU (optional, but recommended for faster training)

Software Requirements:
• Operating System: Windows 10/11, Linux, or macOS
• Python: Version 3.8 or higher
• Web Browser: Chrome, Firefox, or Edge (for web interface)
• Internet Connection: Required for initial dependency installation

================================================================================

3. INSTALLATION INSTRUCTIONS

Step 3.1: Verify Python Installation

Open Command Prompt (Windows) or Terminal (Linux/macOS) and verify Python version:

    python --version

Expected output: Python 3.8.x or higher


Step 3.2: Navigate to Project Directory

    cd C:\Users\Admin1\Documents\naresh\ultimato


Step 3.3: Install Required Dependencies

Install all required Python packages using pip:

    pip install -r requirements.txt

Expected time: 2-5 minutes

Dependencies include:
• PyTorch: Deep learning framework
• torchvision: Computer vision utilities
• Flask: Web framework
• OpenCV: Image processing
• NumPy, Pandas: Data manipulation
• Matplotlib, Seaborn: Visualization
• Albumentations: Data augmentation
• scikit-learn: Machine learning utilities


Step 3.4: Verify Installation

Check if PyTorch is properly installed:

    python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

================================================================================

4. DATASET PREPARATION

The dataset must be organized into training, validation, and test sets with proper folder structure.

Step 4.1: Understanding Dataset Structure

The required folder structure:

    data/
    ├── train/          (70% of images)
    │   ├── Healthy/
    │   ├── Yellow/
    │   ├── RedRot/
    │   ├── Rust/
    │   ├── Mosaic/
    │   └── Other/
    ├── val/            (15% of images)
    │   └── (same structure)
    └── test/           (15% of images)
        └── (same structure)


Step 4.2: Automatic Dataset Split (Recommended)

If you have images organized by class in a single folder:

    python utils.py --split path/to/your_images --data-dir data

This command automatically:
• Creates the required folder structure
• Splits images into train (70%), validation (15%), and test (15%) sets
• Maintains class balance across splits


Step 4.3: Manual Dataset Organization (Alternative)

If you prefer manual organization:

    python utils.py --create-structure --data-dir data

Then manually copy images to appropriate folders.


Step 4.4: Verify Dataset

Verify the dataset is correctly organized:

    python utils.py --verify --count

Expected output: Summary of images in each class for train/val/test sets


Dataset Requirements:
• Minimum images per class: 100-200 images recommended
• Image formats: JPG, PNG, BMP
• Image quality: Clear, well-lit, focused images
• Diversity: Include various lighting conditions and angles

================================================================================

5. MODEL TRAINING

Step 5.1: Configure Training Parameters (Optional)

Edit config.py to customize training settings:

    IMAGE_SIZE = (224, 224)
    GRID_SIZE = (4, 4)
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001


Step 5.2: Start Training

Run the training script:

    python train.py


Step 5.3: Monitor Training Progress

The training script displays:
• Current epoch and progress
• Training loss and accuracy
• Validation loss and accuracy
• Estimated time remaining

Sample output:

    Epoch 1/50
    Training: 100%|████████████| 75/75 [02:34<00:00]
    Train Loss: 1.2345, Train Acc: 0.6543
    Val Loss: 1.1234, Val Acc: 0.7123

    Best model saved! (Val Acc: 0.7123)


Step 5.4: Training Duration

Expected training time:
• With GPU: 10-20 minutes for 3000 images
• Without GPU: 30-60 minutes for 3000 images


Step 5.5: Training Outputs

Training generates the following files in models/ directory:

1. best_model.pth - Model with best validation accuracy
2. final_model.pth - Model after all epochs
3. training_history.png - Loss and accuracy curves
4. checkpoint_epoch_X.pth - Checkpoints every 10 epochs

================================================================================

6. MODEL EVALUATION

Step 6.1: Generate Confusion Matrix

After training completes, evaluate the model:

    python generate_confusion_matrix.py


Step 6.2: Understanding Evaluation Results

The script displays:

    ============================================================
    Generating Confusion Matrix from Trained Model
    ============================================================

    [OK] Model loaded (trained for 50 epochs)
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


Step 6.3: Evaluation Outputs

Generated files:
• confusion_matrix.png - Visual representation of model performance
• classification_report.txt - Detailed metrics (precision, recall, F1-score)

================================================================================

7. RUNNING THE APPLICATION

Step 7.1: Launch Web Server

Start the Flask web application:

    python app.py

Expected output:

    * Running on http://127.0.0.1:5000
    * Running on http://localhost:5000


Step 7.2: Access Web Interface

Open your web browser and navigate to:

    http://localhost:5000


Step 7.3: Web Interface Features

The web application provides:

1. Image Upload
   • Drag and drop image upload
   • Or click to browse and select image
   • Supports JPG, PNG, BMP formats

2. Real-time Analysis
   • Automatic disease detection
   • Progress indication during processing
   • Results displayed within 2-5 seconds

3. Visual Results
   • Original and annotated images
   • Disease probability chart
   • Grid analysis statistics
   • Detailed heatmap report
   • Confusion matrix visualization

4. Automatic Organization
   • Uploaded images automatically saved to data/classification/{disease}/
   • Useful for building classified image database

================================================================================

8. TESTING THE SYSTEM

Step 8.1: Single Image Prediction (Command-line)

Test with a single image:

    python inference.py --image path/to/test_image.jpg --output output/


Step 8.2: Batch Processing

Process multiple images at once:

    python inference.py --batch path/to/test_images/ --output output/


Step 8.3: Detailed Analysis

Generate comprehensive report with heatmaps:

    python inference.py --image test.jpg --detailed --output output/


Step 8.4: Understanding Output Files

For each processed image:

1. {filename}_annotated.png - Image with highlighted diseased regions
2. {filename}_result.png - Side-by-side comparison
3. {filename}_detailed_report.png - Comprehensive analysis with statistics


Step 8.5: Sample Detection Output

Console output example:

    ==============================================================
    DETECTION RESULTS
    ==============================================================

    Overall Classification:
      Prediction: Rust
      Confidence: 94.32%

    Class Probabilities:
      Healthy      : 3.21%
      Yellow       : 4.55%
      RedRot       : 1.87%
      Rust         : 94.32%
      Mosaic       : 2.11%
      Other        : 3.94%

    Grid Analysis:
      Total Grids: 16
      Diseased Grids: 4

    Diseased Grid Details:
      1. Grid 5: Rust (Confidence: 92.45%)
      2. Grid 6: Rust (Confidence: 89.73%)
      3. Grid 9: Rust (Confidence: 91.28%)
      4. Grid 10: Rust (Confidence: 88.91%)

    ==============================================================
    ASSESSMENT: Disease Detected - Rust
    Affected regions: 4 grid(s)
    ==============================================================

================================================================================

9. EXPECTED RESULTS

Model Performance Metrics:

With 3000+ well-labeled images:
• Overall Accuracy: 95-98%
• Per-class Accuracy: 90-100%
• Processing Time: 2-5 seconds per image
• F1-Score: 0.93-0.98


Confusion Matrix Interpretation:

The confusion matrix shows:
• Diagonal values (dark): Correct predictions
• Off-diagonal values (light): Misclassifications
• Row percentages: Recall (sensitivity)
• Column percentages: Precision


Grid Localization Accuracy:

• Grid size: 4x4 (16 cells)
• Localization precision: 85-90%
• False positive rate: <5%

================================================================================

10. TROUBLESHOOTING

Issue 1: "Model not found" Error

Solution:
    python train.py
    dir models\best_model.pth


Issue 2: "Port 5000 already in use"

Solution: Change port in app.py (line 222):
    app.run(host='0.0.0.0', port=5001, debug=False)


Issue 3: CUDA Out of Memory

Solution: Reduce batch size in config.py:
    BATCH_SIZE = 16


Issue 4: Poor Accuracy (< 85%)

Possible causes and solutions:
• Insufficient training data → Add more images (100+ per class)
• Mislabeled images → Verify dataset labels
• Class imbalance → Ensure similar number of images per class
• Underfitting → Increase epochs or use larger model


Issue 5: Slow Training (No GPU)

Solution:
• Reduce image size in config.py
• Use lighter model (ResNet18 or MobileNet-V2)
• Be patient (30-60 minutes expected)


Issue 6: Import Errors

Solution:
    pip install -r requirements.txt --upgrade


Issue 7: Confusion Matrix Not Showing

Solution:
    python generate_confusion_matrix.py
    dir models\confusion_matrix.png

================================================================================

SUMMARY CHECKLIST

Before submitting, ensure you have completed:

☐ Installed all dependencies (pip install -r requirements.txt)
☐ Prepared dataset with proper folder structure
☐ Verified dataset (python utils.py --verify)
☐ Trained model successfully (python train.py)
☐ Generated confusion matrix (python generate_confusion_matrix.py)
☐ Tested web interface (python app.py)
☐ Verified model predictions on test images
☐ Reviewed output visualizations and reports

================================================================================

PROJECT DIRECTORY STRUCTURE

    ultimato/
    ├── app.py                          # Flask web application
    ├── train.py                        # Model training script
    ├── inference.py                    # Command-line inference
    ├── generate_confusion_matrix.py    # Evaluation script
    ├── config.py                       # Configuration settings
    ├── model.py                        # Model architectures
    ├── dataset.py                      # Dataset handling
    ├── grid_localization.py            # Grid analysis
    ├── visualization.py                # Visualization utilities
    ├── utils.py                        # Dataset utilities
    ├── requirements.txt                # Dependencies
    ├── README.md                       # Documentation
    ├── QUICKSTART.md                   # Quick start guide
    ├── data/                           # Dataset directory
    │   ├── train/
    │   ├── val/
    │   ├── test/
    │   └── classification/
    ├── models/                         # Trained models
    │   ├── best_model.pth
    │   ├── training_history.png
    │   └── confusion_matrix.png
    ├── output/                         # Inference outputs
    ├── uploads/                        # Web app uploads
    └── static/                         # Web interface files

================================================================================

CONCLUSION

This document provides comprehensive instructions to execute the Sugarcane Disease Detection System. The system successfully implements deep learning-based disease detection with high accuracy and provides multiple interfaces for user interaction.

Key Achievements:

1. High Accuracy: 95-98% classification accuracy
2. User-Friendly: Web interface for easy interaction
3. Comprehensive Analysis: Grid-based localization and detailed reports
4. Flexible Deployment: Multiple usage modes (web, API, CLI)
5. Automatic Organization: Classified images saved for database building

Future Enhancements:

• Mobile application development
• Real-time video stream analysis
• Integration with agricultural databases
• Multi-language support
• Cloud deployment

================================================================================

END OF DOCUMENT

Note: For any issues or questions during execution, refer to the comprehensive README.md file in the project directory or consult the troubleshooting section above.
