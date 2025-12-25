"""
Configuration file for Sugarcane Disease Detection Project
"""

import torch

# Dataset Configuration
CLASSES = ['Healthy', 'Yellow', 'RedRot', 'Rust', 'Mosaic', 'Other']
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: cls for idx, cls in enumerate(CLASSES)}

# Image Configuration
IMAGE_SIZE = (224, 224)
GRID_SIZE = (4, 4)  # Divide image into 4x4 grid for localization

# Training Configuration
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4

# Model Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_SAVE_PATH = 'models/best_model.pth'

# Data Paths
TRAIN_DIR = 'data/train'
VAL_DIR = 'data/val'
TEST_DIR = 'data/test'

# Augmentation Configuration
TRAIN_AUGMENT = True
VAL_AUGMENT = False

# Grid Detection Configuration
CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence to highlight a grid
GRID_COLOR = (255, 0, 0)  # Red color for highlighting diseased grids
GRID_THICKNESS = 3

# Random Image Detection
OTHER_CLASS_THRESHOLD = 0.5  # If 'Other' class confidence > threshold, it's a random image
