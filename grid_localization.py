"""
Grid-based Disease Localization System
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import config
from dataset import get_inference_transforms


class GridLocalizer:
    """Localizes disease in image using grid-based approach"""

    def __init__(self, model, grid_size=config.GRID_SIZE, confidence_threshold=config.CONFIDENCE_THRESHOLD):
        """
        Args:
            model: Trained classification model
            grid_size: Tuple (rows, cols) for grid division
            confidence_threshold: Minimum confidence to mark grid as diseased
        """
        self.model = model
        self.model.eval()
        self.grid_size = grid_size
        self.confidence_threshold = confidence_threshold
        self.transform = get_inference_transforms()

    def divide_image_into_grids(self, image):
        """
        Divide image into grid patches

        Args:
            image: PIL Image or numpy array

        Returns:
            grid_patches: List of grid patches
            grid_coords: List of (x, y, w, h) coordinates for each grid
        """
        if isinstance(image, Image.Image):
            image = np.array(image)

        h, w = image.shape[:2]
        grid_rows, grid_cols = self.grid_size

        grid_h = h // grid_rows
        grid_w = w // grid_cols

        grid_patches = []
        grid_coords = []

        for i in range(grid_rows):
            for j in range(grid_cols):
                y1 = i * grid_h
                y2 = (i + 1) * grid_h if i < grid_rows - 1 else h
                x1 = j * grid_w
                x2 = (j + 1) * grid_w if j < grid_cols - 1 else w

                patch = image[y1:y2, x1:x2]
                grid_patches.append(patch)
                grid_coords.append((x1, y1, x2 - x1, y2 - y1))

        return grid_patches, grid_coords

    def classify_grids(self, grid_patches):
        """
        Classify each grid patch

        Args:
            grid_patches: List of image patches

        Returns:
            predictions: List of (class_idx, confidence, probabilities) for each grid
        """
        predictions = []

        with torch.no_grad():
            for patch in grid_patches:
                # Transform patch
                transformed = self.transform(image=patch)
                patch_tensor = transformed['image'].unsqueeze(0).to(config.DEVICE)

                # Get prediction
                logits = self.model(patch_tensor)
                probabilities = F.softmax(logits, dim=1)
                confidence, class_idx = torch.max(probabilities, dim=1)

                predictions.append({
                    'class_idx': class_idx.item(),
                    'class_name': config.IDX_TO_CLASS[class_idx.item()],
                    'confidence': confidence.item(),
                    'probabilities': probabilities.cpu().numpy()[0]
                })

        return predictions

    def localize_disease(self, image_path):
        """
        Main function to localize disease in image

        Args:
            image_path: Path to input image

        Returns:
            results: Dictionary containing:
                - overall_prediction: Overall image classification
                - diseased_grids: List of grid indices with disease
                - grid_predictions: Predictions for all grids
                - grid_coords: Coordinates of all grids
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        original_image = np.array(image)

        # Classify whole image first (for overall prediction)
        transformed = self.transform(image=original_image)
        image_tensor = transformed['image'].unsqueeze(0).to(config.DEVICE)

        with torch.no_grad():
            logits = self.model(image_tensor)
            probabilities = F.softmax(logits, dim=1)
            confidence, class_idx = torch.max(probabilities, dim=1)

            overall_prediction = {
                'class_idx': class_idx.item(),
                'class_name': config.IDX_TO_CLASS[class_idx.item()],
                'confidence': confidence.item(),
                'probabilities': probabilities.cpu().numpy()[0]
            }

        # Divide into grids and classify each
        grid_patches, grid_coords = self.divide_image_into_grids(original_image)
        grid_predictions = self.classify_grids(grid_patches)

        # Find diseased grids (exclude 'Healthy' and 'Other')
        diseased_grids = []
        for idx, pred in enumerate(grid_predictions):
            # Check if grid shows disease (not Healthy, not Other, and high confidence)
            if (pred['class_name'] not in ['Healthy', 'Other'] and
                    pred['confidence'] >= self.confidence_threshold):
                diseased_grids.append({
                    'grid_idx': idx,
                    'coords': grid_coords[idx],
                    'disease': pred['class_name'],
                    'confidence': pred['confidence']
                })

        results = {
            'overall_prediction': overall_prediction,
            'diseased_grids': diseased_grids,
            'grid_predictions': grid_predictions,
            'grid_coords': grid_coords,
            'original_image': original_image
        }

        return results

    def get_attention_map(self, image_path):
        """
        Generate attention map using Grad-CAM (optional alternative visualization)

        Args:
            image_path: Path to input image

        Returns:
            attention_map: Heatmap showing areas of interest
        """
        # This is a simplified version - can be enhanced with actual Grad-CAM
        image = Image.open(image_path).convert('RGB')
        original_image = np.array(image)

        transformed = self.transform(image=original_image)
        image_tensor = transformed['image'].unsqueeze(0).to(config.DEVICE)
        image_tensor.requires_grad = True

        # Forward pass
        logits = self.model(image_tensor)
        pred_class = logits.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        class_score = logits[0, pred_class]
        class_score.backward()

        # Get gradients
        gradients = image_tensor.grad.data.abs()
        attention = gradients.squeeze().mean(dim=0).cpu().numpy()

        # Normalize
        attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)

        return attention
