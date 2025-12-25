"""
Generate Confusion Matrix from Existing Trained Model
This script evaluates the trained model on the validation set and generates a confusion matrix.
No retraining required!
"""

import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

import config
from model import create_model
from dataset import get_dataloaders


def generate_confusion_matrix(model_path='models/best_model.pth', model_type='resnet50', save_dir='models'):
    """
    Generate confusion matrix from existing trained model

    Args:
        model_path: Path to trained model checkpoint
        model_type: Model architecture type
        save_dir: Directory to save confusion matrix
    """
    print("="*60)
    print("Generating Confusion Matrix from Trained Model")
    print("="*60)

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found at {model_path}")
        print("Please train the model first using: python train.py")
        return

    print(f"\n[OK] Found model at: {model_path}")

    # Create model
    print("\nLoading model...")
    model = create_model(model_type=model_type, pretrained=False)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=config.DEVICE)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[OK] Model loaded (trained for {checkpoint.get('epoch', 'unknown')} epochs)")
        if 'val_acc' in checkpoint:
            print(f"  Validation Accuracy: {checkpoint['val_acc']:.4f}")
    else:
        model.load_state_dict(checkpoint)
        print("[OK] Model loaded")

    model = model.to(config.DEVICE)
    model.eval()

    # Get validation data loader
    print("\nLoading validation dataset...")
    _, val_loader = get_dataloaders()
    print(f"[OK] Loaded {len(val_loader.dataset)} validation samples")

    # Evaluate model
    print("\nEvaluating model on validation set...")
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Evaluating'):
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate accuracy
    accuracy = (all_preds == all_labels).mean()
    print(f"\n[OK] Evaluation complete!")
    print(f"  Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Generate confusion matrix
    print("\nGenerating confusion matrix...")
    cm = confusion_matrix(all_labels, all_preds)

    # Create confusion matrix plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=config.CLASSES,
                yticklabels=config.CLASSES,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix - Validation Set', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save confusion matrix
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(save_path, dpi=150)
    print(f"[OK] Confusion matrix saved to: {save_path}")
    plt.close()

    # Print classification report
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(all_labels, all_preds, target_names=config.CLASSES))

    # Print per-class accuracy
    print("\nPer-Class Accuracy:")
    print("-"*60)
    for i, class_name in enumerate(config.CLASSES):
        class_mask = all_labels == i
        if class_mask.sum() > 0:
            class_acc = (all_preds[class_mask] == all_labels[class_mask]).mean()
            class_count = class_mask.sum()
            print(f"  {class_name:12s} : {class_acc:.4f} ({class_acc*100:.2f}%) - {class_count} samples")

    print("\n" + "="*60)
    print("[OK] Confusion matrix generation complete!")
    print("="*60)
    print("\nYou can now view the confusion matrix in the web interface.")
    print("Run: python app.py")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate confusion matrix from trained model')
    parser.add_argument('--model', '-m', type=str, default='models/best_model.pth',
                        help='Path to trained model')
    parser.add_argument('--model_type', type=str, default='resnet50',
                        choices=['resnet50', 'resnet18', 'efficientnet_b0', 'mobilenet_v2', 'custom'],
                        help='Model architecture type')
    parser.add_argument('--save_dir', '-s', type=str, default='models',
                        help='Directory to save confusion matrix')

    args = parser.parse_args()

    generate_confusion_matrix(
        model_path=args.model,
        model_type=args.model_type,
        save_dir=args.save_dir
    )
