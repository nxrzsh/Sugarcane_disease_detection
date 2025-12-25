"""
Utility functions for dataset management and helpers
"""

import os
import shutil
from pathlib import Path
import random


def create_directory_structure(base_path='data'):
    """
    Create the required directory structure for the dataset

    Args:
        base_path: Base directory path
    """
    splits = ['train', 'val', 'test']
    classes = ['Healthy', 'Yellow', 'RedRot', 'Rust', 'Mosaic', 'Other']

    for split in splits:
        for cls in classes:
            dir_path = os.path.join(base_path, split, cls)
            os.makedirs(dir_path, exist_ok=True)

    print(f"✓ Directory structure created at: {base_path}")
    print(f"  Splits: {', '.join(splits)}")
    print(f"  Classes: {', '.join(classes)}")


def split_dataset(source_dir, output_dir='data', train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split dataset into train/val/test sets

    Args:
        source_dir: Directory containing class subdirectories with images
        output_dir: Output directory for split dataset
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"

    # Create output structure
    create_directory_structure(output_dir)

    # Get all class directories
    class_dirs = [d for d in os.listdir(source_dir)
                  if os.path.isdir(os.path.join(source_dir, d))]

    print(f"\nSplitting dataset from: {source_dir}")
    print(f"Train: {train_ratio:.0%} | Val: {val_ratio:.0%} | Test: {test_ratio:.0%}")
    print("-" * 60)

    for cls in class_dirs:
        cls_path = os.path.join(source_dir, cls)

        # Get all images
        images = [f for f in os.listdir(cls_path)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        # Shuffle
        random.shuffle(images)

        # Calculate split indices
        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]

        # Copy files
        for img in train_imgs:
            src = os.path.join(cls_path, img)
            dst = os.path.join(output_dir, 'train', cls, img)
            shutil.copy2(src, dst)

        for img in val_imgs:
            src = os.path.join(cls_path, img)
            dst = os.path.join(output_dir, 'val', cls, img)
            shutil.copy2(src, dst)

        for img in test_imgs:
            src = os.path.join(cls_path, img)
            dst = os.path.join(output_dir, 'test', cls, img)
            shutil.copy2(src, dst)

        print(f"{cls:12s} : {n_train:4d} train | {n_val:4d} val | {len(test_imgs):4d} test")

    print("-" * 60)
    print("✓ Dataset split completed!")


def count_dataset_samples(data_dir='data'):
    """
    Count samples in each split and class

    Args:
        data_dir: Dataset directory
    """
    splits = ['train', 'val', 'test']
    classes = ['Healthy', 'Yellow', 'RedRot', 'Rust', 'Mosaic', 'Other']

    print("\nDataset Statistics")
    print("=" * 70)

    for split in splits:
        print(f"\n{split.upper()}")
        print("-" * 70)

        total = 0
        for cls in classes:
            cls_path = os.path.join(data_dir, split, cls)

            if not os.path.exists(cls_path):
                count = 0
            else:
                count = len([f for f in os.listdir(cls_path)
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])

            print(f"  {cls:12s} : {count:5d} images")
            total += count

        print(f"  {'TOTAL':12s} : {total:5d} images")

    print("=" * 70)


def verify_dataset(data_dir='data'):
    """
    Verify dataset structure and report any issues

    Args:
        data_dir: Dataset directory
    """
    print("\nVerifying dataset structure...")
    print("=" * 70)

    issues = []
    splits = ['train', 'val', 'test']
    classes = ['Healthy', 'Yellow', 'RedRot', 'Rust', 'Mosaic', 'Other']

    # Check directory structure
    for split in splits:
        split_path = os.path.join(data_dir, split)
        if not os.path.exists(split_path):
            issues.append(f"Missing split directory: {split_path}")
            continue

        for cls in classes:
            cls_path = os.path.join(split_path, cls)
            if not os.path.exists(cls_path):
                issues.append(f"Missing class directory: {cls_path}")

    # Check for empty directories
    for split in splits:
        for cls in classes:
            cls_path = os.path.join(data_dir, split, cls)
            if os.path.exists(cls_path):
                images = [f for f in os.listdir(cls_path)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                if len(images) == 0:
                    issues.append(f"Empty directory: {cls_path}")

    # Report results
    if issues:
        print("⚠ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✓ Dataset structure is valid!")

    print("=" * 70)

    return len(issues) == 0


def get_class_weights(data_dir='data', split='train'):
    """
    Calculate class weights for imbalanced dataset

    Args:
        data_dir: Dataset directory
        split: Split to calculate weights for

    Returns:
        class_weights: Dictionary of class weights
    """
    import torch
    from collections import Counter

    classes = ['Healthy', 'Yellow', 'RedRot', 'Rust', 'Mosaic', 'Other']
    counts = []

    for cls in classes:
        cls_path = os.path.join(data_dir, split, cls)
        if os.path.exists(cls_path):
            count = len([f for f in os.listdir(cls_path)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        else:
            count = 0
        counts.append(count)

    # Calculate weights (inverse frequency)
    total = sum(counts)
    weights = [total / (len(classes) * count) if count > 0 else 0 for count in counts]

    print("\nClass Weights:")
    print("-" * 40)
    for cls, count, weight in zip(classes, counts, weights):
        print(f"{cls:12s} : {count:5d} samples | weight: {weight:.4f}")

    return torch.FloatTensor(weights)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Dataset utilities')
    parser.add_argument('--create-structure', action='store_true',
                        help='Create dataset directory structure')
    parser.add_argument('--split', type=str,
                        help='Split dataset from source directory')
    parser.add_argument('--count', action='store_true',
                        help='Count samples in dataset')
    parser.add_argument('--verify', action='store_true',
                        help='Verify dataset structure')
    parser.add_argument('--class-weights', action='store_true',
                        help='Calculate class weights')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Dataset directory')

    args = parser.parse_args()

    if args.create_structure:
        create_directory_structure(args.data_dir)

    if args.split:
        split_dataset(args.split, args.data_dir)

    if args.count:
        count_dataset_samples(args.data_dir)

    if args.verify:
        verify_dataset(args.data_dir)

    if args.class_weights:
        weights = get_class_weights(args.data_dir)

    if not any([args.create_structure, args.split, args.count, args.verify, args.class_weights]):
        parser.print_help()
