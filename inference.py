"""
Inference Script for Sugarcane Disease Detection with Grid Localization
"""

import os
import argparse
import torch
import cv2
import numpy as np
from PIL import Image

import config
from model import create_model
from grid_localization import GridLocalizer
from visualization import Visualizer


class SugarcaneDiseaseDetector:
    """Main class for disease detection inference"""

    def __init__(self, model_path, model_type='resnet50', device=None):
        """
        Args:
            model_path: Path to trained model checkpoint
            model_type: Type of model architecture
            device: Device to run inference on
        """
        self.device = device or config.DEVICE
        self.model_type = model_type

        # Load model
        print(f"Loading model from {model_path}...")
        self.model = self._load_model(model_path)
        self.model.eval()

        # Initialize localizer and visualizer
        self.localizer = GridLocalizer(self.model)
        self.visualizer = Visualizer()

        print("Model loaded successfully!")

    def _load_model(self, model_path):
        """Load trained model from checkpoint"""
        # Create model
        model = create_model(model_type=self.model_type, pretrained=False)

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)

        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        return model

    def detect(self, image_path, output_dir='output', detailed_report=True):
        """
        Perform disease detection on a single image

        Args:
            image_path: Path to input image
            output_dir: Directory to save outputs
            detailed_report: Whether to generate detailed report

        Returns:
            results: Detection results dictionary
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nProcessing image: {image_path}")
        print("-" * 60)

        # Perform localization
        results = self.localizer.localize_disease(image_path)

        # Display results
        self._display_results(results)

        # Generate base filename for outputs
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        # Save visualizations
        if detailed_report:
            report_path = os.path.join(output_dir, f"{base_name}_detailed_report.png")
            self.visualizer.create_detailed_report(results, save_path=report_path)
        else:
            viz_path = os.path.join(output_dir, f"{base_name}_result.png")
            self.visualizer.create_result_visualization(results, save_path=viz_path, show=False)

        # Save annotated image
        annotated_path = os.path.join(output_dir, f"{base_name}_annotated.png")
        self.visualizer.save_annotated_image(results, annotated_path)

        print("-" * 60)
        return results

    def _display_results(self, results):
        """Display detection results in console"""
        overall_pred = results['overall_prediction']
        diseased_grids = results['diseased_grids']

        print("\n" + "="*60)
        print("DETECTION RESULTS")
        print("="*60)

        # Overall prediction
        print(f"\nOverall Classification:")
        print(f"  Prediction: {overall_pred['class_name']}")
        print(f"  Confidence: {overall_pred['confidence']:.2%}")

        # Class probabilities
        print(f"\nClass Probabilities:")
        for i, cls in enumerate(config.CLASSES):
            prob = overall_pred['probabilities'][i]
            bar = '█' * int(prob * 30)
            print(f"  {cls:12s} : {bar:30s} {prob:.2%}")

        # Disease localization
        print(f"\nGrid Analysis:")
        print(f"  Total Grids: {len(results['grid_predictions'])}")
        print(f"  Diseased Grids: {len(diseased_grids)}")

        if diseased_grids:
            print(f"\nDiseased Grid Details:")
            for i, grid_info in enumerate(diseased_grids, 1):
                print(f"  {i}. Grid {grid_info['grid_idx']}: {grid_info['disease']} "
                      f"(Confidence: {grid_info['confidence']:.2%})")

        # Final assessment
        print(f"\n" + "="*60)
        print("ASSESSMENT:")
        if overall_pred['class_name'] == 'Healthy':
            print("✓ The leaf appears to be HEALTHY")
        elif overall_pred['class_name'] == 'Other':
            print("⚠ This does not appear to be a sugarcane leaf")
            print("  Please provide a clear image of a sugarcane leaf")
        else:
            print(f"⚠ Disease Detected: {overall_pred['class_name']}")
            if diseased_grids:
                print(f"  Affected regions: {len(diseased_grids)} grid(s)")
                print("  Check the output visualization for affected areas")
        print("="*60 + "\n")

    def batch_detect(self, image_dir, output_dir='output', detailed_report=False):
        """
        Perform detection on multiple images

        Args:
            image_dir: Directory containing images
            output_dir: Directory to save outputs
            detailed_report: Whether to generate detailed reports
        """
        # Get all image files
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        image_files = [f for f in os.listdir(image_dir)
                       if f.lower().endswith(image_extensions)]

        if not image_files:
            print(f"No images found in {image_dir}")
            return

        print(f"Found {len(image_files)} images to process")

        # Process each image
        all_results = []
        for i, img_file in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] Processing {img_file}...")
            img_path = os.path.join(image_dir, img_file)

            try:
                results = self.detect(img_path, output_dir, detailed_report)
                all_results.append({
                    'filename': img_file,
                    'results': results
                })
            except Exception as e:
                print(f"Error processing {img_file}: {str(e)}")

        # Generate summary report
        self._generate_batch_summary(all_results, output_dir)

    def _generate_batch_summary(self, all_results, output_dir):
        """Generate summary report for batch processing"""
        summary_path = os.path.join(output_dir, 'batch_summary.txt')

        with open(summary_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("BATCH PROCESSING SUMMARY\n")
            f.write("="*70 + "\n\n")

            # Statistics
            total = len(all_results)
            disease_counts = {cls: 0 for cls in config.CLASSES}

            for item in all_results:
                pred_class = item['results']['overall_prediction']['class_name']
                disease_counts[pred_class] += 1

            f.write(f"Total Images Processed: {total}\n\n")
            f.write("Disease Distribution:\n")
            for cls, count in disease_counts.items():
                percentage = (count / total * 100) if total > 0 else 0
                f.write(f"  {cls:12s} : {count:3d} ({percentage:5.1f}%)\n")

            f.write("\n" + "="*70 + "\n")
            f.write("INDIVIDUAL RESULTS\n")
            f.write("="*70 + "\n\n")

            # Individual results
            for i, item in enumerate(all_results, 1):
                filename = item['filename']
                pred = item['results']['overall_prediction']
                diseased_grids = len(item['results']['diseased_grids'])

                f.write(f"{i}. {filename}\n")
                f.write(f"   Prediction: {pred['class_name']}\n")
                f.write(f"   Confidence: {pred['confidence']:.2%}\n")
                f.write(f"   Diseased Grids: {diseased_grids}\n\n")

        print(f"\nBatch summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Sugarcane Disease Detection')
    parser.add_argument('--image', '-i', type=str, help='Path to input image')
    parser.add_argument('--batch', '-b', type=str, help='Path to directory with images')
    parser.add_argument('--model', '-m', type=str, default='models/best_model.pth',
                        help='Path to trained model')
    parser.add_argument('--model_type', type=str, default='resnet50',
                        choices=['resnet50', 'resnet18', 'efficientnet_b0', 'mobilenet_v2', 'custom'],
                        help='Model architecture type')
    parser.add_argument('--output', '-o', type=str, default='output',
                        help='Output directory')
    parser.add_argument('--detailed', action='store_true',
                        help='Generate detailed reports')

    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        print("Please train the model first using train.py")
        return

    # Initialize detector
    detector = SugarcaneDiseaseDetector(
        model_path=args.model,
        model_type=args.model_type
    )

    # Run detection
    if args.image:
        detector.detect(args.image, args.output, args.detailed)
    elif args.batch:
        detector.batch_detect(args.batch, args.output, args.detailed)
    else:
        print("Please provide either --image or --batch argument")
        parser.print_help()


if __name__ == '__main__':
    main()
