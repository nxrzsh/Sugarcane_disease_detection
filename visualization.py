"""
Visualization Utilities for Disease Detection and Grid Highlighting
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import config


class Visualizer:
    """Visualize disease detection results with grid highlighting"""

    def __init__(self, grid_color=config.GRID_COLOR, grid_thickness=config.GRID_THICKNESS):
        """
        Args:
            grid_color: Color for highlighting diseased grids (B, G, R)
            grid_thickness: Thickness of grid borders
        """
        self.grid_color = grid_color
        self.grid_thickness = grid_thickness

    def draw_grid_overlay(self, image, grid_coords, diseased_grids=None):
        """
        Draw grid overlay on image with diseased grids highlighted

        Args:
            image: Original image (numpy array)
            grid_coords: List of (x, y, w, h) for all grids
            diseased_grids: List of dictionaries with diseased grid info

        Returns:
            annotated_image: Image with grid overlay
        """
        annotated_image = image.copy()

        # Draw all grid lines (light gray)
        for x, y, w, h in grid_coords:
            cv2.rectangle(
                annotated_image,
                (x, y),
                (x + w, y + h),
                (200, 200, 200),
                1
            )

        # Highlight diseased grids
        if diseased_grids:
            for grid_info in diseased_grids:
                x, y, w, h = grid_info['coords']

                # Draw thick colored border
                cv2.rectangle(
                    annotated_image,
                    (x, y),
                    (x + w, y + h),
                    self.grid_color,
                    self.grid_thickness
                )

                # Add semi-transparent overlay
                overlay = annotated_image.copy()
                cv2.rectangle(
                    overlay,
                    (x, y),
                    (x + w, y + h),
                    self.grid_color,
                    -1
                )
                cv2.addWeighted(overlay, 0.2, annotated_image, 0.8, 0, annotated_image)

                # Add disease label
                label = f"{grid_info['disease']}"
                conf = grid_info['confidence']

                # Background for text
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
                cv2.rectangle(
                    annotated_image,
                    (x, y - text_height - 10),
                    (x + text_width + 10, y),
                    self.grid_color,
                    -1
                )

                # Text
                cv2.putText(
                    annotated_image,
                    label,
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )

        return annotated_image

    def create_result_visualization(self, results, save_path=None, show=True):
        """
        Create comprehensive visualization of detection results

        Args:
            results: Results dictionary from GridLocalizer
            save_path: Optional path to save visualization
            show: Whether to display the visualization

        Returns:
            fig: Matplotlib figure
        """
        image = results['original_image']
        overall_pred = results['overall_prediction']
        diseased_grids = results['diseased_grids']
        grid_coords = results['grid_coords']

        # Create annotated image
        annotated_image = self.draw_grid_overlay(image, grid_coords, diseased_grids)

        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Original image
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')

        # Annotated image with grids
        axes[1].imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
        axes[1].set_title('Disease Localization', fontsize=14, fontweight='bold')
        axes[1].axis('off')

        # Add overall prediction text
        pred_text = f"Overall Prediction: {overall_pred['class_name']}\n"
        pred_text += f"Confidence: {overall_pred['confidence']:.2%}\n\n"

        if diseased_grids:
            pred_text += f"Diseased Grids Found: {len(diseased_grids)}\n"
            for i, grid_info in enumerate(diseased_grids[:5]):  # Show up to 5
                pred_text += f"  Grid {grid_info['grid_idx']}: {grid_info['disease']} "
                pred_text += f"({grid_info['confidence']:.2%})\n"
        else:
            if overall_pred['class_name'] == 'Healthy':
                pred_text += "Status: Healthy leaf detected"
            elif overall_pred['class_name'] == 'Other':
                pred_text += "Status: Not a sugarcane leaf"
            else:
                pred_text += "Status: No localized disease detected"

        fig.text(0.5, 0.02, pred_text, ha='center', fontsize=11,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout(rect=[0, 0.1, 1, 1])

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")

        if show:
            plt.show()

        return fig

    def create_detailed_report(self, results, save_path=None):
        """
        Create detailed multi-panel visualization report

        Args:
            results: Results dictionary from GridLocalizer
            save_path: Path to save the report
        """
        image = results['original_image']
        overall_pred = results['overall_prediction']
        diseased_grids = results['diseased_grids']
        grid_coords = results['grid_coords']
        grid_predictions = results['grid_predictions']

        # Create annotated image
        annotated_image = self.draw_grid_overlay(image, grid_coords, diseased_grids)

        # Create figure
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # Original image
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax1.set_title('Original Image', fontsize=12, fontweight='bold')
        ax1.axis('off')

        # Annotated image
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
        ax2.set_title('Disease Localization', fontsize=12, fontweight='bold')
        ax2.axis('off')

        # Grid heatmap (disease confidence per grid)
        ax3 = fig.add_subplot(gs[0, 2])
        grid_rows, grid_cols = config.GRID_SIZE
        heatmap = np.zeros((grid_rows, grid_cols))

        for i, pred in enumerate(grid_predictions):
            row = i // grid_cols
            col = i % grid_cols
            # Use max disease probability (excluding Healthy and Other)
            disease_probs = [pred['probabilities'][j] for j in range(len(config.CLASSES))
                             if config.CLASSES[j] not in ['Healthy', 'Other']]
            heatmap[row, col] = max(disease_probs) if disease_probs else 0

        im = ax3.imshow(heatmap, cmap='hot', interpolation='nearest')
        ax3.set_title('Disease Probability Heatmap', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax3)
        ax3.set_xticks([])
        ax3.set_yticks([])

        # Overall prediction probabilities
        ax4 = fig.add_subplot(gs[1, :2])
        classes = config.CLASSES
        probs = overall_pred['probabilities']
        colors = ['green' if c == overall_pred['class_name'] else 'steelblue' for c in classes]

        bars = ax4.barh(classes, probs, color=colors)
        ax4.set_xlabel('Probability', fontsize=11)
        ax4.set_title('Overall Class Probabilities', fontsize=12, fontweight='bold')
        ax4.set_xlim([0, 1])

        # Add probability values on bars
        for i, (bar, prob) in enumerate(zip(bars, probs)):
            ax4.text(prob + 0.02, i, f'{prob:.2%}', va='center', fontsize=9)

        # Summary statistics
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis('off')

        summary_text = "DETECTION SUMMARY\n" + "="*40 + "\n\n"
        summary_text += f"Overall Prediction:\n  {overall_pred['class_name']}\n\n"
        summary_text += f"Confidence:\n  {overall_pred['confidence']:.2%}\n\n"
        summary_text += f"Total Grids Analyzed:\n  {len(grid_predictions)}\n\n"
        summary_text += f"Diseased Grids:\n  {len(diseased_grids)}\n\n"

        if diseased_grids:
            disease_counts = {}
            for grid_info in diseased_grids:
                disease = grid_info['disease']
                disease_counts[disease] = disease_counts.get(disease, 0) + 1

            summary_text += "Disease Distribution:\n"
            for disease, count in disease_counts.items():
                summary_text += f"  {disease}: {count} grid(s)\n"

        ax5.text(0.1, 0.9, summary_text, fontsize=10, verticalalignment='top',
                 family='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Detailed report saved to: {save_path}")

        return fig

    def save_annotated_image(self, results, save_path):
        """
        Save just the annotated image with grid highlights

        Args:
            results: Results dictionary from GridLocalizer
            save_path: Path to save the image
        """
        image = results['original_image']
        diseased_grids = results['diseased_grids']
        grid_coords = results['grid_coords']

        annotated_image = self.draw_grid_overlay(image, grid_coords, diseased_grids)

        cv2.imwrite(save_path, annotated_image)
        print(f"Annotated image saved to: {save_path}")

    def create_performance_report(self, confusion_matrix_path, save_path=None):
        """
        Create a model performance visualization including confusion matrix

        Args:
            confusion_matrix_path: Path to saved confusion matrix image
            save_path: Optional path to save the report

        Returns:
            fig: Matplotlib figure
        """
        if not os.path.exists(confusion_matrix_path):
            print(f"Confusion matrix not found at {confusion_matrix_path}")
            return None

        # Load confusion matrix image
        cm_img = cv2.imread(confusion_matrix_path)
        cm_img_rgb = cv2.cvtColor(cm_img, cv2.COLOR_BGR2RGB)

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.imshow(cm_img_rgb)
        ax.set_title('Model Performance - Confusion Matrix', fontsize=14, fontweight='bold')
        ax.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Performance report saved to: {save_path}")

        return fig
