# Quick Start Guide

Get the Sugarcane Disease Detection system running in **5 simple steps**. Total time: ~30 minutes.

## Prerequisites

- Python 3.8+
- 8GB RAM recommended
- GPU optional (speeds up training)

## Step 1: Install Dependencies (2 minutes)

```bash
pip install -r requirements.txt
```

## Step 2: Prepare Dataset (5 minutes)

### Option A: Auto-split existing images (Recommended)

If your images are organized by class in a single folder:

```bash
python utils.py --split path/to/your_images --data-dir data
```

This automatically creates train (70%), validation (15%), and test (15%) splits.

### Option B: Manual organization

```bash
# Create folder structure
python utils.py --create-structure --data-dir data

# Then manually copy images to:
# data/train/{Healthy,Yellow,RedRot,Rust,Mosaic,Other}/
# data/val/{same structure}
# data/test/{same structure}
```

### Verify dataset

```bash
python utils.py --verify --count
```

## Step 3: Train Model (10-30 minutes)

```bash
python train.py
```

**What to expect:**
- With GPU: ~10-20 minutes for 3000 images
- Without GPU: ~30-60 minutes
- Best model saved to `models/best_model.pth`
- Training curves saved to `models/training_history.png`

**Tip:** You can stop early with `Ctrl+C` if validation accuracy stops improving.

## Step 4: Generate Confusion Matrix (2 minutes)

```bash
python generate_confusion_matrix.py
```

This evaluates your model and saves the confusion matrix to `models/confusion_matrix.png`.

## Step 5: Launch Web Interface

```bash
python app.py
```

Open your browser to: **http://localhost:5000**

**Features:**
- Drag & drop image upload
- Real-time disease prediction
- Visual grid analysis showing affected regions
- Confidence scores and probability charts
- Confusion matrix visualization
- **Automatic image organization** - classified images are saved to `data/classification/{predicted_class}/`

## Alternative: Command-Line Inference

### Single image

```bash
python inference.py --image path/to/image.jpg --output results/
```

### Batch processing

```bash
python inference.py --batch path/to/images/ --output results/
```

### Detailed reports with heatmaps

```bash
python inference.py --image test.jpg --detailed --output results/
```

## Complete Example Workflow

```bash
# 1. Install
pip install -r requirements.txt

# 2. Prepare data
python utils.py --split my_dataset --data-dir data
python utils.py --verify --count

# 3. Train
python train.py

# 4. Evaluate
python generate_confusion_matrix.py

# 5. Launch web interface
python app.py

# Then open http://localhost:5000 and start detecting!
```

## New Feature: Automatic Image Organization

When you classify images through the web interface:
- Images are automatically saved to `data/classification/{predicted_class}/`
- Example: An image classified as "Yellow" goes to `data/classification/Yellow/`
- Useful for building a database, quality control, and creating new training datasets

## Quick Troubleshooting

| Issue | Solution |
|-------|----------|
| "Model not found" | Run `python train.py` first |
| "CUDA out of memory" | Reduce `BATCH_SIZE` in `config.py` |
| "Port 5000 in use" | Change port in `app.py` line 222 |
| Poor accuracy | Ensure 100+ images per class, check labels |
| Confusion matrix missing | Run `python generate_confusion_matrix.py` |

## Expected Results

With 3000 well-labeled images:
- Overall accuracy: 95-98%
- Per-class accuracy: 90-100%
- Processing time: 2-5 seconds per image

## Next Steps

- **Improve model:** Add more training data, adjust hyperparameters in `config.py`
- **Deploy:** Use the REST API (`/api/predict`) to integrate with other applications
- **Batch process:** Use command-line tools to process multiple images
- **Review classifications:** Check `data/classification/` folders to see organized results

## Quick Command Reference

```bash
# Setup
pip install -r requirements.txt
python utils.py --split source_folder --data-dir data

# Training
python train.py
python generate_confusion_matrix.py

# Inference
python app.py                                    # Web interface
python inference.py --image test.jpg             # Single image
python inference.py --batch test_folder/         # Batch processing
```

## Need More Help?

See **README.md** for comprehensive documentation including:
- REST API endpoints and examples
- Advanced configuration options
- Python module integration
- Deployment strategies
- Detailed troubleshooting

---

**That's it!** You now have a working disease detection system with automatic image organization. Start with the web interface at http://localhost:5000 after completing the 5 steps above.
