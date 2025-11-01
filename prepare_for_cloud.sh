#!/bin/bash
# Script to prepare dataset for Google Colab/Kaggle upload

echo "📦 Preparing YOLO-UDD Dataset for Cloud Upload"
echo "=============================================="
echo ""

# Set paths
PROJECT_DIR="/home/student/MIR/Project/YOLO-UDD-v2.0"
DATASET_DIR="$PROJECT_DIR/data/trashcan"
OUTPUT_DIR="$PROJECT_DIR/cloud_upload"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "📊 Checking dataset..."
if [ ! -d "$DATASET_DIR" ]; then
    echo "❌ ERROR: Dataset not found at $DATASET_DIR"
    exit 1
fi

# Check annotations
if [ ! -f "$DATASET_DIR/annotations/train.json" ]; then
    echo "❌ ERROR: train.json not found!"
    exit 1
fi

if [ ! -f "$DATASET_DIR/annotations/val.json" ]; then
    echo "❌ ERROR: val.json not found!"
    exit 1
fi

echo "✅ Dataset found!"
echo ""

# Count images
TRAIN_COUNT=$(ls "$DATASET_DIR/images/train" 2>/dev/null | wc -l)
VAL_COUNT=$(ls "$DATASET_DIR/images/val" 2>/dev/null | wc -l)

echo "📊 Dataset Statistics:"
echo "   Training images: $TRAIN_COUNT"
echo "   Validation images: $VAL_COUNT"
echo ""

# Calculate size
DATASET_SIZE=$(du -sh "$DATASET_DIR" | cut -f1)
echo "   Dataset size: $DATASET_SIZE"
echo ""

# Create zip file
echo "📦 Creating zip file for upload..."
echo "   This may take a few minutes..."
echo ""

cd "$PROJECT_DIR/data"
zip -r "$OUTPUT_DIR/trashcan_dataset.zip" trashcan/ -x "*.git*" "*.DS_Store" 2>&1 | grep -v "adding:" | head -20

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Dataset zip created successfully!"
    echo ""
    
    # Show zip file info
    ZIP_SIZE=$(du -sh "$OUTPUT_DIR/trashcan_dataset.zip" | cut -f1)
    echo "📄 Zip file details:"
    echo "   Location: $OUTPUT_DIR/trashcan_dataset.zip"
    echo "   Size: $ZIP_SIZE"
    echo ""
    
    # Instructions
    echo "📤 Next Steps for Google Colab:"
    echo "================================"
    echo ""
    echo "1. Go to https://drive.google.com"
    echo ""
    echo "2. Create this folder structure:"
    echo "   My Drive/"
    echo "   └── YOLO_UDD/"
    echo "       └── dataset/"
    echo ""
    echo "3. Upload the zip file:"
    echo "   $OUTPUT_DIR/trashcan_dataset.zip"
    echo "   to: My Drive/YOLO_UDD/dataset/"
    echo ""
    echo "4. In Google Drive, right-click the zip file"
    echo "   → Extract here (or unzip manually)"
    echo ""
    echo "5. Open the Colab notebook:"
    echo "   $PROJECT_DIR/YOLO_UDD_Training_Colab.ipynb"
    echo ""
    echo "6. Upload it to https://colab.research.google.com"
    echo ""
    echo "7. Enable GPU and run all cells!"
    echo ""
    echo "================================"
    echo ""
    echo "📤 Next Steps for Kaggle:"
    echo "================================"
    echo ""
    echo "1. Go to https://www.kaggle.com/datasets"
    echo ""
    echo "2. Click 'New Dataset'"
    echo ""
    echo "3. Upload:"
    echo "   $OUTPUT_DIR/trashcan_dataset.zip"
    echo ""
    echo "4. Set dataset name: 'trashcan-underwater-debris'"
    echo ""
    echo "5. Make it private or public"
    echo ""
    echo "6. Click 'Create'"
    echo ""
    echo "7. In Kaggle notebook:"
    echo "   - Add your dataset"
    echo "   - Enable GPU"
    echo "   - Clone the GitHub repo"
    echo "   - Start training!"
    echo ""
    echo "✨ Dataset is ready for upload!"
    
else
    echo "❌ ERROR: Failed to create zip file"
    exit 1
fi
