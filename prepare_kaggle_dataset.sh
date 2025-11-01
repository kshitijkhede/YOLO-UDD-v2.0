#!/bin/bash

# YOLO-UDD v2.0 - Kaggle Dataset Preparation Script
# This script prepares your dataset for upload to Kaggle

set -e  # Exit on error

echo "======================================"
echo "📦 KAGGLE DATASET PREPARATION"
echo "======================================"
echo ""

# Navigate to data directory
cd /home/student/MIR/Project/YOLO-UDD-v2.0/data

# Check if trashcan directory exists
if [ ! -d "trashcan" ]; then
    echo "❌ Error: trashcan directory not found!"
    exit 1
fi

echo "✅ Found trashcan directory"
echo ""

# Verify dataset structure
echo "📋 Verifying dataset structure..."
echo ""

if [ ! -f "trashcan/annotations/train.json" ]; then
    echo "❌ Error: train.json not found!"
    exit 1
fi

if [ ! -f "trashcan/annotations/val.json" ]; then
    echo "❌ Error: val.json not found!"
    exit 1
fi

if [ ! -d "trashcan/images/train" ]; then
    echo "❌ Error: train images directory not found!"
    exit 1
fi

if [ ! -d "trashcan/images/val" ]; then
    echo "❌ Error: val images directory not found!"
    exit 1
fi

echo "✅ All required files and directories found"
echo ""

# Count files
TRAIN_IMAGES=$(ls -1 trashcan/images/train | wc -l)
VAL_IMAGES=$(ls -1 trashcan/images/val | wc -l)
TRAIN_JSON_SIZE=$(du -h trashcan/annotations/train.json | cut -f1)
VAL_JSON_SIZE=$(du -h trashcan/annotations/val.json | cut -f1)

echo "📊 Dataset Statistics:"
echo "   Training images: $TRAIN_IMAGES"
echo "   Validation images: $VAL_IMAGES"
echo "   train.json size: $TRAIN_JSON_SIZE"
echo "   val.json size: $VAL_JSON_SIZE"
echo ""

# Calculate total size
TOTAL_SIZE=$(du -sh trashcan | cut -f1)
echo "   Total dataset size: $TOTAL_SIZE"
echo ""

# Ask for confirmation
read -p "📦 Create trashcan_dataset.zip for Kaggle upload? (y/n): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Cancelled by user"
    exit 0
fi

# Remove old zip if exists
if [ -f "trashcan_dataset.zip" ]; then
    echo "🗑️  Removing old trashcan_dataset.zip..."
    rm trashcan_dataset.zip
fi

# Create zip file
echo ""
echo "📦 Creating trashcan_dataset.zip..."
echo "   This may take 2-5 minutes..."
echo ""

zip -r trashcan_dataset.zip trashcan/ -x "*.DS_Store" "*.git*" "__pycache__/*"

if [ $? -eq 0 ]; then
    ZIP_SIZE=$(du -h trashcan_dataset.zip | cut -f1)
    echo ""
    echo "======================================"
    echo "✅ SUCCESS!"
    echo "======================================"
    echo ""
    echo "📦 Dataset archive created:"
    echo "   File: trashcan_dataset.zip"
    echo "   Size: $ZIP_SIZE"
    echo "   Location: $(pwd)/trashcan_dataset.zip"
    echo ""
    echo "🚀 Next Steps:"
    echo "   1. Go to https://www.kaggle.com/datasets"
    echo "   2. Click 'New Dataset'"
    echo "   3. Upload: trashcan_dataset.zip"
    echo "   4. Title: 'TrashCAN Underwater Debris Dataset'"
    echo "   5. Click 'Create'"
    echo ""
    echo "📖 For complete instructions, see:"
    echo "   KAGGLE_COMPLETE_GUIDE.md"
    echo ""
else
    echo "❌ Error: Failed to create zip file!"
    exit 1
fi
