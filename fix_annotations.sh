#!/bin/bash
# Quick fix script to generate proper annotations for YOLO-UDD v2.0

echo "🔧 YOLO-UDD v2.0 Dataset Annotation Fix Script"
echo "=============================================="
echo ""

# Define paths
EXTERNAL_DATASET="/home/student/MIR/Project/mir dataset/archive/dataset"
PROJECT_DIR="/home/student/MIR/Project/YOLO-UDD-v2.0"
TARGET_ANN_DIR="$PROJECT_DIR/data/trashcan/annotations"

echo "📁 Checking paths..."
if [ ! -d "$EXTERNAL_DATASET" ]; then
    echo "❌ ERROR: External dataset not found at: $EXTERNAL_DATASET"
    exit 1
fi

if [ ! -d "$PROJECT_DIR" ]; then
    echo "❌ ERROR: Project directory not found at: $PROJECT_DIR"
    exit 1
fi

echo "✅ Paths verified"
echo ""

# Navigate to external dataset
cd "$EXTERNAL_DATASET"

echo "📊 Available dataset versions:"
echo "  1. instance_version - Objects labeled by instance (recommended for detection)"
echo "  2. material_version - Objects labeled by material type"
echo ""

# Check if conversion script exists
if [ ! -f "scripts/trash_can_coco.py" ]; then
    echo "❌ ERROR: Conversion script not found!"
    echo "   Expected: $EXTERNAL_DATASET/scripts/trash_can_coco.py"
    exit 1
fi

echo "🔄 Converting TrashCAN annotations to COCO format..."
echo "   Using: instance_version (recommended for object detection)"
echo ""

# Run conversion script
cd "$EXTERNAL_DATASET"
python scripts/trash_can_coco.py instance

if [ $? -ne 0 ]; then
    echo "❌ ERROR: Annotation conversion failed!"
    echo "   Please check the conversion script output above"
    exit 1
fi

echo "✅ Conversion complete!"
echo ""

# Check if annotations were generated
if [ ! -f "instance_version/annotations/train.json" ]; then
    echo "❌ ERROR: train.json not generated!"
    exit 1
fi

if [ ! -f "instance_version/annotations/val.json" ]; then
    echo "❌ ERROR: val.json not generated!"
    exit 1
fi

echo "📋 Generated annotation files:"
TRAIN_SIZE=$(wc -c < "instance_version/annotations/train.json")
VAL_SIZE=$(wc -c < "instance_version/annotations/val.json")
echo "  ✅ train.json ($(numfmt --to=iec-i --suffix=B $TRAIN_SIZE))"
echo "  ✅ val.json ($(numfmt --to=iec-i --suffix=B $VAL_SIZE))"
echo ""

# Create backup of empty files
echo "💾 Backing up empty annotation files..."
mkdir -p "$TARGET_ANN_DIR/backup"
if [ -f "$TARGET_ANN_DIR/train.json" ]; then
    mv "$TARGET_ANN_DIR/train.json" "$TARGET_ANN_DIR/backup/train.json.old"
fi
if [ -f "$TARGET_ANN_DIR/val.json" ]; then
    mv "$TARGET_ANN_DIR/val.json" "$TARGET_ANN_DIR/backup/val.json.old"
fi
echo "✅ Backup complete"
echo ""

# Copy new annotations
echo "📦 Copying annotations to project..."
cp "instance_version/annotations/train.json" "$TARGET_ANN_DIR/"
cp "instance_version/annotations/val.json" "$TARGET_ANN_DIR/"

if [ $? -eq 0 ]; then
    echo "✅ Annotations copied successfully!"
else
    echo "❌ ERROR: Failed to copy annotations"
    exit 1
fi
echo ""

# Verify the files
echo "🔍 Verifying annotations..."
cd "$PROJECT_DIR"

# Check file sizes
NEW_TRAIN_SIZE=$(wc -c < "$TARGET_ANN_DIR/train.json")
NEW_VAL_SIZE=$(wc -c < "$TARGET_ANN_DIR/val.json")

echo "  📄 train.json: $(numfmt --to=iec-i --suffix=B $NEW_TRAIN_SIZE)"
echo "  📄 val.json: $(numfmt --to=iec-i --suffix=B $NEW_VAL_SIZE)"

if [ $NEW_TRAIN_SIZE -lt 1000 ]; then
    echo "  ⚠️  WARNING: train.json seems too small!"
fi

if [ $NEW_VAL_SIZE -lt 1000 ]; then
    echo "  ⚠️  WARNING: val.json seems too small!"
fi
echo ""

# Run dataset verification
echo "✅ Running dataset verification..."
python scripts/verify_dataset.py --dataset-dir data/trashcan

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 SUCCESS! Your dataset is ready for training!"
    echo ""
    echo "📋 Next steps:"
    echo "  1. Review the verification output above"
    echo "  2. Start training with: ./train.sh"
    echo "  3. Monitor with TensorBoard: tensorboard --logdir=runs"
    echo ""
else
    echo ""
    echo "⚠️  Dataset verification found issues. Please review the output above."
    echo ""
fi

# Offer to sync with GitHub
echo "🔄 Do you want to sync this fix with GitHub? (y/n)"
read -r response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo "📤 Syncing with GitHub..."
    git add data/trashcan/annotations/*.json
    git commit -m "Fix: Add proper COCO format annotations for TrashCAN dataset"
    git push origin main
    echo "✅ Synced with GitHub!"
else
    echo "ℹ️  Skipping GitHub sync. Run ./sync_github.sh later to sync."
fi

echo ""
echo "✨ All done! Happy training! ✨"
