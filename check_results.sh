#!/bin/bash
echo "======================================================================"
echo "🔍 YOLO-UDD Training Results Verification"
echo "======================================================================"

# Check for checkpoint files
echo -e "\n📦 Checkpoint Files:"
find runs/ -name "*.pt" -exec ls -lh {} \; 2>/dev/null | while read line; do
    echo "   $line"
done

checkpoint_count=$(find runs/ -name "*.pt" 2>/dev/null | wc -l)
echo -e "\n   Total: $checkpoint_count checkpoint(s)"

# Check for logs
echo -e "\n📊 TensorBoard Logs:"
log_count=$(find runs/ -name "events.out.tfevents.*" 2>/dev/null | wc -l)
echo "   Found: $log_count event file(s)"

if [ $log_count -gt 0 ]; then
    latest_log=$(find runs/ -name "events.out.tfevents.*" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
    echo "   Latest: $latest_log"
fi

# Directory structure
echo -e "\n📁 Directory Structure:"
find runs/ -type d 2>/dev/null | head -10 | while read dir; do
    echo "   $dir"
done

# Summary
echo -e "\n======================================================================"
echo "📋 VERIFICATION SUMMARY"
echo "======================================================================"

if [ $checkpoint_count -gt 0 ]; then
    echo "✅ Training completed successfully!"
    echo "✅ $checkpoint_count model checkpoint(s) saved"
    
    # Find latest checkpoint
    latest_ckpt=$(find runs/ -name "*.pt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
    if [ ! -z "$latest_ckpt" ]; then
        size=$(du -h "$latest_ckpt" | cut -f1)
        echo "✅ Latest checkpoint: $latest_ckpt ($size)"
    fi
    
    if [ $log_count -gt 0 ]; then
        echo "✅ TensorBoard logs available"
    fi
    
    echo -e "\n🎯 Next Steps:"
    echo "   1. Run evaluation on test dataset"
    echo "   2. Test inference on new images"
    echo "   3. View training curves: tensorboard --logdir=runs/"
else
    echo "❌ No checkpoint files found!"
    echo "   Training may have failed or not completed"
fi

echo "======================================================================"
