#!/bin/bash
# Quick training status checker

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║              🔍 YOLO-UDD Training Status Checker                     ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if training is running
if pgrep -f "scripts/train.py" > /dev/null; then
    echo "✅ Training is RUNNING"
    echo "   Process ID: $(pgrep -f 'scripts/train.py')"
    echo ""
else
    echo "⏸️  Training is NOT running"
    echo ""
fi

# Check for training directories
if [ -d "runs" ]; then
    echo "📁 Training Runs Found:"
    for dir in runs/*/; do
        if [ -d "$dir" ]; then
            dirname=$(basename "$dir")
            checkpoint_count=$(find "$dir/checkpoints" -name "epoch_*.pt" 2>/dev/null | wc -l)
            latest=$(find "$dir/checkpoints" -name "latest.pt" 2>/dev/null)
            
            if [ -f "$latest" ]; then
                size=$(du -h "$latest" | cut -f1)
                modified=$(stat -c %y "$latest" | cut -d'.' -f1)
                echo "   • $dirname:"
                echo "     - Epochs saved: $checkpoint_count"
                echo "     - Latest checkpoint: $size (modified: $modified)"
            fi
        fi
    done
    echo ""
else
    echo "📁 No training runs found yet"
    echo ""
fi

# Check GPU status
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 GPU Status:"
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader | \
        awk -F, '{printf "   • %s\n   • Memory: %s / %s\n   • Utilization: %s\n", $1, $2, $3, $4}'
    echo ""
fi

# Show most recent training log
latest_run=$(ls -td runs/*/ 2>/dev/null | head -1)
if [ -d "$latest_run" ]; then
    echo "📊 Latest Run: $(basename $latest_run)"
    if [ -d "$latest_run/logs" ]; then
        log_count=$(find "$latest_run/logs" -name "events.out.tfevents.*" | wc -l)
        echo "   • TensorBoard logs: $log_count files"
    fi
    echo ""
fi

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                         💡 Quick Commands                            ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo "Start training:    python3 run_full_training.py"
echo "View logs:         tensorboard --logdir=$latest_run/logs/"
echo "Check GPU:         watch -n 1 nvidia-smi"
echo "Monitor progress:  watch -n 5 ./check_training_status.sh"
echo ""
