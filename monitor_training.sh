#!/bin/bash
# Training Monitor Script for YOLO-UDD v2.0

echo "=========================================="
echo "  YOLO-UDD v2.0 Training Monitor"
echo "=========================================="
echo ""

# Check if training is running
if ps aux | grep -v grep | grep "python.*train.py" > /dev/null; then
    echo "✅ Training is RUNNING"
    echo ""
    ps aux | grep -v grep | grep "python.*train.py" | awk '{print "PID:", $2, "| CPU:", $3"%", "| MEM:", $4"%"}'
    echo ""
else
    echo "⚠️  Training is NOT running"
    echo ""
fi

# Show recent log output
if [ -f "live_training.log" ]; then
    echo "=========================================="
    echo "  Latest Training Output:"
    echo "=========================================="
    tail -30 live_training.log
elif [ -f "training.log" ]; then
    echo "=========================================="
    echo "  Latest Training Output:"
    echo "=========================================="
    tail -30 training.log
else
    echo "No log file found yet."
fi

echo ""
echo "=========================================="
echo "  Training Directory Contents:"
echo "=========================================="
if [ -d "runs/train" ]; then
    ls -lh runs/train/
else
    echo "Training directory not created yet."
fi

echo ""
echo "=========================================="
echo "  Commands:"
echo "=========================================="
echo "  Watch logs: tail -f live_training.log"
echo "  Kill training: pkill -f 'python.*train.py'"
echo "  Check GPU: nvidia-smi"
echo "=========================================="
