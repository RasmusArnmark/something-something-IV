#!/bin/bash
# Monitor training progress

LOG_FILE="/Users/rasmusarnmark/Desktop/skku/IV/final_project/something-something-IV/training.log"
CHECKPOINT_DIR="/Users/rasmusarnmark/Desktop/skku/IV/final_project/something-something-IV/checkpoints/test_2d"

echo "==================================================="
echo "Training Monitor"
echo "==================================================="
echo

# Check if process is running
PROCESS_COUNT=$(ps aux | grep "train_2d.py" | grep -v grep | wc -l)
if [ $PROCESS_COUNT -gt 0 ]; then
    echo "✓ Training is RUNNING"
    ps aux | grep "train_2d.py" | grep -v grep | head -n 1
    echo
else
    echo "✗ Training is NOT running"
    echo
fi

# Check log file size
if [ -f "$LOG_FILE" ]; then
    LOG_SIZE=$(ls -lh "$LOG_FILE" | awk '{print $5}')
    echo "Log file size: $LOG_SIZE"
    echo "Last 20 lines of log:"
    echo "---------------------------------------------------"
    tail -n 20 "$LOG_FILE"
else
    echo "Log file not found yet"
fi

echo
echo "---------------------------------------------------"

# Check for checkpoints
if [ -d "$CHECKPOINT_DIR" ]; then
    CHECKPOINT_COUNT=$(ls -1 "$CHECKPOINT_DIR" 2>/dev/null | wc -l)
    echo "Checkpoints created: $CHECKPOINT_COUNT"
    if [ $CHECKPOINT_COUNT -gt 0 ]; then
        echo "Latest checkpoint:"
        ls -lht "$CHECKPOINT_DIR" | head -n 2
    fi
else
    echo "No checkpoints directory yet"
fi

echo
echo "==================================================="
echo "To monitor in real-time, run:"
echo "  tail -f $LOG_FILE"
echo "==================================================="
