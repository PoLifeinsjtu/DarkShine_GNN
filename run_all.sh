#!/bin/bash

# Script to run the entire tracking GNN training and testing process
set -e

SCRIPT_DIR=$(dirname "$(realpath "$0")")

# Record the start time
START_TIME=$(date +%s)
START_TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

echo "========================================="
echo "Start the task: $START_TIMESTAMP"
echo "========================================="

# Start to execute train
echo "Start to execute trkgnn to train..."
cd "$SCRIPT_DIR/trkgnn" || exit
./run.sh
echo "Training in trkgnn completed."

# Go back to the script directory
cd "$SCRIPT_DIR" || exit

# Start to execute test
echo "Start to execute new_data to test..."
cd "$SCRIPT_DIR/new_data" || exit
./run.sh
echo "Testing in new_data completed."

# Record the end time
END_TIME=$(date +%s)
END_TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
ELAPSED_TIME=$((END_TIME - START_TIME))

echo "========================================="
echo "Finish the task: $END_TIMESTAMP"
echo "Cost time in all: $(date -u -d @$ELAPSED_TIME +"%H:%M:%S")"
echo "========================================="