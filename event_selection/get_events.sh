#!/bin/bash

# Set SSH details
USER="maira"  # Change this to your username
HOST="correlator3.FNAL.GOV"  # Change this to the actual correlator's hostname or IP
SOURCE_BASE="/data/bfonseca/AcousticNpyData/MQXFS1d"  # Change this to the actual path on the correlator
DEST="/home/maira/Magnets/code_revamp/tmp"

# Get ramp number as an argument
RAMP=$1

# Ensure the ramp number is provided
if [ -z "$RAMP" ]; then
    echo "Error: Please provide a ramp number (e.g., ./run_ramp_analysis.sh 01)"
    exit 1
fi

# Construct the source directory
RAMP_DIR="Ramp${RAMP}"
SOURCE_DIR="$SOURCE_BASE/$RAMP_DIR"

# Check if the remote directory exists
ssh "$USER@$HOST" "[ -d '$SOURCE_DIR' ]"
if [ $? -ne 0 ]; then
    echo "Error: Remote directory $SOURCE_DIR does not exist."
    exit 1
fi

echo "Copying files from $RAMP_DIR..."

# Copy files from the remote directory to the local destination
scp "$USER@$HOST:$SOURCE_DIR/ai"*.npy "$USER@$HOST:$SOURCE_DIR/curr.npy" \
    "$USER@$HOST:$SOURCE_DIR/time.npy" "$DEST"

echo "Files from $RAMP_DIR copied successfully."

echo "Running Python script..."
python3 /home/maira/Magnets/code_revamp/event_finder_v2.py "$RAMP"

echo "Cleaning up temporary files..."
rm -f /home/maira/Magnets/code_revamp/tmp/ai*.npy /home/maira/Magnets/code_revamp/tmp/curr.npy /home/maira/Magnets/code_revamp/tmp/time.npy

echo "Temporary files removed from /tmp/"

echo "Processing complete."

