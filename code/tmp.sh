#!/bin/bash

# Define the command you want to run
COMMAND_TO_RUN="python run_downstream.py --my-tmp --ds-train --ds-trainmode finetune --simu-exp --ds-nsimroom 1 --ds-task TDOA --time 11251422 --gpu-id 0," 

# Define the sleep time in seconds (30 minutes * 60 seconds/minute = 1800 seconds)
SLEEP_TIME=1800 

while true; do
    echo "--- Running command at $(date) ---"
    
    # Run the command
    eval "$COMMAND_TO_RUN"
    eval "rm -rf ../exp/TDOA/11251422/finetune-all-mlp-100-*"
    
    # Check the exit status of the command (optional, but good practice)
    if [ $? -ne 0 ]; then
        echo "Command failed with exit code $?"
        # You might add logic here to break or handle the error
    fi

    echo "--- Command finished. Waiting for 30 minutes... ---"
    
    # Wait for 30 minutes (1800 seconds)
    sleep $SLEEP_TIME 

done
