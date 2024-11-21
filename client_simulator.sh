#!/bin/bash

NUM_CLIENTS=5

CLIENT_DIR="/Users/ritvikasonawane/Documents/r24k8xbv/0F24/QFL/embedded-devices"

CLIENT_SCRIPT="client_pytorch.py"

CONDA_ENV="flowerfl"

SERVER_IP=$(ipconfig getifaddr en0)
SERVER_ADDRESS="${SERVER_IP}:8080"

for ((i=0; i<NUM_CLIENTS; i++))
do
    osascript -e "tell application \"Terminal\" to do script \"cd $CLIENT_DIR && conda activate $CONDA_ENV && python3 $CLIENT_SCRIPT --cid=$i --server_address=$SERVER_ADDRESS --mnist\""
    sleep 1
done