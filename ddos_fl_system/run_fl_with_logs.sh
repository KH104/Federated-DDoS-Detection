#!/bin/bash
export PYTHONPATH=.

"/Users/kartikeyahazela/research paper/Technical Code/.venv/bin/python" -m federated.server > server.log 2>&1 &
SERVER_PID=$!
sleep 3

"/Users/kartikeyahazela/research paper/Technical Code/.venv/bin/python" -m federated.client > client1.log 2>&1 &
CLIENT1_PID=$!
sleep 2

"/Users/kartikeyahazela/research paper/Technical Code/.venv/bin/python" -m federated.client > client2.log 2>&1 &
CLIENT2_PID=$!

wait $SERVER_PID
wait $CLIENT1_PID
wait $CLIENT2_PID
