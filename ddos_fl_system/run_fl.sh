#!/bin/bash
"/Users/kartikeyahazela/research paper/Technical Code/.venv/bin/python" federated/server.py &
SERVER_PID=$!
sleep 2

"/Users/kartikeyahazela/research paper/Technical Code/.venv/bin/python" federated/client.py &
CLIENT1_PID=$!
sleep 1

"/Users/kartikeyahazela/research paper/Technical Code/.venv/bin/python" federated/client.py &
CLIENT2_PID=$!

wait $SERVER_PID
wait $CLIENT1_PID
wait $CLIENT2_PID
