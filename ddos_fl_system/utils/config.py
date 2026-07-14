import os

# Data Configuration
DATA_DIR = "data"
DUMMY_DATA_SAMPLES = 10000
DUMMY_DATA_FEATURES = 20

# Model Configuration
INPUT_DIM = DUMMY_DATA_FEATURES
HIDDEN_DIM_1 = 64
HIDDEN_DIM_2 = 32
OUTPUT_DIM = 2 # Binary classification (0: Normal, 1: DDoS)
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 5

# Federated Learning Configuration
FL_ROUNDS = 3
MIN_AVAILABLE_CLIENTS = 2
FRACTION_FIT = 1.0

# Anomaly Detection Configuration
CONTAMINATION = 0.05 # Expected percentage of anomalies
