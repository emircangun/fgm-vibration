import os
import torch.nn as nn
import numpy as np

EXP = "1"

# FILE PATHS
DATA_FILE = os.path.join(os.getcwd(), "data", "non-local-data.csv")
MODEL_WEIGHTS_FILE = os.path.join(os.getcwd(), "model_weights", f"weights_{EXP}.pt")
K_NORMS_FILE = os.path.join(os.getcwd(), "model_weights", f"k_norm_values_{EXP}.np")
LOSS_FILE = os.path.join(os.getcwd(), "losses", f"train_and_eval_loss_{EXP}.png")

# MODEL PARAMETERS
HIDDEN_SIZES    = [25, 25, 25]
# HIDDEN_SIZES    = [30, 30, 30]
ACTIVATION      = nn.ReLU
DROPOUT_RATE    = 0.1

# OPTIMIZER PARAMETERS
OPTIMIZER_PARAMS = {
    'lr': 0.01,
    'momentum': 0.9
}

# TRAIN PARAMETERS
EPOCHS = 100
TRAIN_TEST_SPLIT_RATIO = 0.01
BATCH_SIZE = 128


# DATA MAPPINGS
METHOD_DICT = {
    "CBT":0,
    "TBT":1,
    "PSDBT":2,
    "ESDBT":3,
    "HSDBT":4,
    "TSDBT":5,
    "ASDBT":6,
    "ISDBT":7,
    "ICDBT":8,
    "ITDBT":9,
    "PESDBT1":10,
    "PESDBT5":11,
    "PESDBT10":12
}

BEAM_TYPE_DICT = {
    "S-S FG":0,
    "C-F FG":1,
    "C-C FG":2
}

L_H_DICT = {
    5:0,
    10:1,
    20:2,
    30:3,
    100:4,
}