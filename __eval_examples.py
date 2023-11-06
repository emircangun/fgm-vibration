import torch

from dataset import *
from train_and_evaluate import *
from model import NeuralNetwork
from plot import *
import config


model = load_model(config.MODEL_WEIGHTS_FILE)

examples = [
    ["HSDBT", "S-S", 5, 0.],
    ["HSDBT", "S-S", 5, 0.5],
    ["HSDBT", "S-S", 5, 1.],
    ["HSDBT", "S-S", 5, 2.],
    ["HSDBT", "S-S", 5, 5.],
    ["HSDBT", "S-S", 5, 10.],

    ["CBT", "S-S", 5, 0.],
    ["CBT", "S-S", 5, 0.5],
    ["CBT", "S-S", 5, 1.],
    ["CBT", "S-S", 5, 2.],
    ["CBT", "S-S", 5, 5.],
    ["CBT", "S-S", 5, 10.],

    ["HSDBT", "S-S", 20, 0.],
    ["HSDBT", "S-S", 20, 0.5],
    ["HSDBT", "S-S", 20, 1.],
    ["HSDBT", "S-S", 20, 2.],
    ["HSDBT", "S-S", 20, 5.],
    ["HSDBT", "S-S", 20, 10.],

    ["CBT", "S-S", 20, 0.],
    ["CBT", "S-S", 20, 0.5],
    ["CBT", "S-S", 20, 1.],
    ["CBT", "S-S", 20, 2.],
    ["CBT", "S-S", 20, 5.],
    ["CBT", "S-S", 20, 10.],
]

ground_truths = [
    5.1527,4.4107,3.9904,3.6265,3.4014,3.2817,
    5.3953,4.5931,4.1484,3.7793,3.5949,3.4921,
    5.4603,4.6511,4.2051,3.8361,3.6485,3.5390,
    5.4777,4.6641,4.2163,3.8472,3.6628,3.5547
]
example_results = test_examples(model, examples).reshape(-1)
errors = torch.abs(example_results - torch.tensor(ground_truths)) / torch.tensor(ground_truths) * 100.0

for i in range(len(examples)):
    print(errors[i].item())
