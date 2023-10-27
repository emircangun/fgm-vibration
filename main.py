import numpy as np
import random
import torch

from dataset import *
from train_and_evaluate import *
from model import NeuralNetwork
import config


# manualSeed = 1773

# np.random.seed(manualSeed)
# random.seed(manualSeed)
# torch.manual_seed(manualSeed)


if __name__ == '__main__':
    training_dataloader, test_dataloader = get_dataloaders(config.DATA_FILE)

    model = NeuralNetwork(hidden_sizes=config.HIDDEN_SIZES,
                          activation=config.ACTIVATION,
                          dropout_rate=config.DROPOUT_RATE)
    
    optimizer = optim.SGD(model.parameters(),
                          **config.OPTIMIZER_PARAMS)

    criterion = nn.MSELoss()

    train_losses = train(model,
                         optimizer,
                         criterion,
                         training_dataloader,
                         test_dataloader,
                         n_epochs=config.EPOCHS)


    # some examples to validate
    examples = [
        ["HSDBT", "S-S FG", 20, 0.5]
    ]
    example_results = test_examples(model, examples)
    print(example_results)