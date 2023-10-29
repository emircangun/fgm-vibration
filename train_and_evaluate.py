import torch
import os

from config import METHOD_DICT, BEAM_TYPE_DICT, L_H_DICT
import config
from dataset import transform_examples
from model import *


def evaluate(model, test_dataloader):
    total_samples = 0
    total_mse = 0.0 

    model.eval()
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            outputs = model(inputs)

            mse = ((outputs - labels).pow(2).sum()).item()

            total_mse += mse
            total_samples += labels.size(0)

    # Calculate the average MSE over the entire dataset
    mean_mse = total_mse / total_samples

    return mean_mse


def train(model, optimizer, criterion, training_dataloader, test_dataloader, n_epochs=150):
    train_losses = []
    eval_losses = []
    for epoch in range(n_epochs):
        cur_loss = 0
        total_len = 0
        
        model.train()
        for data, target in training_dataloader:  # Iterate through your data
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.float(), target.float())
            loss.backward()
            optimizer.step()

            cur_loss += loss
            total_len += data.shape[0]
        
        train_losses.append((cur_loss/total_len).item())
        eval_losses.append(evaluate(model, test_dataloader))

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            test_mse = evaluate(model, test_dataloader)
            print(f"Epoch: {epoch} \t Loss {cur_loss/total_len} \t Test Mean MSE: {test_mse}")
        
        if epoch == n_epochs - 1:
            save_model(model, config.MODEL_WEIGHTS_FILE)

    return train_losses, eval_losses


def test_examples(model, examples):
    output = model(transform_examples(examples))
    return output
