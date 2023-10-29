import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import config
import os

def plot_losses(train_losses, eval_losses):
    plt.plot(train_losses, label="Train Loss")
    plt.plot(eval_losses, label="Evaluation Loss")
    plt.xlabel("# epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.savefig(config.LOSS_FILE)

    print(f"Loss file saved as {config.LOSS_FILE}!")
    
    plt.close()