from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from skorch import NeuralNetRegressor
from sklearn.utils import parallel_backend
from skorch import helper
import torch
import torch.nn as nn
import torch.optim as optim

import warnings
warnings.filterwarnings('ignore')

# with parallel_backend('multiprocessing'):

from model import NeuralNetwork
from dataset import *
import config

df = load_df_and_map_categoricals(config.DATA_FILE)
training_dataset, test_dataset = df_to_train_and_test_datasets(df)

model = NeuralNetRegressor(
    NeuralNetwork,
    criterion=nn.MSELoss,
    max_epochs=config.EPOCHS,
    batch_size=config.BATCH_SIZE,
    verbose=False,
    optimizer=optim.SGD,
    train_split=helper.predefined_split(test_dataset)
)

param_grid = {
    ##### for optimizer
    'optimizer__lr': [0.01, 0.001],
    # 'optimizer__momentum': [0.9],

    ##### for module
    'module__hidden_sizes':[[25, 25, 25], [30, 30, 30]],
    'module__activation': [nn.ReLU, nn.Tanh, nn.Sigmoid],
    # 'module__dropout_rate': [0.1, 0.15]
}

# Create your model, criterion, and grid search object
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)

# Fit the grid search to find the best configuration
grid_result = grid_search.fit(training_dataset.tensors[0], training_dataset.tensors[1])

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))