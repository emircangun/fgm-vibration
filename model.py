import torch
import torch.nn as nn

from config import METHOD_DICT, BEAM_TYPE_DICT, L_H_DICT


input_size = len(METHOD_DICT.keys()) + len(BEAM_TYPE_DICT.keys()) + len(L_H_DICT.keys()) + 1


class NeuralNetwork(nn.Module):
    def __init__(self,
                 hidden_sizes=[20, 20],
                #  weight_init=torch.nn.init.xavier_uniform_,
                 activation=nn.ReLU,
                 dropout_rate=0.1):
        super(NeuralNetwork, self).__init__()

        layers = []
        
        # input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(activation())
        layers.append(nn.Dropout(dropout_rate))
        
        # hidden layers
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            layers.append(activation())
            layers.append(nn.Dropout(dropout_rate))
        
        # output layer
        layers.append(nn.Linear(hidden_sizes[-1], 1))
        layers.append(nn.ReLU())
        
        self.model = nn.Sequential(*layers)


    def forward(self, x):
        return self.model(x)
    

def save_model(model, path_to_save):
    torch.save(model, path_to_save)

def load_model(path_to_load):
    model = torch.load(path_to_load)
    model.eval()
    return model