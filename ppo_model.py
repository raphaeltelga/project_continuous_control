import torch
import torch.nn as nn
import torch.nn.functional as F
    
def init_weights(layer):
    if type(layer) == nn.Linear:
        torch.nn.init.kaiming_uniform_(layer.weight)
        layer.bias.data.fill_(0.01)
    
class PolicyNetwork(nn.Module):
    def __init__(self, input_units, output_units, hidden_units):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(nn.Linear(input_units,hidden_units),
                                  nn.BatchNorm1d(hidden_units),
                                  nn.ReLU(),
                                  nn.Linear(hidden_units, hidden_units),
                                  nn.BatchNorm1d(hidden_units),
                                  nn.ReLU(),
                                  nn.Linear(hidden_units, output_units))
        self.stds = nn.Parameter(torch.ones(1,output_units))

    def reset_param(self):
        self.network.apply(init_weights)
    
    def forward(self, state):
        means = F.tanh(self.network(state))
        return torch.distributions.Normal(means, self.stds), means 

    

class CriticNetwork(nn.Module):
    def __init__(self, input_units, hidden_units):
        super(CriticNetwork,self).__init__()
        self.network = nn.Sequential(nn.Linear(input_units,hidden_units),
                                  nn.BatchNorm1d(hidden_units),
                                  nn.ReLU(),
                                  nn.Linear(hidden_units, hidden_units),
                                  nn.BatchNorm1d(hidden_units),
                                  nn.ReLU(),
                                  nn.Linear(hidden_units, 1))

    
    def reset_param(self):
        self.network.apply(init_weights)
    
    def forward(self, state):
        state_value = self.network(state) 
        return state_value
      