import torch
import torch.nn as nn
from matplotlib.pyplot import *

## phi = sin(x)sin(y)
## 5 or more with relu
## More layers for resnet
## make sure to introduce dropout lyers fro generalization and make sure to initialize the weights correctly
## zevian distribution for initialization or gaussian.

class Res(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Res, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x1 = torch.tanh(self.fc1(x)) ## do relu instead unless strong reason for tanh
        x2 = torch.tanh(self.fc2(1.0 * x + x1)) 
        x3 = self.fc3(x2)
        return x3

class SimpleReluNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SimpleReluNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x1 = nn.Relu(self.fc1(x))
        x2 = nn.Relu(self.dropout(self.fc2(x1)))
        x3 = nn.Relu(self.fc3(x2))
        x4 = nn.Relu(self.dropout(self.fc4(x3)))
        x5 = nn.Relu(self.fc5(x4))
        return x5
    
class GeneralReluNet(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super(GeneralReluNet, self).__init__()
        layers = []
        layer_sizes = [input_size] + hidden_sizes + [input_size]
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)