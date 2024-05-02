import torch
import torch.nn as nn
from matplotlib.pyplot import *

## phi = sin(x)sin(y)
## 5 or more with relu
## More layers for resnet
## make sure to introduce dropout lyers fro generalization and make sure to initialize the weights correctly
## xaviar distribution for initialization or gaussian.
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        # m.bias.data.fill_(0.01)

class Res(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Res, self).__init__()
        ## Layer initialization
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout1 = nn.Dropout(0.5)

        ## weight initialization
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        

    def forward(self, x):
        x1 = torch.tanh(self.fc1(x))
        x2 = torch.tanh(self.dropout1(self.fc2(x1))) 
        x3 = self.fc3(x2 + x1)
        return x3

class SimpleReluNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleReluNet, self).__init__()
        ## Layer initialization
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, output_dim)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.25)

        ## weight initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
        nn.init.xavier_uniform_(self.fc5.weight)

    def forward(self, x):
        x1 = nn.functional.relu(self.fc1(x))
        x2 = nn.functional.relu(self.dropout1(self.fc2(x1)))
        x3 = nn.functional.relu(self.fc3(x2))
        x4 = nn.functional.relu(self.dropout2(self.fc4(x3)))
        x5 = self.fc5(x4)
        return x5

class GeneralReLuNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(GeneralReLuNet, self).__init__()
        layers = []
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                if torch.rand() < 0.1:
                    print(f"Dropout at layer {i}")
                    layers.append(nn.Dropout(0.25 + torch.rand()/4))
                layers.append(nn.ReLu())
        self.network = nn.Sequential(*layers)
        self.network.apply(init_weights)

    def forward(self, x):
        return self.network(x)