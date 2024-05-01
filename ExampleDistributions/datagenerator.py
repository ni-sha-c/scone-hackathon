import torch
import numpy as np
from torch.utils.data import Dataset
from gaussian_example import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


target_w = [0.8, 0.2]
target_mu = [[1,0],[5,-1]]
target_sigma = [[[1, 0],[0,1]],[[2,0],[0,1]]]
source_gaussian = Gaussiannd([1], [[0, 0]], [np.eye(2)])
target_gaussian = Gaussiannd([1], [[0, 0]], [np.eye(2)])


## We use 2d_unimodal as source
def source_score(x):
    out = source_gaussian.logprob_grad(x)
    return out
     

## We use 2d_bimodal gaussian as target
def target_score(x):
    out = source_gaussian.logprob_grad(x)
    return out

class PDEDataset(Dataset):
    def __init__(self, x_gr):
        self.x = x_gr
        self.y = pde_rhs(self.x)  # Assign the entire tensor at once
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def pde_rhs(x):
    with torch.no_grad():
        rhs = source_score(x) - target_score(x)
    return rhs

def create_dataset(x_gr):
    dataset = PDEDataset(x_gr)
    return dataset

num_samples = 100000
dim = 2
x_max = 10
x_min = -10
x_gr = torch.rand(num_samples, 2) * (abs(x_max) + abs(x_min)) + x_min
dataset = create_dataset(x_gr)
torch.save(dataset, "dataset2.pth")