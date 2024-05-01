## assume phi = sin(x)sin(y)
## we have q = [1, 1]
## p = [cos(x+y) - 2cos(x)sin(y) + 1, cos(x+y) - 2sin(x)cos(y) + 1]
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

device = "cuda" if torch.cuda.is_available() else "cpu"

def phi(x : torch.tensor) -> torch.tensor:
    return torch.prod(torch.sin(x), dim = 0)

def q(x : torch.tensor) -> torch.tensor:
    return torch.ones(x.shape)

def p(x : torch.tensor) -> torch.tensor:
    sum_cos = torch.cos(torch.sum(x, dim = 1))
    sinx = torch.sin(x)
    cosx = torch.cos(x)
    t = torch.stack((cosx[:, 0]*sinx[:, 1], sinx[:, 0]*cosx[:, 1]), dim = 1)
    return sum_cos.reshape(-1, 1) + 1 - 2*t

def plot_contour(func):
    x = torch.linspace(0,2*torch.pi, 100)
    X,Y = torch.meshgrid(x, x)
    Z = torch.empty_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func(torch.tensor([X[i, j], Y[i, j]]))
    plt.contourf(X, Y, Z)
    plt.colorbar()
    plt.show()

## tested
def dphi(x : torch.tensor) -> torch.tensor:
    return torch.func.jacrev(phi)(x)

## tested
def d_dphidotq(x : torch.tensor) -> torch.tensor:
    f = lambda x : torch.dot(dphi(x), q(x))
    return torch.func.jacrev(f)(x)

##tested
def divergence_dphi(x : torch.tensor) -> torch.tensor:
    return torch.trace(torch.func.jacrev(dphi)(x)) ## can replace with -2*phi

##tested
def d_divergence_dphi(x : torch.tensor) -> torch.tensor:
    return torch.func.jacrev(divergence_dphi)(x) ## can replace with -2*dphi


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
        rhs = p(x) - q(x)
    return rhs

def create_dataset(x_gr):
    dataset = PDEDataset(x_gr)
    return dataset

num_samples = 2500
dim = 2
x_max = 2*torch.pi
x_min = 0
x_gr = torch.rand(num_samples, 2) * (abs(x_max) + abs(x_min)) + x_min
dataset = create_dataset(x_gr)
torch.save(dataset, "dataset_sin.pth")

# test = torch.rand(5,2)