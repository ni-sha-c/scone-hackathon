# fc + residual block

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tqdm as progressbar
from matplotlib.pyplot import *
import torch.nn.functional as F
import numpy as np
from numpy import *
from datagenerator import *

torch.pi = torch.acos(torch.zeros(1)).item() * 2

class Res(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Res, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x1 = torch.tanh(self.fc1(x))
        x2 = torch.tanh(self.fc2(1.0 * x + x1))
        x3 = self.fc3(x2)
        return x3

def pde(tar_sc, dtar_sc, model, x):
    x= x.requires_grad_(True)  # Create a separate variable for use in this function
    x_numpy = x.cpu().detach()
    q = torch.tensor(tar_sc(x_numpy), device=x.device)
    q_prime = torch.tensor(dtar_sc(x_numpy), device=x.device)
    y = model(x)
    dvx_dx = torch.autograd.grad(y, x, create_graph=True)[0]
    d2vx_dx2 = torch.autograd.grad(dvx_dx, x, create_graph=True)[0]
    pde_lhs = d2vx_dx2 + dvx_dx * q+ y * q_prime
    return pde_lhs



def train_pde(model, x_gr, p_gr, q_gr, tar_sc, dtar_sc, pde, lr, batchsize, device, xmin=torch.tensor([-10.0]), xmax=torch.tensor([10.0]), m=100, iter_num=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataset = create_dataset(x_gr, p_gr, q_gr, m, xmin, xmax)
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True)
    print("Training the model...")
    with progressbar.tqdm (range(iter_num), unit="epoch") as pbar:
        for epoch in range(iter_num):
            optimizer.zero_grad()
            for t, (x, y) in enumerate(dataloader):
                x_g = x.to(device)
                y_g = y.to(device)
                y_g = y_g.unsqueeze(1)
                output = torch.zeros(batchsize, 1, requires_grad=True)
                output = output.to(device)
                new_output = torch.zeros_like(output)  # Create a new tensor with the same shape and device
                for i, x_i in enumerate(x_g):
                    x_i = x_i.detach().clone().to(torch.float32).unsqueeze(0)
                    new_output[i] = pde(tar_sc, dtar_sc, model, x_i)
                output = new_output  # Assign the new tensor to output
                pde_loss = torch.nn.functional.mse_loss(output, y_g.float())
                boundary_loss = 1/2 * model(torch.tensor([-10.0], device=device, dtype=torch.float32))**2 + 1/2*model(torch.tensor([10.0], device=device, dtype=torch.float32))**2
                loss = pde_loss + boundary_loss
                print("loss:", loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if epoch % 25 == 0:
                pbar.write(f"Epoch {epoch}: loss={loss.item():.3e}")
            pbar.update()
    return model

def solve_newton_step_pinn(x_gr, p_gr, q_gr, tar_sc, dtar_sc, pde, lr, batchsize, device, xmin, xmax, m, iter_num=25):
    model = Res(input_dim=1, hidden_dim=50).to(torch.float32)
    model = model.to(device)
    model = train_pde(model, x_gr, p_gr, q_gr, tar_sc, dtar_sc, pde, lr, batchsize, device, xmin, xmax, m, iter_num)
    model.eval()
    with torch.no_grad():
        x_gr_tensor = torch.tensor(x_gr, dtype=torch.float32, device=device).unsqueeze(1)
        model_predictions = model(x_gr_tensor).cpu().numpy().flatten()
    return model_predictions