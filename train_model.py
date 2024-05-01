# fc + residual block

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tqdm as progressbar
from matplotlib.pyplot import *
import torch.nn.functional as F
import numpy as np
from sin_prod import *


## tested in sin_prod.py
def pde(tar_sc , model, x):
    x= x.requires_grad_(True)  # Create a separate variable for use in this function
    dy = torch.func.jacrev(model)
    dy_dotq = lambda x : torch.dot(dy(x), tar_sc(x))
    d_dy_dotq = torch.func.jacrev(dy_dotq)
    divergence_dy = lambda x : torch.trace(torch.func.jacrev(dy)(x))
    d_divergence_dy = torch.func.jacrev(divergence_dy)
    pde_lhs = d_divergence_dy(x) + d_dy_dotq(x)
    return pde_lhs



def train_pde(model, tar_sc, dataset_path, lr, batchsize, device, epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataset = torch.load(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True)
    print("Training the model...")
    with progressbar.tqdm (range(epochs), unit="epoch") as pbar:
        for epoch in range(epochs):
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
                    new_output[i] = pde(tar_sc, model, x_i)
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

def solve_newton_step_pinn(x_gr, p_gr, q_gr, tar_sc, dtar_sc, pde, lr, batchsize, device, xmin, xmax, m, epochs=25):
    model = Res(input_dim=1, hidden_dim=50).to(torch.float32)
    model = model.to(device)
    model = train_pde(model, x_gr, p_gr, q_gr, tar_sc, dtar_sc, pde, lr, batchsize, device, xmin, xmax, m, epochs)
    model.eval()
    with torch.no_grad():
        x_gr_tensor = torch.tensor(x_gr, dtype=torch.float32, device=device).unsqueeze(1)
        model_predictions = model(x_gr_tensor).cpu().numpy().flatten()
    return model_predictions