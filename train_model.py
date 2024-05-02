# fc + residual block

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tqdm as progressbar
from matplotlib.pyplot import *
import torch.nn.functional as F
import numpy as np
from sin_prod import *
from models import *

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

## remains to fix
def train_pde(model, tar_sc, dataloader, optimizer, batchsize, device, epochs=10):
    print("Training the model...")
    with progressbar.tqdm (range(epochs), unit="epoch") as pbar:
        for epoch in range(epochs):
            optimizer.zero_grad()
            for t, (x, y) in enumerate(dataloader):
                # x_g = x.to(device)
                # y_g = y.to(device)
                x_g, y_g = x, y
                # y_g = y_g.unsqueeze(1)
                output = torch.zeros(batchsize, 1, requires_grad=True)
                # output = output.to(device)
                new_output = pde(tar_sc, model, x_g)
                # new_output = torch.empty_like(output)  # Create a new tensor with the same shape and device
                # for i, x_i in enumerate(x_g):
                #     x_i = x_i.detach().clone().to(torch.float32).unsqueeze(0)
                #     new_output[i] = pde(tar_sc, model, x_i)
                output = new_output  # Assign the new tensor to output
                pde_loss = torch.nn.functional.mse_loss(output, y_g.float())
                # loss = pde_loss
                print("loss:", pde_loss)
                optimizer.zero_grad()
                pde_loss.backward()
                optimizer.step()
            if epoch % 25 == 0:
                pbar.write(f"Epoch {epoch}: loss={pde_loss.item():.3e}")
            pbar.update()
    return model

def solve_newton_step_pinn(tar_sc, lr, batchsize, device, epochs=25):
    model = Res(input_dim=2, hidden_dim=50, output_dim=1)
    model = model.to(device)
    dataset_path = "dataset_sin_10000.pth"
    dataset = torch.load(dataset_path)
    train_set, test_set = torch.utils.data.random_split(dataset, [0.7, 0.3])
    train_dataloader = DataLoader(train_set, batch_size=batchsize, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=batchsize, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model = train_pde(model, tar_sc, train_dataloader, optimizer, batchsize, epochs)
    model.eval()
    with torch.no_grad():
        model_predictions = np.empty(len(test_dataloader) * batchsize)
        for t, (x_gr, y_gr) in enumerate(test_dataloader):
            model_predictions = model(x_gr).cpu().numpy().flatten()
    return model_predictions

solve_newton_step_pinn(q, 0.01, 100, "cpu", epochs = 2)