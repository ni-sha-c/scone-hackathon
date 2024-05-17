# fc + residual block

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tqdm as progressbar
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from sin_prod import *
from models import *
from FNO import FNO1d

## tested in sin_prod.py
# fc + residual block
# fc + residual block
def pde(tar_sc , model, x):
    x= x.requires_grad_(True)  # Create a separate variable for use in this function
    dy = lambda p : torch.func.jacrev(model)(p).squeeze()
    dy_dotq = lambda x : torch.dot(dy(x), tar_sc(x))
    d_dy_dotq = torch.func.jacrev(dy_dotq)
    divergence_dy = lambda x : torch.trace(torch.func.jacrev(dy)(x))
    d_divergence_dy = torch.func.jacrev(divergence_dy)
    pde_lhs = d_divergence_dy(x) + d_dy_dotq(x)
    return pde_lhs

## remains to fix
def train_pde(model, tar_sc, dataloader, optimizer, batchsize, device, epochs=10, fno = False):
    loss_list = []
    print("Training the model...")
    output = torch.empty((batchsize, 2), device = device)
    for epoch in tqdm(range(epochs)):
        # optimizer.zero_grad()
        for t, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()
            x_g = x.to(device)
            y_g = y.to(device)
            new_output = torch.empty_like(output)
            for i, x_i in enumerate(x_g):
                x_out = x_i.detach().clone()
                # if fno :
                #     x_out = x_out.reshape(1, -1)
                new_output[i] = pde(tar_sc, model, x_out)
            output = new_output
            pde_loss = nn.functional.mse_loss(output, y_g.float())
            # loss = pde_loss
            pde_loss.backward()
            optimizer.step()
        print("loss:", pde_loss.item())
        loss_list.append(pde_loss.item())
    return loss_list

def solve_newton_step_pinn(tar_sc, lr, batchsize, device, num_train, num_test, model_type, epochs=25):
    if model_type == "res":
        model = Res(input_dim=2, hidden_dim=50, output_dim=1)
        network = "res"
    elif model_type == "general":
        model = GeneralReLuNet(input_size = 2, hidden_sizes = [50, 50, 64, 25], output_size = 1)
        network = "General ReLu"
    elif model_type == "FNO":
        model = FNO1d(1)
        network = "FNO"
    else:
        model = SimpleReluNet(2, 50, 1)
        network = "Simple ReLu"
    model = model.to(device)
    x_max = 2*torch.pi
    x_min = 0
    ## train
    x_gr = torch.rand(num_train, 2) * (abs(x_max) + abs(x_min)) + x_min
    train_dataset = create_dataset(x_gr)
    train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_lst = train_pde(model, tar_sc, train_dataloader, optimizer, batchsize, device, epochs, fno = (model_type == 'FNO'))
    plt.plot(loss_lst)
    plt.title(f"Loss plot for {network} network")
    plt.xlabel("The epoch")
    plt.ylabel("The MSE loss")
    plt.show()
    return model, loss_lst

## tbd
def test_model(model, num_test):
    ## test
    x_gr = torch.rand(num_test, 2) * (abs(x_max) + abs(x_min)) + x_min
    test_dataset = create_dataset(x_gr)
    model.eval()
    with torch.no_grad():
        model_predictions = model(x_gr).cpu().numpy().flatten()

solve_newton_step_pinn(q, 0.01, 100, "cpu", 2000, 1000, "", epochs = 2)