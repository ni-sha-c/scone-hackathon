import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels * out_channels))
        # self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))
        # weight is stored as real to avoid issue with Adam not working on complex parameters
        # FNO code initializes with rand but we initializes with randn as that seems more natural.
        self.weights1 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, self.modes1, 2))
        # Need unsafe=True since we're changing the dtype of the parameters
        # parametrize.register_parametrization(self, 'weights1', RealToComplex(), unsafe=True)

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x, norm='ortho')

        # Multiply relevant Fourier modes
        # out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        # out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)
        weights1 = torch.view_as_complex(self.weights1)
        out_ft = F.pad(self.compl_mul1d(x_ft[:, :, :self.modes1], weights1),
                       (0, x_ft.shape[-1] - self.modes1))

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1), norm='ortho')
        return x

class FourierOperator1d(nn.Module):

    def __init__(self, modes, width):
        super().__init__()
        self.modes = modes
        self.width = width
        self.conv = SpectralConv1d(self.width, self.width, self.modes)
        self.w = nn.Conv1d(self.width, self.width, 1)

    def forward(self, x):
        return F.gelu(self.conv(x) + self.w(x))


class FNO1d(nn.Module):
    def __init__(self, modes, width = 128, nlayers=4, padding=0):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.nlayers = nlayers
        self.padding = padding  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(2, self.width)  # input channel is 2: (a(x), x)

        self.layers = nn.Sequential(*[FourierOperator1d(self.modes1, self.width)
                                      for _ in range(self.nlayers)])

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.stack([x, grid], dim=-1)
        x = self.fc0(x)
        x = rearrange(x, 'b x c -> b c x')
        if self.padding != 0:
            x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        # FNO code doesn't apply activation on the last block, but we do for code's simplicity.
        # Performance seems about the same.
        x = self.layers(x)

        if self.padding != 0:
            x = x[..., :-self.padding] # pad the domain if input is non-periodic
        x = rearrange(x, 'b c x -> b x c')
        x = self.fc2(F.gelu(self.fc1(x)))
        return rearrange(x, 'b x 1 -> b x')

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        return repeat(torch.linspace(0, 1, size_x, dtype=torch.float, device=device),
                      'x -> b x', b=batchsize)
