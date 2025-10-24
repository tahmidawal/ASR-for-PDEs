import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parameter as Parameter
import numpy as np
from functools import partial
import operator
from functools import reduce



class LpLoss(object):
    def __init__(self, d=1, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms/ y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


def compl_mul1d(input, weights):
    """
    Complex multiplication helper in Fourier domain for spectral operations.
    
    Args:
        input: (batch, in_channels, x//2+1, 2) - real/imag components
        weights: (in_channels, out_channels, modes, 2) - learnable complex weights
    
    Returns:
        (batch, out_channels, modes, 2) - complex multiplication result
    """
    op = partial(torch.einsum, "bix,iox->box")
    return torch.stack([
        op(input[..., 0], weights[..., 0]) - op(input[..., 1], weights[..., 1]),
        op(input[..., 1], weights[..., 0]) + op(input[..., 0], weights[..., 1])
    ], dim=-1)


'''
1D Fourier Layer. Does FFT, does the linear transformation , and doe stehe inversse FFT

The key insight is that since so many PDEs have sparse representation then we need to ensure to linear transform and do inverse FFT

We only need to learn weihgs for the lowest modes in that case
'''

class SpectralConv1d(nn.Module):
    """
    1D Fourier layer. Does FFT, linear transformation, and inverse FFT.
    
    Key insight: PDEs often have sparse representations in Fourier space.
    We only need to learn weights for the lowest 'modes' frequencies.
    """
    
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConv1d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # Number of Fourier modes to multiply, at most floor(N/2) + 1

        # Initialize weight scaling for stability
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes, 2)
        )

    def forward(self, x):
        batchsize = x.shape[0]

        # 1. Compute Fourier coefficients
        x_ft = torch.fft.rfft(x, dim=-1)

        # Convert to old format for compatibility (real, imag as last dim)
        x_ft_old = torch.stack([x_ft.real, x_ft.imag], dim=-1)

        # 2. Multiply relevant Fourier modes with learnable weights
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1, 2, 
                           device=x.device, dtype=x.dtype)

        # Only transform the first 'modes' frequencies (low-pass filter effect)
        out_ft[:, :, :self.modes] = compl_mul1d(
            x_ft_old[:, :, :self.modes], 
            self.weights1
        )

        # 3. Convert back to complex format
        out_ft_complex = torch.view_as_complex(out_ft)

        # 4. Inverse FFT to get back to physical space
        x = torch.fft.irfft(out_ft_complex, n=x.size(-1), dim=-1)

        return x

'''

One FNO block combining, 

1, S[ectral] Convolution (global in fourier space)
2. Local convolution - 1x1 convolution, acts as skip connection
3. Batch Normalization
4. Activation

'''


class FNOBlock1d(nn.Module):
    """
    One FNO block combining:
    1. Spectral convolution (global, in Fourier space)
    2. Local convolution (1x1 conv, acts as skip connection)
    3. Batch normalization
    4. Activation
    """
    
    def __init__(self, modes, width):
        super(FNOBlock1d, self).__init__()

        self.modes = modes
        self.width = width

        # Spectral convolution
        self.conv = SpectralConv1d(self.width, self.width, self.modes)

        # Local convolution (skip connection in physical space)
        self.w = nn.Conv1d(self.width, self.width, 1)

        # Batch normalization for stability
        self.bn = nn.BatchNorm1d(self.width)

    def forward(self, x):
        # Path 1: Spectral convolution in Fourier space
        x1 = self.conv(x)
        
        # Path 2: Local transformation (skip connection)
        x2 = self.w(x)

        # Combine paths and normalize
        x = self.bn(x1 + x2)

        # Activation
        x = F.gelu(x)

        return x

'''

Complete FNO model

Lift the input to the width dimension
Apply FNI blocks 
Project the output to the out_dim dimension
'''



class FNO1d(nn.Module):

    def __init__(self, modes, width, in_dim = 2, out_dim = 1, depth = 4):
        super(FNO1d, self).__init__()

        self.modes = modes
        self.width = width
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.depth = depth

        self.fc0 = nn.Conv1d(in_dim, width, 1) # Here our input dimension is 2 because we provide the input value along with the coordinate

        self.blocks = nn.ModuleList([FNOBlock1d(modes, width) for _ in range(depth)])

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, self.out_dim)

    def forward(self, x):
        """
        Args:
            x: (batch, 2, grid_points) where dim 1 is [source, coordinate]
            
        Returns:
            (batch, grid_points) - solution values
        """
        # 1. Lift to hidden dimension
        x = self.fc0(x)  # (batch, width, grid_points)
        
        # 2. Apply FNO blocks
        for block in self.blocks:
            x = block(x)
        
        # 3. Transpose for final projection
        x = x.permute(0, 2, 1)  # (batch, grid_points, width)
        
        # 4. Final projection
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        
        return x.squeeze()  # (batch, grid_points)


    def count_params(self):
        c = 0
        for p in self.parameters():
            c += p.numel()
        return c



class UnitGaussianNormalizer(object):

    '''
    Something I didn't know. Apparently we need ot to have zero amean and unit variance otherwise thwe can't have stable FNO training. 
    '''

    def __init__(self, x, eps=0.00001): 
        super(UnitGaussianNormalizer, self).__init__()

        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x):
        """Transform back to original space"""
        x = (x * (self.std + self.eps)) + self.mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()
        return self

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()
        return self
