"""
Ultra-lightweight models for 1D Poisson equation

Models tested:
1. Dense Net (MLP): ~5-10K params - fully connected layers
2. Micro CNN: ~10-15K params - minimal convolutional network
3. Nano U-Net: ~20-30K params - minimal encoder-decoder with skip connections

Goal: Test if simple dense networks can match CNNs for this smooth PDE problem.
"""

import torch
import torch.nn as nn
import numpy as np

from models import register


@register('mlp-dense')
class MLPDense(nn.Module):
    """
    Simple Multi-Layer Perceptron (Dense Network) for 1D Poisson
    
    Hypothesis: Since the 1D Poisson equation produces smooth solutions,
    a fully-connected network might be sufficient without explicit spatial structure.
    
    Architecture:
        Input: (B, 1, 256) source f(x)
        Flatten → FC layers → Reshape
        Output: (B, 1, 256) solution u(x)
    
    Args:
        input_size: Input spatial resolution (default: 256)
        hidden_dims: List of hidden layer dimensions
        activation: 'relu' or 'gelu'
    """
    
    def __init__(
        self,
        input_size=256,
        hidden_dims=[128, 256, 512, 256],
        activation='relu'
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_dims = hidden_dims
        
        # Build fully connected layers
        layers = []
        in_dim = input_size
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            
            if activation == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation == 'gelu':
                layers.append(nn.GELU())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            
            in_dim = hidden_dim
        
        # Final output layer
        layers.append(nn.Linear(in_dim, input_size))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, source):
        """
        Args:
            source: (B, 1, N) - source term f(x)
        
        Returns:
            solution: (B, 1, N) - predicted solution u(x)
        """
        B, C, N = source.shape
        
        # Flatten: (B, 1, N) → (B, N)
        x = source.squeeze(1)
        
        # Pass through MLP
        x = self.net(x)
        
        # Reshape back: (B, N) → (B, 1, N)
        return x.unsqueeze(1)


@register('cnn-micro')
class CNNMicro(nn.Module):
    """
    Micro CNN for 1D Poisson equation
    
    Ultra-lightweight convolutional network with 3-4 layers.
    Target: ~10-15K parameters
    
    Architecture:
        Stack of Conv1d layers with small channel counts
        Preserves spatial dimensions throughout
    
    Args:
        hidden_channels: List of channel sizes [8, 16, 32]
        kernel_size: Convolution kernel size (default: 5)
        padding_mode: 'reflect', 'zeros', or 'circular'
        activation: 'relu' or 'gelu'
    """
    
    def __init__(
        self,
        hidden_channels=[8, 16, 32],
        kernel_size=5,
        padding_mode='reflect',
        activation='relu'
    ):
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding_mode = padding_mode
        
        layers = []
        in_channels = 1
        padding = kernel_size // 2
        
        # Build convolutional layers
        for out_channels in hidden_channels:
            layers.append(nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                padding_mode=padding_mode
            ))
            
            if activation == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation == 'gelu':
                layers.append(nn.GELU())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            
            in_channels = out_channels
        
        # Final output layer: map to solution (1 channel)
        layers.append(nn.Conv1d(
            in_channels,
            1,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode=padding_mode
        ))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, source):
        """
        Args:
            source: (B, 1, N) - source term f(x)
        
        Returns:
            solution: (B, 1, N) - predicted solution u(x)
        """
        return self.net(source)


@register('unet-nano')
class UNetNano(nn.Module):
    """
    Nano U-Net: Minimal encoder-decoder with skip connections
    
    Target: ~20-30K parameters
    
    Architecture:
        2-level U-Net only:
        - Encoder: 256 → 128 (with pooling)
        - Decoder: 128 → 256 (with upsampling)
        - Skip connections preserve details
    
    This tests whether U-Net's multi-scale architecture helps
    even at extremely low parameter counts.
    
    Args:
        base_channels: Number of channels in first layer (8 or 12)
        kernel_size: Convolution kernel size
        padding_mode: 'reflect', 'zeros', or 'circular'
        activation: 'relu' or 'gelu'
    """
    
    def __init__(
        self,
        base_channels=12,
        kernel_size=5,
        padding_mode='reflect',
        activation='relu'
    ):
        super().__init__()
        
        self.base_channels = base_channels
        padding = kernel_size // 2
        
        # Choose activation function
        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation == 'gelu':
            act_fn = nn.GELU
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Encoder level 1: 256 resolution
        self.encoder1 = nn.Sequential(
            nn.Conv1d(1, base_channels, kernel_size, padding=padding, padding_mode=padding_mode),
            act_fn(inplace=True),
            nn.Conv1d(base_channels, base_channels, kernel_size, padding=padding, padding_mode=padding_mode),
            act_fn(inplace=True)
        )
        self.pool1 = nn.MaxPool1d(2)
        
        # Encoder level 2: 128 resolution (bottleneck)
        self.encoder2 = nn.Sequential(
            nn.Conv1d(base_channels, base_channels * 2, kernel_size, padding=padding, padding_mode=padding_mode),
            act_fn(inplace=True),
            nn.Conv1d(base_channels * 2, base_channels * 2, kernel_size, padding=padding, padding_mode=padding_mode),
            act_fn(inplace=True)
        )
        
        # Decoder: Upsample back to 256
        self.upconv1 = nn.ConvTranspose1d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        
        # Decoder conv (receives concatenated skip connection from encoder1)
        self.decoder1 = nn.Sequential(
            nn.Conv1d(base_channels * 2, base_channels, kernel_size, padding=padding, padding_mode=padding_mode),
            act_fn(inplace=True),
            nn.Conv1d(base_channels, base_channels, kernel_size, padding=padding, padding_mode=padding_mode),
            act_fn(inplace=True)
        )
        
        # Final output layer
        self.output_conv = nn.Conv1d(base_channels, 1, kernel_size=1)
    
    def forward(self, source):
        """
        Args:
            source: (B, 1, N) - source term f(x)
        
        Returns:
            solution: (B, 1, N) - predicted solution u(x)
        """
        # Encoder
        enc1 = self.encoder1(source)  # (B, base_ch, 256)
        x = self.pool1(enc1)           # (B, base_ch, 128)
        
        enc2 = self.encoder2(x)        # (B, base_ch*2, 128)
        
        # Decoder with skip connection
        x = self.upconv1(enc2)         # (B, base_ch, 256)
        
        # Handle potential size mismatch
        if x.shape[2] != enc1.shape[2]:
            diff = enc1.shape[2] - x.shape[2]
            x = nn.functional.pad(x, (diff // 2, diff - diff // 2))
        
        # Concatenate skip connection
        x = torch.cat([x, enc1], dim=1)  # (B, base_ch*2, 256)
        
        # Final decoder conv
        x = self.decoder1(x)             # (B, base_ch, 256)
        
        # Output
        return self.output_conv(x)       # (B, 1, 256)


@register('unet-nano-deep')
class UNetNanoDeep(nn.Module):
    """
    Slightly deeper Nano U-Net variant
    
    Same as UNetNano but with 3 conv blocks per level instead of 2.
    Target: ~35-45K parameters
    
    This tests if going slightly deeper helps at low parameter counts.
    """
    
    def __init__(
        self,
        base_channels=10,
        kernel_size=5,
        padding_mode='reflect',
        activation='relu'
    ):
        super().__init__()
        
        self.base_channels = base_channels
        padding = kernel_size // 2
        
        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation == 'gelu':
            act_fn = nn.GELU
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Encoder level 1: 256 resolution (3 conv blocks)
        self.encoder1 = nn.Sequential(
            nn.Conv1d(1, base_channels, kernel_size, padding=padding, padding_mode=padding_mode),
            act_fn(inplace=True),
            nn.Conv1d(base_channels, base_channels, kernel_size, padding=padding, padding_mode=padding_mode),
            act_fn(inplace=True),
            nn.Conv1d(base_channels, base_channels, kernel_size, padding=padding, padding_mode=padding_mode),
            act_fn(inplace=True)
        )
        self.pool1 = nn.MaxPool1d(2)
        
        # Encoder level 2: 128 resolution (3 conv blocks)
        self.encoder2 = nn.Sequential(
            nn.Conv1d(base_channels, base_channels * 2, kernel_size, padding=padding, padding_mode=padding_mode),
            act_fn(inplace=True),
            nn.Conv1d(base_channels * 2, base_channels * 2, kernel_size, padding=padding, padding_mode=padding_mode),
            act_fn(inplace=True),
            nn.Conv1d(base_channels * 2, base_channels * 2, kernel_size, padding=padding, padding_mode=padding_mode),
            act_fn(inplace=True)
        )
        
        # Decoder
        self.upconv1 = nn.ConvTranspose1d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        
        self.decoder1 = nn.Sequential(
            nn.Conv1d(base_channels * 2, base_channels, kernel_size, padding=padding, padding_mode=padding_mode),
            act_fn(inplace=True),
            nn.Conv1d(base_channels, base_channels, kernel_size, padding=padding, padding_mode=padding_mode),
            act_fn(inplace=True),
            nn.Conv1d(base_channels, base_channels, kernel_size, padding=padding, padding_mode=padding_mode),
            act_fn(inplace=True)
        )
        
        self.output_conv = nn.Conv1d(base_channels, 1, kernel_size=1)
    
    def forward(self, source):
        """
        Args:
            source: (B, 1, N) - source term f(x)
        
        Returns:
            solution: (B, 1, N) - predicted solution u(x)
        """
        # Encoder
        enc1 = self.encoder1(source)
        x = self.pool1(enc1)
        
        enc2 = self.encoder2(x)
        
        # Decoder with skip connection
        x = self.upconv1(enc2)
        
        if x.shape[2] != enc1.shape[2]:
            diff = enc1.shape[2] - x.shape[2]
            x = nn.functional.pad(x, (diff // 2, diff - diff // 2))
        
        x = torch.cat([x, enc1], dim=1)
        x = self.decoder1(x)
        
        return self.output_conv(x)


