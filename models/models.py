

import torch
import torch.nn as nn
import numpy as np
from .fno1d import FNO1d as FNO1dImpl

# Model registry
_models = {}

def register(name):
    def decorator(cls):
        _models[name] = cls
        return cls
    return decorator

def make(spec, args=None):
    if args is not None:
        model = _models[spec['name']](**args)
    else:
        model = _models[spec['name']](**spec.get('args', {}))
    return model


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


@register('cnn-better')
class CNNBetter(nn.Module):
    """
    Better CNN with residual connections for 1D Poisson equation

    Improvements over basic CNN:
    - Deeper network (5-7 layers)
    - Larger kernels for bigger receptive field
    - Residual connections every 2 layers
    - GELU activation (better for smooth functions)

    Target: ~80-100K parameters
    Expected performance: 0.1-0.5% relative L2 error

    Args:
        channels: List of channel sizes (default: [32, 64, 128, 64, 32])
        kernel_sizes: List of kernel sizes (default: [9, 7, 5, 7, 9])
        padding_mode: 'reflect', 'zeros', or 'circular'
        activation: 'relu' or 'gelu' (gelu recommended)
        use_residual: Add skip connections every 2 layers
    """

    def __init__(
        self,
        channels=[32, 64, 128, 64, 32],
        kernel_sizes=[9, 7, 5, 7, 9],
        padding_mode='reflect',
        activation='gelu',
        use_residual=True
    ):
        super().__init__()

        self.channels = channels
        self.use_residual = use_residual

        # Choose activation
        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation == 'gelu':
            act_fn = nn.GELU
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Initial conv: 1 → channels[0]
        padding_0 = kernel_sizes[0] // 2
        self.input_conv = nn.Sequential(
            nn.Conv1d(1, channels[0], kernel_size=kernel_sizes[0],
                     padding=padding_0, padding_mode=padding_mode),
            act_fn()
        )

        # Build main layers
        self.layers = nn.ModuleList()
        self.skip_convs = nn.ModuleList() if use_residual else None

        for i in range(len(channels) - 1):
            padding = kernel_sizes[i + 1] // 2

            layer = nn.Sequential(
                nn.Conv1d(channels[i], channels[i + 1], kernel_size=kernel_sizes[i + 1],
                         padding=padding, padding_mode=padding_mode),
                act_fn()
            )
            self.layers.append(layer)

            # Add skip connection projection if needed (every 2 layers)
            if use_residual and i % 2 == 1 and channels[i - 1] != channels[i + 1]:
                skip_conv = nn.Conv1d(channels[i - 1], channels[i + 1], kernel_size=1)
                self.skip_convs.append(skip_conv)
            else:
                self.skip_convs.append(None) if use_residual else None

        # Output conv: channels[-1] → 1
        self.output_conv = nn.Conv1d(channels[-1], 1, kernel_size=1)

    def forward(self, source):
        """
        Args:
            source: (B, 1, N) - source term f(x)

        Returns:
            solution: (B, 1, N) - predicted solution u(x)
        """
        x = self.input_conv(source)

        # Store for residual connections
        residual = x

        for i, layer in enumerate(self.layers):
            x = layer(x)

            # Add residual connection every 2 layers
            if self.use_residual and i % 2 == 1:
                if self.skip_convs[i] is not None:
                    residual = self.skip_convs[i](residual)
                x = x + residual
                residual = x

        return self.output_conv(x)


@register('unet-medium')
class UNetMedium(nn.Module):
    """
    Medium-sized U-Net for 1D Poisson equation

    Improvements over nano U-Net:
    - 3 levels instead of 2: 256 → 128 → 64 → 128 → 256
    - More conv blocks per level (3 instead of 2)
    - Larger base channels (32)
    - GELU activation for smooth functions

    Target: ~100-150K parameters
    Expected performance: 0.05-0.2% relative L2 error

    Args:
        base_channels: Number of channels in first layer (default: 32)
        num_levels: Number of U-Net levels (default: 3)
        kernel_size: Convolution kernel size
        blocks_per_level: Conv blocks per encoder/decoder level
        padding_mode: 'reflect', 'zeros', or 'circular'
        activation: 'relu' or 'gelu'
    """

    def __init__(
        self,
        base_channels=32,
        num_levels=3,
        kernel_size=7,
        blocks_per_level=3,
        padding_mode='reflect',
        activation='gelu'
    ):
        super().__init__()

        self.num_levels = num_levels
        padding = kernel_size // 2

        # Choose activation
        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation == 'gelu':
            act_fn = nn.GELU
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()

        in_ch = 1
        for level in range(num_levels):
            out_ch = base_channels * (2 ** level)

            # Build conv blocks for this level
            blocks = []
            for block in range(blocks_per_level):
                blocks.append(nn.Conv1d(
                    in_ch if block == 0 else out_ch,
                    out_ch,
                    kernel_size=kernel_size,
                    padding=padding,
                    padding_mode=padding_mode
                ))
                blocks.append(act_fn())

            self.encoders.append(nn.Sequential(*blocks))

            # Add pooling (except for last level - bottleneck)
            if level < num_levels - 1:
                self.pools.append(nn.MaxPool1d(2))

            in_ch = out_ch

        # Build decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for level in range(num_levels - 2, -1, -1):
            in_ch = base_channels * (2 ** (level + 1))
            out_ch = base_channels * (2 ** level)

            # Upsampling
            self.upconvs.append(nn.ConvTranspose1d(in_ch, out_ch, kernel_size=2, stride=2))

            # Decoder blocks (receives skip connection, so in_ch * 2)
            blocks = []
            for block in range(blocks_per_level):
                blocks.append(nn.Conv1d(
                    out_ch * 2 if block == 0 else out_ch,
                    out_ch,
                    kernel_size=kernel_size,
                    padding=padding,
                    padding_mode=padding_mode
                ))
                blocks.append(act_fn())

            self.decoders.append(nn.Sequential(*blocks))

        # Output layer
        self.output_conv = nn.Conv1d(base_channels, 1, kernel_size=1)

    def forward(self, source):
        """
        Args:
            source: (B, 1, N) - source term f(x)

        Returns:
            solution: (B, 1, N) - predicted solution u(x)
        """
        # Encoder
        skip_connections = []
        x = source

        for level in range(self.num_levels):
            x = self.encoders[level](x)

            # Save skip connection (except for bottleneck)
            if level < self.num_levels - 1:
                skip_connections.append(x)
                x = self.pools[level](x)

        # Decoder
        for level in range(self.num_levels - 2, -1, -1):
            x = self.upconvs[self.num_levels - 2 - level](x)

            # Handle size mismatch
            skip = skip_connections[level]
            if x.shape[2] != skip.shape[2]:
                diff = skip.shape[2] - x.shape[2]
                x = nn.functional.pad(x, (diff // 2, diff - diff // 2))

            # Concatenate skip connection
            x = torch.cat([x, skip], dim=1)

            # Decoder conv
            x = self.decoders[self.num_levels - 2 - level](x)

        return self.output_conv(x)


@register('fno-1d')
class FNO1D(nn.Module):
    """
    Fourier Neural Operator for 1D Poisson equation

    FNO learns operators in the Fourier domain, making it highly effective
    for smooth PDEs with spectral structure. For 1D Poisson with sinusoidal
    sources, FNO should naturally learn the relationship u_k = f_k / k^2.

    Architecture:
    - Lifting: project input from 1 channel to 'width' channels
    - FNO layers: spectral convolution in Fourier space + skip connection
    - Projection: project from 'width' channels back to 1 channel

    Target: ~50-80K parameters
    Expected performance: 0.01-0.1% relative L2 error

    Args:
        modes: Number of Fourier modes to use (frequency cutoff)
        width: Number of channels in hidden layers
        num_layers: Number of FNO layers
        activation: 'relu' or 'gelu'

    Reference:
        Li et al., "Fourier Neural Operator for Parametric Partial
        Differential Equations", ICLR 2021
    """

    def __init__(
        self,
        modes=16,
        width=64,
        num_layers=4,
        activation='gelu'
    ):
        super().__init__()

        self.modes = modes
        self.width = width
        self.num_layers = num_layers

        # Choose activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Lifting layer: 1 → width
        self.lifting = nn.Conv1d(1, width, kernel_size=1)

        # FNO layers
        self.fno_layers = nn.ModuleList([
            SpectralConv1d(width, width, modes) for _ in range(num_layers)
        ])

        # Skip connection layers
        self.skip_layers = nn.ModuleList([
            nn.Conv1d(width, width, kernel_size=1) for _ in range(num_layers)
        ])

        # Projection layer: width → 1
        self.projection = nn.Sequential(
            nn.Conv1d(width, width // 2, kernel_size=1),
            self.activation,
            nn.Conv1d(width // 2, 1, kernel_size=1)
        )

    def forward(self, source):
        """
        Args:
            source: (B, 1, N) - source term f(x)

        Returns:
            solution: (B, 1, N) - predicted solution u(x)
        """
        # Lifting
        x = self.lifting(source)

        # FNO layers
        for i in range(self.num_layers):
            # Spectral convolution in Fourier space
            x_fno = self.fno_layers[i](x)

            # Skip connection in physical space
            x_skip = self.skip_layers[i](x)

            # Combine and activate
            x = self.activation(x_fno + x_skip)

        # Projection
        return self.projection(x)


class SpectralConv1d(nn.Module):
    """
    1D Spectral Convolution (Fourier layer) for FNO

    Performs convolution in Fourier space:
    1. FFT: transform to frequency domain
    2. Multiply by learnable weights (for lowest 'modes' frequencies)
    3. IFFT: transform back to physical space

    Args:
        in_channels: Input channels
        out_channels: Output channels
        modes: Number of Fourier modes to keep
    """

    def __init__(self, in_channels, out_channels, modes):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        # Learnable weights in Fourier space (complex-valued)
        # We only store modes for positive frequencies (real FFT symmetry)
        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            torch.randn(in_channels, out_channels, modes, 2) * scale
        )

    def forward(self, x):
        """
        Args:
            x: (B, in_channels, N) input

        Returns:
            (B, out_channels, N) output
        """
        B, C, N = x.shape

        # FFT: transform to frequency domain
        # Use rfft for real-valued input (more efficient)
        x_ft = torch.fft.rfft(x, dim=-1)  # (B, C, N//2+1)

        # Initialize output in Fourier space
        out_ft = torch.zeros(B, self.out_channels, N // 2 + 1,
                            dtype=torch.cfloat, device=x.device)

        # Multiply by learnable weights (only for low modes)
        # Convert weights to complex
        weights_complex = torch.view_as_complex(self.weights)  # (in_ch, out_ch, modes)

        # Matrix multiply in Fourier space (for modes 0 to self.modes-1)
        modes_to_use = min(self.modes, N // 2 + 1)
        out_ft[:, :, :modes_to_use] = torch.einsum(
            'bix,iox->box',
            x_ft[:, :, :modes_to_use],
            weights_complex[:, :, :modes_to_use]
        )

        # IFFT: transform back to physical space
        x_out = torch.fft.irfft(out_ft, n=N, dim=-1)

        return x_out


@register('fno-1d-advanced')
class FNO1DAdvanced(nn.Module):
    """
    Advanced Fourier Neural Operator with all optimizations

    Enhancements over basic FNO:
    - More Fourier modes (32) for higher frequency resolution
    - Deeper network (6 layers) for more expressiveness
    - Wider channels (96) for more capacity
    - Residual connections between FNO layers
    - Learnable frequency weights for adaptive mode selection
    - Layer normalization for training stability

    Target: ~120-150K parameters
    Expected performance: 0.001-0.05% relative L2 error

    Args:
        modes: Number of Fourier modes (default: 32)
        width: Hidden channel width (default: 96)
        num_layers: Number of FNO layers (default: 6)
        activation: 'relu' or 'gelu'
        use_layer_norm: Apply layer normalization
        use_freq_weights: Use learnable frequency importance weights
    """

    def __init__(
        self,
        modes=32,
        width=96,
        num_layers=6,
        activation='gelu',
        use_layer_norm=True,
        use_freq_weights=True
    ):
        super().__init__()

        self.modes = modes
        self.width = width
        self.num_layers = num_layers
        self.use_layer_norm = use_layer_norm
        self.use_freq_weights = use_freq_weights

        # Choose activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Lifting: 1 → width (with residual path)
        self.lifting = nn.Sequential(
            nn.Conv1d(1, width // 2, kernel_size=1),
            self.activation,
            nn.Conv1d(width // 2, width, kernel_size=1)
        )

        # FNO layers with residual connections
        self.fno_layers = nn.ModuleList([
            SpectralConv1dAdvanced(width, width, modes, use_freq_weights)
            for _ in range(num_layers)
        ])

        # Skip connection layers
        self.skip_layers = nn.ModuleList([
            nn.Conv1d(width, width, kernel_size=1) for _ in range(num_layers)
        ])

        # Layer normalization (optional)
        if use_layer_norm:
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(width) for _ in range(num_layers)
            ])
        else:
            self.layer_norms = None

        # Projection: width → 1 (with deeper projection head)
        self.projection = nn.Sequential(
            nn.Conv1d(width, width // 2, kernel_size=1),
            self.activation,
            nn.Conv1d(width // 2, width // 4, kernel_size=1),
            self.activation,
            nn.Conv1d(width // 4, 1, kernel_size=1)
        )

    def forward(self, source):
        """
        Args:
            source: (B, 1, N) - source term f(x)

        Returns:
            solution: (B, 1, N) - predicted solution u(x)
        """
        # Lifting
        x = self.lifting(source)

        # FNO layers with residual connections
        for i in range(self.num_layers):
            residual = x

            # Spectral convolution
            x_fno = self.fno_layers[i](x)

            # Skip connection
            x_skip = self.skip_layers[i](x)

            # Combine
            x = x_fno + x_skip

            # Add residual connection (every layer)
            x = x + residual

            # Layer norm (optional)
            if self.layer_norms is not None:
                # LayerNorm expects (B, N, C) format
                x = x.transpose(1, 2)
                x = self.layer_norms[i](x)
                x = x.transpose(1, 2)

            # Activation
            x = self.activation(x)

        # Projection
        return self.projection(x)


class SpectralConv1dAdvanced(nn.Module):
    """
    Advanced 1D Spectral Convolution with learnable frequency weights

    Enhancement: adds learnable importance weights for each frequency mode,
    allowing the network to adaptively focus on relevant frequencies.

    Args:
        in_channels: Input channels
        out_channels: Output channels
        modes: Number of Fourier modes to keep
        use_freq_weights: Use learnable per-mode importance weights
    """

    def __init__(self, in_channels, out_channels, modes, use_freq_weights=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.use_freq_weights = use_freq_weights

        # Learnable weights in Fourier space (complex-valued)
        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            torch.randn(in_channels, out_channels, modes, 2) * scale
        )

        # Learnable frequency importance weights (optional)
        if use_freq_weights:
            self.freq_weights = nn.Parameter(torch.ones(modes))
        else:
            self.freq_weights = None

    def forward(self, x):
        """
        Args:
            x: (B, in_channels, N) input

        Returns:
            (B, out_channels, N) output
        """
        B, C, N = x.shape

        # FFT
        x_ft = torch.fft.rfft(x, dim=-1)

        # Initialize output
        out_ft = torch.zeros(B, self.out_channels, N // 2 + 1,
                            dtype=torch.cfloat, device=x.device)

        # Convert weights to complex
        weights_complex = torch.view_as_complex(self.weights)

        # Apply frequency weights if enabled
        modes_to_use = min(self.modes, N // 2 + 1)
        if self.use_freq_weights:
            # Apply learnable per-frequency scaling
            freq_weights = self.freq_weights[:modes_to_use].unsqueeze(0).unsqueeze(0)
            weighted_input = x_ft[:, :, :modes_to_use] * freq_weights
        else:
            weighted_input = x_ft[:, :, :modes_to_use]

        # Spectral convolution
        out_ft[:, :, :modes_to_use] = torch.einsum(
            'bix,iox->box',
            weighted_input,
            weights_complex[:, :, :modes_to_use]
        )

        # IFFT
        x_out = torch.fft.irfft(out_ft, n=N, dim=-1)

        return x_out


@register('fno-1d-official')
class FNO1dOfficial(nn.Module):
    """
    Official FNO implementation following the original paper closely
    
    This is our custom implementation based on the official FNO codebase
    with proper spectral convolution, normalization, and training practices.
    
    Expected to achieve much better accuracy than the existing FNO models.
    """
    
    def __init__(
        self,
        modes=24,
        width=96,
        depth=6,
        activation='gelu'
    ):
        super().__init__()
        
        # Use our custom FNO implementation
        self.fno = FNO1dImpl(
            modes=modes,
            width=width,
            in_dim=2,  # source + coordinate
            out_dim=1,  # solution
            depth=depth
        )
    
    def forward(self, source):
        """
        Args:
            source: (B, 1, N) - source term f(x)
        
        Returns:
            solution: (B, 1, N) - predicted solution u(x)
        """
        B, C, N = source.shape
        
        # Add grid coordinates (CRITICAL for FNO!)
        grid = torch.linspace(0, 1, N, device=source.device, dtype=source.dtype)
        grid = grid.reshape(1, 1, N).repeat(B, 1, 1)
        
        # Concatenate: (B, 2, N) - [source, coordinate]
        x = torch.cat([source, grid], dim=1)
        
        # Forward through FNO
        out = self.fno(x)  # (B, N)
        
        # Reshape to match expected output format
        return out.unsqueeze(1)  # (B, 1, N)
