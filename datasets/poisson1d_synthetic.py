"""
On-the-fly synthetic dataset for 1D Poisson problems

Generates analytical source-solution pairs dynamically during training.
No disk storage required!

For the 1D Poisson equation: -d²u/dx² = f(x) with u(0) = u(1) = 0

We use source functions: f(x) = sin(2πkx) where k is a frequency
Analytical solution: u(x) = sin(2πkx) / (2πk)²
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets import register


@register('poisson1d-synthetic')
class Poisson1DSynthetic(Dataset):
    """
    Generate analytical 1D Poisson solutions on-the-fly

    Args:
        num_samples: Total number of samples per epoch
        resolution: Number of spatial points (N)
        freq_range: (min_freq, max_freq) for random frequency selection
        use_superposition: If True, use sum of multiple frequencies
        num_components: Number of frequency components (if use_superposition=True)
        use_float64: Generate in float64 precision
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        num_samples=10000,
        resolution=256,
        freq_range=(1, 20),
        use_superposition=True,
        num_components=3,
        use_float64=False,
        seed=None,
        normalize_solution=False
    ):
        self.num_samples = num_samples
        self.resolution = resolution
        self.freq_range = freq_range
        self.use_superposition = use_superposition
        self.num_components = num_components
        self.use_float64 = use_float64
        self.seed = seed
        self.normalize_solution = normalize_solution

        # Set dtype
        self.dtype = np.float64 if use_float64 else np.float32

        # Create spatial grid
        self.x = np.linspace(0, 1, resolution, dtype=self.dtype)

        # Random number generator (for reproducibility)
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return self.num_samples

    def analytical_solution_1d_poisson_sine(self, x, k):
        """
        Exact analytical solution for -d²u/dx² = sin(2πkx) with u(0)=u(1)=0

        Solution: u(x) = sin(2πkx) / (2πk)²

        This is the EXACT solution - no numerical approximation!
        The network will learn to match this exact solution.
        """
        return np.sin(2 * np.pi * k * x) / (2 * np.pi * k) ** 2

    def generate_sample(self):
        """
        Generate a random source-solution pair

        Returns:
            source: (1, N) array - f(x)
            solution: (1, N) array - u(x)
        """
        x = self.x
        N = self.resolution

        if self.use_superposition:
            # Generate random superposition of frequencies
            num_freqs = self.rng.randint(1, self.num_components + 1)

            # Random frequencies
            frequencies = self.rng.uniform(
                self.freq_range[0],
                self.freq_range[1],
                size=num_freqs
            )

            # Random amplitudes (normalized to keep solution bounded)
            amplitudes = self.rng.uniform(0.3, 1.0, size=num_freqs)
            amplitudes = amplitudes / np.sum(amplitudes)  # Normalize

            # Build superposition
            source = np.zeros(N, dtype=self.dtype)
            solution = np.zeros(N, dtype=self.dtype)

            for k, amp in zip(frequencies, amplitudes):
                source += amp * np.sin(2 * np.pi * k * x)
                solution += amp * self.analytical_solution_1d_poisson_sine(x, k)

        else:
            # Single frequency
            k = self.rng.uniform(self.freq_range[0], self.freq_range[1])
            source = np.sin(2 * np.pi * k * x)
            solution = self.analytical_solution_1d_poisson_sine(x, k)

        # Optionally scale analytical solution to [-1, 1] per-sample
        if self.normalize_solution:
            s_min = solution.min()
            s_max = solution.max()
            if s_max > s_min:
                # Scale to [-1, 1]
                solution = 2.0 * (solution - s_min) / (s_max - s_min) - 1.0
            else:
                solution = np.zeros_like(solution)

        # Reshape to (1, N)
        source = source.reshape(1, -1)
        solution = solution.reshape(1, -1)

        return source, solution

    def __getitem__(self, idx):
        """
        Generate a sample on-the-fly

        Note: idx is used only to seed the random generation for this sample
        """
        # Use idx to seed this specific sample (for reproducibility during validation)
        sample_seed = (self.seed if self.seed is not None else 0) + idx
        local_rng = np.random.RandomState(sample_seed)
        self.rng = local_rng

        # Generate sample
        source, solution = self.generate_sample()

        # Convert to torch tensors
        if self.use_float64:
            source_t = torch.from_numpy(source).double()
            solution_t = torch.from_numpy(solution).double()
        else:
            source_t = torch.from_numpy(source).float()
            solution_t = torch.from_numpy(solution).float()

        return {
            'source': source_t,
            'solution': solution_t
        }


@register('poisson1d-synthetic-simple')
class Poisson1DSyntheticSimple(Dataset):
    """
    Simpler version: single frequency per sample

    Useful for debugging or testing specific frequency ranges
    """

    def __init__(
        self,
        num_samples=10000,
        resolution=256,
        freq_range=(1, 20),
        use_float64=False,
        seed=None,
        normalize_solution=False
    ):
        self.num_samples = num_samples
        self.resolution = resolution
        self.freq_range = freq_range
        self.use_float64 = use_float64
        self.seed = seed
        self.normalize_solution = normalize_solution

        self.dtype = np.float64 if use_float64 else np.float32
        self.x = np.linspace(0, 1, resolution, dtype=self.dtype)
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return self.num_samples

    def analytical_solution_1d_poisson_sine(self, x, k):
        return np.sin(2 * np.pi * k * x) / (2 * np.pi * k) ** 2

    def __getitem__(self, idx):
        # Reproducible sample generation
        sample_seed = (self.seed if self.seed is not None else 0) + idx
        local_rng = np.random.RandomState(sample_seed)

        # Single frequency
        k = local_rng.uniform(self.freq_range[0], self.freq_range[1])

        x = self.x
        source = np.sin(2 * np.pi * k * x).reshape(1, -1)
        solution = self.analytical_solution_1d_poisson_sine(x, k)

        if self.normalize_solution:
            s_min = solution.min()
            s_max = solution.max()
            if s_max > s_min:
                solution = 2.0 * (solution - s_min) / (s_max - s_min) - 1.0
            else:
                solution = np.zeros_like(solution)

        solution = solution.reshape(1, -1)

        # Convert to torch
        if self.use_float64:
            source_t = torch.from_numpy(source).double()
            solution_t = torch.from_numpy(solution).double()
        else:
            source_t = torch.from_numpy(source).float()
            solution_t = torch.from_numpy(solution).float()

        return {
            'source': source_t,
            'solution': solution_t
        }


