"""
Count parameters for all lightweight models

Verifies that models meet target parameter counts:
- Dense Net (MLP): ~5-10K
- Micro CNN: ~10-15K
- Nano U-Net: ~20-30K
- Nano U-Net Deep: ~35-45K
"""

import torch
import numpy as np
import models


def count_parameters(model):
    """Count total parameters in model"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def format_params(n):
    """Format parameter count for display"""
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.1f}K"
    else:
        return f"{n}"


def test_model(name, model_spec):
    """Create model and count parameters"""
    print(f"\n{name}:")
    print(f"  Config: {model_spec}")
    
    model = models.make(model_spec)
    total, trainable = count_parameters(model)
    
    print(f"  Total params:      {format_params(total):>10} ({total:,})")
    print(f"  Trainable params:  {format_params(trainable):>10} ({trainable:,})")
    
    return total


def main():
    print("=" * 70)
    print("Ultra-Lightweight Model Parameter Counts")
    print("=" * 70)
    
    results = {}
    
    # 1. Dense Net (MLP)
    total = test_model(
        "1. Dense Net (MLP)",
        {
            'name': 'mlp-dense',
            'args': {
                'input_size': 256,
                'hidden_dims': [128, 256, 512, 256],
                'activation': 'relu'
            }
        }
    )
    results['Dense Net'] = total
    print(f"  Target: 5-10K ✓" if 5000 <= total <= 10000 else f"  Target: 5-10K ✗")
    
    # 2. Micro CNN
    total = test_model(
        "2. Micro CNN",
        {
            'name': 'cnn-micro',
            'args': {
                'hidden_channels': [8, 16, 32],
                'kernel_size': 5,
                'padding_mode': 'reflect',
                'activation': 'relu'
            }
        }
    )
    results['Micro CNN'] = total
    print(f"  Target: 10-15K ✓" if 10000 <= total <= 15000 else f"  Target: 10-15K ✗")
    
    # 3. Nano U-Net
    total = test_model(
        "3. Nano U-Net",
        {
            'name': 'unet-nano',
            'args': {
                'base_channels': 12,
                'kernel_size': 5,
                'padding_mode': 'reflect',
                'activation': 'relu'
            }
        }
    )
    results['Nano U-Net'] = total
    print(f"  Target: 20-30K ✓" if 20000 <= total <= 30000 else f"  Target: 20-30K ✗")
    
    # 4. Nano U-Net Deep
    total = test_model(
        "4. Nano U-Net Deep",
        {
            'name': 'unet-nano-deep',
            'args': {
                'base_channels': 10,
                'kernel_size': 5,
                'padding_mode': 'reflect',
                'activation': 'relu'
            }
        }
    )
    results['Nano U-Net Deep'] = total
    print(f"  Target: 35-45K ✓" if 35000 <= total <= 45000 else f"  Target: 35-45K ✗")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'Model':<20} {'Parameters':<15} {'Target':<15} {'Status'}")
    print("-" * 70)
    
    targets = {
        'Dense Net': (5000, 10000, "5-10K"),
        'Micro CNN': (10000, 15000, "10-15K"),
        'Nano U-Net': (20000, 30000, "20-30K"),
        'Nano U-Net Deep': (35000, 45000, "35-45K")
    }
    
    for name, total in results.items():
        min_p, max_p, target_str = targets[name]
        status = "✓" if min_p <= total <= max_p else "✗"
        print(f"{name:<20} {format_params(total):<15} {target_str:<15} {status}")
    
    print("=" * 70)
    print(f"Total parameters across all models: {format_params(sum(results.values()))}")
    print("=" * 70)


if __name__ == '__main__':
    main()


