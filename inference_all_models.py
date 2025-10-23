"""
Inference script for all lightweight Poisson models

For each trained model:
1. Load the best checkpoint
2. Generate 5 random test samples
3. Create comprehensive plots showing:
   - Input signal (source term f(x))
   - Analytical solution (ground truth)
   - Model predicted solution
   - Error: |prediction - analytical|
   - Relative error

Generates comparison plots for all 4 models.
"""

import argparse
import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import datasets
import models


def load_model_checkpoint(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model spec
    model_spec = checkpoint['model']
    
    # Create model
    model = models.make(model_spec)
    model = model.to(device)
    
    # Convert to double precision (models were trained in float64)
    model = model.double()
    
    # Load weights (remove 'sd' key from state dict)
    if 'sd' in model_spec:
        model.load_state_dict(model_spec['sd'])
    
    model.eval()
    return model


def generate_test_samples(num_samples=5, resolution=256, seed=42, normalize_solution=False):
    """Generate test samples with known analytical solutions.

    If normalize_solution is True, scale each analytical solution to [-1, 1]
    per-sample for consistency with training normalization.
    """
    np.random.seed(seed)
    samples = []
    
    x = np.linspace(0, 1, resolution, dtype=np.float64)
    
    for i in range(num_samples):
        # Generate random frequency components (3-5 components)
        num_freqs = np.random.randint(3, 6)
        frequencies = np.random.uniform(1, 20, size=num_freqs)
        amplitudes = np.random.uniform(0.3, 1.0, size=num_freqs)
        amplitudes = amplitudes / np.sum(amplitudes)  # Normalize
        
        # Build source and solution
        source = np.zeros(resolution, dtype=np.float64)
        solution = np.zeros(resolution, dtype=np.float64)
        
        for k, amp in zip(frequencies, amplitudes):
            source += amp * np.sin(2 * np.pi * k * x)
            solution += amp * np.sin(2 * np.pi * k * x) / (2 * np.pi * k) ** 2
        
        if normalize_solution:
            s_min = solution.min()
            s_max = solution.max()
            if s_max > s_min:
                solution = 2.0 * (solution - s_min) / (s_max - s_min) - 1.0
            else:
                solution = np.zeros_like(solution)

        samples.append({
            'x': x,
            'source': source,
            'solution': solution,
            'frequencies': frequencies,
            'amplitudes': amplitudes
        })
    
    return samples


def predict_with_model(model, source, device='cuda'):
    """Run inference with a model"""
    # Convert to tensor
    source_t = torch.from_numpy(source).reshape(1, 1, -1).double().to(device)
    
    with torch.no_grad():
        prediction = model(source_t)
    
    # Convert back to numpy
    prediction_np = prediction.cpu().numpy().squeeze()
    
    return prediction_np


def plot_single_model_results(samples, predictions, model_name, save_path):
    """Create comprehensive plot for a single model across 5 samples"""
    num_samples = len(samples)
    
    fig = plt.figure(figsize=(20, 4 * num_samples))
    gs = GridSpec(num_samples, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    for i, (sample, pred) in enumerate(zip(samples, predictions)):
        x = sample['x']
        source = sample['source']
        solution = sample['solution']
        error = pred - solution
        rel_error = np.abs(error) / (np.abs(solution) + 1e-10)
        
        # Column 1: Input signal (source)
        ax1 = fig.add_subplot(gs[i, 0])
        ax1.plot(x, source, 'b-', linewidth=2)
        ax1.set_xlabel('x')
        ax1.set_ylabel('f(x)')
        ax1.set_title(f'Sample {i+1}: Input Signal (Source)')
        ax1.grid(True, alpha=0.3)
        
        # Add frequency info
        freqs_str = ', '.join([f'{f:.1f}' for f in sample['frequencies'][:3]])
        if len(sample['frequencies']) > 3:
            freqs_str += ', ...'
        ax1.text(0.02, 0.98, f'Freqs: {freqs_str}', 
                transform=ax1.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Column 2: Analytical vs Predicted
        ax2 = fig.add_subplot(gs[i, 1])
        ax2.plot(x, solution, 'g-', linewidth=2, label='Analytical', alpha=0.7)
        ax2.plot(x, pred, 'r--', linewidth=2, label='Predicted', alpha=0.7)
        ax2.set_xlabel('x')
        ax2.set_ylabel('u(x)')
        ax2.set_title(f'Sample {i+1}: Solution Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add MSE info
        mse = np.mean(error ** 2)
        ax2.text(0.02, 0.02, f'MSE: {mse:.2e}', 
                transform=ax2.transAxes, fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Column 3: Absolute Error
        ax3 = fig.add_subplot(gs[i, 2])
        ax3.plot(x, np.abs(error), 'r-', linewidth=2)
        ax3.set_xlabel('x')
        ax3.set_ylabel('|Error|')
        ax3.set_title(f'Sample {i+1}: Absolute Error')
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
        
        # Add max error info
        max_err = np.max(np.abs(error))
        ax3.text(0.02, 0.98, f'Max: {max_err:.2e}', 
                transform=ax3.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='salmon', alpha=0.8))
        
        # Column 4: Relative Error
        ax4 = fig.add_subplot(gs[i, 3])
        ax4.plot(x, rel_error, 'm-', linewidth=2)
        ax4.set_xlabel('x')
        ax4.set_ylabel('Relative Error')
        ax4.set_title(f'Sample {i+1}: Relative Error')
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
        
        # Add mean relative error
        mean_rel_err = np.mean(rel_error)
        ax4.text(0.02, 0.98, f'Mean: {mean_rel_err:.2e}', 
                transform=ax4.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='plum', alpha=0.8))
    
    plt.suptitle(f'{model_name} - Inference Results on 5 Test Samples', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot: {save_path}")


def plot_all_models_comparison(samples, all_predictions, model_names, save_path):
    """Create comparison plot showing all models on the same samples"""
    num_samples = len(samples)
    num_models = len(model_names)
    
    fig = plt.figure(figsize=(24, 4 * num_samples))
    gs = GridSpec(num_samples, 6, figure=fig, hspace=0.3, wspace=0.4)
    
    for i, sample in enumerate(samples):
        x = sample['x']
        source = sample['source']
        solution = sample['solution']
        
        # Column 1: Input signal
        ax1 = fig.add_subplot(gs[i, 0])
        ax1.plot(x, source, 'b-', linewidth=2)
        ax1.set_xlabel('x')
        ax1.set_ylabel('f(x)')
        ax1.set_title(f'Sample {i+1}: Source')
        ax1.grid(True, alpha=0.3)
        
        # Column 2: Analytical solution
        ax2 = fig.add_subplot(gs[i, 1])
        ax2.plot(x, solution, 'g-', linewidth=2)
        ax2.set_xlabel('x')
        ax2.set_ylabel('u(x)')
        ax2.set_title(f'Sample {i+1}: Analytical')
        ax2.grid(True, alpha=0.3)
        
        # Column 3: All model predictions
        ax3 = fig.add_subplot(gs[i, 2])
        ax3.plot(x, solution, 'g-', linewidth=3, label='Analytical', alpha=0.5)
        
        colors = ['r', 'b', 'm', 'c']
        linestyles = ['-', '--', '-.', ':']
        for j, (name, preds) in enumerate(zip(model_names, all_predictions)):
            pred = preds[i]
            ax3.plot(x, pred, color=colors[j % len(colors)], 
                    linestyle=linestyles[j % len(linestyles)],
                    linewidth=2, label=name, alpha=0.7)
        
        ax3.set_xlabel('x')
        ax3.set_ylabel('u(x)')
        ax3.set_title(f'Sample {i+1}: All Predictions')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # Columns 4-6: Error comparison for each model
        for col_offset, error_type in enumerate(['Absolute', 'Relative', 'MSE']):
            ax = fig.add_subplot(gs[i, 3 + col_offset])
            
            if error_type == 'MSE':
                # Bar plot of MSE for all models
                mses = []
                for preds in all_predictions:
                    pred = preds[i]
                    mse = np.mean((pred - solution) ** 2)
                    mses.append(mse)
                
                bars = ax.bar(range(len(model_names)), mses, 
                             color=['r', 'b', 'm', 'c'][:len(model_names)], alpha=0.7)
                ax.set_xticks(range(len(model_names)))
                ax.set_xticklabels([name.split()[0] for name in model_names], 
                                  rotation=45, ha='right', fontsize=8)
                ax.set_ylabel('MSE')
                ax.set_title(f'Sample {i+1}: MSE Comparison')
                ax.set_yscale('log')
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add values on bars
                for bar, mse in zip(bars, mses):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{mse:.2e}', ha='center', va='bottom', fontsize=7)
            
            elif error_type == 'Absolute':
                for j, (name, preds) in enumerate(zip(model_names, all_predictions)):
                    pred = preds[i]
                    error = np.abs(pred - solution)
                    ax.plot(x, error, color=colors[j % len(colors)], 
                           linestyle=linestyles[j % len(linestyles)],
                           linewidth=1.5, label=name, alpha=0.7)
                
                ax.set_xlabel('x')
                ax.set_ylabel('|Error|')
                ax.set_title(f'Sample {i+1}: Absolute Error')
                ax.set_yscale('log')
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)
            
            else:  # Relative
                for j, (name, preds) in enumerate(zip(model_names, all_predictions)):
                    pred = preds[i]
                    rel_error = np.abs(pred - solution) / (np.abs(solution) + 1e-10)
                    ax.plot(x, rel_error, color=colors[j % len(colors)], 
                           linestyle=linestyles[j % len(linestyles)],
                           linewidth=1.5, label=name, alpha=0.7)
                
                ax.set_xlabel('x')
                ax.set_ylabel('Relative Error')
                ax.set_title(f'Sample {i+1}: Relative Error')
                ax.set_yscale('log')
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)
    
    plt.suptitle('All Models Comparison - 5 Test Samples', 
                 fontsize=18, fontweight='bold', y=0.998)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comparison plot: {save_path}")


def compute_model_statistics(samples, predictions, model_name):
    """Compute overall statistics for a model"""
    mses = []
    maes = []
    max_errors = []
    rel_l2s = []
    
    for sample, pred in zip(samples, predictions):
        solution = sample['solution']
        error = pred - solution
        
        mse = np.mean(error ** 2)
        mae = np.mean(np.abs(error))
        max_err = np.max(np.abs(error))
        
        l2_error = np.sqrt(np.sum(error ** 2))
        l2_norm = np.sqrt(np.sum(solution ** 2))
        rel_l2 = l2_error / (l2_norm + 1e-10)
        
        mses.append(mse)
        maes.append(mae)
        max_errors.append(max_err)
        rel_l2s.append(rel_l2)
    
    stats = {
        'model': model_name,
        'mean_mse': np.mean(mses),
        'std_mse': np.std(mses),
        'mean_mae': np.mean(maes),
        'std_mae': np.std(maes),
        'mean_max_error': np.mean(max_errors),
        'std_max_error': np.std(max_errors),
        'mean_rel_l2': np.mean(rel_l2s),
        'std_rel_l2': np.std(rel_l2s)
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Inference for all lightweight Poisson models')
    parser.add_argument('--save_dir', default='save', help='Directory with trained models')
    parser.add_argument('--output_dir', default='inference_results', help='Output directory for plots')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of test samples')
    parser.add_argument('--seed', type=int, default=12345, help='Random seed for test samples')
    parser.add_argument('--device', default='cuda', help='Device to use')
    parser.add_argument('--normalize_solution', action='store_true', help='Scale analytical solutions to [-1,1]')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Model configurations
    model_configs = [
        {'name': 'poisson1d_dense', 'display_name': 'Dense Net (MLP)'},
        {'name': 'poisson1d_micro_cnn', 'display_name': 'Micro CNN'},
        {'name': 'poisson1d_nano_unet', 'display_name': 'Nano U-Net'},
        {'name': 'poisson1d_nano_unet_deep', 'display_name': 'Nano U-Net Deep'},
    ]
    
    # Generate test samples
    print(f"Generating {args.num_samples} test samples...")
    samples = generate_test_samples(num_samples=args.num_samples, seed=args.seed, normalize_solution=args.normalize_solution)
    print(f"Generated {len(samples)} samples")
    
    # Process each model
    all_predictions = []
    all_stats = []
    model_names = []
    
    for config in model_configs:
        model_name = config['name']
        display_name = config['display_name']
        checkpoint_path = os.path.join(args.save_dir, model_name, 'epoch-best.pth')
        
        if not os.path.exists(checkpoint_path):
            print(f"WARNING: Checkpoint not found for {display_name}: {checkpoint_path}")
            print(f"Skipping {display_name}...")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing: {display_name}")
        print(f"{'='*60}")
        
        # Load model
        print(f"Loading model from: {checkpoint_path}")
        model = load_model_checkpoint(checkpoint_path, device=args.device)
        print(f"Model loaded successfully")
        
        # Run inference on all samples
        predictions = []
        for i, sample in enumerate(samples):
            pred = predict_with_model(model, sample['source'], device=args.device)
            predictions.append(pred)
            mse = np.mean((pred - sample['solution']) ** 2)
            print(f"  Sample {i+1}: MSE = {mse:.4e}")
        
        # Store predictions
        all_predictions.append(predictions)
        model_names.append(display_name)
        
        # Compute statistics
        stats = compute_model_statistics(samples, predictions, display_name)
        all_stats.append(stats)
        
        print(f"\nStatistics for {display_name}:")
        print(f"  Mean MSE: {stats['mean_mse']:.4e} ± {stats['std_mse']:.4e}")
        print(f"  Mean MAE: {stats['mean_mae']:.4e} ± {stats['std_mae']:.4e}")
        print(f"  Mean Max Error: {stats['mean_max_error']:.4e} ± {stats['std_max_error']:.4e}")
        print(f"  Mean Rel L2: {stats['mean_rel_l2']:.4e} ± {stats['std_rel_l2']:.4e}")
        
        # Create individual model plot
        plot_path = os.path.join(args.output_dir, f'{model_name}_inference.png')
        plot_single_model_results(samples, predictions, display_name, plot_path)
    
    # Create comparison plot if we have multiple models
    if len(all_predictions) > 1:
        print(f"\n{'='*60}")
        print("Creating comparison plot for all models...")
        print(f"{'='*60}")
        comparison_path = os.path.join(args.output_dir, 'all_models_comparison.png')
        plot_all_models_comparison(samples, all_predictions, model_names, comparison_path)
    
    # Print summary table
    print(f"\n{'='*60}")
    print("SUMMARY TABLE")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'Mean MSE':<15} {'Mean MAE':<15} {'Mean Rel L2':<15}")
    print("-" * 65)
    for stats in all_stats:
        print(f"{stats['model']:<20} {stats['mean_mse']:<15.4e} {stats['mean_mae']:<15.4e} {stats['mean_rel_l2']:<15.4e}")
    
    print(f"\n{'='*60}")
    print(f"Inference complete! Results saved in: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

