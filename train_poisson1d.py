"""
Training script for 1D Poisson models

Supports various architectures:
- Dense Net (MLP): ~5-10K params
- Micro CNN: ~10-15K params
- Nano U-Net: ~20-30K params
- Better CNN: ~80-100K params (with residual connections)
- UNet Medium: ~100-150K params (3-level U-Net)
- FNO-1D: ~50-80K params (Fourier Neural Operator)
- FNO-1D Advanced: ~120-150K params (enhanced FNO)

Supports configurable loss functions:
- 'mse': Standard MSE (L2 norm) loss [default]
- 'relative_mse': Relative MSE loss (scale-invariant)

Uses comprehensive metrics tracking: MSE, MAE, L_inf, Relative L2.
"""

import argparse
import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import datasets
import models
import utils


def l2_loss(pred, target):
    """
    Compute L2 loss: sqrt(sum((pred - target)^2))

    Args:
        pred: (B, 1, N) predicted solution
        target: (B, 1, N) ground truth solution

    Returns:
        scalar loss value
    """
    return torch.sqrt(torch.sum((pred - target) ** 2))

def relative_mse_loss(pred, target, eps=1e-10):
    """
    Compute relative MSE loss: mean(((pred - target) / |target|)^2)

    This loss is scale-invariant and focuses on relative errors,
    which is useful when solution magnitudes vary across samples.

    Args:
        pred: (B, 1, N) predicted solution
        target: (B, 1, N) ground truth solution
        eps: small constant to avoid division by zero

    Returns:
        scalar loss value
    """
    relative_error = (pred - target) / (torch.abs(target) + eps)
    return torch.mean(relative_error ** 2)


def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])

    log('{} dataset: size={}'.format(tag, len(dataset)))
    sample = dataset[0]
    for k, v in sample.items():
        log('  {}: shape={}'.format(k, tuple(v.shape)))

    loader = DataLoader(
        dataset,
        batch_size=spec['batch_size'],
        shuffle=(tag == 'train'),
        num_workers=8,
        pin_memory=True
    )
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader


def prepare_training():
    model = models.make(config['model']).cuda()

    # Convert model to double precision if specified
    if config.get('use_double_precision', False):
        model = model.double()
        log('Model converted to float64 (double precision)')

    optimizer = utils.make_optimizer(model.parameters(), config['optimizer'])
    epoch_start = 1

    # Learning rate scheduler
    lr_scheduler = None
    if config.get('reduce_lr_on_plateau') is not None:
        lr_scheduler = ReduceLROnPlateau(optimizer, **config['reduce_lr_on_plateau'])

    log('model: #params={}'.format(utils.compute_num_params(model, text=True)))

    # Log loss type
    loss_type = config.get('loss_type', 'mse')
    log('loss function: {}'.format(loss_type))

    return model, optimizer, epoch_start, lr_scheduler


def compute_metrics(pred, target):
    """
    Compute comprehensive error metrics

    Args:
        pred: (B, 1, N) predicted solution
        target: (B, 1, N) ground truth solution

    Returns:
        dict of metrics
    """
    # MSE
    mse = F.mse_loss(pred, target).item()

    # L2 loss (raw)
    l2_loss_val = torch.sqrt(torch.sum((pred - target) ** 2)).item()

    # Relative L2 error
    l2_norm = torch.sqrt(torch.sum(target ** 2))
    relative_l2 = (l2_loss_val / (l2_norm + 1e-10)).item()

    # Relative MSE
    relative_error = (pred - target) / (torch.abs(target) + 1e-10)
    relative_mse = torch.mean(relative_error ** 2).item()

    return {
        'mse': mse,
        'l2': l2_loss_val,
        'relative_l2': relative_l2,
        'relative_mse': relative_mse
    }


def train(train_loader, model, optimizer):
    model.train()

    # Use double precision for loss computation if specified
    use_double = config.get('use_double_precision', False)

    metrics_acc = {
        'mse': utils.Averager(),
        'l2': utils.Averager(),
        'relative_l2': utils.Averager(),
        'relative_mse': utils.Averager()
    }

    for batch in tqdm(train_loader, leave=False, desc='train'):
        for k, v in batch.items():
            batch[k] = v.cuda()

        source = batch['source']  # (B, 1, N)
        target = batch['solution']  # (B, 1, N)

        # Forward pass
        pred = model(source)

        # Compute loss (support both MSE and relative MSE)
        loss_type = config.get('loss_type', 'l2')
        if use_double:
            pred_loss = pred.double()
            target_loss = target.double()
        else:
            pred_loss = pred
            target_loss = target

        if loss_type == 'relative_mse':
            loss = relative_mse_loss(pred_loss, target_loss)
        elif loss_type == 'l2':
            loss = l2_loss(pred_loss, target_loss)
        else:  # default: 'mse'
            loss = F.mse_loss(pred_loss, target_loss)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Optional gradient clipping
        if config.get('grad_clip') is not None:
            nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])

        optimizer.step()

        # Compute metrics (on float32 for consistency)
        with torch.no_grad():
            metrics = compute_metrics(pred.float(), target.float())
            for k, v in metrics.items():
                metrics_acc[k].add(v)

    return {k: v.item() for k, v in metrics_acc.items()}


def evaluate(val_loader, model):
    """Evaluate model on validation set"""
    model.eval()
    use_double = config.get('use_double_precision', False)

    metrics_acc = {
        'mse': utils.Averager(),
        'l2': utils.Averager(),
        'relative_l2': utils.Averager(),
        'relative_mse': utils.Averager()
    }

    with torch.no_grad():
        for batch in tqdm(val_loader, leave=False, desc='val'):
            for k, v in batch.items():
                batch[k] = v.cuda()

            source = batch['source']
            target = batch['solution']

            pred = model(source)

            # Compute metrics
            metrics = compute_metrics(pred.float(), target.float())
            for k, v in metrics.items():
                metrics_acc[k].add(v)

    return {k: v.item() for k, v in metrics_acc.items()}


def plot_metrics(metrics_history, save_path):
    """Plot training metrics with log scale"""
    epochs = list(range(1, len(metrics_history['train_mse']) + 1))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # MSE (log scale)
    ax = axes[0, 0]
    ax.plot(epochs, metrics_history['train_mse'], 'b-', linewidth=2, label='Train')
    if metrics_history['val_mse']:
        val_epochs = [e for e in epochs if e <= len(metrics_history['val_mse'])]
        ax.plot(val_epochs, metrics_history['val_mse'][:len(val_epochs)], 'r-', linewidth=2, label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE')
    ax.set_title('Mean Squared Error (log scale)')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # L2 Loss (log scale)
    ax = axes[0, 1]
    ax.plot(epochs, metrics_history['train_l2'], 'b-', linewidth=2, label='Train')
    if metrics_history['val_l2']:
        val_epochs = [e for e in epochs if e <= len(metrics_history['val_l2'])]
        ax.plot(val_epochs, metrics_history['val_l2'][:len(val_epochs)], 'r-', linewidth=2, label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('L2')
    ax.set_title('L2 Loss (log scale)')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Relative L2 error (log scale)
    ax = axes[1, 0]
    ax.plot(epochs, metrics_history['train_relative_l2'], 'b-', linewidth=2, label='Train')
    if metrics_history['val_relative_l2']:
        val_epochs = [e for e in epochs if e <= len(metrics_history['val_relative_l2'])]
        ax.plot(val_epochs, metrics_history['val_relative_l2'][:len(val_epochs)], 'r-', linewidth=2, label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Relative L2 Error')
    ax.set_title('Relative L2 Error (log scale)')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Learning rate (log scale)
    ax = axes[1, 1]
    ax.plot(epochs, metrics_history['lr'], 'g-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate (log scale)')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    plt.suptitle('1D Poisson Training Progress', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Save metrics history
    np.save(os.path.join(save_path, 'metrics_history.npy'), metrics_history)


def main(config_, save_path):
    global config, log, writer
    config = config_
    log, writer = utils.set_save_path(save_path, remove=False)

    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, val_loader = make_data_loaders()
    model, optimizer, epoch_start, lr_scheduler = prepare_training()

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val', 1)

    # Early stopping
    patience = config.get('early_stopping_patience', 50)
    best_val_mse = float('inf')
    best_epoch = 0
    patience_counter = 0

    # Metrics history
    metrics_history = {
        'train_mse': [],
        'train_l2': [],
        'train_relative_l2': [],
        'train_relative_mse': [],
        'val_mse': [],
        'val_l2': [],
        'val_relative_l2': [],
        'val_relative_mse': [],
        'lr': [],
        'best_val_mse': best_val_mse,
        'best_epoch': best_epoch
    }

    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

        # Current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        metrics_history['lr'].append(current_lr)
        writer.add_scalar('lr', current_lr, epoch)

        # Training
        train_metrics = train(train_loader, model, optimizer)

        for k, v in train_metrics.items():
            metrics_history['train_{}'.format(k)].append(v)

        log_info.append('train: mse={:.2e}, l2={:.2e}, rel_l2={:.2e}, rel_mse={:.2e}'.format(
            train_metrics['mse'],
            train_metrics['l2'],
            train_metrics['relative_l2'],
            train_metrics['relative_mse']
        ))

        writer.add_scalars('mse', {'train': train_metrics['mse']}, epoch)

        # Prepare checkpoint
        model_spec = config['model']
        model_spec['sd'] = model.state_dict()
        optimizer_spec = config['optimizer']
        optimizer_spec['sd'] = optimizer.state_dict()
        sv_file = {
            'model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': epoch
        }

        # Save last checkpoint
        torch.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))

        # Validation
        if (epoch_val is not None) and (epoch % epoch_val == 0):
            val_metrics = evaluate(val_loader, model)

            for k, v in val_metrics.items():
                metrics_history['val_{}'.format(k)].append(v)

            log_info.append('val: mse={:.2e}, l2={:.2e}, rel_l2={:.2e}, rel_mse={:.2e}'.format(
                val_metrics['mse'],
                val_metrics['l2'],
                val_metrics['relative_l2'],
                val_metrics['relative_mse']
            ))

            writer.add_scalars('mse', {'train': train_metrics['mse'], 'val': val_metrics['mse']}, epoch)

            # Check for improvement
            if val_metrics['mse'] < best_val_mse:
                best_val_mse = val_metrics['mse']
                best_epoch = epoch
                patience_counter = 0
                metrics_history['best_val_mse'] = best_val_mse
                metrics_history['best_epoch'] = best_epoch
                torch.save(sv_file, os.path.join(save_path, 'epoch-best.pth'))
                log_info.append('(best)')
            else:
                patience_counter += 1

            # LR scheduler step
            if lr_scheduler is not None:
                old_lr = optimizer.param_groups[0]['lr']
                lr_scheduler.step(val_metrics['mse'])
                new_lr = optimizer.param_groups[0]['lr']
                if new_lr < old_lr:
                    log('Learning rate reduced from {:.2e} to {:.2e}'.format(old_lr, new_lr))

            # Early stopping
            if patience_counter >= patience:
                log('Early stopping triggered! No improvement for {} epochs.'.format(patience))
                log('Best Val MSE: {:.2e} at epoch {}'.format(best_val_mse, best_epoch))
                break

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

        log(', '.join(log_info))
        writer.flush()

        # Plot metrics periodically
        if epoch % 10 == 0 or epoch == epoch_max:
            plot_metrics(metrics_history, save_path)

    # Final plot
    plot_metrics(metrics_history, save_path)
    log('Training completed! Best Val MSE: {:.2e} at epoch {}'.format(best_val_mse, best_epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--name', default=None)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    save_path = os.path.join('./save', save_name)

    main(config, save_path)

