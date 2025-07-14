#!/usr/bin/env python
# Optimized training script for GPU/CPU instances with ResNet18

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import os
import logging
import sys
import time
import argparse

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

# Import debugging libraries
try:
    import smdebug.pytorch as smd
    SMDEBUG_AVAILABLE = True
except ImportError:
    SMDEBUG_AVAILABLE = False
    logger.warning("smdebug not available, debugging features will be disabled")


def test(model, test_loader, criterion, device, use_amp=False, hook=None):
    """Evaluation on test data with GPU optimization"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    # Use more batches on GPU for better evaluation
    max_batches = 50 if device.type == 'cuda' else 10

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            if batch_idx >= max_batches:
                break

            data, target = data.to(device), target.to(device)
            
            # Use mixed precision on GPU
            if use_amp and device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    output = model(data)
                    loss = criterion(output, target)
            else:
                output = model(data)
                loss = criterion(output, target)
                
            test_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)

    avg_loss = test_loss / total
    accuracy = 100. * correct / total

    # Use the exact format expected by the regex patterns
    logger.info(f'Test set: Average loss: {avg_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.0f}%)')

    return avg_loss, accuracy


def train(model, train_loader, valid_loader, criterion, optimizer, scheduler, args, device, hook=None):
    """Optimized training function for GPU/CPU"""
    epochs = args.epochs
    best_loss = float('inf')
    
    # Enable mixed precision training on GPU
    use_amp = device.type == 'cuda' and args.use_amp
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    # Different batch limits for GPU vs CPU
    max_batches_per_epoch = None if device.type == 'cuda' else min(100, len(train_loader))

    for epoch in range(1, epochs + 1):
        model.train()
        
        # Set the hook mode to TRAIN if available
        if hook:
            hook.set_mode(smd.modes.TRAIN)

        running_loss = 0.0
        correct = 0
        total = 0
        epoch_start_time = time.time()

        for batch_idx, (data, target) in enumerate(train_loader):
            if max_batches_per_epoch and batch_idx >= max_batches_per_epoch:
                break

            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

            # Use automatic mixed precision on GPU
            if use_amp:
                with torch.cuda.amp.autocast():
                    output = model(data)
                    loss = criterion(output, target)
                scaler.scale(loss).backward()
                
                # Record gradients if hook is available
                if hook:
                    hook.record_tensor_value(tensor_name="loss", tensor_value=loss)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                
                # Record gradients if hook is available
                if hook:
                    hook.record_tensor_value(tensor_name="loss", tensor_value=loss)
                
                optimizer.step()

            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            # Log progress
            log_interval = 10 if device.type == 'cuda' else 20
            if batch_idx % log_interval == 0:
                current = batch_idx * len(data)
                dataset_size = len(train_loader.dataset) if max_batches_per_epoch is None else max_batches_per_epoch * args.batch_size
                # Use exact format for loss logging
                logger.info(f'Train Epoch: {epoch} [{current}/{dataset_size} '
                           f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

        # Calculate epoch metrics
        epoch_loss = running_loss / (batch_idx + 1)
        epoch_acc = 100. * correct / total
        epoch_time = time.time() - epoch_start_time

        # Log epoch summary with the exact format expected by regex
        logger.info(f'Epoch {epoch}: Loss: {epoch_loss:.4f}, Accuracy: {correct}/{total} ({epoch_acc:.0f}%), Time: {epoch_time:.1f}s')

        # Step the learning rate scheduler
        scheduler.step()
        
        # Set hook mode to EVAL if available
        if hook:
            hook.set_mode(smd.modes.EVAL)
        
        # Validation after each epoch
        test_loss, test_acc = test(model, valid_loader, criterion, device, use_amp, hook)

        # Save best model
        if test_loss < best_loss:
            best_loss = test_loss
            logger.info(f"New best model found at epoch {epoch}")
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_loss,
            }, os.path.join(args.model_dir, 'checkpoint.pth'))

    return model


def net(dropout_rate=0.2, num_classes=133):
    """ResNet18 optimized for GPU/CPU training"""
    logger.info("Initializing ResNet18...")

    # Use pretrained weights from torchvision
    model = models.resnet18(pretrained=True)

    # Freeze early layers - unfreeze more layers on GPU for better fine-tuning
    if torch.cuda.is_available():
        # On GPU: only freeze first 3 blocks
        for name, param in model.named_parameters():
            if 'layer4' not in name and 'fc' not in name and 'layer3.1' not in name:
                param.requires_grad = False
    else:
        # On CPU: freeze more layers for faster training
        for name, param in model.named_parameters():
            if 'layer4' not in name and 'fc' not in name:
                param.requires_grad = False

    # Replace the last fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(num_features, num_classes)
    )

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

    return model


def get_optimizer(model, args):
    """Create optimizer with different defaults for GPU/CPU"""
    # Only optimize parameters that require gradients
    params_to_update = [p for p in model.parameters() if p.requires_grad]
    
    optimizer_name = args.optimizer.lower()
    
    # Adjust learning rate based on device
    lr = args.lr
    if torch.cuda.is_available() and args.auto_lr:
        lr = lr * 2  # Can use higher LR on GPU
    
    if optimizer_name == 'adam':
        logger.info(f"Using Adam optimizer with lr={lr}, weight_decay={args.weight_decay}")
        optimizer = optim.Adam(params_to_update, lr=lr, weight_decay=args.weight_decay)
    elif optimizer_name == 'sgd':
        logger.info(f"Using SGD optimizer with lr={lr}, momentum={args.momentum}, weight_decay={args.weight_decay}")
        optimizer = optim.SGD(params_to_update, lr=lr, momentum=args.momentum, 
                            weight_decay=args.weight_decay, nesterov=True)
    elif optimizer_name == 'adamw':
        logger.info(f"Using AdamW optimizer with lr={lr}, weight_decay={args.weight_decay}")
        optimizer = optim.AdamW(params_to_update, lr=lr, weight_decay=args.weight_decay)
    elif optimizer_name == 'rmsprop':
        logger.info(f"Using RMSprop optimizer with lr={lr}, momentum={args.momentum}, weight_decay={args.weight_decay}")
        optimizer = optim.RMSprop(params_to_update, lr=lr, momentum=args.momentum, 
                                weight_decay=args.weight_decay)
    else:
        logger.warning(f"Unknown optimizer '{args.optimizer}', defaulting to Adam")
        optimizer = optim.Adam(params_to_update, lr=lr, weight_decay=args.weight_decay)
    
    return optimizer


def create_data_loaders(data_dir, batch_size, data_pct, device):
    """Create data loaders optimized for GPU/CPU"""
    # Data augmentation - more aggressive on GPU
    if device.type == 'cuda':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # Simpler augmentation for CPU
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_dataset = torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train'),
        transform=train_transform
    )
    valid_dataset = torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'valid'),
        transform=test_transform
    )
    test_dataset = torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'test'),
        transform=test_transform
    )

    # Use subset based on data_pct parameter
    train_subset_size = int(len(train_dataset) * data_pct)
    train_indices = torch.randperm(len(train_dataset))[:train_subset_size]
    train_subset = Subset(train_dataset, train_indices)
    
    logger.info(f"Using {data_pct*100:.0f}% of training data ({train_subset_size} samples)")

    # Optimize number of workers based on device
    if device.type == 'cuda':
        num_workers = min(8, os.cpu_count() or 1)
        pin_memory = True
        persistent_workers = True if num_workers > 0 else False
    else:
        num_workers = min(4, os.cpu_count() or 1)
        pin_memory = False
        persistent_workers = True if num_workers > 0 else False

    # Create data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True  # For stable batch norm
    )

    # Larger batch size for validation/test (no gradients)
    eval_batch_size = batch_size * 2 if device.type == 'cuda' else batch_size

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=2 if num_workers > 0 else None
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=2 if num_workers > 0 else None
    )

    return train_loader, valid_loader, test_loader


def main(args):
    start_time = time.time()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Log configuration
    logger.info(f"Starting training on {device}")
    logger.info(f"Hyperparameters:")
    logger.info(f"  epochs: {args.epochs}")
    logger.info(f"  batch_size: {args.batch_size}")
    logger.info(f"  learning_rate: {args.lr}")
    logger.info(f"  dropout: {args.dropout}")
    logger.info(f"  momentum: {args.momentum}")
    logger.info(f"  optimizer: {args.optimizer}")
    logger.info(f"  weight_decay: {args.weight_decay}")
    logger.info(f"  data_pct: {args.data_pct}")
    logger.info(f"  use_amp: {args.use_amp and device.type == 'cuda'}")

    # Enable cudnn benchmarking for better performance on GPU
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Initialize the SMDebug hook if available
    hook = None
    if SMDEBUG_AVAILABLE:
        try:
            hook = smd.Hook.create_from_json_file()
            logger.info("SMDebug hook created successfully")
        except Exception as e:
            logger.warning(f"Failed to create SMDebug hook: {e}")
            hook = None

    # Initialize model
    model = net(dropout_rate=args.dropout).to(device)
    
    # Register model with hook if available
    if hook:
        hook.register_module(model)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, args)
    
    # Register loss and optimizer with hook if available
    if hook:
        hook.register_loss(criterion)
    
    # Learning rate scheduler - different for GPU/CPU
    if device.type == 'cuda':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    # Create data loaders
    train_loader, valid_loader, test_loader = create_data_loaders(
        args.data_dir,
        args.batch_size,
        args.data_pct,
        device
    )

    # Train model
    model = train(model, train_loader, valid_loader, criterion, optimizer, scheduler, args, device, hook)

    # Final evaluation on test set
    if hook:
        hook.set_mode(smd.modes.EVAL)
    
    test_loss, test_accuracy = test(model, test_loader, criterion, device, args.use_amp and device.type == 'cuda', hook)
    logger.info(f"Final Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    # Save model
    model_path = os.path.join(args.model_dir, 'model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'test_accuracy': test_accuracy,
        'test_loss': test_loss,
        'hyperparameters': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'dropout': args.dropout,
            'momentum': args.momentum,
            'optimizer': args.optimizer,
            'weight_decay': args.weight_decay,
            'data_pct': args.data_pct
        },
        'args': args
    }, model_path)

    # Save model for inference
    model_inference_path = os.path.join(args.model_dir, 'model.pt')
    # Move model to CPU before saving for better compatibility
    model.cpu()
    torch.save(model, model_inference_path)

    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time/60:.1f} minutes")
    logger.info(f"Model saved to {model_path}")


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    
    # Training hyperparameters
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    
    # Model hyperparameters
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout rate for final layer (default: 0.2)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum for SGD/RMSprop optimizer (default: 0.9)')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd', 'adamw', 'rmsprop'],
                        help='optimizer type (default: adam)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='weight decay (L2 penalty) (default: 1e-4)')
    
    # Data configuration
    parser.add_argument('--data_pct', type=float, default=1.0,
                        help='percentage of training data to use (default: 1.0)')
    
    # Training configuration
    parser.add_argument('--use-amp', action='store_true', default=True,
                        help='use automatic mixed precision on GPU (default: True)')
    parser.add_argument('--auto-lr', action='store_true', default=True,
                        help='automatically adjust learning rate for GPU (default: True)')
    
    # Environment and directory arguments
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))
    parser.add_argument('--data-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING', './data'))
    parser.add_argument('--num-gpus', type=int, default=os.environ.get('SM_NUM_GPUS', 0))
    parser.add_argument('--output-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)