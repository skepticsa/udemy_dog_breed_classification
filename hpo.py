#!/usr/bin/env python
# Optimized HPO script for faster training with ResNet18

import argparse
import logging
import os
import sys
import warnings
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from PIL import ImageFile
import numpy as np

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings('ignore', category=UserWarning)

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class FastImageFolder(torchvision.datasets.ImageFolder):
    """Optimized ImageFolder with caching and error handling"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = {}

    def __getitem__(self, index):
        if index in self.cache:
            return self.cache[index]

        path, target = self.samples[index]

        try:
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)

            # Cache the result
            self.cache[index] = (sample, target)
            return sample, target

        except Exception as e:
            logger.warning(f"Error loading image {path}: {str(e)}")
            # Return a black image of the correct size as fallback
            sample = torch.zeros((3, 224, 224))
            return sample, target


def test(model, test_loader, criterion, device):
    '''Fast evaluation on a subset of test data during HPO'''
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    # Only evaluate on first 10 batches for speed during HPO
    max_batches = min(10, len(test_loader))

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            if batch_idx >= max_batches:
                break

            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() * data.size(0)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    avg_loss = test_loss / total if total > 0 else float('inf')
    accuracy = 100. * correct / total if total > 0 else 0.0

    # Extrapolate to full dataset for consistent metrics
    full_dataset_loss = avg_loss

    # IMPORTANT: Log metrics in a consistent format for SageMaker to capture
    logger.info(f'Test set: Average loss: {full_dataset_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.0f}%)')

    # Also log in simpler formats to ensure SageMaker catches at least one
    logger.info(f'test_loss: {full_dataset_loss:.4f}')
    logger.info(f'Loss: {full_dataset_loss:.4f}')

    # Print to stdout as well (sometimes SageMaker captures stdout better)
    print(f'Test set: Average loss: {full_dataset_loss:.4f}')

    return full_dataset_loss


def train(model, train_loader, criterion, optimizer, epoch, device):
    '''Fast training for one epoch'''
    model.train()
    running_loss = 0.0
    batches_processed = 0

    # Train on subset of data for faster HPO
    max_batches = min(100, len(train_loader))  # Limit batches per epoch

    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx >= max_batches:
            break

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        batches_processed += 1

        if batch_idx % 20 == 0:
            logger.info(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{max_batches * train_loader.batch_size} '
                       f'({100. * batch_idx / max_batches:.0f}%)]\tLoss: {loss.item():.6f}')

    avg_loss = running_loss / batches_processed if batches_processed > 0 else float('inf')
    logger.info(f'Training Epoch: {epoch}, Average Loss: {avg_loss:.4f}')


def net(dropout_rate=0.2):
    '''Initialize ResNet18 with pretrained weights - consistent with train_model.py'''
    logger.info("Initializing ResNet18 model (faster than ResNet50)...")

    # Use ResNet18 for faster training
    model = models.resnet18(pretrained=True)

    # Freeze all layers except the last block and FC layer
    for name, param in model.named_parameters():
        if 'layer4' not in name and 'fc' not in name:
            param.requires_grad = False

    # Replace the last fully connected layer with configurable dropout
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(num_features, 133)
    )

    return model


def create_data_loaders(data_pct, data_dir, batch_size):
    '''Create data loaders with optimizations for speed'''

    # Simplified transforms for faster processing
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Direct resize, no cropping
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
    logger.info(f"Loading data from {data_dir}")

    train_path = os.path.join(data_dir, 'train')
    test_path = os.path.join(data_dir, 'test')

    train_dataset = FastImageFolder(train_path, transform=train_transform)
    test_dataset = FastImageFolder(test_path, transform=test_transform)

    # Use 30% of training data - consistent with train_model.py
    subset_pct = data_pct  # How much data to use for training (as a float in percentage)
    train_subset_size = int(len(train_dataset) * subset_pct)
    train_indices = np.random.choice(len(train_dataset), train_subset_size, replace=False)
    train_subset = Subset(train_dataset, train_indices)

    # Use subset of test data too
    test_subset_size = min(500, len(test_dataset))
    test_indices = np.random.choice(len(test_dataset), test_subset_size, replace=False)
    test_subset = Subset(test_dataset, test_indices)

    logger.info(f"Using subset for HPO - Train: {len(train_subset)} ({100*subset_pct} of data), Test: {len(test_subset)}")

    # Determine number of workers
    num_workers = min(4, os.cpu_count() or 1) if torch.cuda.is_available() else 0

    # Create data loaders with optimizations
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if num_workers > 0 else False
    )

    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size * 2,  # Larger batch for testing
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if num_workers > 0 else False
    )

    return train_loader, test_loader


def get_optimizer(model, args):
    '''Create optimizer based on the specified type with appropriate hyperparameters'''
    # Only optimize parameters that require gradients
    params_to_optimize = filter(lambda p: p.requires_grad, model.parameters())

    optimizer_name = args.optimizer.lower()

    if optimizer_name == 'adam':
        logger.info(f"Using Adam optimizer with lr={args.lr}, weight_decay={args.weight_decay}")
        optimizer = optim.Adam(params_to_optimize, lr=args.lr, weight_decay=args.weight_decay)
    elif optimizer_name == 'sgd':
        logger.info(f"Using SGD optimizer with lr={args.lr}, momentum={args.momentum}, weight_decay={args.weight_decay}")
        optimizer = optim.SGD(params_to_optimize, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif optimizer_name == 'adamw':
        logger.info(f"Using AdamW optimizer with lr={args.lr}, weight_decay={args.weight_decay}")
        optimizer = optim.AdamW(params_to_optimize, lr=args.lr, weight_decay=args.weight_decay)
    elif optimizer_name == 'rmsprop':
        logger.info(f"Using RMSprop optimizer with lr={args.lr}, momentum={args.momentum}, weight_decay={args.weight_decay}")
        optimizer = optim.RMSprop(params_to_optimize, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        logger.warning(f"Unknown optimizer '{args.optimizer}', defaulting to Adam")
        optimizer = optim.Adam(params_to_optimize, lr=args.lr, weight_decay=args.weight_decay)

    return optimizer


def main(args):
    '''Main training function optimized for speed'''

    try:
        start_time = time.time()
        logger.info("Starting FAST training job...")
        logger.info(f"Hyperparameters: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
        logger.info(f"Additional hyperparameters: dropout={args.dropout}, momentum={args.momentum}, optimizer={args.optimizer}, weight_decay={args.weight_decay}")

        # Set device and enable optimization
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        if torch.cuda.is_available():
            # Enable cuDNN autotuner for better performance
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True

        # Initialize model with dropout hyperparameter
        model = net(dropout_rate=args.dropout)
        model = model.to(device)

        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

        # Create loss function
        criterion = nn.CrossEntropyLoss()

        # Create optimizer with hyperparameters
        optimizer = get_optimizer(model, args)

        # Create data loaders
        train_loader, test_loader = create_data_loaders(
            args.data_pct,
            args.data_dir,
            args.batch_size)

        # Training loop - reduced epochs for HPO
        hpo_epochs = min(args.epochs, 3)  # Max 3 epochs for HPO
        logger.info(f"Running {hpo_epochs} epochs for HPO")

        best_loss = float('inf')
        for epoch in range(1, hpo_epochs + 1):
            epoch_start = time.time()

            train(model, train_loader, criterion, optimizer, epoch, device)
            test_loss = test(model, test_loader, criterion, device)

            epoch_time = time.time() - epoch_start
            logger.info(f"Epoch {epoch} completed in {epoch_time:.1f} seconds")

            if test_loss < best_loss:
                best_loss = test_loss

        # Save model
        logger.info(f"Saving model to {args.model_dir}")
        model_path = os.path.join(args.model_dir, 'model.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'best_loss': best_loss,
            'hyperparameters': {
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'lr': args.lr,
                'dropout': args.dropout,
                'momentum': args.momentum,
                'optimizer': args.optimizer,
                'weight_decay': args.weight_decay
            }
        }, model_path)

        total_time = time.time() - start_time
        logger.info(f"Training completed successfully in {total_time:.1f} seconds!")

    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument('--data_pct', type=float, default=1)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.5)
    parser.add_argument('--optimizer', type=str, default="adam")
    parser.add_argument('--weight-decay', type=float, default=1e-3)

    # Directories
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--data-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--output-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))

    args = parser.parse_args()

    main(args)