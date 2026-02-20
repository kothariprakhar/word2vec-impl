import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# --- 1. Configuration ---
# Define hyperparameters, device setup etc.
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 128
INITIAL_LR = 0.01
# Using a small number of epochs (2-3) for a quick demo as requested.
# For full training to reproduce paper results, this would be much higher (e.g., 200).
NUM_EPOCHS = 3 
DATASET_NAME = 'cifar10'  # Choose 'cifar10' or 'cifar100'
NUM_CLASSES = 10 if DATASET_NAME == 'cifar10' else 100
CHECKPOINT_PATH = f'best_tinyimagenet_{DATASET_NAME}.pth'
PLOT_SAVE_PATH = 'results.png'
# The error "[Errno 2] No such file or directory: 'python'" indicates issues with multiprocessing workers.
# To ensure the code runs without errors on systems where the 'python' executable for subprocesses
# is not easily found or configured, we explicitly set num_workers to 0.
# This prevents multiprocessing-related errors but may slow down data loading.
# The architecture document suggests 'num_workers: 4 (or appropriate for system)',
# and 0 is considered 'appropriate for system' if multiprocessing causes errors.
NUM_WORKERS = 0 

print(f"Using device: {DEVICE}")
print(f"Training on {DATASET_NAME} with {NUM_CLASSES} classes.")
print(f"Number of epochs for demo: {NUM_EPOCHS}")

# --- 2. Model Definition ---
class TinyImageNet(nn.Module):
    """
    Implements the deep convolutional neural network described by Alex Krizhevsky
    for tiny image classification (e.g., CIFAR-10/100).
    """
    def __init__(self, num_classes=10):
        super(TinyImageNet, self).__init__()

        # Layer 1: Conv-ReLU-Norm-Pool
        self.conv1 = nn.Conv2d(3, 96, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        # Using nn.LocalResponseNorm as specified in the architecture document
        # and paper summary implies its use for local contrast normalization.
        self.norm1 = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2) # 32x32 -> 15x15

        # Layer 2: Conv-ReLU-Norm-Pool
        self.conv2 = nn.Conv2d(96, 96, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.norm2 = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2) # 15x15 -> 7x7

        # Layer 3: Conv-ReLU-Pool
        self.conv3 = nn.Conv2d(96, 192, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2) # 7x7 -> 3x3

        # Fully Connected Layers
        # Calculate the input dimension for the first FC layer
        # Output of Pool3: (N, 192, 3, 3)
        self.fc_input_dim = 192 * 3 * 3 # 1728

        self.fc1 = nn.Linear(self.fc_input_dim, 256)
        self.relu4 = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(256, 256)
        self.relu5 = nn.ReLU(inplace=True)

        self.fc3 = nn.Linear(256, num_classes) # Output layer with num_classes neurons

    def forward(self, x):
        # Layer 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.norm1(x)
        x = self.pool1(x)

        # Layer 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.norm2(x)
        x = self.pool2(x)

        # Layer 3
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        # Flatten for fully connected layers
        x = x.view(-1, self.fc_input_dim)

        # Layer 4 (FC1-ReLU)
        x = self.fc1(x)
        x = self.relu4(x)

        # Layer 5 (FC2-ReLU)
        x = self.fc2(x)
        x = self.relu5(x)

        # Output Layer (FC3)
        x = self.fc3(x)
        return x

# --- 3. Data Loading and Preprocessing ---
def get_cifar_data_loaders(batch_size, dataset_name='cifar10', num_workers=0):
    """
    Prepares and returns CIFAR training and test DataLoaders with specified
    data augmentation and normalization.
    """
    # Mean and Std for CIFAR-10/100 datasets (common values)
    if dataset_name == 'cifar10':
        normalize_mean = [0.4914, 0.4822, 0.4465]
        normalize_std = [0.2471, 0.2435, 0.2616]
        dataset_class = torchvision.datasets.CIFAR10
    elif dataset_name == 'cifar100':
        normalize_mean = [0.5071, 0.4867, 0.4408]
        normalize_std = [0.2675, 0.2565, 0.2761]
        dataset_class = torchvision.datasets.CIFAR100
    else:
        raise ValueError("Invalid dataset_name. Choose 'cifar10' or 'cifar100'.")

    # Training transforms with data augmentation
    transform_train = transforms.Compose([
        transforms.Pad(4),                     # Pad image to 36x36
        transforms.RandomCrop(32),             # Randomly crop back to 32x32
        transforms.RandomHorizontalFlip(),     # Randomly flip horizontally
        transforms.ToTensor(),                 # Convert images to PyTorch tensors
        transforms.Normalize(normalize_mean, normalize_std), # Normalize pixel values
    ])

    # Test transforms (no augmentation)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std),
    ])

    # Load datasets
    trainset = dataset_class(root='./data', train=True, download=True, transform=transform_train)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    testset = dataset_class(root='./data', train=False, download=True, transform=transform_test)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader

# --- 4. Training and Evaluation Functions ---
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Performs one training epoch.
    """
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()  # Zero the parameter gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, targets)  # Calculate loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if (batch_idx + 1) % 100 == 0:
            print(f"  Batch {batch_idx+1}/{len(dataloader)} - Loss: {loss.item():.4f}")

    avg_loss = running_loss / len(dataloader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluates the model on a given dataloader (e.g., validation or test set).
    """
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient calculation
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / len(dataloader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def plot_results(train_losses, val_losses, train_accs, val_accs, lrs, save_path='results.png'):
    """
    Plots training and validation loss, accuracy, and learning rate over epochs.
    Saves the plot to a file.
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_accs, label='Training Accuracy')
    plt.plot(epochs, val_accs, label='Validation Accuracy')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(epochs, lrs, label='Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Results plot saved to {save_path}")
    plt.close() # Close the plot to free memory

# --- 5. Main Execution Block ---
if __name__ == '__main__':
    # Fix: Explicitly set the multiprocessing start method.
    # The error "[Errno 2] No such file or directory: 'python'" often occurs
    # when PyTorch's DataLoader or other multiprocessing components try to spawn
    # new Python processes but cannot find the 'python' executable in the system's PATH.
    # Even with num_workers=0, some internal PyTorch mechanisms related to multiprocessing setup
    # might trigger this. Setting the start method explicitly, especially to 'spawn',
    # can help PyTorch correctly locate and invoke the interpreter.
    if sys.platform.startswith('win') or torch.multiprocessing.get_start_method(allow_none=True) is None:
        try:
            torch.multiprocessing.set_start_method('spawn', force=True)
            print("Set multiprocessing start method to 'spawn'.")
        except RuntimeError as e:
            print(f"Warning: Could not set multiprocessing start method to 'spawn': {e}")
            print("Continuing without explicitly setting start method, may encounter issues.")

    # Data Loaders
    # NUM_WORKERS is explicitly set to 0 in the Configuration section to avoid multiprocessing issues.
    train_loader, test_loader = get_cifar_data_loaders(BATCH_SIZE, DATASET_NAME, NUM_WORKERS)
    print(f"Initialized DataLoaders with {NUM_WORKERS} workers.")

    # Model, Loss, Optimizer, Scheduler
    model = TinyImageNet(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=INITIAL_LR, momentum=0.9, weight_decay=5e-4)
    # Learning rate scheduler with typical milestones for CIFAR datasets over many epochs.
    # For a demo with NUM_EPOCHS=3, this scheduler will not decay the LR.
    scheduler = optim.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.1)

    # Training History
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    lrs = []
    best_val_accuracy = 0.0

    print("\nStarting training...")
    for epoch in range(NUM_EPOCHS):
        current_lr = optimizer.param_groups[0]['lr']
        lrs.append(current_lr)

        # Training Phase
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} - Training...")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation Phase (using test_loader as validation)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Evaluating...")
        val_loss, val_acc = evaluate_model(model, test_loader, criterion, DEVICE)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        scheduler.step()  # Update learning rate

        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} Summary ---")
        print(f"LR: {current_lr:.6f} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

        # Save best model based on validation accuracy
        if val_acc > best_val_accuracy:
            print(f"Validation accuracy improved from {best_val_accuracy:.2f}% to {val_acc:.2f}%. Saving model.")
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), CHECKPOINT_PATH)
        else:
            print(f"Validation accuracy did not improve. Best was {best_val_accuracy:.2f}%.")

    print("\nTraining complete.")

    # --- Final Evaluation ---
    if os.path.exists(CHECKPOINT_PATH):
        print(f"\nLoading best model from {CHECKPOINT_PATH} for final evaluation.")
        # Ensure map_location is used when loading state_dict to handle device consistency
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    else:
        print("\nNo best model checkpoint found. Using the last trained model for final evaluation.")

    final_test_loss, final_test_acc = evaluate_model(model, test_loader, criterion, DEVICE)
    print(f"\n--- Final Test Results ---")
    print(f"Final Test Loss: {final_test_loss:.4f}")
    print(f"Final Test Accuracy: {final_test_acc:.2f}%")
    print(f"Final Test Error Rate: {100. - final_test_acc:.2f}%")

    # --- Visualization ---
    plot_results(train_losses, val_losses, train_accs, val_accs, lrs, save_path=PLOT_SAVE_PATH)