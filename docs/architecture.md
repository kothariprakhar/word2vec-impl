This document outlines the architecture and implementation details for reproducing the model described in "Learning Multiple Layers of Features from Tiny Images" by Alex Krizhevsky, using PyTorch. The goal is to achieve state-of-the-art performance on CIFAR-10 and CIFAR-100 datasets through careful replication of the proposed network structure, training methodologies, and evaluation strategies.

---

# Architecture Document: Learning Multiple Layers of Features from Tiny Images (PyTorch Implementation)

## 1. Overview

This implementation focuses on building a deep convolutional neural network (CNN) in PyTorch, replicating the architecture proposed by Krizhevsky for tiny image classification. The core components include sequential blocks of convolutional layers, ReLU activations, local contrast normalization (implemented as Local Response Normalization), and max-pooling, followed by fully connected layers. The training pipeline will leverage PyTorch's data loading, optimization, and scheduling capabilities, incorporating key techniques like extensive data augmentation (random crops and horizontal flips) and a standard cross-entropy loss function. Evaluation will primarily rely on the test error rate, supplemented by visual analytics of training progress.

## 2. Module Breakdown

This section details the custom `nn.Module` classes and components required for the network.

### `LocalResponseNormalization` (Custom Wrapper or `torch.nn.LocalResponseNorm`)

*   **Name:** `LocalResponseNormalization` (if custom) or `torch.nn.LocalResponseNorm`
*   **Purpose:** Implements local contrast normalization, which helps with generalization by normalizing across feature maps in a local neighborhood. It computes a normalized output for each element by dividing it by a factor that depends on the sum of squares of elements in a local neighborhood across feature maps.
*   **Key Parameters:**
    *   `size`: The number of channels to sum over (n in the paper's formula). Default: 5
    *   `alpha`: Scaling parameter (α). Default: 1e-4
    *   `beta`: Exponent parameter (β). Default: 0.75
    *   `k`: Additive constant (k). Default: 2
*   **Input Shape:** `(N, C, H, W)`
*   **Output Shape:** `(N, C, H, W)`

### `TinyImageNet` (Main Model)

This class represents the complete convolutional neural network.

*   **Name:** `TinyImageNet`
*   **Purpose:** Implements the end-to-end convolutional neural network as described in the paper, from raw image input to class probabilities.
*   **Key Parameters:**
    *   `num_classes`: Integer, number of output classes (e.g., 10 for CIFAR-10, 100 for CIFAR-100).
*   **Input Shape:** `(N, 3, 32, 32)` (batch of RGB 32x32 images)
*   **Output Shape:** `(N, num_classes)` (logits for each class)

#### Internal Layers and Tensor Shapes:

The `TinyImageNet` module will consist of the following sequential layers:

1.  **Input Layer:** `(N, 3, 32, 32)`
2.  **Layer 1 (Conv-ReLU-Norm-Pool):**
    *   `self.conv1 = nn.Conv2d(3, 96, kernel_size=3, padding=1)`
        *   Input: `(N, 3, 32, 32)`
        *   Output: `(N, 96, 32, 32)`
    *   `self.relu1 = nn.ReLU(inplace=True)`
        *   Input: `(N, 96, 32, 32)`
        *   Output: `(N, 96, 32, 32)`
    *   `self.norm1 = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2)`
        *   Input: `(N, 96, 32, 32)`
        *   Output: `(N, 96, 32, 32)`
    *   `self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)`
        *   Input: `(N, 96, 32, 32)`
        *   Output: `(N, 96, 15, 15)` (calculated as `floor((32 - 3)/2) + 1 = 15`)
3.  **Layer 2 (Conv-ReLU-Norm-Pool):**
    *   `self.conv2 = nn.Conv2d(96, 96, kernel_size=3, padding=1)`
        *   Input: `(N, 96, 15, 15)`
        *   Output: `(N, 96, 15, 15)`
    *   `self.relu2 = nn.ReLU(inplace=True)`
        *   Input: `(N, 96, 15, 15)`
        *   Output: `(N, 96, 15, 15)`
    *   `self.norm2 = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2)`
        *   Input: `(N, 96, 15, 15)`
        *   Output: `(N, 96, 15, 15)`
    *   `self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)`
        *   Input: `(N, 96, 15, 15)`
        *   Output: `(N, 96, 7, 7)` (calculated as `floor((15 - 3)/2) + 1 = 7`)
4.  **Layer 3 (Conv-ReLU-Pool):**
    *   `self.conv3 = nn.Conv2d(96, 192, kernel_size=3, padding=1)`
        *   Input: `(N, 96, 7, 7)`
        *   Output: `(N, 192, 7, 7)`
    *   `self.relu3 = nn.ReLU(inplace=True)`
        *   Input: `(N, 192, 7, 7)`
        *   Output: `(N, 192, 7, 7)`
    *   `self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)`
        *   Input: `(N, 192, 7, 7)`
        *   Output: `(N, 192, 3, 3)` (calculated as `floor((7 - 3)/2) + 1 = 3`)
5.  **Flattening Layer:**
    *   Before fully connected layers, the convolutional output `(N, 192, 3, 3)` is flattened to `(N, 192 * 3 * 3) = (N, 1728)`.
6.  **Layer 4 (Fully Connected-ReLU):**
    *   `self.fc1 = nn.Linear(192 * 3 * 3, 256)`
        *   Input: `(N, 1728)`
        *   Output: `(N, 256)`
    *   `self.relu4 = nn.ReLU(inplace=True)`
        *   Input: `(N, 256)`
        *   Output: `(N, 256)`
7.  **Layer 5 (Fully Connected-ReLU):**
    *   `self.fc2 = nn.Linear(256, 256)`
        *   Input: `(N, 256)`
        *   Output: `(N, 256)`
    *   `self.relu5 = nn.ReLU(inplace=True)`
        *   Input: `(N, 256)`
        *   Output: `(N, 256)`
8.  **Output Layer (Softmax - implicitly handled by loss):**
    *   `self.fc3 = nn.Linear(256, num_classes)`
        *   Input: `(N, 256)`
        *   Output: `(N, num_classes)` (logits)

## 3. Training Pipeline

### Data Loading Strategy

The CIFAR-10 and CIFAR-100 datasets will be loaded using `torchvision.datasets`.
Crucially, data augmentation is applied during training to prevent overfitting.

*   **Dataset:** `torchvision.datasets.CIFAR10` or `torchvision.datasets.CIFAR100`
*   **Transforms (Training):**
    1.  `torchvision.transforms.Pad(4)`: Pad image to 36x36.
    2.  `torchvision.transforms.RandomCrop(32)`: Randomly crop back to 32x32.
    3.  `torchvision.transforms.RandomHorizontalFlip()`: Randomly flip images horizontally.
    4.  `torchvision.transforms.ToTensor()`: Convert images to PyTorch tensors.
    5.  `torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])`: Normalize pixel values using CIFAR-specific means and standard deviations.
*   **Transforms (Validation/Test):**
    1.  `torchvision.transforms.ToTensor()`
    2.  `torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])`
*   **DataLoader:** `torch.utils.data.DataLoader`
    *   `batch_size`: 128 (default, can be tuned)
    *   `shuffle`: `True` for training, `False` for validation/test
    *   `num_workers`: 4 (or appropriate for system)
    *   `pin_memory`: `True` (if using GPU)

### Loss Function(s)

*   **Primary Loss:** `torch.nn.CrossEntropyLoss`
    *   This loss function combines `LogSoftmax` and `NLLLoss` in one single class. It expects raw logits from the model and target class indices.

### Optimizer & Scheduler

*   **Optimizer:** `torch.optim.SGD` (Stochastic Gradient Descent)
    *   `lr`: 0.01 (initial learning rate, will be decayed)
    *   `momentum`: 0.9
    *   `weight_decay`: 5e-4 (L2 regularization, common for CNNs)
*   **Learning Rate Scheduler:** `torch.optim.lr_scheduler.MultiStepLR` or `StepLR`
    *   `MultiStepLR`: Decays the learning rate by a factor (e.g., 0.1) at specified epoch milestones.
        *   `milestones`: [60, 120, 160] (example, typical for CIFAR datasets, adjust based on total epochs)
        *   `gamma`: 0.1 (multiplicative factor of learning rate decay)
    *   `StepLR`: Decays the learning rate by a factor every `step_size` epochs.
        *   `step_size`: 50
        *   `gamma`: 0.1

### Training Loop Pseudocode

```python
# Initialize model, loss, optimizer, and scheduler
model = TinyImageNet(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
scheduler = optim.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.1) # Example milestones

num_epochs = 200 # Example

for epoch in range(num_epochs):
    # --- Training Phase ---
    model.train() # Set model to training mode
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad() # Zero the parameter gradients
        outputs = model(inputs) # Forward pass
        loss = criterion(outputs, targets) # Calculate loss
        loss.backward() # Backward pass
        optimizer.step() # Update weights

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total_train += targets.size(0)
        correct_train += predicted.eq(targets).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct_train / total_train
    print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')

    scheduler.step() # Update learning rate

    # --- Validation Phase ---
    model.eval() # Set model to evaluation mode
    val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad(): # Disable gradient calculation for validation
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total_val += targets.size(0)
            correct_val += predicted.eq(targets).sum().item()

    val_loss = val_loss / len(val_loader)
    val_accuracy = 100. * correct_val / total_val
    print(f'Epoch {epoch+1}/{num_epochs} - Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')

    # Save best model checkpoint
    # if val_accuracy > best_accuracy:
    #     torch.save(model.state_dict(), 'best_model.pth')
    #     best_accuracy = val_accuracy
```

## 4. Evaluation

### Metrics to Compute

*   **Test Error Rate:** The primary metric, calculated as `(1 - accuracy) * 100` on the test dataset.
*   **Accuracy:** Percentage of correctly classified images on the training, validation, and test sets.
*   **Loss:** Average `CrossEntropyLoss` on training, validation, and test sets.

### Visualization (Plots, Sample Outputs)

*   **Loss Curves:** Plot training and validation loss per epoch.
*   **Accuracy Curves:** Plot training and validation accuracy per epoch.
*   **Learning Rate Schedule:** Plot the learning rate over epochs to visualize decay.
*   **Sample Predictions:** Display a grid of test images along with their true labels and the model's predicted labels (and confidence scores if desired). Highlight misclassifications.
*   **Confusion Matrix:** For CIFAR-10, a confusion matrix can provide insights into which classes are being confused.

## 5. File Structure

A single Python file named `cifar_krizhevsky.py` is suggested for this implementation, organized with clear section markers.

```
# cifar_krizhevsky.py

# --- 0. Imports ---
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

# --- 1. Configuration ---
# Define hyperparameters, device setup etc.
# For example:
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# BATCH_SIZE = 128
# LEARNING_RATE = 0.01
# NUM_EPOCHS = 200
# DATASET_NAME = 'cifar10' # or 'cifar100'
# NUM_CLASSES = 10 # or 100

# --- 2. Model Definition ---
# Defines the TinyImageNet class and LocalResponseNormalization
class TinyImageNet(nn.Module):
    def __init__(self, num_classes=10):
        super(TinyImageNet, self).__init__()
        # Layer 1
        self.conv1 = nn.Conv2d(3, 96, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.norm1 = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        # Layer 2
        self.conv2 = nn.Conv2d(96, 96, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.norm2 = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        # Layer 3
        self.conv3 = nn.Conv2d(96, 192, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        # Fully Connected Layers
        # Calculate the size after conv and pooling layers
        # Input 32x32 -> Pool1 (15x15) -> Pool2 (7x7) -> Pool3 (3x3)
        self.fc_input_dim = 192 * 3 * 3 # 192 channels, 3x3 feature map
        self.fc1 = nn.Linear(self.fc_input_dim, 256)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256, 256)
        self.relu5 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(self.norm1(self.relu1(self.conv1(x))))
        x = self.pool2(self.norm2(self.relu2(self.conv2(x))))
        x = self.pool3(self.relu3(self.conv3(x)))

        x = x.view(-1, self.fc_input_dim) # Flatten
        x = self.relu4(self.fc1(x))
        x = self.relu5(self.fc2(x))
        x = self.fc3(x) # Output logits
        return x

# --- 3. Data Loading and Preprocessing ---
# Define transforms and create DataLoader instances
def get_cifar_data_loaders(batch_size, dataset_name='cifar10'):
    # Mean and Std for CIFAR10/100
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

    transform_train = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std),
    ])

    trainset = dataset_class(root='./data', train=True, download=True, transform=transform_train)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    testset = dataset_class(root='./data', train=False, download=True, transform=transform_test)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader

# --- 4. Training and Evaluation Functions ---
# Define train_one_epoch, validate_one_epoch, test_model, and plot_results functions
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    return running_loss / len(dataloader), 100. * correct / total

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return running_loss / len(dataloader), 100. * correct / total

def plot_results(train_losses, val_losses, train_accs, val_accs, lrs):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(15, 5))

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
    plt.show()

# --- 5. Main Execution Block ---
# Setup, training loop, and final evaluation
if __name__ == '__main__':
    # Configuration
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 128
    INITIAL_LR = 0.01
    NUM_EPOCHS = 200
    DATASET_NAME = 'cifar10' # 'cifar10' or 'cifar100'
    NUM_CLASSES = 10 if DATASET_NAME == 'cifar10' else 100
    CHECKPOINT_PATH = 'best_tinyimagenet.pth'

    print(f"Using device: {DEVICE}")
    print(f"Training on {DATASET_NAME} with {NUM_CLASSES} classes.")

    # Data Loaders
    train_loader, test_loader = get_cifar_data_loaders(BATCH_SIZE, DATASET_NAME)

    # Model, Loss, Optimizer, Scheduler
    model = TinyImageNet(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=INITIAL_LR, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.1)

    # Training History
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    lrs = []
    best_val_accuracy = 0.0

    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        lrs.append(optimizer.param_groups[0]['lr'])
        
        # Training
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation (using test_loader as validation)
        val_loss, val_acc = evaluate_model(model, test_loader, criterion, DEVICE)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        scheduler.step()

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | LR: {lrs[-1]:.6f} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_accuracy:
            print(f"Validation accuracy improved from {best_val_accuracy:.2f}% to {val_acc:.2f}%. Saving model.")
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), CHECKPOINT_PATH)

    print("\nTraining complete.")

    # Load best model for final testing
    print(f"Loading best model from {CHECKPOINT_PATH} for final evaluation.")
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    final_test_loss, final_test_acc = evaluate_model(model, test_loader, criterion, DEVICE)
    print(f"Final Test Loss: {final_test_loss:.4f}, Final Test Accuracy: {final_test_acc:.2f}%")
    print(f"Final Test Error Rate: {100. - final_test_acc:.2f}%")

    # Plotting results
    plot_results(train_losses, val_losses, train_accs, val_accs, lrs)
```