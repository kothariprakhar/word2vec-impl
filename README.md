# Learning Multiple Layers of Features from Tiny Images: A PyTorch Implementation

## A Reproduction of Krizhevsky's Deep CNN for CIFAR-10/100

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kothariprakhar/word2vec-impl/blob/main/notebook.ipynb)

## 📖 Overview

This repository presents a PyTorch implementation of the deep convolutional neural network described in Alex Krizhevsky's influential paper, "**Learning Multiple Layers of Features from Tiny Images**" (2012). This work demonstrated the efficacy of deep CNNs for image classification on small-resolution datasets like CIFAR-10 and CIFAR-100, achieving state-of-the-art results at the time of publication.

Our implementation meticulously replicates the proposed network architecture, including the use of Rectified Linear Units (ReLUs) for faster training, local contrast normalization (Local Response Normalization), extensive data augmentation strategies (random cropping and horizontal flipping), and training on GPUs. The goal is to provide a clear, executable, and reproducible baseline for this classic CNN model.

**Original Paper:** [Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf) (Note: This is an earlier technical report, but it contains the core ideas. The results mentioned in the abstract are often attributed to later work or the context of the larger deep learning advancements Krizhevsky was involved in.)

## 📐 Architecture

The model is a deep convolutional neural network specifically designed for 32x32 color images, comprising multiple layers of convolutions, pooling, normalization, and fully connected layers.

### Model Diagram

```mermaid
graph TD
    Input[Input Image (32x32x3)] --> Conv1[Conv2d (3->96, 3x3, P=1)]
    Conv1 --> ReLU1[ReLU]
    ReLU1 --> Norm1[LocalResponseNorm]
    Norm1 --> Pool1[MaxPool2d (3x3, S=2)]
    Pool1 --> Conv2[Conv2d (96->96, 3x3, P=1)]
    Conv2 --> ReLU2[ReLU]
    ReLU2 --> Norm2[LocalResponseNorm]
    Norm2 --> Pool2[MaxPool2d (3x3, S=2)]
    Pool2 --> Conv3[Conv2d (96->192, 3x3, P=1)]
    Conv3 --> ReLU3[ReLU]
    ReLU3 --> Pool3[MaxPool2d (3x3, S=2)]
    Pool3 --> Flatten[Flatten (192*3*3 = 1728)]
    Flatten --> FC1[Linear (1728->256)]
    FC1 --> ReLU4[ReLU]
    ReLU4 --> FC2[Linear (256->256)]
    FC2 --> ReLU5[ReLU]
    ReLU5 --> FC3[Linear (256->num_classes)]
    FC3 --> Output[Logits]
```

## 🚀 Quick Start

Follow these steps to set up the environment and run the training script.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/kothariprakhar/word2vec-impl.git
    cd word2vec-impl
    ```

2.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install torch torchvision matplotlib
    ```

### Usage

To train and evaluate the model on CIFAR-10 (default) or CIFAR-100:

```bash
# To train on CIFAR-10 (default settings)
python cifar_krizhevsky.py

# To train on CIFAR-100
python cifar_krizhevsky.py --dataset cifar100

# You can also adjust other parameters like batch size, learning rate, epochs, etc.
# For example:
# python cifar_krizhevsky.py --dataset cifar100 --epochs 100 --batch_size 64 --lr 0.005
```

The script will:
*   Download the specified CIFAR dataset.
*   Initialize and train the Krizhevsky CNN.
*   Apply data augmentation during training.
*   Decay the learning rate at specified milestones.
*   Save the best performing model based on validation accuracy.
*   Print training/validation metrics per epoch.
*   Finally, evaluate the best model on the test set and plot learning curves.

## 📊 Results

The following table will be populated with actual performance metrics after running the training script.

| Dataset    | Epochs | Initial LR | Weight Decay | Best Val. Accuracy (%) | Final Test Accuracy (%) | Test Error Rate (%) |
| :--------- | :----- | :--------- | :----------- | :--------------------- | :---------------------- | :------------------ |
| CIFAR-10   | 200    | 0.01       | 5e-4         | `[To be filled]`       | `[To be filled]`        | `[To be filled]`    |
| CIFAR-100  | 200    | 0.01       | 5e-4         | `[To be filled]`       | `[To be filled]`        | `[To be filled]`    |

## 📁 Project Structure

```
.
├── cifar_krizhevsky.py    # Main script for model definition, training, and evaluation
├── data/                  # Directory to store downloaded CIFAR datasets (created automatically)
├── best_tinyimagenet.pth  # Saved model weights (created after first run)
├── LICENSE                # MIT License file
└── README.md              # This README file
```

## 📝 Citation

If you find this implementation useful, please consider citing the original paper:

```bibtex
@techreport{krizhevsky2009learning,
  title={Learning Multiple Layers of Features from Tiny Images},
  author={Krizhevsky, Alex},
  year={2009},
  institution={University of Toronto}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.