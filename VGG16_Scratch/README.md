# VGG16-From-Scratch

This repository contains the implementation of the **VGG16** architecture from scratch using both **PyTorch** and **TensorFlow**. It provides the source code, training scripts, and utilities for building, training, and evaluating the VGG16 model on custom datasets or ImageNet.

## Table of Contents

- [Project Overview](#project-overview)
- [Folder Structure](#folder-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Testing](#testing)
- [License](#license)

## Project Overview

VGG16 is a deep convolutional neural network architecture, widely used for image classification tasks. This repository provides implementations of VGG16 in both **PyTorch** and **TensorFlow**, with training scripts and evaluation pipelines. It also includes notebooks for exploration and testing, utilities for data preprocessing, and configurations for different experiment setups.

## Folder Structure

    VGG16-From-Scratch/
    │── data/                   # Dataset storage (if applicable)
    │   ├── raw/                # Raw dataset (if needed)
    │   ├── processed/          # Preprocessed dataset
    │
    │── notebooks/              # Jupyter notebooks for exploration & testing
    │   ├── vgg16_pytorch.ipynb
    │   ├── vgg16_tensorflow.ipynb
    │
    │── models/                 # Model definitions
    │   ├── vgg16_pytorch.py    # VGG16 implementation in PyTorch
    │   ├── vgg16_tensorflow.py # VGG16 implementation in TensorFlow
    │
    │── utils/                  # Utility functions (data loading, preprocessing, etc.)
    │   ├── dataset_loader.py   # Custom dataloaders
    │   ├── preprocessing.py    # Preprocessing utilities
    │
    │── experiments/            # Training logs, results, and saved models
    │   ├── pytorch/
    │   │   ├── checkpoints/    # PyTorch model checkpoints
    │   │   ├── logs/           # PyTorch training logs
    │   │
    │   ├── tensorflow/
    │       ├── checkpoints/    # TensorFlow model checkpoints
    │       ├── logs/           # TensorFlow training logs
    │
    │── scripts/                # Training and evaluation scripts
    │   ├── train_pytorch.py    # Training script for PyTorch
    │   ├── train_tensorflow.py # Training script for TensorFlow
    │   ├── evaluate.py         # Evaluation script
    │
    │── tests/                  # Unit tests for model, data pipeline, etc.
    │   ├── test_vgg16.py       # Testing script
    │
    │── requirements.txt        # Python dependencies
    │── README.md               # Documentation
    │── .gitignore              # Ignore unnecessary files
    │── config.py               # Config file (e.g., hyperparameters)
  

## Installation

To set up the project locally, clone the repository and install the required dependencies:

1. Clone the repository:
    ```bash
    git clone https://github.com/netra212/VGG16-From-Scratch.git
    cd VGG16-From-Scratch
    ```

2. Install the dependencies from `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

3. For TensorFlow and PyTorch, ensure you have the respective framework installed:
    - PyTorch: [Installation guide](https://pytorch.org/get-started/locally/)
    - TensorFlow: [Installation guide](https://www.tensorflow.org/install)

## Usage

### Jupyter Notebooks

- **vgg16_pytorch.ipynb**: A Jupyter notebook for experimenting with the PyTorch version of VGG16.
- **vgg16_tensorflow.ipynb**: A Jupyter notebook for experimenting with the TensorFlow version of VGG16.

You can run these notebooks to visualize the architecture, load datasets, and try different configurations.

### Model Definitions

- **models/vgg16_pytorch.py**: Contains the PyTorch implementation of VGG16.
- **models/vgg16_tensorflow.py**: Contains the TensorFlow implementation of VGG16.

### Training

- **scripts/train_pytorch.py**: Train the VGG16 model using PyTorch.
- **scripts/train_tensorflow.py**: Train the VGG16 model using TensorFlow.

For both, you can configure the hyperparameters (e.g., batch size, learning rate, etc.) in `config.py`.

```bash
python scripts/train_pytorch.py
python scripts/train_tensorflow.py
```

### Evaluation

Once training is complete, you can evaluate your model using the following script:
```python scripts/evaluate.py```

### Testing

To test the implemented model for unit tests and validation, use the following:
```python -m unittest tests.test_vgg16```

###  License

This project is open source and available for anyone to use, modify, and distribute under the MIT License.



