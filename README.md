# NeuroCore

A deep learning engine implemented from scratch in C++ for handwritten digit classification on the MNIST dataset. This project implements core neural network algorithms including backpropagation, gradient descent, and proper weight initialization without relying on machine learning frameworks.

## Overview

NeuroCore demonstrates understanding of neural network fundamentals by implementing:

- Custom matrix operations with OpenMP parallelization
- Manual backpropagation algorithm using the chain rule
- ReLU and Sigmoid activation functions with proper derivatives
- He initialization for ReLU layers
- Stochastic Gradient Descent (SGD) optimizer
- Model serialization and training visualization

The network achieves 88% accuracy on MNIST test set after 20 epochs of training.

## Architecture

The project uses a hybrid C++/Python architecture:

- C++ Core Engine: Neural network implementation with custom matrix class
- Python Scripts: Data preprocessing and visualization utilities
- OpenMP: Parallel matrix operations for improved performance

### Network Configuration

- Input Layer: 784 neurons (28×28 MNIST images)
- Hidden Layer: 128 neurons with ReLU activation
- Output Layer: 10 neurons (digits 0-9) with Sigmoid activation
- Learning Rate: 0.001
- Weight Initialization: He initialization for ReLU layers

## Technologies

- C++17
- OpenMP (parallel computing)
- CMake (build system)
- Python 3 (data processing, visualization)
- NumPy, Pandas, Matplotlib, scikit-learn

## Building and Running

### Prerequisites

- C++ compiler with C++17 support (GCC 7+, Clang 5+)
- CMake 3.12+
- OpenMP
- Python 3.7+ with virtual environment

### Setup

1. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install pandas numpy scikit-learn matplotlib
```

2. Download and prepare MNIST data:
```bash
python scripts/data_loader.py
```

3. Build the C++ engine:
```bash
mkdir build && cd build
cmake ..
make
```

4. Train the network:
```bash
./NeuroCore
```

5. Visualize training results:
```bash
python ../scripts/visualize.py
```

## Implementation Details

### Backpropagation

The backpropagation algorithm implements the chain rule for gradient computation:

```
δ_L = (y_pred - y_true) · σ'(z_L)          # Output layer error
δ_l = (W_l+1^T · δ_l+1) · σ'(z_l)          # Hidden layer error
∂L/∂W_l = δ_l · a_l-1^T                    # Weight gradient
W_l = W_l - α · ∂L/∂W_l                    # SGD weight update
```

### Loss Function

Mean Squared Error (MSE):
```
L = (1/2) · ||y_pred - y_true||²
```

## Performance

- Final accuracy: 88% on MNIST test set (1,000 samples)
- Training time: ~5-10 minutes for 20 epochs (10,000 training samples)
- Uses OpenMP for parallel matrix operations

## Output Files

After training:
- data/training_log.csv: Epoch-by-epoch loss and accuracy
- data/neurocore_model.json: Serialized model weights and biases
- data/training_curves.png: Training visualization plots

## Skills Demonstrated

- Neural network implementation from first principles
- C++ memory management and performance optimization
- Parallel computing with OpenMP
- Numerical algorithms (matrix operations, gradient descent)
- Build systems (CMake)
- Hybrid C++/Python system design

## License

Educational purposes - demonstrating deep learning fundamentals.

## Acknowledgments

- MNIST dataset by Yann LeCun
- OpenMP for parallel computing standards
