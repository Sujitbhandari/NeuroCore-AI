# NeuroCore: High-Performance Deep Learning Engine

A high-performance, hybrid AI system built from scratch in C++ (for speed/training) wrapped with Python (for data/visualization). This project demonstrates deep understanding of the calculus and memory management behind AI systems.

## What Makes This Impressive

- Raw C++ Implementation: No external libraries like Eigen or PyTorch - you implement the math yourself
- OpenMP Parallelization: Multithreaded matrix operations using all CPU cores
- Manual Backpropagation: Full implementation of the chain rule and gradient descent
- Hybrid Architecture: C++ for computation, Python for data handling and visualization

## Architecture

### Core Engine (C++17)
- Matrix Class: Custom implementation using std::vector<double> with OpenMP parallelization
- Neural Network: Fully-connected feedforward network with:
  - Variable hidden layers
  - Sigmoid and ReLU activation functions
  - Manual backpropagation with Stochastic Gradient Descent (SGD)
  - Model serialization to JSON

### Data Pipeline (Python)
- Downloads and preprocesses MNIST dataset
- Normalizes pixel values (0-1)
- Exports to CSV format for C++ consumption

### Visualization (Python)
- Plots training loss and accuracy curves
- Generates publication-ready figures

## Directory Structure

```
NeuroCore AI/
├── src/              # C++ source files
│   ├── main.cpp      # Training entry point
│   ├── Matrix.cpp    # Matrix operations
│   └── Network.cpp   # Neural network implementation
├── include/          # C++ header files
│   ├── Matrix.h
│   └── Network.h
├── scripts/         # Python utilities
│   ├── data_loader.py    # MNIST data preparation
│   └── visualize.py      # Training visualization
├── data/            # Data files (CSV, logs, models)
├── CMakeLists.txt   # Build configuration
└── README.md        # This file
```

## Quick Start

### Prerequisites

- C++ Compiler with C++17 support (GCC 7+, Clang 5+)
- CMake (3.12+)
- OpenMP (usually included with compiler)
- Python 3.7+ with packages:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib

### Installation

1. Create virtual environment and install Python dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   pip install pandas numpy scikit-learn matplotlib
   ```

2. Prepare the data:
   ```bash
   python scripts/data_loader.py
   ```
   This downloads MNIST and creates data/mnist_train.csv and data/mnist_test.csv.

3. Build the C++ engine:
   ```bash
   mkdir build
   cd build
   cmake ..
   make
   ```

4. Train the network:
   ```bash
   ./NeuroCore
   ```
   The network will train for 20 epochs and print accuracy at each step.

5. Visualize results:
   ```bash
   python ../scripts/visualize.py
   ```

## Network Architecture

Default configuration:
- Input Layer: 784 neurons (28×28 MNIST images)
- Hidden Layer: 128 neurons with ReLU activation
- Output Layer: 10 neurons (digits 0-9) with Sigmoid activation
- Learning Rate: 0.001
- Weight Initialization: He initialization for ReLU layers

## Mathematical Foundation

### Forward Pass
```
z_l = W_l · a_l-1 + b_l
a_l = σ(z_l)
```
where σ is the activation function (Sigmoid or ReLU).

### Backpropagation (Chain Rule)
```
δ_L = (y_pred - y_true) · σ'(z_L)          # Output layer error
δ_l = (W_l+1^T · δ_l+1) · σ'(z_l)          # Hidden layer error
∂L/∂W_l = δ_l · a_l-1^T                    # Weight gradient
W_l = W_l - α · ∂L/∂W_l                    # Weight update (SGD)
```

### Loss Function
Mean Squared Error (MSE):
```
L = (1/2) · ||y_pred - y_true||²
```

## Customization

### Change Network Architecture

Edit src/main.cpp:
```cpp
std::vector<size_t> layerSizes = {784, 256, 128, 64, 10};  # Add more layers
```

### Change Activation Function

```cpp
Network network(layerSizes, ActivationType::RELU, 0.01);  # Use ReLU
```

### Adjust Learning Rate

```cpp
Network network(layerSizes, ActivationType::SIGMOID, 0.001);  # Lower learning rate
```

## Performance Features

- OpenMP Parallelization: Matrix operations use all CPU cores
- Memory Efficient: Uses std::vector for safe memory management
- Optimized Build: Compiles with -O3 optimization flags

## Output Files

After training, you'll find:
- data/training_log.csv: Epoch-by-epoch loss and accuracy
- data/neurocore_model.json: Trained weights and biases
- data/training_curves.png: Visualization plots

## Performance Results

The network achieves approximately 88% accuracy on MNIST test set after 20 epochs of training.

## Troubleshooting

### OpenMP not found
```bash
# On macOS with Homebrew
brew install libomp

# On Ubuntu/Debian
sudo apt-get install libomp-dev
```

### Build errors
- Ensure CMake version ≥ 3.12
- Check that your compiler supports C++17
- Verify OpenMP is installed

### Data loading errors
- Run python scripts/data_loader.py first
- Check that data/ directory exists
- Verify internet connection for MNIST download

## License

This project is for educational purposes, demonstrating deep learning fundamentals.

## Acknowledgments

- MNIST dataset by Yann LeCun
- OpenMP for parallel computing standards
