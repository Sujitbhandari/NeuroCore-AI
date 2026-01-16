#include "../include/Network.h"
#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>
#include <sstream>
#include <iomanip>

// Constructor
Network::Network(const std::vector<size_t>& layerSizes, 
                 ActivationType activation,
                 double learningRate)
    : layerSizes_(layerSizes),
      numLayers_(layerSizes.size() - 1),
      activation_(activation),
      learningRate_(learningRate) {
    
    // Initialize per-layer activations
    // Default: ReLU for hidden layers, Sigmoid for output layer
    layerActivations_.resize(numLayers_);
    for (size_t i = 0; i < numLayers_; ++i) {
        // Last layer (output): Sigmoid for classification
        // Hidden layers: ReLU to avoid vanishing gradients
        layerActivations_[i] = (i == numLayers_ - 1) ? ActivationType::SIGMOID : ActivationType::RELU;
    }
    
    // Initialize weights and biases
    weights_.reserve(numLayers_);
    biases_.reserve(numLayers_);
    
    for (size_t i = 0; i < numLayers_; ++i) {
        // Weight matrix: (layerSize[i+1] x layerSize[i])
        // Connects layer i to layer i+1
        weights_.emplace_back(layerSizes_[i + 1], layerSizes_[i]);
        
        // Bias vector: (layerSize[i+1] x 1)
        biases_.emplace_back(layerSizes_[i + 1], 1);
    }
    
    initializeWeights();
}

/**
 * Forward Pass
 * 
 * Computes: a_l = σ(W_l * a_l-1 + b_l)
 * where σ is the activation function.
 */
Matrix Network::forward(const Matrix& input) {
    // Store activations and z-values for backpropagation
    activations_.clear();
    zValues_.clear();
    activations_.reserve(numLayers_ + 1);
    zValues_.reserve(numLayers_);
    
    // Input layer activation (just the input itself)
    Matrix current = input;
    activations_.push_back(current);
    
    // Propagate through each layer
    for (size_t i = 0; i < numLayers_; ++i) {
        // Compute weighted sum: z = W * a + b
        Matrix z = weights_[i].dot_product(current).add(biases_[i]);
        zValues_.push_back(z);
        
        // Apply activation function: a = σ(z)
        // Use layer-specific activation (ReLU for hidden, Sigmoid for output)
        current = applyActivation(z, i);
        activations_.push_back(current);
    }
    
    return current;  // Output layer activation
}

/**
 * Backward Pass (Backpropagation with Chain Rule)
 * 
 * Uses the chain rule to compute gradients and update weights.
 * 
 * Step-by-step:
 * 1. Compute output error: δ_L = (y_pred - y_true) * σ'(z_L)
 * 2. For each hidden layer (backward):
 *    - Compute error: δ_l = (W_l+1^T * δ_l+1) * σ'(z_l)
 *    - Update weights: W_l = W_l - α * δ_l * a_l-1^T
 *    - Update biases: b_l = b_l - α * δ_l
 */
double Network::backward(const Matrix& input, const Matrix& target) {
    // Forward pass (stores activations and z-values)
    Matrix output = forward(input);
    
    // Compute loss: MSE = (1/2) * ||y_pred - y_true||²
    Matrix error = output.subtract(target);
    double loss = 0.0;
    for (size_t i = 0; i < error.getRows(); ++i) {
        loss += error(i, 0) * error(i, 0);
    }
    loss *= 0.5;
    
    // BACKPROPAGATION: Start from output layer and work backward
    // 
    // Output layer error: δ_L = (y_pred - y_true) * σ'(z_L)
    // This is the derivative of loss w.r.t. the output layer's pre-activation
    // Use layer-specific activation (Sigmoid for output)
    Matrix delta = error.multiply_elementwise(
        applyActivationDerivative(zValues_[numLayers_ - 1], numLayers_ - 1)
    );
    
    // Propagate error backward through each layer
    for (int i = numLayers_ - 1; i >= 0; --i) {
        // Gradient of loss w.r.t. weights: ∂L/∂W_i = δ_i * a_i-1^T
        // This comes from: z_i = W_i * a_i-1 + b_i, so ∂z_i/∂W_i = a_i-1^T
        Matrix weightGradient = delta.dot_product(activations_[i].transpose());
        
        // Update weights: W_i = W_i - α * ∂L/∂W_i
        // SGD update rule: subtract gradient scaled by learning rate
        weights_[i] = weights_[i].subtract(weightGradient.scalar_multiply(learningRate_));
        
        // Gradient of loss w.r.t. biases: ∂L/∂b_i = δ_i
        // This comes from: z_i = W_i * a_i-1 + b_i, so ∂z_i/∂b_i = 1
        Matrix biasGradient = delta;  // Bias gradient is just the error
        
        // Update biases: b_i = b_i - α * ∂L/∂b_i
        biases_[i] = biases_[i].subtract(biasGradient.scalar_multiply(learningRate_));
        
        // If not at input layer, propagate error backward
        if (i > 0) {
            // Error for previous layer: δ_i-1 = (W_i^T * δ_i) * σ'(z_i-1)
            // This is the chain rule: error flows backward through weights
            // Use layer-specific activation (ReLU for hidden layers)
            delta = weights_[i].transpose().dot_product(delta)
                    .multiply_elementwise(applyActivationDerivative(zValues_[i - 1], i - 1));
        }
    }
    
    return loss;
}

// Train on a single sample
double Network::trainSample(const Matrix& input, const Matrix& target) {
    return backward(input, target);
}

// Predict class
int Network::predict(const Matrix& input) {
    Matrix output = forward(input);
    
    // Find index of maximum output (argmax)
    int predictedClass = 0;
    double maxValue = output(0, 0);
    
    for (size_t i = 1; i < output.getRows(); ++i) {
        if (output(i, 0) > maxValue) {
            maxValue = output(i, 0);
            predictedClass = i;
        }
    }
    
    return predictedClass;
}

// Evaluate accuracy
double Network::evaluate(const std::vector<Matrix>& inputs, 
                         const std::vector<Matrix>& targets) {
    if (inputs.size() != targets.size()) {
        throw std::invalid_argument("Inputs and targets must have same size");
    }
    
    size_t correct = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        int predicted = predict(inputs[i]);
        
        // Find true class from one-hot target
        int trueClass = 0;
        double maxValue = targets[i](0, 0);
        for (size_t j = 1; j < targets[i].getRows(); ++j) {
            if (targets[i](j, 0) > maxValue) {
                maxValue = targets[i](j, 0);
                trueClass = j;
            }
        }
        
        if (predicted == trueClass) {
            correct++;
        }
    }
    
    return static_cast<double>(correct) / inputs.size();
}

// Save model to file
void Network::saveModel(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }
    
    file << std::fixed << std::setprecision(10);
    file << "{\n";
    file << "  \"num_layers\": " << numLayers_ << ",\n";
    file << "  \"layer_sizes\": [";
    for (size_t i = 0; i < layerSizes_.size(); ++i) {
        file << layerSizes_[i];
        if (i < layerSizes_.size() - 1) file << ", ";
    }
    file << "],\n";
    file << "  \"activation\": " << (activation_ == ActivationType::SIGMOID ? "\"sigmoid\"" : "\"relu\"") << ",\n";
    file << "  \"learning_rate\": " << learningRate_ << ",\n";
    file << "  \"weights\": [\n";
    
    for (size_t i = 0; i < weights_.size(); ++i) {
        file << "    {\n";
        file << "      \"layer\": " << i << ",\n";
        file << "      \"rows\": " << weights_[i].getRows() << ",\n";
        file << "      \"cols\": " << weights_[i].getCols() << ",\n";
        file << "      \"data\": [";
        for (size_t r = 0; r < weights_[i].getRows(); ++r) {
            for (size_t c = 0; c < weights_[i].getCols(); ++c) {
                file << weights_[i](r, c);
                if (r < weights_[i].getRows() - 1 || c < weights_[i].getCols() - 1) {
                    file << ", ";
                }
            }
        }
        file << "]\n";
        file << "    }";
        if (i < weights_.size() - 1) file << ",";
        file << "\n";
    }
    
    file << "  ],\n";
    file << "  \"biases\": [\n";
    
    for (size_t i = 0; i < biases_.size(); ++i) {
        file << "    {\n";
        file << "      \"layer\": " << i << ",\n";
        file << "      \"rows\": " << biases_[i].getRows() << ",\n";
        file << "      \"data\": [";
        for (size_t r = 0; r < biases_[i].getRows(); ++r) {
            file << biases_[i](r, 0);
            if (r < biases_[i].getRows() - 1) {
                file << ", ";
            }
        }
        file << "]\n";
        file << "    }";
        if (i < biases_.size() - 1) file << ",";
        file << "\n";
    }
    
    file << "  ]\n";
    file << "}\n";
    
    file.close();
}

// Load model from file (simplified - would need JSON parser for production)
void Network::loadModel(const std::string& filename) {
    // Note: This is a simplified loader. For production, use a proper JSON parser.
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for reading: " + filename);
    }
    
    // For now, just print a message
    // Full implementation would parse JSON and restore weights/biases
    std::cout << "Note: Model loading from " << filename << " not fully implemented." << std::endl;
    std::cout << "Use saveModel() to export weights, then manually restore if needed." << std::endl;
    
    file.close();
}

// Sigmoid activation: σ(x) = 1 / (1 + e^(-x))
double Network::sigmoid(double x) const {
    // Clamp to prevent overflow
    x = std::max(-500.0, std::min(500.0, x));
    return 1.0 / (1.0 + std::exp(-x));
}

// Sigmoid derivative: σ'(x) = σ(x) * (1 - σ(x))
double Network::sigmoidDerivative(double x) const {
    double s = sigmoid(x);
    return s * (1.0 - s);
}

// ReLU activation: ReLU(x) = max(0, x)
double Network::relu(double x) const {
    return std::max(0.0, x);
}

// ReLU derivative: ReLU'(x) = 1 if x > 0, else 0
double Network::reluDerivative(double x) const {
    return x > 0.0 ? 1.0 : 0.0;
}

// Get activation type for a specific layer
ActivationType Network::getLayerActivation(size_t layerIndex) const {
    if (layerIndex < layerActivations_.size()) {
        return layerActivations_[layerIndex];
    }
    return activation_;  // Fallback to default
}

// Apply activation function to matrix (with optional layer index)
Matrix Network::applyActivation(const Matrix& input, size_t layerIndex) const {
    Matrix result(input.getRows(), input.getCols());
    ActivationType act = (layerIndex != SIZE_MAX) ? getLayerActivation(layerIndex) : activation_;
    
    if (act == ActivationType::SIGMOID) {
        for (size_t i = 0; i < input.getRows(); ++i) {
            for (size_t j = 0; j < input.getCols(); ++j) {
                result(i, j) = sigmoid(input(i, j));
            }
        }
    } else {  // RELU
        for (size_t i = 0; i < input.getRows(); ++i) {
            for (size_t j = 0; j < input.getCols(); ++j) {
                result(i, j) = relu(input(i, j));
            }
        }
    }
    
    return result;
}

// Apply activation derivative to matrix (with optional layer index)
Matrix Network::applyActivationDerivative(const Matrix& input, size_t layerIndex) const {
    Matrix result(input.getRows(), input.getCols());
    ActivationType act = (layerIndex != SIZE_MAX) ? getLayerActivation(layerIndex) : activation_;
    
    if (act == ActivationType::SIGMOID) {
        for (size_t i = 0; i < input.getRows(); ++i) {
            for (size_t j = 0; j < input.getCols(); ++j) {
                result(i, j) = sigmoidDerivative(input(i, j));
            }
        }
    } else {  // RELU
        for (size_t i = 0; i < input.getRows(); ++i) {
            for (size_t j = 0; j < input.getCols(); ++j) {
                result(i, j) = reluDerivative(input(i, j));
            }
        }
    }
    
    return result;
}

// Initialize weights using He initialization (for ReLU) or Xavier (for Sigmoid)
void Network::initializeWeights() {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    for (size_t i = 0; i < numLayers_; ++i) {
        double fanIn = static_cast<double>(layerSizes_[i]);
        ActivationType act = getLayerActivation(i);
        
        double stdDev;
        if (act == ActivationType::RELU) {
            // He Initialization for ReLU: N(0, sqrt(2.0 / fan_in))
            // This prevents "dying ReLU" problem - neurons start with proper scale
            stdDev = std::sqrt(2.0 / fanIn);
        } else {
            // Xavier/Glorot initialization for Sigmoid: N(0, sqrt(2 / (fan_in + fan_out)))
            double fanOut = static_cast<double>(layerSizes_[i + 1]);
            stdDev = std::sqrt(2.0 / (fanIn + fanOut));
        }
        
        // Use normal distribution (not uniform) - better for ReLU
        std::normal_distribution<double> dis(0.0, stdDev);
        
        // Initialize weights
        for (size_t r = 0; r < weights_[i].getRows(); ++r) {
            for (size_t c = 0; c < weights_[i].getCols(); ++c) {
                weights_[i](r, c) = dis(gen);
            }
        }
        
        // Initialize biases to small random values (near zero for ReLU)
        std::normal_distribution<double> biasDis(0.0, 0.01);
        for (size_t r = 0; r < biases_[i].getRows(); ++r) {
            biases_[i](r, 0) = biasDis(gen);
        }
    }
}

// One-hot encode class index
Matrix Network::oneHotEncode(int classIndex, size_t numClasses) const {
    Matrix oneHot(numClasses, 1);
    oneHot.fill(0.0);
    if (classIndex >= 0 && classIndex < static_cast<int>(numClasses)) {
        oneHot(classIndex, 0) = 1.0;
    }
    return oneHot;
}

