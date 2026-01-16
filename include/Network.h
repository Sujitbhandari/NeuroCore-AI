#ifndef NETWORK_H
#define NETWORK_H

#include "Matrix.h"
#include <vector>
#include <string>
#include <fstream>

/**
 * Activation Function Type
 */
enum class ActivationType {
    SIGMOID,
    RELU
};

/**
 * Neural Network Class
 * 
 * A fully-connected feedforward neural network with:
 * - Variable number of hidden layers
 * - Configurable activation functions (Sigmoid, ReLU)
 * - Manual backpropagation with Stochastic Gradient Descent (SGD)
 * - Model serialization support
 */
class Network {
public:
    /**
     * Constructor
     * 
     * @param layerSizes Vector specifying the number of neurons in each layer
     *                   Example: {784, 128, 64, 10} for MNIST
     * @param activation Activation function type (SIGMOID or RELU)
     * @param learningRate Learning rate for SGD
     */
    Network(const std::vector<size_t>& layerSizes, 
            ActivationType activation = ActivationType::SIGMOID,
            double learningRate = 0.01);
    
    /**
     * Forward Pass
     * 
     * Propagates input through the network to produce output.
     * 
     * @param input Input vector (flattened image pixels)
     * @return Output vector (class probabilities)
     */
    Matrix forward(const Matrix& input);
    
    /**
     * Backward Pass (Backpropagation)
     * 
     * Computes gradients using the chain rule and updates weights via SGD.
     * 
     * Mathematical Foundation:
     * 
     * 1. Loss Function: Mean Squared Error (MSE)
     *    L = (1/2) * ||y_pred - y_true||²
     * 
     * 2. Chain Rule for Gradients:
     *    ∂L/∂W = ∂L/∂a * ∂a/∂z * ∂z/∂W
     *    where:
     *      - W: weights
     *      - z: pre-activation (weighted sum)
     *      - a: post-activation (activation function output)
     * 
     * 3. Backpropagation Algorithm:
     *    a) Compute output error: δ_output = (y_pred - y_true) * σ'(z_output)
     *    b) Propagate error backward: δ_l = (W_l+1^T * δ_l+1) * σ'(z_l)
     *    c) Update weights: W_l = W_l - α * δ_l * a_l-1^T
     *       where α is the learning rate
     * 
     * @param input Input vector
     * @param target Target output (one-hot encoded)
     * @return Loss value
     */
    double backward(const Matrix& input, const Matrix& target);
    
    /**
     * Train on a single sample
     * 
     * @param input Input vector
     * @param target Target output (one-hot encoded)
     * @return Loss value
     */
    double trainSample(const Matrix& input, const Matrix& target);
    
    /**
     * Predict class for input
     * 
     * @param input Input vector
     * @return Predicted class index
     */
    int predict(const Matrix& input);
    
    /**
     * Evaluate accuracy on dataset
     * 
     * @param inputs Vector of input matrices
     * @param targets Vector of target matrices (one-hot encoded)
     * @return Accuracy (0.0 to 1.0)
     */
    double evaluate(const std::vector<Matrix>& inputs, 
                    const std::vector<Matrix>& targets);
    
    /**
     * Save model weights to file
     * 
     * Format: JSON-like text file with weights and biases for each layer
     * 
     * @param filename Output file path
     */
    void saveModel(const std::string& filename) const;
    
    /**
     * Load model weights from file
     * 
     * @param filename Input file path
     */
    void loadModel(const std::string& filename);
    
    // Getters
    size_t getNumLayers() const { return numLayers_; }
    const std::vector<Matrix>& getWeights() const { return weights_; }
    const std::vector<Matrix>& getBiases() const { return biases_; }

private:
    // Network architecture
    std::vector<size_t> layerSizes_;
    size_t numLayers_;
    ActivationType activation_;  // Default activation (for backward compatibility)
    double learningRate_;
    
    // Per-layer activation functions (optional: defaults to activation_)
    std::vector<ActivationType> layerActivations_;  // activation_[i] for layer i
    
    // Weights and biases
    // weights_[i] connects layer i to layer i+1
    // biases_[i] is the bias vector for layer i+1
    std::vector<Matrix> weights_;
    std::vector<Matrix> biases_;
    
    // Forward pass storage (for backpropagation)
    std::vector<Matrix> activations_;  // Post-activation values
    std::vector<Matrix> zValues_;      // Pre-activation values (weighted sums)
    
    // Activation functions
    double sigmoid(double x) const;
    double sigmoidDerivative(double x) const;
    double relu(double x) const;
    double reluDerivative(double x) const;
    
    // Apply activation function to matrix (with optional layer index)
    Matrix applyActivation(const Matrix& input, size_t layerIndex = SIZE_MAX) const;
    Matrix applyActivationDerivative(const Matrix& input, size_t layerIndex = SIZE_MAX) const;
    
    // Get activation type for a specific layer
    ActivationType getLayerActivation(size_t layerIndex) const;
    
    // Initialize weights (Xavier/Glorot initialization)
    void initializeWeights();
    
    // Helper: Convert class index to one-hot vector
    Matrix oneHotEncode(int classIndex, size_t numClasses) const;
};

#endif // NETWORK_H

