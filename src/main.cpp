#include "../include/Network.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <random>
#include <numeric>

/**
 * Load MNIST data from CSV file
 * 
 * Format: Label, Pixel1, Pixel2, ..., Pixel784
 * 
 * @param filename Path to CSV file
 * @param inputs Output vector of input matrices (784x1 each)
 * @param targets Output vector of target matrices (10x1 one-hot encoded)
 * @param maxSamples Maximum number of samples to load (0 = all)
 */
void loadMNIST(const std::string& filename,
               std::vector<Matrix>& inputs,
               std::vector<Matrix>& targets,
               size_t maxSamples = 0) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    std::string line;
    size_t count = 0;
    
    std::cout << "Loading data from " << filename << "..." << std::endl;
    
    while (std::getline(file, line) && (maxSamples == 0 || count < maxSamples)) {
        std::istringstream iss(line);
        std::string token;
        std::vector<double> values;
        
        // Parse CSV line
        while (std::getline(iss, token, ',')) {
            values.push_back(std::stod(token));
        }
        
        if (values.size() < 785) {  // 1 label + 784 pixels
            continue;
        }
        
        // Extract label (first value)
        int label = static_cast<int>(values[0]);
        
        // Extract pixels (remaining 784 values)
        std::vector<double> pixels(values.begin() + 1, values.end());
        
        // Create input matrix (784x1)
        Matrix input = Matrix::fromVector(pixels, true);
        inputs.push_back(input);
        
        // Create one-hot encoded target (10x1)
        Matrix target(10, 1);
        target.fill(0.0);
        target(label, 0) = 1.0;
        targets.push_back(target);
        
        count++;
        
        if (count % 1000 == 0) {
            std::cout << "   Loaded " << count << " samples..." << std::endl;
        }
    }
    
    file.close();
    std::cout << "Loaded " << inputs.size() << " samples" << std::endl;
}

/**
 * Main Training Function
 */
int main() {
    std::cout << "NeuroCore: High-Performance Deep Learning Engine" << std::endl;
    std::cout << "===================================================" << std::endl;
    std::cout << std::endl;
    
    // Network architecture: 784 (input) -> 128 -> 10 (output)
    std::vector<size_t> layerSizes = {784, 128, 10};
    
    std::cout << "Building neural network..." << std::endl;
    std::cout << "   Architecture: ";
    for (size_t i = 0; i < layerSizes.size(); ++i) {
        std::cout << layerSizes[i];
        if (i < layerSizes.size() - 1) std::cout << " -> ";
    }
    std::cout << std::endl;
    
    // Create network with ReLU for hidden layers, Sigmoid for output
    Network network(layerSizes, ActivationType::RELU, 0.001);
    std::cout << "   Hidden Activation: ReLU" << std::endl;
    std::cout << "   Output Activation: Sigmoid" << std::endl;
    std::cout << "   Learning Rate: 0.001" << std::endl;
    std::cout << "   Weight Init: He initialization" << std::endl;
    std::cout << std::endl;
    
    // Load training data
    std::vector<Matrix> trainInputs, trainTargets;
    try {
        loadMNIST("data/mnist_train.csv", trainInputs, trainTargets, 10000);  // Use 10k samples for faster training
    } catch (const std::exception& e) {
        std::cerr << "Error loading training data: " << e.what() << std::endl;
        std::cerr << "   Make sure you've run: python scripts/data_loader.py" << std::endl;
        return 1;
    }
    
    // Load test data
    std::vector<Matrix> testInputs, testTargets;
    try {
        loadMNIST("data/mnist_test.csv", testInputs, testTargets, 1000);  // Use 1k samples for testing
    } catch (const std::exception& e) {
        std::cerr << "Warning: Could not load test data: " << e.what() << std::endl;
        std::cerr << "   Continuing with training only..." << std::endl;
    }
    
    // Training log file
    std::ofstream logFile("data/training_log.csv");
    if (logFile.is_open()) {
        logFile << "epoch,loss,accuracy\n";
    }
    
    // Training loop
    const int numEpochs = 20;
    std::cout << "Starting training for " << numEpochs << " epochs..." << std::endl;
    std::cout << std::endl;
    
    for (int epoch = 0; epoch < numEpochs; ++epoch) {
        std::cout << "Epoch " << (epoch + 1) << "/" << numEpochs << std::endl;
        
        double totalLoss = 0.0;
        size_t numSamples = trainInputs.size();
        
        // Shuffle data indices each epoch
        std::vector<size_t> indices(numSamples);
        std::iota(indices.begin(), indices.end(), 0);
        
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);
        
        // Train on all samples (SGD: one sample at a time)
        for (size_t idx = 0; idx < numSamples; ++idx) {
            size_t i = indices[idx];  // Use shuffled index
            double loss = network.trainSample(trainInputs[i], trainTargets[i]);
            totalLoss += loss;
            
            // Progress indicator
            if ((idx + 1) % 1000 == 0) {
                std::cout << "   Processed " << (idx + 1) << "/" << numSamples 
                          << " samples (Loss: " << std::fixed << std::setprecision(6) 
                          << (totalLoss / (idx + 1)) << ")" << std::endl;
            }
        }
        
        double avgLoss = totalLoss / numSamples;
        
        // Evaluate on test set
        double accuracy = 0.0;
        if (!testInputs.empty()) {
            accuracy = network.evaluate(testInputs, testTargets);
            std::cout << "   Test Accuracy: " << std::fixed << std::setprecision(4) 
                      << (accuracy * 100.0) << "%" << std::endl;
        } else {
            // Evaluate on training set if no test set
            accuracy = network.evaluate(trainInputs, trainTargets);
            std::cout << "   Training Accuracy: " << std::fixed << std::setprecision(4) 
                      << (accuracy * 100.0) << "%" << std::endl;
        }
        
        std::cout << "   Average Loss: " << std::fixed << std::setprecision(6) 
                  << avgLoss << std::endl;
        std::cout << std::endl;
        
        // Log to file
        if (logFile.is_open()) {
            logFile << (epoch + 1) << "," << avgLoss << "," << accuracy << "\n";
        }
    }
    
    logFile.close();
    
    // Save trained model
    std::cout << "Saving trained model..." << std::endl;
    try {
        network.saveModel("data/neurocore_model.json");
        std::cout << "Model saved to data/neurocore_model.json" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Warning: Could not save model: " << e.what() << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << "Training complete!" << std::endl;
    std::cout << "Training log saved to data/training_log.csv" << std::endl;
    std::cout << "   Run: python scripts/visualize.py to see the loss curve" << std::endl;
    
    return 0;
}

