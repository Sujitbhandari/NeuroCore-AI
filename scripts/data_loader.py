#!/usr/bin/env python3
"""
NeuroCore Data Loader
Downloads MNIST dataset and converts it to CSV format for C++ engine consumption.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

def download_and_prepare_mnist():
    """
    Downloads MNIST dataset, normalizes pixel values (0-1), and saves as CSV.
    Format: Headerless CSV with columns: Label, Pixel1, Pixel2, ..., Pixel784
    """
    print("Downloading MNIST dataset from OpenML...")
    
    # Download MNIST dataset
    # as_frame=False returns numpy arrays (faster for large datasets)
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    
    X, y = mnist.data, mnist.target.astype(int)
    
    print(f"Downloaded {len(X)} samples")
    print(f"   Image shape: {X.shape[1]} pixels (28x28)")
    print(f"   Labels: {len(np.unique(y))} classes (0-9)")
    
    # Normalize pixel values from [0, 255] to [0, 1]
    print("Normalizing pixel values to [0, 1]...")
    X_normalized = X.astype(np.float64) / 255.0
    
    # Combine label and features into a single array
    # Format: [Label, Pixel1, Pixel2, ..., Pixel784]
    print("Preparing CSV format...")
    data = np.column_stack([y, X_normalized])
    
    # Create data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Save training data (first 60,000 samples)
    train_data = data[:60000]
    train_path = os.path.join(data_dir, 'mnist_train.csv')
    print(f"Saving training data to {train_path}...")
    np.savetxt(train_path, train_data, delimiter=',', fmt='%.6f')
    print(f"Saved {len(train_data)} training samples")
    
    # Save test data (last 10,000 samples)
    test_data = data[60000:]
    test_path = os.path.join(data_dir, 'mnist_test.csv')
    print(f"Saving test data to {test_path}...")
    np.savetxt(test_path, test_data, delimiter=',', fmt='%.6f')
    print(f"Saved {len(test_data)} test samples")
    
    print("\nData preparation complete!")
    print(f"   Training file: {train_path}")
    print(f"   Test file: {test_path}")
    print("\nCSV Format: Label, Pixel1, Pixel2, ..., Pixel784 (headerless)")

if __name__ == "__main__":
    try:
        download_and_prepare_mnist()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
