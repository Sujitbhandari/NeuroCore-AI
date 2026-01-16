#!/usr/bin/env python3
"""
NeuroCore Visualization Script
Reads training logs and plots loss vs epoch curve.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_training_results():
    """
    Reads training_log.csv and plots loss vs epoch curve.
    """
    # Get the data directory path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    log_path = os.path.join(project_root, 'data', 'training_log.csv')
    
    # Check if log file exists
    if not os.path.exists(log_path):
        print(f"Error: Training log not found at {log_path}")
        print("   Make sure you've run the NeuroCore training first!")
        sys.exit(1)
    
    print(f"Loading training log from {log_path}...")
    
    try:
        # Read CSV file
        df = pd.read_csv(log_path)
        
        if df.empty:
            print("Error: Training log is empty")
            sys.exit(1)
        
        print(f"Loaded {len(df)} epochs of training data")
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Loss vs Epoch
        epochs = df['epoch'].values
        losses = df['loss'].values
        
        ax1.plot(epochs, losses, 'b-o', linewidth=2, markersize=8, label='Training Loss')
        ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Loss (MSE)', fontsize=12, fontweight='bold')
        ax1.set_title('Training Loss vs Epoch', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        ax1.set_xlim(left=0.5, right=len(epochs) + 0.5)
        
        # Plot 2: Accuracy vs Epoch
        if 'accuracy' in df.columns:
            accuracies = df['accuracy'].values * 100  # Convert to percentage
            
            ax2.plot(epochs, accuracies, 'g-s', linewidth=2, markersize=8, label='Test Accuracy')
            ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
            ax2.set_title('Test Accuracy vs Epoch', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=10)
            ax2.set_xlim(left=0.5, right=len(epochs) + 0.5)
            ax2.set_ylim(bottom=0, top=100)
            
            # Add value annotations
            for i, (epoch, acc) in enumerate(zip(epochs, accuracies)):
                ax2.annotate(f'{acc:.1f}%', 
                           (epoch, acc), 
                           textcoords="offset points", 
                           xytext=(0,10), 
                           ha='center',
                           fontsize=9)
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(project_root, 'data', 'training_curves.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
        
        # Display summary statistics
        print("\nTraining Summary:")
        print(f"   Initial Loss: {losses[0]:.6f}")
        print(f"   Final Loss: {losses[-1]:.6f}")
        print(f"   Loss Reduction: {((losses[0] - losses[-1]) / losses[0] * 100):.2f}%")
        
        if 'accuracy' in df.columns:
            print(f"   Initial Accuracy: {accuracies[0]:.2f}%")
            print(f"   Final Accuracy: {accuracies[-1]:.2f}%")
            print(f"   Accuracy Improvement: {accuracies[-1] - accuracies[0]:.2f}%")
        
        # Show plot
        print("\nDisplaying plot...")
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    try:
        plot_training_results()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
