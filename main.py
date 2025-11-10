"""
Main entry script
Used to run Hopfield network and RBM experiments
"""

import os
import argparse
import time
import numpy as np
import torch

from src.hopfield.experiments import HopfieldExperiments
from src.boltzmann.experiments import RBMExperiments
import config


def run_hopfield_experiments():
    """Run Hopfield network experiments"""
    print("=" * 50)
    print("Running Hopfield Network Experiments")
    print("=" * 50)
    
    # Create experiment object
    hopfield_exp = HopfieldExperiments()
    
    # Run all experiments
    results = hopfield_exp.run_all_experiments()
    
    print("\nHopfield Network Experiments completed!")
    print(f"Results saved to: {config.RESULTS_DIR}/hopfield")
    
    return results


def run_rbm_experiments():
    """Run RBM experiments"""
    print("=" * 50)
    print("Running RBM Experiments")
    print("=" * 50)
    
    # Create experiment object
    rbm_exp = RBMExperiments()
    
    # Run all experiments
    results = rbm_exp.run_all_experiments()
    
    print("\nRBM Experiments completed!")
    print(f"Results saved to: {config.RESULTS_DIR}/rbm")
    
    return results


def run_hopfield_memory_test():
    """Run Hopfield network memory test"""
    print("=" * 50)
    print("Running Hopfield Network Memory Test")
    print("=" * 50)
    
    # Create experiment object
    hopfield_exp = HopfieldExperiments()
    
    # Load MNIST data
    patterns = hopfield_exp.loader.get_specific_digit_samples(
        digits=[0, 1, 2, 3, 4], 
        num_samples=config.HOPFIELD_CONFIG['num_patterns'],
        binary_values={-1, 1}
    )
    
    # Train network
    print(f"Training Hopfield network with {len(patterns)} patterns...")
    hopfield_exp.network.train(patterns)
    
    # Test memory recall
    print("Testing memory recall...")
    for i, pattern in enumerate(patterns):
        # Add noise
        noise_level = 0.2
        noisy_pattern = hopfield_exp.loader.add_noise(pattern, noise_level, binary_values={-1, 1})
        
        # Recover pattern
        recovered_pattern, energy_history = hopfield_exp.network.recover(
            noisy_pattern, 
            max_iterations=config.HOPFIELD_CONFIG['max_iterations'],
            mode='asynchronous'
        )
        
        # Calculate similarity
        similarity = hopfield_exp.network.calculate_similarity(pattern, recovered_pattern)
        
        print(f"Pattern {i+1}: Original vs Recovered similarity = {similarity:.4f}")
        
        # Save results
        hopfield_exp.visualizer.plot_recovery_comparison(
            pattern, noisy_pattern, recovered_pattern, 
            save_path=os.path.join(config.RESULTS_DIR, 'hopfield', f'memory_test_{i}.png')
        )
    
    print("\nHopfield Network Memory Test completed!")


def run_rbm_generation_test():
    """Run RBM generation test"""
    print("=" * 50)
    print("Running RBM Generation Test")
    print("=" * 50)
    
    # Create experiment object
    rbm_exp = RBMExperiments()
    
    # Train RBM
    print("Training RBM...")
    training_results = rbm_exp.train_rbm(n_epochs=30, batch_size=64)
    
    # Generate samples
    print("Generating samples...")
    generation_results = rbm_exp.generate_samples(n_samples=64, n_gibbs_steps=1000)
    
    # Analyze weights
    print("Analyzing weights...")
    weight_results = rbm_exp.analyze_weights()
    
    # Visualize features
    print("Visualizing learned features...")
    feature_results = rbm_exp.visualize_features(n_samples=1000)
    
    print("\nRBM Generation Test completed!")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Hopfield Network and RBM Experiments')
    parser.add_argument('--mode', type=str, default='all', 
                        choices=['hopfield', 'rbm', 'hopfield_test', 'rbm_test', 'all'],
                        help='Experiment mode to run')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='Device to use for computation')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Run experiments based on mode
    start_time = time.time()
    
    if args.mode == 'hopfield' or args.mode == 'all':
        run_hopfield_experiments()
    
    if args.mode == 'rbm' or args.mode == 'all':
        run_rbm_experiments()
    
    if args.mode == 'hopfield_test':
        run_hopfield_memory_test()
    
    if args.mode == 'rbm_test':
        run_rbm_generation_test()
    
    total_time = time.time() - start_time
    print(f"\nAll experiments completed in {total_time:.2f} seconds!")


if __name__ == "__main__":
    main()