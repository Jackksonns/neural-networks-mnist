"""
Hopfield network experiment scripts
Pattern storage and recovery experiments
Noise interference experiments
Capacity analysis experiments
"""

import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
from sklearn.metrics import confusion_matrix

from .network import HopfieldNetwork
from .visualizer import HopfieldVisualizer
from ..utils.data_loader import MNISTLoader
from ..utils.preprocessing import add_gaussian_noise, add_binary_noise, hamming_distance
import config


class HopfieldExperiments:
    """
    Hopfield network experiment class
    """
    
    def __init__(self):
        """Initialize experiment environment"""
        self.loader = MNISTLoader()
        self.network = HopfieldNetwork(config.HOPFIELD_CONFIG['n_neurons'])
        self.visualizer = HopfieldVisualizer()
        self.results = {}
        
        # Create results save directory
        os.makedirs(os.path.join(config.RESULTS_DIR, 'hopfield'), exist_ok=True)
    
    def run_pattern_storage_experiment(self, digits: List[int] = [0, 1, 2]) -> Dict:
        """
        Pattern storage and recovery experiment
        
        Args:
            digits: List of digits to store
            
        Returns:
            Experiment result dictionary
        """
        print(f"Running pattern storage experiment for digits: {digits}")
        
        # Load samples of specified digits
        patterns = []
        for digit in digits:
            sample = self.loader.get_specific_digit_sample(digit, binary_values={-1, 1})
            patterns.append(sample)
        
        patterns_tensor = torch.stack(patterns)
        
        # Train network
        self.network.train(patterns_tensor)
        
        # Test perfect recovery
        perfect_recovery_results = self._test_perfect_recovery(patterns_tensor)
        
        # Save results
        results = {
            'digits': digits,
            'patterns': patterns_tensor,
            'perfect_recovery': perfect_recovery_results
        }
        
        # Visualize results
        self._visualize_storage_results(results)
        
        print(f"Pattern storage experiment completed. Perfect recovery rate: {perfect_recovery_results['recovery_rate']:.2f}")
        return results
    
    def run_noise_interference_experiment(self, digits: List[int] = [0, 1, 2],
                                         noise_levels: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5]) -> Dict:
        """
        Noise interference experiment
        
        Args:
            digits: List of digits to store
            noise_levels: List of noise levels
            
        Returns:
            Experiment result dictionary
        """
        print(f"Running noise interference experiment for digits: {digits}")
        
        # Load samples of specified digits
        patterns = []
        for digit in digits:
            sample = self.loader.get_specific_digit_sample(digit, binary_values={-1, 1})
            patterns.append(sample)
        
        patterns_tensor = torch.stack(patterns)
        
        # Train network
        self.network.train(patterns_tensor)
        
        # Test recovery under different noise levels
        recovery_results = {}
        
        for noise_level in noise_levels:
            print(f"Testing noise level: {noise_level}")
            
            # Add noise to each pattern
            noisy_patterns = []
            for pattern in patterns_tensor:
                noisy = add_binary_noise(pattern, noise_level)
                noisy_patterns.append(noisy)
            
            noisy_tensor = torch.stack(noisy_patterns)
            
            # Recover patterns
            recovered_tensor = []
            for noisy in noisy_tensor:
                recovered, energy_history = self.network.recover(noisy, max_iter=100)
                recovered_tensor.append(recovered)
            
            recovered_tensor = torch.stack(recovered_tensor)
            
            # Calculate recovery rate
            recovery_rate = self._calculate_recovery_rate(patterns_tensor, recovered_tensor)
            
            recovery_results[noise_level] = {
                'noisy_patterns': noisy_tensor,
                'recovered_patterns': recovered_tensor,
                'recovery_rate': recovery_rate
            }
            
            # Visualize recovery results for each noise level
            self._visualize_noise_recovery_results(
                patterns_tensor, noisy_tensor, recovered_tensor, 
                noise_level, recovery_rate
            )
        
        # Plot noise-recovery rate curve
        self._plot_noise_recovery_curve(recovery_results)
        
        # Save results
        results = {
            'digits': digits,
            'patterns': patterns_tensor,
            'recovery_results': recovery_results
        }
        
        print(f"Noise interference experiment completed.")
        return results
    
    def run_capacity_analysis_experiment(self, max_patterns: int = 20) -> Dict:
        """
        Capacity analysis experiment
        
        Args:
            max_patterns: Maximum number of patterns
            
        Returns:
            Experiment result dictionary
        """
        print(f"Running capacity analysis experiment with max patterns: {max_patterns}")
        
        # Test with different numbers of patterns
        pattern_counts = list(range(1, max_patterns + 1))
        recovery_rates = []
        
        for n_patterns in pattern_counts:
            print(f"Testing with {n_patterns} patterns")
            
            # Load random digit samples
            patterns = []
            for i in range(n_patterns):
                digit = i % 10  # Cycle through digits 0-9
                sample = self.loader.get_specific_digit_sample(digit, binary_values={-1, 1})
                patterns.append(sample)
            
            patterns_tensor = torch.stack(patterns)
            
            # Train network
            self.network.train(patterns_tensor)
            
            # Test perfect recovery
            recovered_tensor = []
            for pattern in patterns_tensor:
                recovered, _ = self.network.recover(pattern, max_iter=100)
                recovered_tensor.append(recovered)
            
            recovered_tensor = torch.stack(recovered_tensor)
            
            # Calculate recovery rate
            recovery_rate = self._calculate_recovery_rate(patterns_tensor, recovered_tensor)
            recovery_rates.append(recovery_rate)
        
        # Plot capacity analysis curve
        self._plot_capacity_analysis_curve(pattern_counts, recovery_rates)
        
        # Calculate theoretical capacity
        theoretical_capacity = 0.138 * self.network.n_neurons  # Empirical formula: 0.138N
        
        # Save results
        results = {
            'pattern_counts': pattern_counts,
            'recovery_rates': recovery_rates,
            'theoretical_capacity': theoretical_capacity
        }
        
        print(f"Capacity analysis experiment completed. Theoretical capacity: {theoretical_capacity:.2f}")
        return results
    
    def run_attractor_analysis_experiment(self, digits: List[int] = [0, 1, 2]) -> Dict:
        """
        Attractor analysis experiment
        
        Args:
            digits: List of digits to store
            
        Returns:
            Experiment result dictionary
        """
        print(f"Running attractor analysis experiment for digits: {digits}")
        
        # Load samples of specified digits
        patterns = []
        for digit in digits:
            sample = self.loader.get_specific_digit_sample(digit, binary_values={-1, 1})
            patterns.append(sample)
        
        patterns_tensor = torch.stack(patterns)
        
        # Train network
        self.network.train(patterns_tensor)
        
        # Find attractors
        attractors, basins = self.network.find_attractors(random_samples=100)
        
        # Analyze attractors for each stored pattern
        attractor_analysis = {}
        for i, pattern in enumerate(patterns_tensor):
            # Find the attractor closest to the current pattern
            min_distance = float('inf')
            closest_attractor = None
            
            for attractor in attractors:
                distance = hamming_distance(pattern, attractor)
                if distance < min_distance:
                    min_distance = distance
                    closest_attractor = attractor
            
            attractor_analysis[i] = {
                'pattern': pattern,
                'closest_attractor': closest_attractor,
                'distance': min_distance
            }
        
        # Visualize attractors
        self._visualize_attractors(attractors, patterns_tensor)
        
        # Save results
        results = {
            'digits': digits,
            'patterns': patterns_tensor,
            'attractors': attractors,
            'basins': basins,
            'attractor_analysis': attractor_analysis
        }
        
        print(f"Attractor analysis experiment completed. Found {len(attractors)} attractors.")
        return results
    
    def _test_perfect_recovery(self, patterns: torch.Tensor) -> Dict:
        """
        Test perfect recovery
        
        Args:
            patterns: Stored patterns
            
        Returns:
            Perfect recovery result dictionary
        """
        recovered_tensor = []
        energy_histories = []
        
        for pattern in patterns:
            recovered, energy_history = self.network.recover(pattern, max_iter=100)
            recovered_tensor.append(recovered)
            energy_histories.append(energy_history)
        
        recovered_tensor = torch.stack(recovered_tensor)
        
        # Calculate recovery rate
        recovery_rate = self._calculate_recovery_rate(patterns, recovered_tensor)
        
        return {
            'recovered_patterns': recovered_tensor,
            'energy_histories': energy_histories,
            'recovery_rate': recovery_rate
        }
    
    def _calculate_recovery_rate(self, original: torch.Tensor, recovered: torch.Tensor) -> float:
        """
        Calculate recovery rate
        
        Args:
            original: Original patterns
            recovered: Recovered patterns
            
        Returns:
            Recovery rate
        """
        n_patterns = original.shape[0]
        n_recovered = 0
        
        for i in range(n_patterns):
            if torch.equal(original[i], recovered[i]):
                n_recovered += 1
        
        return n_recovered / n_patterns
    
    def _visualize_storage_results(self, results: Dict) -> None:
        """
        Visualize storage results
        
        Args:
            results: Storage experiment results
        """
        patterns = results['patterns']
        perfect_recovery = results['perfect_recovery']
        
        # Plot comparison of original and recovered patterns
        for i, pattern in enumerate(patterns):
            recovered = perfect_recovery['recovered_patterns'][i]
            energy_history = perfect_recovery['energy_histories'][i]
            
            # Pattern comparison
            save_path = os.path.join(
                config.RESULTS_DIR, 'hopfield', 'recovered_images', 
                f'digit_{results["digits"][i]}_comparison.png'
            )
            self.visualizer.plot_recovery_comparison(
                pattern, pattern, recovered, save_path=save_path,
                title=f"Digit {results['digits'][i]} Recovery"
            )
            
            # Energy trajectory
            save_path = os.path.join(
                config.RESULTS_DIR, 'hopfield', 'energy_plots', 
                f'digit_{results["digits"][i]}_energy.png'
            )
            self.visualizer.plot_energy_trajectory(
                energy_history, save_path=save_path,
                title=f"Digit {results['digits'][i]} Energy Trajectory"
            )
    
    def _visualize_noise_recovery_results(self, patterns: torch.Tensor, 
                                         noisy: torch.Tensor, recovered: torch.Tensor,
                                         noise_level: float, recovery_rate: float) -> None:
        """
        Visualize noise recovery results
        
        Args:
            patterns: Original patterns
            noisy: Noisy patterns
            recovered: Recovered patterns
            noise_level: Noise level
            recovery_rate: Recovery rate
        """
        save_path = os.path.join(
            config.RESULTS_DIR, 'hopfield', 'interference_analysis', 
            f'noise_level_{noise_level:.1f}.png'
        )
        
        # Visualize only the first pattern
        self.visualizer.plot_recovery_comparison(
            patterns[0], noisy[0], recovered[0], save_path=save_path,
            title=f"Noise Level {noise_level:.1f}, Recovery Rate: {recovery_rate:.2f}"
        )
    
    def _plot_noise_recovery_curve(self, recovery_results: Dict) -> None:
        """
        Plot noise-recovery rate curve
        
        Args:
            recovery_results: Recovery result dictionary
        """
        noise_levels = list(recovery_results.keys())
        recovery_rates = [recovery_results[nl]['recovery_rate'] for nl in noise_levels]
        
        save_path = os.path.join(
            config.RESULTS_DIR, 'hopfield', 'interference_analysis', 
            'noise_recovery_curve.png'
        )
        
        self.visualizer.plot_noise_recovery_curves(
            noise_levels, recovery_rates, save_path=save_path,
            title="Noise Recovery Performance"
        )
    
    def _plot_capacity_analysis_curve(self, pattern_counts: List[int], 
                                      recovery_rates: List[float]) -> None:
        """
        Plot capacity analysis curve
        
        Args:
            pattern_counts: List of pattern counts
            recovery_rates: List of recovery rates
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot curve
        ax.plot(pattern_counts, recovery_rates, 'bo-', linewidth=2, markersize=8)
        
        # Plot theoretical capacity line
        theoretical_capacity = 0.138 * self.network.n_neurons
        ax.axvline(x=theoretical_capacity, color='r', linestyle='--', 
                  label=f"Theoretical Capacity: {theoretical_capacity:.2f}")
        
        # Set chart properties
        ax.set_xlabel('Number of Patterns', fontsize=12)
        ax.set_ylabel('Recovery Rate', fontsize=12)
        ax.set_title('Hopfield Network Capacity Analysis', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        ax.legend()
        
        # Save chart
        save_path = os.path.join(
            config.RESULTS_DIR, 'hopfield', 'interference_analysis', 
            'capacity_analysis.png'
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _visualize_attractors(self, attractors: List[torch.Tensor], 
                             patterns: torch.Tensor) -> None:
        """
        Visualize attractors
        
        Args:
            attractors: List of attractors
            patterns: Stored patterns
        """
        # Create figure
        n_attractors = len(attractors)
        n_patterns = patterns.shape[0]
        
        fig, axes = plt.subplots(2, max(n_attractors, n_patterns), 
                               figsize=(max(n_attractors, n_patterns) * 2.5, 5))
        
        if n_attractors == 1:
            axes = axes.reshape(1, -1)
        
        # Display stored patterns
        for i in range(n_patterns):
            pattern_np = patterns[i].detach().cpu().numpy().reshape(28, 28)
            axes[0, i].imshow(pattern_np, cmap='binary')
            axes[0, i].set_title(f'Stored Pattern {i}', fontsize=10)
            axes[0, i].axis('off')
        
        # Display attractors
        for i in range(n_attractors):
            attractor_np = attractors[i].detach().cpu().numpy().reshape(28, 28)
            axes[1, i].imshow(attractor_np, cmap='binary')
            axes[1, i].set_title(f'Attractor {i}', fontsize=10)
            axes[1, i].axis('off')
        
        # Hide extra subplots
        for i in range(n_patterns, axes.shape[1]):
            axes[0, i].axis('off')
        
        for i in range(n_attractors, axes.shape[1]):
            axes[1, i].axis('off')
        
        # Set overall title
        fig.suptitle('Stored Patterns and Attractors', fontsize=14)
        plt.tight_layout()
        
        # Save chart
        save_path = os.path.join(
            config.RESULTS_DIR, 'hopfield', 'interference_analysis', 
            'attractors.png'
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_all_experiments(self) -> Dict:
        """
        Run all experiments
        
        Returns:
            Dictionary of all experiment results
        """
        print("Running all Hopfield network experiments...")
        
        # Pattern storage and recovery experiment
        storage_results = self.run_pattern_storage_experiment()
        
        # Noise interference experiment
        noise_results = self.run_noise_interference_experiment()
        
        # Capacity analysis experiment
        capacity_results = self.run_capacity_analysis_experiment()
        
        # Attractor analysis experiment
        attractor_results = self.run_attractor_analysis_experiment()
        
        # Save all results
        all_results = {
            'storage': storage_results,
            'noise': noise_results,
            'capacity': capacity_results,
            'attractor': attractor_results
        }
        
        print("All Hopfield network experiments completed.")
        return all_results