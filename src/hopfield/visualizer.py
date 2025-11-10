"""
Hopfield network visualization tools
Dynamic energy curve plotting
2D/3D energy landscape projection (PCA dimensionality reduction)
Frame-by-frame display of image recovery process
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from typing import List, Tuple, Optional
from ..network import HopfieldNetwork


class HopfieldVisualizer:
    """
    Hopfield network visualization utility class
    """
    
    @staticmethod
    def plot_energy_trajectory(energy_history: List[float], 
                              save_path: Optional[str] = None,
                              title: str = "Hopfield Network Energy Trajectory") -> plt.Figure:
        """
        Plot energy-time curve
        
        Args:
            energy_history: List of energy history
            save_path: Save path
            title: Chart title
            
        Returns:
            matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot energy curve
        ax.plot(energy_history, 'b-', linewidth=2)
        ax.scatter(range(len(energy_history)), energy_history, c='blue', s=20)
        
        # Set chart properties
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Energy', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Save chart
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_energy_landscape_2d(network: HopfieldNetwork, 
                                patterns: torch.Tensor,
                                n_samples: int = 1000,
                                save_path: Optional[str] = None,
                                title: str = "2D Energy Landscape") -> plt.Figure:
        """
        2D energy landscape projection:
        1. Random sampling in state space
        2. Calculate energy at each point
        3. PCA dimensionality reduction to 2D
        4. Draw contour plot
        
        Args:
            network: Hopfield network
            patterns: Stored patterns
            n_samples: Number of sample points
            save_path: Save path
            title: Chart title
            
        Returns:
            matplotlib figure object
        """
        # Generate random state samples
        n_neurons = network.n_neurons
        random_states = torch.randint(0, 2, (n_samples, n_neurons)) * 2 - 1  # Values are {-1, 1}
        
        # Calculate energy at each state point
        energies = []
        for state in random_states:
            energy = network.energy(state)
            energies.append(energy)
        
        # Convert to numpy array
        states_np = random_states.detach().cpu().numpy()
        energies_np = np.array(energies)
        
        # PCA dimensionality reduction to 2D
        pca = PCA(n_components=2)
        states_2d = pca.fit_transform(states_np)
        
        # Also apply PCA to stored patterns
        patterns_np = patterns.detach().cpu().numpy()
        patterns_2d = pca.transform(patterns_np)
        
        # Create grid for contour plot
        x_min, x_max = states_2d[:, 0].min() - 1, states_2d[:, 0].max() + 1
        y_min, y_max = states_2d[:, 1].min() - 1, states_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        
        # Interpolate energy values to grid
        from scipy.interpolate import griddata
        grid_z = griddata(states_2d, energies_np, (xx, yy), method='cubic')
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Draw contour plot
        contour = ax.contourf(xx, yy, grid_z, levels=20, cmap='viridis', alpha=0.8)
        fig.colorbar(contour, ax=ax, label='Energy')
        
        # Plot positions of stored patterns
        ax.scatter(patterns_2d[:, 0], patterns_2d[:, 1], 
                  c='red', s=100, marker='*', label='Stored Patterns', edgecolors='white')
        
        # Set chart properties
        ax.set_xlabel('Principal Component 1', fontsize=12)
        ax.set_ylabel('Principal Component 2', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        
        # Save chart
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_convergence_process(states_sequence: List[torch.Tensor],
                                grid_size: Tuple[int, int] = (28, 28),
                                save_path: Optional[str] = None,
                                title: str = "Convergence Process") -> plt.Figure:
        """
        Frame-by-frame display of state evolution
        
        Args:
            states_sequence: List of state sequence
            grid_size: Image grid size
            save_path: Save path
            title: Chart title
            
        Returns:
            matplotlib figure object
        """
        # Calculate subplot layout
        n_states = len(states_sequence)
        n_cols = min(5, n_states)
        n_rows = (n_states + n_cols - 1) // n_cols
        
        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        # Display each state
        for i, state in enumerate(states_sequence):
            if i >= len(axes):
                break
                
            # Convert to image format
            state_np = state.detach().cpu().numpy().reshape(grid_size)
            
            # Display image
            axes[i].imshow(state_np, cmap='binary')
            axes[i].set_title(f'Step {i}', fontsize=10)
            axes[i].axis('off')
        
        # Hide extra subplots
        for i in range(n_states, len(axes)):
            axes[i].axis('off')
        
        # Set overall title
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        # Save chart
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_recovery_comparison(original: torch.Tensor,
                                noisy: torch.Tensor,
                                recovered: torch.Tensor,
                                grid_size: Tuple[int, int] = (28, 28),
                                save_path: Optional[str] = None,
                                title: str = "Pattern Recovery Comparison") -> plt.Figure:
        """
        Side-by-side comparison of original/corrupted/recovered images
        
        Args:
            original: Original image
            noisy: Noisy image
            recovered: Recovered image
            grid_size: Image grid size
            save_path: Save path
            title: Chart title
            
        Returns:
            matplotlib figure object
        """
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # Convert to image format
        original_np = original.detach().cpu().numpy().reshape(grid_size)
        noisy_np = noisy.detach().cpu().numpy().reshape(grid_size)
        recovered_np = recovered.detach().cpu().numpy().reshape(grid_size)
        
        # Display images
        axes[0].imshow(original_np, cmap='binary')
        axes[0].set_title('Original', fontsize=12)
        axes[0].axis('off')
        
        axes[1].imshow(noisy_np, cmap='binary')
        axes[1].set_title('Noisy', fontsize=12)
        axes[1].axis('off')
        
        axes[2].imshow(recovered_np, cmap='binary')
        axes[2].set_title('Recovered', fontsize=12)
        axes[2].axis('off')
        
        # Set overall title
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        # Save chart
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_weight_matrix(network: HopfieldNetwork,
                          save_path: Optional[str] = None,
                          title: str = "Hopfield Network Weight Matrix") -> plt.Figure:
        """
        Visualize weight matrix
        
        Args:
            network: Hopfield network
            save_path: Save path
            title: Chart title
            
        Returns:
            matplotlib figure object
        """
        if not network.is_trained:
            raise ValueError("Network must be trained before visualizing weights")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Draw weight matrix heatmap
        sns.heatmap(network.weights, cmap='coolwarm', center=0,
                   ax=ax, cbar_kws={'label': 'Weight Value'})
        
        # Set chart properties
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Neuron Index', fontsize=12)
        ax.set_ylabel('Neuron Index', fontsize=12)
        
        # Save chart
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_weight_eigenvalues(network: HopfieldNetwork,
                              save_path: Optional[str] = None,
                              title: str = "Weight Matrix Eigenvalue Spectrum") -> plt.Figure:
        """
        Plot eigenvalue spectrum of weight matrix
        
        Args:
            network: Hopfield network
            save_path: Save path
            title: Chart title
            
        Returns:
            matplotlib figure object
        """
        if not network.is_trained:
            raise ValueError("Network must be trained before analyzing eigenvalues")
        
        # Calculate eigenvalues
        eigenvalues = np.linalg.eigvals(network.weights)
        real_parts = np.real(eigenvalues)
        imag_parts = np.imag(eigenvalues)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot eigenvalue distribution
        ax1.hist(real_parts, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_xlabel('Eigenvalue', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title('Real Part Distribution', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Plot eigenvalue scatter plot (complex plane)
        ax2.scatter(real_parts, imag_parts, alpha=0.7, color='red')
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax2.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        ax2.set_xlabel('Real Part', fontsize=12)
        ax2.set_ylabel('Imaginary Part', fontsize=12)
        ax2.set_title('Eigenvalue Spectrum', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # Set overall title
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        # Save chart
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_noise_recovery_curves(noise_levels: List[float],
                                  recovery_rates: List[float],
                                  save_path: Optional[str] = None,
                                  title: str = "Noise Recovery Performance") -> plt.Figure:
        """
        Plot noise-recovery rate curve
        
        Args:
            noise_levels: List of noise levels
            recovery_rates: Corresponding list of recovery rates
            save_path: Save path
            title: Chart title
            
        Returns:
            matplotlib figure object
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot curve
        ax.plot(noise_levels, recovery_rates, 'bo-', linewidth=2, markersize=8)
        
        # Fill area
        ax.fill_between(noise_levels, 0, recovery_rates, alpha=0.2)
        
        # Set chart properties
        ax.set_xlabel('Noise Level', fontsize=12)
        ax.set_ylabel('Recovery Rate', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Add value labels
        for i, (x, y) in enumerate(zip(noise_levels, recovery_rates)):
            ax.text(x, y + 0.02, f'{y:.2f}', ha='center', va='bottom')
        
        # Save chart
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_confusion_matrix(confusion_matrix: np.ndarray,
                             class_labels: List[str],
                             save_path: Optional[str] = None,
                             title: str = "Confusion Matrix") -> plt.Figure:
        """
        Plot confusion matrix
        
        Args:
            confusion_matrix: Confusion matrix
            class_labels: List of class labels
            save_path: Save path
            title: Chart title
            
        Returns:
            matplotlib figure object
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Draw heatmap
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_labels, yticklabels=class_labels, ax=ax)
        
        # Set chart properties
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(title, fontsize=14)
        
        # Save chart
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig