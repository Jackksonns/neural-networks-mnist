"""
RBM Experiment Script
MNIST Data Training
Generated Sample Quality Assessment
Feature Learning Visualization
Weight Pattern Analysis
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from typing import List, Tuple, Dict, Optional
from torchvision.utils import make_grid

from .rbm import RBM
from .sampler import RBMSampler
from ..utils.data_loader import MNISTLoader
from ..utils.preprocessing import binary_to_image
import config


class RBMExperiments:
    """
    RBM Experiment Class
    """
    
    def __init__(self):
        """Initialize experiment environment"""
        self.loader = MNISTLoader()
        self.rbm = RBM(
            n_visible=config.RBM_CONFIG['n_visible'],
            n_hidden=config.RBM_CONFIG['n_hidden'],
            k=config.RBM_CONFIG['k'],
            learning_rate=config.RBM_CONFIG['learning_rate'],
            momentum=config.RBM_CONFIG['momentum'],
            weight_decay=config.RBM_CONFIG['weight_decay'],
            use_cuda=config.RBM_CONFIG['use_cuda']
        )
        self.sampler = RBMSampler(self.rbm)
        self.results = {}
        
        # Create result save directories
        os.makedirs(os.path.join(config.RESULTS_DIR, 'rbm', 'weights'), exist_ok=True)
        os.makedirs(os.path.join(config.RESULTS_DIR, 'rbm', 'generated_samples'), exist_ok=True)
        os.makedirs(os.path.join(config.RESULTS_DIR, 'rbm', 'feature_viz'), exist_ok=True)
    
    def train_rbm(self, n_epochs: int = 50, batch_size: int = 64) -> Dict:
        """
        Train RBM
        
        Args:
            n_epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            Training result dictionary
        """
        print(f"Training RBM for {n_epochs} epochs with batch size {batch_size}")
        
        # Load MNIST data
        train_data = self.loader.get_train_data(binary_values={0, 1})
        train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        
        # Training history
        train_errors = []
        epoch_times = []
        
        # Training loop
        for epoch in range(n_epochs):
            epoch_start_time = time.time()
            batch_errors = []
            
            for batch_idx, (data, _) in enumerate(train_loader):
                # Flatten and binarize data
                batch = data.view(data.size(0), -1)
                batch = (batch > 0.5).float()  # Binarize
                
                if self.rbm.use_cuda:
                    batch = batch.cuda()
                
                # Train one batch
                error = self.rbm.train_batch(batch)
                batch_errors.append(error)
            
            # Calculate average error
            avg_error = np.mean(batch_errors)
            train_errors.append(avg_error)
            
            # Record training time
            epoch_time = time.time() - epoch_start_time
            epoch_times.append(epoch_time)
            
            print(f"Epoch {epoch+1}/{n_epochs}, Error: {avg_error:.6f}, Time: {epoch_time:.2f}s")
        
        # Save trained model
        model_path = os.path.join(config.RESULTS_DIR, 'rbm', 'trained_rbm.pth')
        self.rbm.save_model(model_path)
        
        # Save results
        results = {
            'train_errors': train_errors,
            'epoch_times': epoch_times,
            'model_path': model_path
        }
        
        # Visualize training process
        self._visualize_training_process(results)
        
        print(f"RBM training completed. Model saved to {model_path}")
        return results
    
    def generate_samples(self, n_samples: int = 64, n_gibbs_steps: int = 1000) -> Dict:
        """
        Generate samples
        
        Args:
            n_samples: Number of samples
            n_gibbs_steps: Number of Gibbs sampling steps
            
        Returns:
            Generation result dictionary
        """
        print(f"Generating {n_samples} samples with {n_gibbs_steps} Gibbs steps")
        
        # Generate samples
        samples = self.rbm.generate_samples(n_samples, n_gibbs_steps)
        
        # Convert samples to image format
        samples_np = samples.detach().cpu().numpy()
        images = binary_to_image(samples_np, (28, 28))
        
        # Visualize generated samples
        self._visualize_generated_samples(images)
        
        # Save results
        results = {
            'samples': samples,
            'images': images
        }
        
        print(f"Sample generation completed.")
        return results
    
    def analyze_weights(self) -> Dict:
        """
        Analyze weights
        
        Returns:
            Weight analysis result dictionary
        """
        print("Analyzing RBM weights")
        
        # Get weight matrix
        weights = self.rbm.get_weights()
        weights_np = weights.detach().cpu().numpy()
        
        # Visualize weight matrix
        self._visualize_weights(weights_np)
        
        # Analyze weight distribution
        weight_stats = {
            'mean': np.mean(weights_np),
            'std': np.std(weights_np),
            'min': np.min(weights_np),
            'max': np.max(weights_np)
        }
        
        # Analyze singular values of weight matrix
        U, s, V = np.linalg.svd(weights_np)
        
        # Save results
        results = {
            'weights': weights,
            'weights_np': weights_np,
            'stats': weight_stats,
            'singular_values': s,
            'U': U,
            'V': V
        }
        
        print(f"Weight analysis completed. Mean: {weight_stats['mean']:.4f}, Std: {weight_stats['std']:.4f}")
        return results
    
    def visualize_features(self, n_samples: int = 1000) -> Dict:
        """
        Visualize learned features
        
        Args:
            n_samples: Number of samples for visualization
            
        Returns:
            Feature visualization result dictionary
        """
        print(f"Visualizing learned features with {n_samples} samples")
        
        # Load MNIST data
        train_data = self.loader.get_train_data(binary_values={0, 1})
        train_loader = data.DataLoader(train_data, batch_size=n_samples, shuffle=True)
        
        # Get a batch of data
        for data, labels in train_loader:
            batch = data.view(data.size(0), -1)
            batch = (batch > 0.5).float()  # Binarize
            break
        
        # Get hidden layer representation
        hidden_repr = self.rbm.get_hidden_representation(batch)
        hidden_np = hidden_repr.detach().cpu().numpy()
        labels_np = labels.numpy()
        
        # Use t-SNE for dimensionality reduction
        tsne = TSNE(n_components=2, random_state=42)
        hidden_tsne = tsne.fit_transform(hidden_np)
        
        # Use PCA for dimensionality reduction
        pca = PCA(n_components=2)
        hidden_pca = pca.fit_transform(hidden_np)
        
        # Visualize dimensionality reduction results
        self._visualize_feature_reduction(hidden_tsne, hidden_pca, labels_np)
        
        # Visualize hidden unit activation patterns
        self._visualize_hidden_activations(hidden_np, labels_np)
        
        # Save results
        results = {
            'hidden_repr': hidden_repr,
            'hidden_np': hidden_np,
            'labels': labels_np,
            'tsne': hidden_tsne,
            'pca': hidden_pca
        }
        
        print(f"Feature visualization completed.")
        return results
    
    def compare_sampling_methods(self, n_samples: int = 64, n_steps: int = 1000) -> Dict:
        """
        Compare different sampling methods
        
        Args:
            n_samples: Number of samples
            n_steps: Number of sampling steps
            
        Returns:
            Sampling method comparison result dictionary
        """
        print(f"Comparing sampling methods with {n_samples} samples and {n_steps} steps")
        
        # Standard Gibbs sampling
        gibbs_samples, _ = self.sampler.gibbs_sample(n_samples, n_steps)
        
        # Block Gibbs sampling
        block_samples, _ = self.sampler.block_gibbs_sample(n_samples, n_steps)
        
        # Tempered sampling
        annealed_samples, _ = self.sampler.tempered_transition_sample(n_samples, n_steps)
        
        # Parallel tempering sampling
        pt_samples, _ = self.sampler.parallel_tempering_sample(n_samples, n_steps)
        
        # Convert samples to image format
        gibbs_images = binary_to_image(gibbs_samples.detach().cpu().numpy(), (28, 28))
        block_images = binary_to_image(block_samples.detach().cpu().numpy(), (28, 28))
        annealed_images = binary_to_image(annealed_samples.detach().cpu().numpy(), (28, 28))
        pt_images = binary_to_image(pt_samples.detach().cpu().numpy(), (28, 28))
        
        # Visualize results of different sampling methods
        self._visualize_sampling_comparison(
            gibbs_images, block_images, annealed_images, pt_images
        )
        
        # Calculate sample quality metrics
        gibbs_quality = self._evaluate_sample_quality(gibbs_samples)
        block_quality = self._evaluate_sample_quality(block_samples)
        annealed_quality = self._evaluate_sample_quality(annealed_samples)
        pt_quality = self._evaluate_sample_quality(pt_samples)
        
        # Save results
        results = {
            'gibbs': {
                'samples': gibbs_samples,
                'images': gibbs_images,
                'quality': gibbs_quality
            },
            'block': {
                'samples': block_samples,
                'images': block_images,
                'quality': block_quality
            },
            'annealed': {
                'samples': annealed_samples,
                'images': annealed_images,
                'quality': annealed_quality
            },
            'pt': {
                'samples': pt_samples,
                'images': pt_images,
                'quality': pt_quality
            }
        }
        
        print(f"Sampling methods comparison completed.")
        return results
    
    def run_dream_experiment(self, digit: int = 0, n_steps: int = 1000) -> Dict:
        """
        Run "dream" experiment
        
        Args:
            digit: Starting digit
            n_steps: Number of Gibbs sampling steps
            
        Returns:
            Dream experiment result dictionary
        """
        print(f"Running dream experiment starting from digit {digit} for {n_steps} steps")
        
        # Get sample of specified digit
        sample = self.loader.get_specific_digit_sample(digit, binary_values={0, 1})
        sample = sample.unsqueeze(0)  # Add batch dimension
        
        # Dream
        dream_sample, dream_history = self.sampler.dream(sample, n_steps)
        
        # Convert samples to image format
        original_image = binary_to_image(sample.detach().cpu().numpy(), (28, 28))[0]
        dream_image = binary_to_image(dream_sample.detach().cpu().numpy(), (28, 28))[0]
        
        # Convert dream history to images
        dream_images = []
        for state in dream_history:
            img = binary_to_image(state.detach().cpu().numpy(), (28, 28))[0]
            dream_images.append(img)
        
        # Visualize dream process
        self._visualize_dream_process(original_image, dream_images, dream_image)
        
        # Save results
        results = {
            'original_sample': sample,
            'dream_sample': dream_sample,
            'dream_history': dream_history,
            'original_image': original_image,
            'dream_image': dream_image,
            'dream_images': dream_images
        }
        
        print(f"Dream experiment completed.")
        return results
    
    def _visualize_training_process(self, results: Dict) -> None:
        """
        Visualize training process
        
        Args:
            results: Training result dictionary
        """
        train_errors = results['train_errors']
        epoch_times = results['epoch_times']
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot training error
        ax1.plot(train_errors, 'b-', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Reconstruction Error', fontsize=12)
        ax1.set_title('Training Error', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Plot training time
        ax2.plot(epoch_times, 'g-', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Time (seconds)', fontsize=12)
        ax2.set_title('Training Time per Epoch', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # Save figure
        save_path = os.path.join(config.RESULTS_DIR, 'rbm', 'feature_viz', 'training_process.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _visualize_generated_samples(self, images: np.ndarray) -> None:
        """
        Visualize generated samples
        
        Args:
            images: Generated image array
        """
        # Create grid image
        n_images = min(64, len(images))
        grid_size = int(np.sqrt(n_images))
        
        # Select images
        selected_images = images[:n_images]
        
        # Create figure
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
        
        if grid_size == 1:
            axes = np.array([[axes]])
        
        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                if idx < len(selected_images):
                    axes[i, j].imshow(selected_images[idx], cmap='binary')
                    axes[i, j].axis('off')
                else:
                    axes[i, j].axis('off')
        
        # Set main title
        fig.suptitle('Generated Samples', fontsize=16)
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(config.RESULTS_DIR, 'rbm', 'generated_samples', 'generated_samples.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _visualize_weights(self, weights_np: np.ndarray) -> None:
        """
        Visualize weight matrix
        
        Args:
            weights_np: Numpy array of weight matrix
        """
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot weight matrix heatmap
        sns.heatmap(weights_np, cmap='coolwarm', center=0, ax=ax1)
        ax1.set_title('Weight Matrix', fontsize=14)
        ax1.set_xlabel('Hidden Units', fontsize=12)
        ax1.set_ylabel('Visible Units', fontsize=12)
        
        # Plot weight distribution histogram
        ax2.hist(weights_np.flatten(), bins=50, alpha=0.7, edgecolor='black')
        ax2.set_title('Weight Distribution', fontsize=14)
        ax2.set_xlabel('Weight Value', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Save figure
        save_path = os.path.join(config.RESULTS_DIR, 'rbm', 'weights', 'weight_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Visualize weight patterns corresponding to hidden units
        n_hidden = weights_np.shape[1]
        n_display = min(64, n_hidden)
        
        # Create figure
        grid_size = int(np.sqrt(n_display))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
        
        if grid_size == 1:
            axes = np.array([[axes]])
        
        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                if idx < n_display:
                    # Reshape weights to image
                    weight_img = weights_np[:, idx].reshape(28, 28)
                    axes[i, j].imshow(weight_img, cmap='seismic')
                    axes[i, j].axis('off')
                    axes[i, j].set_title(f'Hidden {idx}', fontsize=8)
                else:
                    axes[i, j].axis('off')
        
        # Set main title
        fig.suptitle('Hidden Unit Weight Patterns', fontsize=16)
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(config.RESULTS_DIR, 'rbm', 'weights', 'weight_patterns.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _visualize_feature_reduction(self, tsne: np.ndarray, pca: np.ndarray, labels: np.ndarray) -> None:
        """
        Visualize feature dimensionality reduction results
        
        Args:
            tsne: t-SNE dimensionality reduction results
            pca: PCA dimensionality reduction results
            labels: Label array
        """
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot t-SNE results
        scatter1 = ax1.scatter(tsne[:, 0], tsne[:, 1], c=labels, cmap='tab10', alpha=0.7)
        ax1.set_xlabel('t-SNE Component 1', fontsize=12)
        ax1.set_ylabel('t-SNE Component 2', fontsize=12)
        ax1.set_title('t-SNE Visualization of Hidden Representations', fontsize=14)
        plt.colorbar(scatter1, ax=ax1)
        
        # Plot PCA results
        scatter2 = ax2.scatter(pca[:, 0], pca[:, 1], c=labels, cmap='tab10', alpha=0.7)
        ax2.set_xlabel('PCA Component 1', fontsize=12)
        ax2.set_ylabel('PCA Component 2', fontsize=12)
        ax2.set_title('PCA Visualization of Hidden Representations', fontsize=14)
        plt.colorbar(scatter2, ax=ax2)
        
        # Save figure
        save_path = os.path.join(config.RESULTS_DIR, 'rbm', 'feature_viz', 'feature_reduction.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _visualize_hidden_activations(self, hidden_np: np.ndarray, labels: np.ndarray) -> None:
        """
        Visualize hidden layer activations
        
        Args:
            hidden_np: Numpy array of hidden layer activations
            labels: Label array
        """
        # Calculate average hidden activation for each digit
        n_hidden = hidden_np.shape[1]
        n_digits = 10
        
        digit_activations = np.zeros((n_digits, n_hidden))
        digit_counts = np.zeros(n_digits)
        
        for i, label in enumerate(labels):
            digit_activations[label] += hidden_np[i]
            digit_counts[label] += 1
        
        # Calculate averages
        for digit in range(n_digits):
            if digit_counts[digit] > 0:
                digit_activations[digit] /= digit_counts[digit]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot heatmap
        sns.heatmap(digit_activations, cmap='viridis', ax=ax)
        ax.set_xlabel('Hidden Units', fontsize=12)
        ax.set_ylabel('Digits', fontsize=12)
        ax.set_title('Average Hidden Activations per Digit', fontsize=14)
        
        # Save figure
        save_path = os.path.join(config.RESULTS_DIR, 'rbm', 'feature_viz', 'hidden_activations.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _visualize_sampling_comparison(self, gibbs_images: np.ndarray, 
                                       block_images: np.ndarray,
                                       annealed_images: np.ndarray,
                                       pt_images: np.ndarray) -> None:
        """
        Visualize comparison of different sampling methods
        
        Args:
            gibbs_images: Gibbs sampling images
            block_images: Block Gibbs sampling images
            annealed_images: Tempered sampling images
            pt_images: Parallel tempering sampling images
        """
        # Select number of images to display
        n_display = 16
        
        # Create figure
        fig, axes = plt.subplots(4, n_display, figsize=(20, 8))
        
        # Display Gibbs sampling results
        for i in range(n_display):
            if i < len(gibbs_images):
                axes[0, i].imshow(gibbs_images[i], cmap='binary')
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('Gibbs Sampling', fontsize=12)
        
        # Display block Gibbs sampling results
        for i in range(n_display):
            if i < len(block_images):
                axes[1, i].imshow(block_images[i], cmap='binary')
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('Block Gibbs Sampling', fontsize=12)
        
        # Display tempered sampling results
        for i in range(n_display):
            if i < len(annealed_images):
                axes[2, i].imshow(annealed_images[i], cmap='binary')
            axes[2, i].axis('off')
            if i == 0:
                axes[2, i].set_title('Tempered Transition', fontsize=12)
        
        # Display parallel tempering sampling results
        for i in range(n_display):
            if i < len(pt_images):
                axes[3, i].imshow(pt_images[i], cmap='binary')
            axes[3, i].axis('off')
            if i == 0:
                axes[3, i].set_title('Parallel Tempering', fontsize=12)
        
        # Set main title
        fig.suptitle('Sampling Methods Comparison', fontsize=16)
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(config.RESULTS_DIR, 'rbm', 'generated_samples', 'sampling_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _evaluate_sample_quality(self, samples: torch.Tensor) -> Dict:
        """
        Evaluate sample quality
        
        Args:
            samples: Generated samples
            
        Returns:
            Sample quality metrics dictionary
        """
        samples_np = samples.detach().cpu().numpy()
        
        # Calculate average activation
        avg_activation = np.mean(samples_np)
        
        # Calculate activation variance
        activation_var = np.var(samples_np)
        
        # Calculate average free energy
        free_energy = self.rbm.free_energy(samples).mean().item()
        
        return {
            'avg_activation': avg_activation,
            'activation_var': activation_var,
            'free_energy': free_energy
        }
    
    def _visualize_dream_process(self, original_image: np.ndarray, 
                                dream_images: List[np.ndarray],
                                final_image: np.ndarray) -> None:
        """
        Visualize dream process
        
        Args:
            original_image: Original image
            dream_images: List of images during dream process
            final_image: Final image
        """
        # Select key frames to display
        n_frames = min(8, len(dream_images))
        frame_indices = np.linspace(0, len(dream_images)-1, n_frames, dtype=int)
        
        # Create figure
        fig, axes = plt.subplots(2, n_frames+1, figsize=(15, 6))
        
        # Display original image
        axes[0, 0].imshow(original_image, cmap='binary')
        axes[0, 0].set_title('Original', fontsize=12)
        axes[0, 0].axis('off')
        axes[1, 0].axis('off')
        
        # Display key frames during dream process
        for i, idx in enumerate(frame_indices):
            axes[0, i+1].imshow(dream_images[idx], cmap='binary')
            axes[0, i+1].set_title(f'Step {idx}', fontsize=10)
            axes[0, i+1].axis('off')
            
            # Calculate difference from original image
            diff = np.abs(dream_images[idx] - original_image)
            axes[1, i+1].imshow(diff, cmap='hot')
            axes[1, i+1].set_title(f'Diff: {np.mean(diff):.3f}', fontsize=10)
            axes[1, i+1].axis('off')
        
        # Set main title
        fig.suptitle('Dream Process', fontsize=16)
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(config.RESULTS_DIR, 'rbm', 'generated_samples', 'dream_process.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_all_experiments(self) -> Dict:
        """
        Run all experiments
        
        Returns:
            All experiment results dictionary
        """
        print("Running all RBM experiments...")
        
        # Train RBM
        training_results = self.train_rbm()
        
        # Generate samples
        generation_results = self.generate_samples()
        
        # Analyze weights
        weight_results = self.analyze_weights()
        
        # Visualize features
        feature_results = self.visualize_features()
        
        # Compare sampling methods
        sampling_results = self.compare_sampling_methods()
        
        # Run dream experiment
        dream_results = self.run_dream_experiment()
        
        # Save all results
        all_results = {
            'training': training_results,
            'generation': generation_results,
            'weights': weight_results,
            'features': feature_results,
            'sampling': sampling_results,
            'dream': dream_results
        }
        
        print("All RBM experiments completed.")
        return all_results