"""
Data preprocessing module
Provides image preprocessing, feature extraction, and data transformation functions
"""

import numpy as np
import torch
from typing import Tuple, Optional
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def normalize_images(images: torch.Tensor, 
                    method: str = 'min_max') -> torch.Tensor:
    """
    Normalize image data
    
    Args:
        images: Input image tensor
        method: Normalization method ('min_max', 'z_score')
        
    Returns:
        Normalized image tensor
    """
    if method == 'min_max':
        # Min-Max normalization to [0,1]
        min_val = torch.min(images)
        max_val = torch.max(images)
        normalized = (images - min_val) / (max_val - min_val + 1e-8)
    elif method == 'z_score':
        # Z-Score standardization
        mean_val = torch.mean(images)
        std_val = torch.std(images)
        normalized = (images - mean_val) / (std_val + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")
        
    return normalized


def calculate_hamming_distance(x1: torch.Tensor, x2: torch.Tensor) -> float:
    """
    Calculate Hamming distance between two binary vectors
    
    Args:
        x1: First vector
        x2: Second vector
        
    Returns:
        Hamming distance (ratio of different elements)
    """
    # Ensure vectors have the same shape
    assert x1.shape == x2.shape, "Vectors must have the same shape"
    
    # Count different elements
    diff = torch.sum(x1 != x2).item()
    
    # Calculate Hamming distance ratio
    hamming_dist = diff / x1.numel()
    
    return hamming_dist


def calculate_ssim(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """
    Calculate Structural Similarity Index (simplified version)
    
    Args:
        original: Original image
        reconstructed: Reconstructed image
        
    Returns:
        SSIM value (range [0,1], higher means more similar)
    """
    # Ensure images have the same shape
    assert original.shape == reconstructed.shape, "Images must have the same shape"
    
    # Calculate means
    mu_orig = torch.mean(original)
    mu_recon = torch.mean(reconstructed)
    
    # Calculate variances and covariance
    sigma_orig_sq = torch.var(original)
    sigma_recon_sq = torch.var(reconstructed)
    sigma_orig_recon = torch.mean((original - mu_orig) * (reconstructed - mu_recon))
    
    # SSIM constants (for stability)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    
    # Calculate SSIM
    numerator = (2 * mu_orig * mu_recon + c1) * (2 * sigma_orig_recon + c2)
    denominator = (mu_orig ** 2 + mu_recon ** 2 + c1) * (sigma_orig_sq + sigma_recon_sq + c2)
    
    ssim = numerator / denominator
    
    return ssim.item()


def pca_reduce_dimensionality(data: torch.Tensor, 
                             n_components: int = 2) -> Tuple[np.ndarray, PCA]:
    """
    Reduce dimensionality using PCA
    
    Args:
        data: Input data with shape (n_samples, n_features)
        n_components: Number of dimensions after reduction
        
    Returns:
        (reduced_data, pca_model): Reduced data and PCA model
    """
    # Convert to numpy array
    if isinstance(data, torch.Tensor):
        data_np = data.detach().cpu().numpy()
    else:
        data_np = data
    
    # Standardize data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_np)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data_scaled)
    
    return reduced_data, pca


def calculate_mutual_information(x: torch.Tensor, y: torch.Tensor, 
                                n_bins: int = 20) -> float:
    """
    Calculate mutual information between two variables (simplified version)
    
    Args:
        x: First variable
        y: Second variable
        n_bins: Number of bins for histogram
        
    Returns:
        Mutual information value
    """
    # Convert to numpy arrays
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy().flatten()
    else:
        x_np = x.flatten()
        
    if isinstance(y, torch.Tensor):
        y_np = y.detach().cpu().numpy().flatten()
    else:
        y_np = y.flatten()
    
    # Calculate marginal histograms
    hist_xy, x_edges, y_edges = np.histogram2d(x_np, y_np, bins=n_bins)
    hist_x, _ = np.histogram(x_np, bins=x_edges)
    hist_y, _ = np.histogram(y_np, bins=y_edges)
    
    # Convert to probabilities
    p_xy = hist_xy / float(np.sum(hist_xy))
    p_x = hist_x / float(np.sum(hist_x))
    p_y = hist_y / float(np.sum(hist_y))
    
    # Calculate mutual information
    mi = 0.0
    for i in range(n_bins):
        for j in range(n_bins):
            if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                mi += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j]))
    
    return mi


def calculate_sparsity(data: torch.Tensor, threshold: float = 0.1) -> float:
    """
    Calculate sparsity of data
    
    Args:
        data: Input data
        threshold: Threshold to consider as "active"
        
    Returns:
        Sparsity value (ratio of active elements)
    """
    # Calculate ratio of active elements
    active_elements = torch.sum(torch.abs(data) > threshold).item()
    total_elements = data.numel()
    
    sparsity = active_elements / total_elements
    
    return sparsity


def create_noise_patterns(shape: Tuple[int, ...], 
                         noise_type: str = 'gaussian') -> torch.Tensor:
    """
    Create noise patterns
    
    Args:
        shape: Shape of noise tensor
        noise_type: Type of noise ('gaussian', 'uniform', 'binary')
        
    Returns:
        Noise tensor
    """
    if noise_type == 'gaussian':
        # Gaussian noise
        noise = torch.randn(shape)
    elif noise_type == 'uniform':
        # Uniform noise
        noise = torch.rand(shape) * 2 - 1  # Range [-1, 1]
    elif noise_type == 'binary':
        # Binary noise
        noise = torch.randint(0, 2, shape) * 2 - 1  # Values are {-1, 1}
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
        
    return noise


def create_occlusion_mask(shape: Tuple[int, ...], 
                         occlusion_ratio: float = 0.2) -> torch.Tensor:
    """
    Create occlusion mask
    
    Args:
        shape: Image shape
        occlusion_ratio: Ratio of occlusion
        
    Returns:
        Occlusion mask (1 for keep, 0 for occlusion)
    """
    mask = torch.ones(shape)
    
    # Only create block occlusion for 2D images
    if len(shape) >= 2:
        height, width = shape[-2:]
        occlusion_size = int(height * np.sqrt(occlusion_ratio))
        
        # Randomly select occlusion position
        start_h = np.random.randint(0, height - occlusion_size + 1)
        start_w = np.random.randint(0, width - occlusion_size + 1)
        
        # Apply occlusion
        mask[..., start_h:start_h+occlusion_size, start_w:start_w+occlusion_size] = 0
    
    return mask


def shuffle_pixels(images: torch.Tensor, 
                  shuffle_ratio: float = 0.1) -> torch.Tensor:
    """
    Randomly shuffle pixel positions
    
    Args:
        images: Input images
        shuffle_ratio: Ratio of pixels to shuffle
        
    Returns:
        Images with shuffled pixels
    """
    shuffled_images = images.clone()
    batch_size, height, width = images.shape
    
    # Calculate number of pixels to shuffle
    total_pixels = height * width
    n_shuffle = int(total_pixels * shuffle_ratio)
    
    for i in range(batch_size):
        # Randomly select pixel positions to shuffle
        indices = np.random.choice(total_pixels, n_shuffle, replace=False)
        
        # Randomly select target positions
        target_indices = np.random.choice(total_pixels, n_shuffle, replace=False)
        
        # Swap pixel values
        for src_idx, tgt_idx in zip(indices, target_indices):
            src_h, src_w = src_idx // width, src_idx % width
            tgt_h, tgt_w = tgt_idx // width, tgt_idx % width
            
            # Swap pixel values
            shuffled_images[i, src_h, src_w], shuffled_images[i, tgt_h, tgt_w] = \
                shuffled_images[i, tgt_h, tgt_w], shuffled_images[i, src_h, src_w]
    
    return shuffled_images