"""
MNIST data loading module
Encapsulates MNIST data loading logic
Provides preprocessing functions like binarization, noise addition, occlusion
Returns formats suitable for Hopfield/RBM
"""

import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from typing import Tuple, Optional, List


class MNISTLoader:
    """
    MNIST data loader
    Provides loading, preprocessing, and specific digit sample extraction for training and test data
    """
    
    def __init__(self, path: str, binary_threshold: float = 0.5, download: bool = True):
        """
        Initialize MNIST loader
        
        Args:
            path: Data save path
            binary_threshold: Binarization threshold
            download: Whether to automatically download data
        """
        self.path = path
        self.binary_threshold = binary_threshold
        self.download = download
        
        # Ensure data directory exists
        os.makedirs(self.path, exist_ok=True)
        
        # Define data transformation
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Convert to tensor in range [0,1]
        ])
        
        # Load datasets
        self.train_dataset = torchvision.datasets.MNIST(
            root=self.path,
            train=True,
            download=self.download,
            transform=self.transform
        )
        
        self.test_dataset = torchvision.datasets.MNIST(
            root=self.path,
            train=False,
            download=self.download,
            transform=self.transform
        )
    
    def load_train_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load complete training set
        
        Returns:
            (images, labels): Training images and labels
        """
        train_loader = DataLoader(self.train_dataset, batch_size=len(self.train_dataset))
        images, labels = next(iter(train_loader))
        return images, labels
    
    def load_test_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load test set
        
        Returns:
            (images, labels): Test images and labels
        """
        test_loader = DataLoader(self.test_dataset, batch_size=len(self.test_dataset))
        images, labels = next(iter(test_loader))
        return images, labels
    
    def get_digit_samples(self, digit: int, n_samples: int = 10, 
                         dataset: str = 'train') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get samples of specific digit
        
        Args:
            digit: Target digit (0-9)
            n_samples: Number of samples
            dataset: Dataset to use ('train' or 'test')
            
        Returns:
            (images, labels): Images and labels of specified digit
        """
        if dataset == 'train':
            dataset_obj = self.train_dataset
        else:
            dataset_obj = self.test_dataset
            
        # Filter indices of specified digit
        indices = [i for i, (_, label) in enumerate(dataset_obj) if label == digit]
        
        # If requested samples exceed available samples, use all available samples
        n_samples = min(n_samples, len(indices))
        indices = indices[:n_samples]
        
        # Create subset
        subset = Subset(dataset_obj, indices)
        loader = DataLoader(subset, batch_size=n_samples)
        images, labels = next(iter(loader))
        
        return images, labels
    
    def get_random_samples(self, n_samples: int = 100, 
                          dataset: str = 'train') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get random samples
        
        Args:
            n_samples: Number of samples
            dataset: Dataset to use ('train' or 'test')
            
        Returns:
            (images, labels): Random images and labels
        """
        if dataset == 'train':
            dataset_obj = self.train_dataset
            max_samples = len(self.train_dataset)
        else:
            dataset_obj = self.test_dataset
            max_samples = len(self.test_dataset)
            
        n_samples = min(n_samples, max_samples)
        
        # Randomly select indices
        indices = np.random.choice(max_samples, n_samples, replace=False)
        
        # Create subset
        subset = Subset(dataset_obj, indices)
        loader = DataLoader(subset, batch_size=n_samples)
        images, labels = next(iter(loader))
        
        return images, labels
    
    def binarize(self, images: torch.Tensor, 
                 threshold: Optional[float] = None) -> torch.Tensor:
        """
        Binarize images to {-1, 1}
        
        Args:
            images: Input image tensor, range [0, 1]
            threshold: Binarization threshold, defaults to threshold used during class initialization
            
        Returns:
            Binarized image tensor with values {-1, 1}
        """
        if threshold is None:
            threshold = self.binary_threshold
            
        # Convert range [0,1] to {-1,1}
        binary_images = torch.where(images > threshold, 
                                   torch.ones_like(images), 
                                   -torch.ones_like(images))
        
        return binary_images
    
    def binarize_zero_one(self, images: torch.Tensor, 
                          threshold: Optional[float] = None) -> torch.Tensor:
        """
        Binarize images to {0, 1}
        
        Args:
            images: Input image tensor, range [0, 1]
            threshold: Binarization threshold, defaults to threshold used during class initialization
            
        Returns:
            Binarized image tensor with values {0, 1}
        """
        if threshold is None:
            threshold = self.binary_threshold
            
        # Keep range [0,1] as {0,1}
        binary_images = torch.where(images > threshold, 
                                   torch.ones_like(images), 
                                   torch.zeros_like(images))
        
        return binary_images
    
    def add_noise(self, images: torch.Tensor, noise_level: float, binary_values: set = {-1, 1}) -> torch.Tensor:
        """
        Add salt-and-pepper noise
        
        Args:
            images: Input image tensor, values should be {-1, 1} or {0, 1}
            noise_level: Noise level (0-1), proportion of pixels to flip
            binary_values: Set of binary values used in the images
            
        Returns:
            Image tensor with added noise
        """
        noisy_images = images.clone()
        
        # Calculate number of pixels to flip
        total_pixels = images.numel()
        n_flip = int(total_pixels * noise_level)
        
        # Randomly select pixel positions to flip
        flat_indices = torch.randperm(total_pixels)[:n_flip]
        
        # Flip selected pixels
        for idx in flat_indices:
            # Convert linear index to multi-dimensional index
            multi_idx = np.unravel_index(idx, images.shape)
            # Flip pixel value
            if images[multi_idx] == max(binary_values):
                noisy_images[multi_idx] = min(binary_values)
            else:
                noisy_images[multi_idx] = max(binary_values)
                
        return noisy_images
    
    def add_occlusion(self, images: torch.Tensor, occlusion_ratio: float = 0.2) -> torch.Tensor:
        """
        Add occlusion block
        
        Args:
            images: Input image tensor, values should be {-1, 1} or {0, 1}
            occlusion_ratio: Proportion of image covered by occlusion block (0-1)
            
        Returns:
            Image tensor with added occlusion
        """
        occluded_images = images.clone()
        
        # Calculate occlusion block size
        _, height, width = images.shape
        occlusion_size = int(height * np.sqrt(occlusion_ratio))
        
        for i in range(images.shape[0]):
            # Randomly select occlusion position
            start_h = np.random.randint(0, height - occlusion_size + 1)
            start_w = np.random.randint(0, width - occlusion_size + 1)
            
            # Apply occlusion (set to 0 or -1)
            if torch.min(images) < 0:
                occluded_images[i, start_h:start_h+occlusion_size, 
                               start_w:start_w+occlusion_size] = -1
            else:
                occluded_images[i, start_h:start_h+occlusion_size, 
                               start_w:start_w+occlusion_size] = 0
                
        return occluded_images
    
    def flatten_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Flatten images to one-dimensional vectors
        
        Args:
            images: Input image tensor, shape (batch_size, height, width)
            
        Returns:
            Flattened image tensor, shape (batch_size, height*width)
        """
        batch_size = images.shape[0]
        return images.view(batch_size, -1)
    
    def unflatten_images(self, flat_images: torch.Tensor, 
                        height: int = 28, width: int = 28) -> torch.Tensor:
        """
        Restore one-dimensional vectors to images
        
        Args:
            flat_images: Flattened image tensor, shape (batch_size, height*width)
            height: Image height
            width: Image width
            
        Returns:
            Restored image tensor, shape (batch_size, height, width)
        """
        batch_size = flat_images.shape[0]
        return flat_images.view(batch_size, height, width)
    
    def get_specific_digit_samples(self, digits: List[int], num_samples: int = 10, 
                                  binary_values: set = {-1, 1}) -> List[torch.Tensor]:
        """
        Get samples of specific digits with specified binary values
        
        Args:
            digits: List of target digits (0-9)
            num_samples: Number of samples per digit
            binary_values: Binary values to use for the images
            
        Returns:
            List of binary image tensors
        """
        patterns = []
        for digit in digits:
            images, _ = self.get_digit_samples(digit, num_samples)
            
            # Binarize to specified binary values
            if binary_values == {0, 1}:
                binary_images = self.binarize_zero_one(images)
            else:  # Default to {-1, 1}
                binary_images = self.binarize(images)
                
            # Flatten and select one sample
            flat_images = self.flatten_images(binary_images)
            patterns.append(flat_images[0])
            
        return patterns