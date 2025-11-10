"""
Restricted Boltzmann Machine (RBM) Core Implementation
Contrastive Divergence (CD-k) Algorithm
Energy Function Calculation
Parameter Initialization Strategies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import config


class RBM(nn.Module):
    """
    Restricted Boltzmann Machine (RBM) Implementation
    """
    
    def __init__(self, n_visible: int, n_hidden: int, 
                 k: int = 1, learning_rate: float = 0.01,
                 momentum: float = 0.5, weight_decay: float = 0.0001,
                 use_cuda: bool = False):
        """
        Initialize RBM
        
        Args:
            n_visible: Number of visible layer units
            n_hidden: Number of hidden layer units
            k: Number of steps for contrastive divergence algorithm
            learning_rate: Learning rate
            momentum: Momentum coefficient
            weight_decay: Weight decay coefficient
            use_cuda: Whether to use GPU
        """
        super(RBM, self).__init__()
        
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.k = k
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.use_cuda = use_cuda and torch.cuda.is_available()
        
        # Initialize weights and biases
        self._initialize_parameters()
        
        # Momentum terms
        self.W_momentum = torch.zeros_like(self.W)
        self.v_bias_momentum = torch.zeros_like(self.v_bias)
        self.h_bias_momentum = torch.zeros_like(self.h_bias)
        
        # Training history
        self.training_history = {
            'reconstruction_error': [],
            'free_energy': [],
            'hidden_probs': []
        }
        
        # If using GPU, move model to GPU
        if self.use_cuda:
            self.cuda()
    
    def _initialize_parameters(self):
        """Initialize model parameters"""
        # Weight initialization: uniform distribution in range [-0.01, 0.01]
        self.W = nn.Parameter(torch.FloatTensor(self.n_visible, self.n_hidden).uniform_(-0.01, 0.01))
        
        # Visible layer bias initialized to 0
        self.v_bias = nn.Parameter(torch.zeros(self.n_visible))
        
        # Hidden layer bias initialized to 0
        self.h_bias = nn.Parameter(torch.zeros(self.n_hidden))
    
    def sample_h(self, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample hidden layer from visible layer
        
        Args:
            v: Visible layer state
            
        Returns:
            Hidden layer probabilities and hidden layer samples
        """
        # Calculate hidden layer activation probabilities
        h_activation = torch.matmul(v, self.W) + self.h_bias
        h_probs = torch.sigmoid(h_activation)
        
        # Sample from Bernoulli distribution
        h_sample = torch.bernoulli(h_probs)
        
        return h_probs, h_sample
    
    def sample_v(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample visible layer from hidden layer
        
        Args:
            h: Hidden layer state
            
        Returns:
            Visible layer probabilities and visible layer samples
        """
        # Calculate visible layer activation probabilities
        v_activation = torch.matmul(h, self.W.t()) + self.v_bias
        v_probs = torch.sigmoid(v_activation)
        
        # Sample from Bernoulli distribution
        v_sample = torch.bernoulli(v_probs)
        
        return v_probs, v_sample
    
    def contrastive_divergence(self, v0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Contrastive Divergence (CD-k) Algorithm
        
        Args:
            v0: Input visible layer vector
            
        Returns:
            Weight update, visible layer bias update, hidden layer bias update
        """
        # Forward pass
        ph0, h0 = self.sample_h(v0)
        
        # Gibbs sampling
        hk = h0.clone()
        for _ in range(self.k):
            vk, _ = self.sample_v(hk)
            phk, hk = self.sample_h(vk)
        
        # Calculate gradients
        dW = (torch.matmul(v0.t(), ph0) - torch.matmul(vk.t(), phk)) / v0.size(0)
        dv_bias = (v0 - vk).mean(0)
        dh_bias = (ph0 - phk).mean(0)
        
        # Apply weight decay
        dW -= self.weight_decay * self.W
        
        # Apply momentum
        self.W_momentum = self.momentum * self.W_momentum + self.learning_rate * dW
        self.v_bias_momentum = self.momentum * self.v_bias_momentum + self.learning_rate * dv_bias
        self.h_bias_momentum = self.momentum * self.h_bias_momentum + self.learning_rate * dh_bias
        
        return self.W_momentum, self.v_bias_momentum, self.h_bias_momentum
    
    def free_energy(self, v: torch.Tensor) -> torch.Tensor:
        """
        Calculate free energy
        
        Args:
            v: Visible layer vector
            
        Returns:
            Free energy
        """
        # Visible layer term
        visible_term = torch.matmul(v, self.v_bias.t())
        
        # Hidden layer term
        hidden_term = torch.sum(
            torch.log(1 + torch.exp(torch.matmul(v, self.W) + self.h_bias)),
            dim=1
        )
        
        return -visible_term - hidden_term
    
    def reconstruct(self, v: torch.Tensor, n_gibbs_steps: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reconstruct visible layer
        
        Args:
            v: Input visible layer vector
            n_gibbs_steps: Number of Gibbs sampling steps
            
        Returns:
            Reconstructed visible layer vector and reconstruction error
        """
        vk = v.clone()
        
        for _ in range(n_gibbs_steps):
            _, h = self.sample_h(vk)
            vk, _ = self.sample_v(h)
        
        # Calculate reconstruction error
        reconstruction_error = F.mse_loss(vk, v)
        
        return vk, reconstruction_error
    
    def train_batch(self, batch: torch.Tensor) -> float:
        """
        Train one batch
        
        Args:
            batch: Input batch
            
        Returns:
            Reconstruction error
        """
        # Contrastive divergence
        dW, dv_bias, dh_bias = self.contrastive_divergence(batch)
        
        # Update parameters
        self.W.data += dW
        self.v_bias.data += dv_bias
        self.h_bias.data += dh_bias
        
        # Calculate reconstruction error
        _, reconstruction_error = self.reconstruct(batch)
        
        # Calculate average free energy
        avg_free_energy = self.free_energy(batch).mean().item()
        
        # Calculate average hidden layer activation probability
        ph0, _ = self.sample_h(batch)
        avg_hidden_prob = ph0.mean().item()
        
        # Record training history
        self.training_history['reconstruction_error'].append(reconstruction_error.item())
        self.training_history['free_energy'].append(avg_free_energy)
        self.training_history['hidden_probs'].append(avg_hidden_prob)
        
        return reconstruction_error.item()
    
    def get_hidden_representation(self, v: torch.Tensor) -> torch.Tensor:
        """
        Get hidden layer representation
        
        Args:
            v: Visible layer vector
            
        Returns:
            Hidden layer representation
        """
        ph, _ = self.sample_h(v)
        return ph
    
    def generate_samples(self, n_samples: int, n_gibbs_steps: int = 1000) -> torch.Tensor:
        """
        Generate samples
        
        Args:
            n_samples: Number of samples
            n_gibbs_steps: Number of Gibbs sampling steps
            
        Returns:
            Generated samples
        """
        # Randomly initialize visible layer
        v = torch.bernoulli(torch.ones(n_samples, self.n_visible) * 0.5)
        
        if self.use_cuda:
            v = v.cuda()
        
        # Gibbs sampling
        for _ in range(n_gibbs_steps):
            _, h = self.sample_h(v)
            v, _ = self.sample_v(h)
        
        return v
    
    def pseudo_likelihood(self, v: torch.Tensor) -> torch.Tensor:
        """
        Calculate pseudo-likelihood
        
        Args:
            v: Visible layer vector
            
        Returns:
            Pseudo-likelihood
        """
        # Randomly select a visible unit to flip
        i = torch.randint(0, self.n_visible, (1,)).item()
        
        # Calculate energy difference before and after flip
        v_flipped = v.clone()
        v_flipped[:, i] = 1 - v_flipped[:, i]
        
        energy_diff = self.free_energy(v) - self.free_energy(v_flipped)
        
        # Calculate pseudo-likelihood
        pl = -torch.mean(torch.log(torch.sigmoid(energy_diff) + 1e-10))
        
        return pl
    
    def get_weights(self) -> torch.Tensor:
        """
        Get weight matrix
        
        Returns:
            Weight matrix
        """
        return self.W.data.clone()
    
    def get_biases(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get biases
        
        Returns:
            Visible layer bias and hidden layer bias
        """
        return self.v_bias.data.clone(), self.h_bias.data.clone()
    
    def save_model(self, filepath: str) -> None:
        """
        Save model
        
        Args:
            filepath: Save path
        """
        torch.save({
            'state_dict': self.state_dict(),
            'n_visible': self.n_visible,
            'n_hidden': self.n_hidden,
            'k': self.k,
            'learning_rate': self.learning_rate,
            'momentum': self.momentum,
            'weight_decay': self.weight_decay,
            'training_history': self.training_history
        }, filepath)
    
    @classmethod
    def load_model(cls, filepath: str, use_cuda: bool = False) -> 'RBM':
        """
        Load model
        
        Args:
            filepath: Model file path
            use_cuda: Whether to use GPU
            
        Returns:
            Loaded RBM model
        """
        checkpoint = torch.load(filepath, map_location='cpu')
        
        model = cls(
            n_visible=checkpoint['n_visible'],
            n_hidden=checkpoint['n_hidden'],
            k=checkpoint['k'],
            learning_rate=checkpoint['learning_rate'],
            momentum=checkpoint['momentum'],
            weight_decay=checkpoint['weight_decay'],
            use_cuda=use_cuda
        )
        
        model.load_state_dict(checkpoint['state_dict'])
        model.training_history = checkpoint['training_history']
        
        if use_cuda and torch.cuda.is_available():
            model.cuda()
        
        return model