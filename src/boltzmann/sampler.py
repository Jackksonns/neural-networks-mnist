"""
RBM Sampler Implementation
Gibbs Sampling
Conditional Sampling
Annealed Sampling
"""

import torch
import numpy as np
from typing import Tuple, Optional, List, Union
from .rbm import RBM


class RBMSampler:
    """
    RBM Sampler Class
    """
    
    def __init__(self, rbm: RBM):
        """
        Initialize sampler
        
        Args:
            rbm: RBM model
        """
        self.rbm = rbm
    
    def gibbs_sample(self, n_samples: int, n_steps: int = 1000, 
                     init_visible: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Gibbs sampling
        
        Args:
            n_samples: Number of samples
            n_steps: Number of Gibbs sampling steps
            init_visible: Initial visible layer state, if None then randomly initialize
            
        Returns:
            Final samples and sampling history
        """
        # Initialize visible layer
        if init_visible is not None:
            v = init_visible.clone()
        else:
            v = torch.bernoulli(torch.ones(n_samples, self.rbm.n_visible) * 0.5)
        
        if self.rbm.use_cuda:
            v = v.cuda()
        
        # Record sampling history
        sample_history = []
        
        # Gibbs sampling
        for step in range(n_steps):
            # Sample hidden layer from visible layer
            _, h = self.rbm.sample_h(v)
            
            # Sample visible layer from hidden layer
            v, _ = self.rbm.sample_v(h)
            
            # Record sampling history (record every certain number of steps)
            if step % 10 == 0:
                sample_history.append(v.clone())
        
        return v, sample_history
    
    def block_gibbs_sample(self, n_samples: int, n_steps: int = 1000,
                          block_size: int = None,
                          init_visible: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Block Gibbs sampling
        
        Args:
            n_samples: Number of samples
            n_steps: Number of Gibbs sampling steps
            block_size: Block size, if None then use half of visible units
            init_visible: Initial visible layer state, if None then randomly initialize
            
        Returns:
            Final samples and sampling history
        """
        # Initialize visible layer
        if init_visible is not None:
            v = init_visible.clone()
        else:
            v = torch.bernoulli(torch.ones(n_samples, self.rbm.n_visible) * 0.5)
        
        if self.rbm.use_cuda:
            v = v.cuda()
        
        # Set block size
        if block_size is None:
            block_size = self.rbm.n_visible // 2
        
        # Record sampling history
        sample_history = []
        
        # Block Gibbs sampling
        for step in range(n_steps):
            # Randomly select blocks to update
            block_indices = torch.randperm(self.rbm.n_visible)[:block_size]
            
            # Calculate hidden layer conditional probabilities
            ph, h = self.rbm.sample_h(v)
            
            # Calculate visible layer conditional probabilities
            pv, _ = self.rbm.sample_v(h)
            
            # Only update selected blocks
            v[:, block_indices] = pv[:, block_indices]
            
            # Record sampling history (record every certain number of steps)
            if step % 10 == 0:
                sample_history.append(v.clone())
        
        return v, sample_history
    
    def tempered_transition_sample(self, n_samples: int, n_steps: int = 1000,
                                  temperatures: List[float] = None,
                                  init_visible: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Annealed sampling (Tempered Transition)
        
        Args:
            n_samples: Number of samples
            n_steps: Number of sampling steps
            temperatures: Temperature list, if None then use default temperature sequence
            init_visible: Initial visible layer state, if None then randomly initialize
            
        Returns:
            Final samples and sampling history
        """
        # Initialize visible layer
        if init_visible is not None:
            v = init_visible.clone()
        else:
            v = torch.bernoulli(torch.ones(n_samples, self.rbm.n_visible) * 0.5)
        
        if self.rbm.use_cuda:
            v = v.cuda()
        
        # Set temperature sequence
        if temperatures is None:
            temperatures = [2.0, 1.5, 1.0, 0.8, 0.6, 0.4, 0.2, 0.1]
        
        # Record sampling history
        sample_history = []
        
        # Annealed sampling
        for step in range(n_steps):
            # Forward annealing
            v_forward = v.clone()
            for temp in temperatures:
                # Calculate hidden layer conditional probabilities (considering temperature)
                h_activation = (torch.matmul(v_forward, self.rbm.W) + self.rbm.h_bias) / temp
                ph = torch.sigmoid(h_activation)
                h = torch.bernoulli(ph)
                
                # Calculate visible layer conditional probabilities (considering temperature)
                v_activation = (torch.matmul(h, self.rbm.W.t()) + self.rbm.v_bias) / temp
                pv = torch.sigmoid(v_activation)
                v_forward = torch.bernoulli(pv)
            
            # Backward annealing
            v_backward = v_forward.clone()
            for temp in reversed(temperatures):
                # Calculate hidden layer conditional probabilities (considering temperature)
                h_activation = (torch.matmul(v_backward, self.rbm.W) + self.rbm.h_bias) / temp
                ph = torch.sigmoid(h_activation)
                h = torch.bernoulli(ph)
                
                # Calculate visible layer conditional probabilities (considering temperature)
                v_activation = (torch.matmul(h, self.rbm.W.t()) + self.rbm.v_bias) / temp
                pv = torch.sigmoid(v_activation)
                v_backward = torch.bernoulli(pv)
            
            # Metropolis-Hastings accept/reject
            # Calculate energy difference
            energy_v = self.rbm.free_energy(v)
            energy_v_backward = self.rbm.free_energy(v_backward)
            
            # Calculate acceptance probability
            log_accept_prob = energy_v - energy_v_backward
            accept_prob = torch.exp(torch.clamp(log_accept_prob, min=-20, max=20))
            
            # Randomly accept or reject
            random_uniform = torch.rand(n_samples, 1)
            if self.rbm.use_cuda:
                random_uniform = random_uniform.cuda()
            
            accept = random_uniform < accept_prob
            accept = accept.expand_as(v)
            
            # Update state
            v = torch.where(accept, v_backward, v)
            
            # Record sampling history (record every certain number of steps)
            if step % 10 == 0:
                sample_history.append(v.clone())
        
        return v, sample_history
    
    def parallel_tempering_sample(self, n_samples: int, n_steps: int = 1000,
                                 temperatures: List[float] = None,
                                 swap_interval: int = 10) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Parallel tempering sampling
        
        Args:
            n_samples: Number of samples
            n_steps: Number of sampling steps
            temperatures: Temperature list, if None then use default temperature sequence
            swap_interval: Swap interval
            
        Returns:
            Final samples and sampling history
        """
        # Set temperature sequence
        if temperatures is None:
            temperatures = [2.0, 1.5, 1.0, 0.8, 0.6, 0.4, 0.2, 0.1]
        
        n_chains = len(temperatures)
        
        # Initialize visible layer for each temperature chain
        v_chains = []
        for _ in range(n_chains):
            v = torch.bernoulli(torch.ones(n_samples, self.rbm.n_visible) * 0.5)
            if self.rbm.use_cuda:
                v = v.cuda()
            v_chains.append(v)
        
        # Record sampling history
        sample_history = []
        
        # Parallel tempering sampling
        for step in range(n_steps):
            # Perform Gibbs sampling for each chain
            for i in range(n_chains):
                temp = temperatures[i]
                v = v_chains[i]
                
                # Calculate hidden layer conditional probabilities (considering temperature)
                h_activation = (torch.matmul(v, self.rbm.W) + self.rbm.h_bias) / temp
                ph = torch.sigmoid(h_activation)
                h = torch.bernoulli(ph)
                
                # Calculate visible layer conditional probabilities (considering temperature)
                v_activation = (torch.matmul(h, self.rbm.W.t()) + self.rbm.v_bias) / temp
                pv = torch.sigmoid(v_activation)
                v_chains[i] = torch.bernoulli(pv)
            
            # Try to swap states of adjacent chains
            if step % swap_interval == 0:
                for i in range(n_chains - 1):
                    # Calculate swap acceptance probability
                    v_i = v_chains[i]
                    v_j = v_chains[i + 1]
                    
                    # Calculate energy difference
                    energy_i = self.rbm.free_energy(v_i)
                    energy_j = self.rbm.free_energy(v_j)
                    
                    # Calculate energy after swap
                    energy_i_swapped = self.rbm.free_energy(v_j)
                    energy_j_swapped = self.rbm.free_energy(v_i)
                    
                    # Calculate acceptance probability
                    temp_i = temperatures[i]
                    temp_j = temperatures[i + 1]
                    
                    log_accept_prob = (1/temp_i - 1/temp_j) * (energy_j - energy_i) + \
                                     (1/temp_j - 1/temp_i) * (energy_i - energy_j)
                    
                    accept_prob = torch.exp(torch.clamp(log_accept_prob, min=-20, max=20))
                    
                    # Randomly accept or reject
                    random_uniform = torch.rand(n_samples, 1)
                    if self.rbm.use_cuda:
                        random_uniform = random_uniform.cuda()
                    
                    accept = random_uniform < accept_prob
                    accept = accept.expand_as(v_i)
                    
                    # Swap states
                    v_chains[i] = torch.where(accept, v_j, v_i)
                    v_chains[i + 1] = torch.where(accept, v_i, v_j)
            
            # Record sampling history (record every certain number of steps)
            if step % 10 == 0:
                # Only record state of chain with temperature 1.0
                if 1.0 in temperatures:
                    idx = temperatures.index(1.0)
                    sample_history.append(v_chains[idx].clone())
                else:
                    # If no chain with temperature 1.0, record chain closest to 1.0
                    closest_idx = min(range(len(temperatures)), 
                                    key=lambda i: abs(temperatures[i] - 1.0))
                    sample_history.append(v_chains[closest_idx].clone())
        
        # Return final state of chain with temperature 1.0
        if 1.0 in temperatures:
            idx = temperatures.index(1.0)
            return v_chains[idx], sample_history
        else:
            # If no chain with temperature 1.0, return chain closest to 1.0
            closest_idx = min(range(len(temperatures)), 
                            key=lambda i: abs(temperatures[i] - 1.0))
            return v_chains[closest_idx], sample_history
    
    def conditional_sample(self, condition_mask: torch.Tensor, 
                          condition_values: torch.Tensor,
                          n_samples: int, n_steps: int = 1000) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Conditional sampling
        
        Args:
            condition_mask: Condition mask, 1 indicates fixed value, 0 indicates variable
            condition_values: Condition values
            n_samples: Number of samples
            n_steps: Number of Gibbs sampling steps
            
        Returns:
            Final samples and sampling history
        """
        # Initialize visible layer
        v = torch.bernoulli(torch.ones(n_samples, self.rbm.n_visible) * 0.5)
        
        if self.rbm.use_cuda:
            v = v.cuda()
            condition_mask = condition_mask.cuda()
            condition_values = condition_values.cuda()
        
        # Apply conditions
        v = torch.where(condition_mask, condition_values, v)
        
        # Record sampling history
        sample_history = []
        
        # Conditional Gibbs sampling
        for step in range(n_steps):
            # Sample hidden layer from visible layer
            _, h = self.rbm.sample_h(v)
            
            # Sample visible layer from hidden layer
            v_new, _ = self.rbm.sample_v(h)
            
            # Apply conditions
            v = torch.where(condition_mask, condition_values, v_new)
            
            # Record sampling history (record every certain number of steps)
            if step % 10 == 0:
                sample_history.append(v.clone())
        
        return v, sample_history
    
    def interpolate(self, v1: torch.Tensor, v2: torch.Tensor, 
                   n_steps: int = 10) -> List[torch.Tensor]:
        """
        Interpolate between two states
        
        Args:
            v1: Start state
            v2: End state
            n_steps: Number of interpolation steps
            
        Returns:
            Interpolation path
        """
        interpolation_path = []
        
        for i in range(n_steps + 1):
            alpha = i / n_steps
            v_interp = (1 - alpha) * v1 + alpha * v2
            
            # Perform Gibbs sampling on interpolated state
            v_sampled, _ = self.gibbs_sample(v_interp.size(0), n_steps=100, init_visible=v_interp)
            interpolation_path.append(v_sampled)
        
        return interpolation_path
    
    def dream(self, init_visible: torch.Tensor, n_steps: int = 1000) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        "Dream" - Perform Gibbs sampling from initial state
        
        Args:
            init_visible: Initial visible layer state
            n_steps: Number of Gibbs sampling steps
            
        Returns:
            Final state and sampling history
        """
        return self.gibbs_sample(init_visible.size(0), n_steps=n_steps, init_visible=init_visible)