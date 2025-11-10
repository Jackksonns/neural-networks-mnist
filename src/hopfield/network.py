"""
Hopfield network core implementation
Implements discrete Hopfield network (energy function, weight update rules)
Supports both asynchronous and synchronous update modes
Computes energy function to track convergence process
"""

import numpy as np
import torch
from typing import List, Tuple, Optional


class HopfieldNetwork:
    """
    Discrete Hopfield network implementation
    
    Core algorithms:
    - Training: Hebbian learning rule W_ij = (1/N) Σ_μ x_i^μ x_j^μ
    - Update: s_i(t+1) = sign(Σ_j W_ij s_j(t))
    - Energy: E = -1/2 Σ_ij W_ij s_i s_j
    """
    
    def __init__(self, n_neurons: int):
        """
        Initialize Hopfield network
        
        Args:
            n_neurons: Number of neurons (784 for MNIST)
        """
        self.n_neurons = n_neurons
        self.weights = None  # Weight matrix (N x N)
        self.is_trained = False
        
    def train(self, patterns: torch.Tensor) -> None:
        """
        Train network using Hebbian rule
        
        W = (1/N) * Σ(p_i * p_i^T) - I
        
        Args:
            patterns: Patterns to store, shape (n_patterns, n_neurons)
                     Values should be {-1, 1} or {0, 1}
        """
        # Ensure patterns have correct shape
        if len(patterns.shape) != 2:
            raise ValueError("Patterns should be a 2D tensor of shape (n_patterns, n_neurons)")
            
        if patterns.shape[1] != self.n_neurons:
            raise ValueError(f"Pattern dimension {patterns.shape[1]} doesn't match network size {self.n_neurons}")
        
        # Convert to numpy array for computation
        if isinstance(patterns, torch.Tensor):
            patterns_np = patterns.detach().cpu().numpy()
        else:
            patterns_np = patterns
            
        n_patterns = patterns_np.shape[0]
        
        # Initialize weight matrix
        self.weights = np.zeros((self.n_neurons, self.n_neurons))
        
        # Hebbian learning rule: W = (1/N) * Σ(p_i * p_i^T)
        for pattern in patterns_np:
            # Outer product
            self.weights += np.outer(pattern, pattern)
        
        # Normalize
        self.weights /= n_patterns
        
        # Remove self-connections (set diagonal elements to 0)
        np.fill_diagonal(self.weights, 0)
        
        self.is_trained = True
        
    def energy(self, state: torch.Tensor) -> float:
        """
        Calculate energy of current state
        
        E = -0.5 * s^T * W * s
        
        Args:
            state: Network state, shape (n_neurons,)
                   Values should be {-1, 1} or {0, 1}
            
        Returns:
            Energy value of current state
        """
        if not self.is_trained:
            raise ValueError("Network must be trained before computing energy")
            
        # Convert to numpy array
        if isinstance(state, torch.Tensor):
            state_np = state.detach().cpu().numpy()
        else:
            state_np = state
            
        # Ensure it's a 1D vector
        state_np = state_np.flatten()
        
        # Calculate energy E = -0.5 * s^T * W * s
        energy = -0.5 * np.dot(state_np, np.dot(self.weights, state_np))
        
        return float(energy)
    
    def _update_rule(self, state: torch.Tensor, neuron_idx: int) -> int:
        """
        Apply update rule to a single neuron
        
        s_i(t+1) = sign(Σ_j W_ij s_j(t))
        
        Args:
            state: Current network state
            neuron_idx: Index of neuron to update
            
        Returns:
            Updated neuron value
        """
        # Convert to numpy array
        if isinstance(state, torch.Tensor):
            state_np = state.detach().cpu().numpy()
        else:
            state_np = state
            
        # Ensure it's a 1D vector
        state_np = state_np.flatten()
        
        # Calculate activation
        activation = np.dot(self.weights[neuron_idx], state_np)
        
        # Apply sign function
        if activation > 0:
            return 1
        elif activation < 0:
            return -1
        else:
            # Keep current value if activation is 0
            return int(state_np[neuron_idx])
    
    def update_async(self, state: torch.Tensor, max_iter: int = 100, 
                    convergence_threshold: float = 0.001,
                    record_energy: bool = True) -> Tuple[torch.Tensor, List[float]]:
        """
        Asynchronous update (one neuron at a time) until convergence
        
        Args:
            state: Initial state
            max_iter: Maximum number of iterations
            convergence_threshold: Convergence threshold (stop when energy change is less than this)
            record_energy: Whether to record energy changes
            
        Returns:
            (final_state, energy_history): Final state and energy history
        """
        if not self.is_trained:
            raise ValueError("Network must be trained before updating")
            
        # Convert to numpy array for manipulation
        if isinstance(state, torch.Tensor):
            current_state = state.detach().cpu().numpy().flatten()
        else:
            current_state = state.flatten().copy()
            
        # Record energy changes
        energy_history = []
        if record_energy:
            energy_history.append(self.energy(torch.from_numpy(current_state)))
        
        # Asynchronous update
        for iteration in range(max_iter):
            # Randomly select a neuron to update
            neuron_idx = np.random.randint(0, self.n_neurons)
            
            # Update that neuron
            current_state[neuron_idx] = self._update_rule(
                torch.from_numpy(current_state), neuron_idx)
            
            # Record energy
            if record_energy and iteration % 5 == 0:  # Record every 5 steps
                energy_history.append(self.energy(torch.from_numpy(current_state)))
            
            # Check convergence
            if record_energy and len(energy_history) > 1:
                energy_change = abs(energy_history[-1] - energy_history[-2])
                if energy_change < convergence_threshold:
                    break
        
        # Convert back to torch tensor
        final_state = torch.from_numpy(current_state)
        
        return final_state, energy_history
    
    def update_sync(self, state: torch.Tensor, max_iter: int = 100,
                   convergence_threshold: float = 0.001,
                   record_energy: bool = True) -> Tuple[torch.Tensor, List[float]]:
        """
        Synchronous update (all neurons at once) until convergence
        
        Args:
            state: Initial state
            max_iter: Maximum number of iterations
            convergence_threshold: Convergence threshold
            record_energy: Whether to record energy changes
            
        Returns:
            (final_state, energy_history): Final state and energy history
        """
        if not self.is_trained:
            raise ValueError("Network must be trained before updating")
            
        # Convert to numpy array for manipulation
        if isinstance(state, torch.Tensor):
            current_state = state.detach().cpu().numpy().flatten()
        else:
            current_state = state.flatten().copy()
            
        # Record energy changes
        energy_history = []
        if record_energy:
            energy_history.append(self.energy(torch.from_numpy(current_state)))
        
        # Synchronous update
        for iteration in range(max_iter):
            # Calculate activations for all neurons
            activations = np.dot(self.weights, current_state)
            
            # Apply sign function
            new_state = np.sign(activations)
            
            # Handle zero activations (keep original values)
            zero_mask = (activations == 0)
            new_state[zero_mask] = current_state[zero_mask]
            
            # Update state
            current_state = new_state
            
            # Record energy
            if record_energy:
                energy_history.append(self.energy(torch.from_numpy(current_state)))
            
            # Check convergence
            if record_energy and len(energy_history) > 1:
                energy_change = abs(energy_history[-1] - energy_history[-2])
                if energy_change < convergence_threshold:
                    break
        
        # Convert back to torch tensor
        final_state = torch.from_numpy(current_state)
        
        return final_state, energy_history
    
    def recall(self, probe: torch.Tensor, update_mode: str = 'async',
               max_iter: int = 100, convergence_threshold: float = 0.001,
               record_energy: bool = True) -> Tuple[torch.Tensor, List[float]]:
        """
        Associative memory: retrieve stored pattern from probe
        
        Args:
            probe: Probe pattern (corrupted input)
            update_mode: Update mode ('async' or 'sync')
            max_iter: Maximum number of iterations
            convergence_threshold: Convergence threshold
            record_energy: Whether to record energy changes
            
        Returns:
            (recalled_state, energy_history): Recalled state and energy history
        """
        if update_mode == 'async':
            return self.update_async(
                probe, max_iter, convergence_threshold, record_energy)
        elif update_mode == 'sync':
            return self.update_sync(
                probe, max_iter, convergence_threshold, record_energy)
        else:
            raise ValueError(f"Unknown update mode: {update_mode}")
    
    def get_attractors(self, n_random_states: int = 100) -> List[torch.Tensor]:
        """
        Find network attractors by evolving from multiple random states
        
        Args:
            n_random_states: Number of random initial states
            
        Returns:
            List of unique attractors
        """
        if not self.is_trained:
            raise ValueError("Network must be trained before finding attractors")
            
        attractors = []
        
        for _ in range(n_random_states):
            # Generate random initial state
            random_state = torch.randint(0, 2, (self.n_neurons,)) * 2 - 1  # Values are {-1, 1}
            
            # Evolve to stable state
            final_state, _ = self.recall(random_state)
            
            # Check if this attractor already exists
            is_duplicate = False
            for attractor in attractors:
                if torch.allclose(final_state, attractor):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                attractors.append(final_state)
        
        return attractors
    
    def calculate_capacity(self) -> float:
        """
        Calculate theoretical capacity of Hopfield network (maximum number of stored patterns)
        
        According to theory, Hopfield network capacity is approximately 0.15N, where N is the number of neurons
        
        Returns:
            Theoretical capacity of the network
        """
        return 0.15 * self.n_neurons
    
    def analyze_weights(self) -> dict:
        """
        Analyze statistical properties of weight matrix
        
        Returns:
            Dictionary with weight matrix statistics
        """
        if not self.is_trained:
            raise ValueError("Network must be trained before analyzing weights")
            
        # Calculate statistics of weight matrix
        weights_flat = self.weights.flatten()
        
        # Calculate eigenvalues
        eigenvalues = np.linalg.eigvals(self.weights)
        
        stats = {
            'mean': float(np.mean(weights_flat)),
            'std': float(np.std(weights_flat)),
            'min': float(np.min(weights_flat)),
            'max': float(np.max(weights_flat)),
            'eigenvalues': {
                'mean': float(np.mean(eigenvalues)),
                'std': float(np.std(eigenvalues)),
                'min': float(np.min(eigenvalues)),
                'max': float(np.max(eigenvalues)),
                'real_parts': [float(ev.real) for ev in eigenvalues],
                'imag_parts': [float(ev.imag) for ev in eigenvalues]
            }
        }
        
        return stats