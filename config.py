"""
Global configuration file
Centralizes management of all hyperparameters for easy experimental adjustments
Separates configurations for Hopfield and RBM
Defines data paths and result save paths
"""

import os

# Project root directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# MNIST configuration
MNIST_PATH = os.path.join(ROOT_DIR, "data")
IMAGE_SIZE = 28 * 28  # 784
BINARY_THRESHOLD = 0.5

# Hopfield network configuration
HOPFIELD_CONFIG = {
    'num_patterns': 3,          # Number of stored patterns
    'noise_levels': [0.1, 0.2, 0.3, 0.4, 0.5],
    'max_iterations': 100,
    'convergence_threshold': 0.001
}

# RBM configuration
RBM_CONFIG = {
    'n_visible': 784,
    'n_hidden': 256,
    'learning_rate': 0.01,
    'batch_size': 64,
    'epochs': 20,
    'k': 1,  # CD-k sampling steps
    'temperatures': [0.5, 1.0, 2.0, 5.0]
}

# Result save paths
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
HOPFIELD_RESULTS_DIR = os.path.join(RESULTS_DIR, "hopfield")
RBM_RESULTS_DIR = os.path.join(RESULTS_DIR, "rbm")

# Hopfield result subdirectories
HOPFIELD_ENERGY_PLOTS_DIR = os.path.join(HOPFIELD_RESULTS_DIR, "energy_plots")
HOPFIELD_RECOVERED_IMAGES_DIR = os.path.join(HOPFIELD_RESULTS_DIR, "recovered_images")
HOPFIELD_INTERFERENCE_ANALYSIS_DIR = os.path.join(HOPFIELD_RESULTS_DIR, "interference_analysis")

# RBM result subdirectories
RBM_WEIGHTS_DIR = os.path.join(RBM_RESULTS_DIR, "weights")
RBM_GENERATED_SAMPLES_DIR = os.path.join(RBM_RESULTS_DIR, "generated_samples")
RBM_FEATURE_VIZ_DIR = os.path.join(RBM_RESULTS_DIR, "feature_viz")

# Ensure result directories exist
for directory in [
    RESULTS_DIR,
    HOPFIELD_RESULTS_DIR,
    HOPFIELD_ENERGY_PLOTS_DIR,
    HOPFIELD_RECOVERED_IMAGES_DIR,
    HOPFIELD_INTERFERENCE_ANALYSIS_DIR,
    RBM_RESULTS_DIR,
    RBM_WEIGHTS_DIR,
    RBM_GENERATED_SAMPLES_DIR,
    RBM_FEATURE_VIZ_DIR
]:
    os.makedirs(directory, exist_ok=True)