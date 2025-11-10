# Neural Networks MNIST: Hopfield & RBM

[![GitHub Repository](https://img.shields.io/badge/GitHub-Jackksonns-blue.svg)](https://github.com/Jackksonns/neural-networks-mnist)

This project implements Hopfield networks and Restricted Boltzmann Machines (RBM) based on the MNIST dataset for research in pattern recognition, memory recovery, and generative models.

## Project Structure

```
kkkbo/
├── data/                           # Data storage directory
├── notebooks/                      # Jupyter notebooks directory
├── results/                        # Experimental results directory
│   ├── hopfield/                   # Hopfield network experiment results
│   │   ├── energy_plots/           # Energy plots
│   │   ├── recovery_plots/         # Recovery process plots
│   │   └── weight_plots/           # Weight visualization plots
│   └── rbm/                        # RBM experiment results
│       ├── weights/                # Weight analysis results
│       ├── generated_samples/      # Generated samples
│       └── feature_viz/            # Feature visualization
├── src/                            # Source code directory
│   ├── hopfield/                   # Hopfield network module
│   │   ├── __init__.py
│   │   ├── network.py              # Hopfield network core implementation
│   │   ├── visualizer.py           # Visualization tools
│   │   └── experiments.py          # Experiment scripts
│   ├── boltzmann/                  # RBM module
│   │   ├── __init__.py
│   │   ├── rbm.py                  # RBM core implementation
│   │   ├── sampler.py              # Sampler
│   │   └── experiments.py          # Experiment scripts
│   └── utils/                      # Utility modules
│       ├── __init__.py
│       ├── data_loader.py          # Data loader
│       └── preprocessing.py        # Data preprocessing tools
├── config.py                       # Global configuration file
├── main.py                         # Main entry script
├── requirements.txt                # Dependency package list
└── README.md                       # Project documentation
```

## Installation and Configuration

### Requirements

- Python 3.7+
- PyTorch 1.8+
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- Jupyter (optional)

### Installation Steps

1. Clone the project to your local machine:
```bash
git clone https://github.com/Jackksonns/neural-networks-mnist.git
cd neural-networks-mnist
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Run All Experiments

```bash
python main.py --mode all
```

### Run Specific Experiments

1. Run Hopfield network experiments:
```bash
python main.py --mode hopfield
```

2. Run RBM experiments:
```bash
python main.py --mode rbm
```

3. Run Hopfield network memory test:
```bash
python main.py --mode hopfield_test
```

4. Run RBM generation test:
```bash
python main.py --mode rbm_test
```

### Specify Computing Device

```bash
python main.py --device cuda  # Use GPU
python main.py --device cpu   # Use CPU
```

## Features

### Hopfield Network

- Training based on Hebbian learning rule
- Asynchronous and synchronous update modes
- Energy function calculation
- Pattern recovery and associative memory
- Robustness testing under noise interference
- Capacity analysis
- Weight matrix visualization

### Restricted Boltzmann Machine (RBM)

- Training with contrastive divergence algorithm
- Multiple sampling methods (Gibbs, block Gibbs, annealed sampling, parallel tempering)
- Free energy calculation
- Feature learning and visualization
- Sample generation and quality assessment
- Weight pattern analysis
- "Dreaming" experiments

### Data Processing

- MNIST data loading and preprocessing
- Image binarization (two methods: {0,1} and {-1,1})
- Noise addition and occlusion
- Data normalization
- Feature dimensionality reduction and visualization

## Experimental Results

### Hopfield Network Experiments

1. **Pattern Storage and Recovery**: Test the network's ability to store and recover digit patterns
2. **Noise Interference Experiments**: Evaluate the network's recovery performance under different noise levels
3. **Capacity Analysis**: Study the relationship between the number of patterns the network can store and the number of neurons
4. **Attractor Analysis**: Visualize energy landscapes and convergence processes

### RBM Experiments

1. **Training Process**: Show the change of reconstruction error with training epochs
2. **Sample Generation**: Generate new MNIST-style digit images
3. **Weight Analysis**: Visualize the weight patterns of hidden units
4. **Feature Learning**: Visualize learned feature representations through t-SNE and PCA
5. **Sampling Method Comparison**: Compare the generation effects of different sampling methods

## Configuration Description

The main configuration parameters of the project are in the `config.py` file:

```python
# MNIST data configuration
MNIST_CONFIG = {
    'download': True,
    'data_dir': './data',
    'batch_size': 64
}

# Hopfield network configuration
HOPFIELD_CONFIG = {
    'num_patterns': 3,      # Number of stored patterns
    'max_iterations': 100,  # Maximum number of iterations
    'convergence_threshold': 0.01  # Convergence threshold
}

# RBM configuration
RBM_CONFIG = {
    'n_visible': 784,       # Number of visible layer units (28*28)
    'n_hidden': 256,        # Number of hidden layer units
    'k': 1,                 # Steps of contrastive divergence
    'learning_rate': 0.01,  # Learning rate
    'momentum': 0.5,        # Momentum
    'weight_decay': 0.0001  # Weight decay
}
```

## Extensions and Customization

### Adding New Experiments

1. Add new experiment methods in the corresponding `experiments.py` file
2. Add command-line options in `main.py`
3. Update the configuration file (if needed)

### Custom Datasets

1. Modify the data loading logic in `src/utils/data_loader.py`
2. Update the data configuration in `config.py`
3. Adjust the network structure to adapt to new data

### Adjusting Network Structure

1. Modify network parameters in `src/hopfield/network.py` or `src/boltzmann/rbm.py`
2. Update the configuration in `config.py`
3. May need to adjust visualization code to adapt to the new network structure

## Frequently Asked Questions

### Q: What to do if training takes too long?

A: You can try the following methods:
- Reduce the number of training epochs
- Increase batch size
- Use GPU acceleration
- Reduce network scale

### Q: The quality of generated samples is not high?

A: You can try:
- Increase the number of training epochs
- Adjust the learning rate
- Increase the number of hidden layer units
- Use more complex sampling methods

### Q: Hopfield network cannot recover patterns?

A: Possible reasons:
- The number of stored patterns exceeds the network capacity
- The similarity between patterns is too high
- The noise level is too high

## References

1. Hopfield, J. J. (1982). Neural networks and physical systems with emergent collective computational abilities. Proceedings of the National Academy of Sciences, 79(8), 2554-2558.

2. Hinton, G. E. (2002). Training products of experts by minimizing contrastive divergence. Neural computation, 14(8), 1771-1800.

3. Ackley, D. H., Hinton, G. E., & Sejnowski, T. J. (1985). A learning algorithm for Boltzmann machines. Cognitive science, 9(1), 147-169.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing

Issue reports and pull requests are welcome. Before submitting, please ensure:

1. Code follows the project style
2. Appropriate tests are added
3. Related documentation is updated

## Contact

For questions or suggestions, please contact us through:

- GitHub: https://github.com/Jackksonns/neural-networks-mnist