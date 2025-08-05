# DeepSeed: A Reinforcement Learning Framework for Learning Initial Solution Distributions in Metaheuristic Optimization

This repository contains the official implementation of the DeepSeed framework, as presented in the paper "DeepSeed: A Reinforcement Learning Framework for Learning Initial Solution Distributions in Metaheuristic Optimization".

## Overview

DeepSeed is a modular and extensible framework that learns initialization strategies for population-based metaheuristics using reinforcement learning. It trains a generative model to propose problem-specific initial populations, leading to improved convergence speed and solution quality.

## Repository Structure

- `src/deepseed/`: Contains the core components of the DeepSeed framework.
  - `generator.py`: The generative model that learns to produce initial populations.
  - `optimizers.py`: Implementations of metaheuristic algorithms (PSO and GA).
  - `benchmark_functions.py`: A collection of benchmark functions for evaluation.
  - `initialization.py`: Standard initialization methods (Random, LHS, OBS).
  - `utils.py`: Utility functions for plotting and calculations.
- `experiments/`: Contains the training and evaluation scripts.
  - `train.py`: The main training script for the DeepSeed framework.
- `results/`: The default output directory for experiment results (plots, logs, etc.).
- `run_experiments.py`: The main entry point for running the experiments.

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- NumPy
- Pandas
- Matplotlib
- SciPy

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/deepseed.git
   cd deepseed
   ```

2. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

### Running the Experiments

To run the experiments, simply execute the `run_experiments.py` script:

```bash
python run_experiments.py
```

The script will run the DeepSeed framework with the default configuration, training a generator for each benchmark function and saving the results in the `results/` directory.

## Citation

If you use this code in your research, please cite the original paper:

```
@article{sayah2025deepseed,
  title={DeepSeed: A Reinforcement Learning Framework for Learning Initial Solution Distributions in Metaheuristic Optimization},
  author={Sayah, Mohamed El amir and Ghedjemis, Fatiha},
  journal={Information Sciences},
  year={2025}
}
```
