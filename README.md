# DeepSeed: A Reinforcement Learning Framework for Learning Initial Solution Distributions in Metaheuristic Optimization

This repository contains the official implementation of the DeepSeed framework, as presented in the paper "DeepSeed: A Reinforcement Learning Framework for Learning Initial Solution Distributions in Metaheuristic Optimization".

## Overview

DeepSeed is a modular and extensible framework that learns initialization strategies for population-based metaheuristics using reinforcement learning. It trains a generative model to propose problem-specific initial populations, leading to improved convergence speed and solution quality.



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

Download the dataset with `dataset_downloader.py`
```bash
dataset_downloader.py
```

Try running the experiments ,`deepseed_adversarial_attack.py` and `deepseed_rastring_benchmarck_function.py`


```bash
python deepseed_adversarial_attack.py
python deepseed_rastring_benchmarck_function.py
```

Those scripts will run the DeepSeed framework with the default configuration, training a generator for each benchmark function and using the trained checkpoint model for the adversarial attack test in  `model_checkpoint/` directory.


