import torch
from scipy.stats import qmc
import random

def random_normal_initialization(pop_size, solution_dim, bounds, device='cpu'):
    """Initializes population using random sampling from a normal distribution scaled to bounds."""
    min_b, max_b = bounds
    # Ensure bounds are floats for calculation
    min_b, max_b = float(min_b), float(max_b)
    mean = (max_b + min_b) / 2.0
    std_dev = (max_b - min_b) / 6.0 # Heuristic: covers ~99.7% of range within 3 std devs
    population = torch.randn(pop_size, solution_dim, device=device) * std_dev + mean
    # Clamp to be strictly within bounds
    min_b_tensor = torch.tensor(min_b, dtype=population.dtype, device=device)
    max_b_tensor = torch.tensor(max_b, dtype=population.dtype, device=device)
    return torch.clamp(population, min_b_tensor, max_b_tensor)

def latin_hypercube_initialization(pop_size, solution_dim, bounds=(-5.12, 5.12), device='cpu'):
    """Initializes population using Latin Hypercube Sampling."""
    min_bound, max_bound = bounds
    # Ensure bounds are floats
    min_bound, max_bound = float(min_bound), float(max_bound)
    # Use a different random seed for LHS in each call to avoid identical initial swarms across runs
    lhc_seed = random.randint(1, 1000000)
    sampler = qmc.LatinHypercube(d=solution_dim, seed=lhc_seed)
    samples_unit_cube = sampler.random(n=pop_size) # Samples in [0, 1]^d
    # Scale samples to the desired bounds
    samples_scaled = qmc.scale(samples_unit_cube, min_bound, max_bound)
    return torch.tensor(samples_scaled, dtype=torch.float32).to(device)

def opposition_based_initialization(pop_size, solution_dim, bounds=(-5.12, 5.12), fitness_function=None, device='cpu'):
    """Initializes population using Opposition-Based Sampling (selection variant)."""
    if fitness_function is None:
        raise ValueError("Opposition-Based Initialization (selection variant) requires a fitness function.")

    min_bound, max_bound = bounds
    # Ensure bounds are floats
    min_bound, max_bound = float(min_bound), float(max_bound)

    # Generate N random points P
    initial_pop = torch.rand(pop_size, solution_dim, device=device) * (max_bound - min_bound) + min_bound
    # Generate N opposition points OP
    opposition_pop = min_bound + max_bound - initial_pop

    # Combine P and OP
    full_candidate_pop = torch.cat((initial_pop, opposition_pop), dim=0)
    # Ensure candidates are within bounds (opposition might slightly exceed due to float precision)
    min_b_tensor = torch.tensor(min_bound, dtype=full_candidate_pop.dtype, device=device)
    max_b_tensor = torch.tensor(max_bound, dtype=full_candidate_pop.dtype, device=device)
    full_candidate_pop = torch.clamp(full_candidate_pop, min_b_tensor, max_b_tensor)

    # Evaluate the fitness (lower benchmark value is better)
    # Ensure input to fitness function is float
    fitness_values = fitness_function(full_candidate_pop.float())

    # Select the best 'pop_size' individuals (lowest fitness values)
    _, top_indices = torch.topk(fitness_values, pop_size, largest=False) # largest=False for minimization
    selected_pop = full_candidate_pop[top_indices]

    return selected_pop.to(device)
