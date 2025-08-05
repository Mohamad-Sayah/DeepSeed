import torch
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import os
import pandas as pd
import time
import matplotlib.pyplot as plt

from deepseed.generator import Generator
from deepseed.optimizers import run_pso_refinement, run_ga_refinement
from deepseed.benchmark_functions import BENCHMARK_FUNCTIONS
from deepseed.initialization import random_normal_initialization, latin_hypercube_initialization, opposition_based_initialization
from deepseed.utils import kl_divergence_std_only, plot_distribution

def train_hybrid_loss_pso_variant(
    initialization_type,    # "generator" (other types won't be called from main loop)
    benchmark_func,
    benchmark_name,
    bounds,
    solution_dim,
    noise_dim,
    output_dir,             # Now the run-specific output directory
    training_iterations,
    swarm_size,
    max_pso_iterations,
    batch_size,
    w_range, c1, c2,
    lambda_objective,       # Weight for initial objective penalty (Generator only)
    lambda_kl,              # Weight for KL divergence penalty (Generator only)
    reward_scale = 1.0,     # Scales the reward signal
    run_index = 0,          # To add to print statements
    save_plots=True         # Control plotting inside the function
    ):

    # Ensure output directory exists FOR THIS RUN
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Ensure initialization_type is "generator" if this function is intended to be used only for it
    # This function can still technically run other types if called directly with them,
    # but the main loop will only call it with "generator".
    print(f"    [Run {run_index+1}/{NUM_RUNS}] Running {benchmark_name} ({initialization_type}) on {device}")


    # --- Initialize History Lists ---
    best_fitness_history = []
    reward_history = []
    policy_loss_hist = []        # Generator specific
    objective_loss_hist = []     # Tracks avg initial objective value (for all types)
    kl_loss_hist = []            # Generator specific
    total_loss_hist = []         # Generator specific

    # --- Setup Generator (if applicable) ---
    generator = None
    optimizer = None

    if initialization_type == "generator":
        generator = Generator(noise_dim, solution_dim).to(device)
        optimizer = optim.Adam(generator.parameters(), lr=1e-4) # Consider making lr a parameter
        optimizer.zero_grad()
        if save_plots:
            initial_gen_samples = generator.sample(500, device)
            plot_distribution(initial_gen_samples,
                              f"Initial Generator Distribution ({benchmark_name})",
                              os.path.join(output_dir, f"{benchmark_name}_Gen_Initial_Distribution.png"),
                              bounds, solution_dim)
    # The following elif branches for "random_normal", etc., are kept for structural completeness
    # but won't be hit if the main loop only calls with initialization_type="generator".
    elif initialization_type in ["random_normal", "latin_hypercube", "opposition_based"]:
        # This part will not be executed based on the user's request to only run the generator method.
        # However, the function structure is kept for now.
        pass
    else:
        raise ValueError(f"Unknown initialization type: {initialization_type}")


    # --- Training Loop ---
    overall_best_fitness_in_run = -float('inf')
    accumulated_loss = torch.tensor(0.0, device=device)

    for it in range(training_iterations):
        log_probs = None
        distribution_objective_term = torch.tensor(0.0, device=device)
        distribution_kl_term = torch.tensor(0.0, device=device)
        initial_best_fitness = -float('inf')

        if initialization_type == "generator":
            generator.train()
            z = torch.randn(swarm_size, noise_dim, device=device)
            mean, std = generator(z)
            dist = Normal(mean, std)
            epsilon = torch.randn_like(std)
            initial_positions = mean + std * epsilon
            initial_positions = initial_positions.float()

            log_probs = dist.log_prob(initial_positions).sum(dim=1)
            initial_objective_vals = benchmark_func(initial_positions)
            distribution_objective_term = initial_objective_vals.mean()
            kl_div = kl_divergence_std_only(mean, std)
            distribution_kl_term = kl_div.mean()
            initial_fitness_vals = -initial_objective_vals.detach()
            if initial_fitness_vals.numel() > 0:
                 initial_best_fitness = initial_fitness_vals.max().item()
        # The following elif block is technically unreachable if main loop only calls 'generator'
        elif initialization_type in ["random_normal", "latin_hypercube", "opposition_based"]:
            if initialization_type == "random_normal":
                initial_positions = random_normal_initialization(swarm_size, solution_dim, bounds, device=device)
            elif initialization_type == "latin_hypercube":
                initial_positions = latin_hypercube_initialization(swarm_size, solution_dim, bounds, device=device)
            elif initialization_type == "opposition_based":
                initial_positions = opposition_based_initialization(swarm_size, solution_dim, bounds, fitness_function=benchmark_func, device=device)

            initial_positions = initial_positions.float()
            initial_objective_vals = benchmark_func(initial_positions)
            initial_fitness_vals = -initial_objective_vals.detach()
            if initial_fitness_vals.numel() > 0:
                 initial_best_fitness = initial_fitness_vals.max().item()
            distribution_objective_term = torch.tensor(initial_objective_vals.mean().item(), device=device)
            log_probs = torch.zeros(swarm_size, device=device) # Placeholder
            distribution_kl_term = torch.tensor(0.0, device=device) # Placeholder
        else: # Should not be reached
            raise ValueError(f"Unsupported initialization type in training loop: {initialization_type}")


        final_gbest_position, fitness_after_pso_iter = run_pso_refinement(
            initial_positions.detach().clone(),
            benchmark_func,
            max_pso_iterations,
            swarm_size,
            solution_dim,
            w_range, c1, c2, bounds
        )

        if fitness_after_pso_iter > overall_best_fitness_in_run:
            overall_best_fitness_in_run = fitness_after_pso_iter

        reward = (fitness_after_pso_iter - initial_best_fitness) * reward_scale
        reward_item = reward if isinstance(reward, (float, int)) else reward.item()

        best_fitness_history.append(fitness_after_pso_iter)
        reward_history.append(reward_item)
        objective_loss_hist.append(distribution_objective_term.item()) # Still useful for generator

        if initialization_type == "generator":
            policy_loss_term = -torch.tensor(reward_item, device=device) * log_probs.mean()
            total_loss_this_iter = (policy_loss_term
                                    + lambda_objective * distribution_objective_term
                                    + lambda_kl * distribution_kl_term)
            accumulated_loss = accumulated_loss + total_loss_this_iter

            policy_loss_hist.append(policy_loss_term.item())
            kl_loss_hist.append(distribution_kl_term.item())
            total_loss_hist.append(total_loss_this_iter.item())

            if (it + 1) % batch_size == 0 and optimizer is not None:
                 if accumulated_loss.requires_grad:
                    average_loss = accumulated_loss / batch_size
                    average_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                 accumulated_loss = torch.tensor(0.0, device=device)
        else: # For non-generator types (not expected to run)
            policy_loss_hist.append(0.0) # Placeholder
            kl_loss_hist.append(0.0) # Placeholder
            total_loss_hist.append(0.0) # Placeholder

        if (it + 1) % 500 == 0:
            print(f"      Iter {it+1}/{training_iterations} ({initialization_type}) | PSO Best Fit (Iter): {fitness_after_pso_iter:.4f} | Reward: {reward_item:.4f}")


    if initialization_type == "generator" and optimizer is not None and (training_iterations % batch_size != 0):
         if accumulated_loss.requires_grad and accumulated_loss.item() != 0:
             remaining_iters_in_batch = training_iterations % batch_size
             if remaining_iters_in_batch > 0:
                 average_loss = accumulated_loss / remaining_iters_in_batch
                 average_loss.backward()
                 optimizer.step()
                 optimizer.zero_grad()

    if initialization_type == "generator" and generator is not None and save_plots:
        final_gen_samples = generator.sample(500, device)
        plot_distribution(final_gen_samples,
                          f"Final Generator Distribution ({benchmark_name})",
                          os.path.join(output_dir, f"{benchmark_name}_Gen_Final_Distribution.png"),
                          bounds, solution_dim)

    print(f"    [Run {run_index+1}/{NUM_RUNS}] Finished {benchmark_name} ({initialization_type}). Final Best Fitness in Run: {overall_best_fitness_in_run:.6f}")

    histories = {
        "best_fitness": best_fitness_history,
        "reward": reward_history,
        "policy_loss": policy_loss_hist,
        "objective_loss": objective_loss_hist,
        "kl_loss": kl_loss_hist,
        "total_loss": total_loss_hist,
    }
    return histories, overall_best_fitness_in_run
