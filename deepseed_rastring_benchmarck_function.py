import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import random
import matplotlib.pyplot as plt

# Ensure reproducibility for comparison (optional)
# torch.manual_seed(42)
# np.random.seed(42)
# random.seed(42)

# ------------------------------
# 1. Rastrigin Function Definition (Unchanged)
# ------------------------------
def rastrigin(x, A=10):
    n = x.shape[-1]
    x = x.float()
    term1 = x**2
    term2 = - A * torch.cos(2 * torch.pi * x)
    return A * n + torch.sum(term1 + term2, dim=-1)

# ------------------------------
# 2. Generator Network (Unchanged)
# ------------------------------
class Generator(nn.Module):
    def __init__(self, noise_dim, solution_dim):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.solution_dim = solution_dim
        self.fc = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(),
            nn.Linear(128, solution_dim * 2)
        )

    def forward(self, z):
        params = self.fc(z)
        mean, log_std = params.chunk(2, dim=1)
        # Add clamping for stability as in the ELBO example
        log_std = torch.clamp(log_std, -5, 2) # Prevents extreme std values
        std = torch.exp(log_std) + 1e-6      # Add epsilon for numerical stability
        return mean, std

# ------------------------------
# 3. PSO Operations (Unchanged)
# ------------------------------
def pso_update(positions, velocities, pbest_positions, gbest_position, w, c1, c2, solution_dim, bounds=None):
    swarm_size = positions.shape[0]
    device = positions.device
    r1 = torch.rand(swarm_size, solution_dim, device=device)
    r2 = torch.rand(swarm_size, solution_dim, device=device)
    inertia_term = w * velocities
    cognitive_term = c1 * r1 * (pbest_positions - positions)
    social_term = c2 * r2 * (gbest_position.unsqueeze(0) - positions)
    new_velocities = inertia_term + cognitive_term + social_term
    new_positions = positions + new_velocities
    if bounds is not None:
        min_bound, max_bound = bounds
        new_positions = torch.clamp(new_positions, min_bound, max_bound)
    return new_positions, new_velocities

def run_pso_refinement(initial_positions, fitness_function, max_pso_iterations, swarm_size, solution_dim, w_range, c1, c2, bounds=None):
    device = initial_positions.device
    positions = initial_positions.clone()
    velocities = torch.zeros_like(positions, device=device)
    # Maximize -rastrigin (equivalent to minimizing rastrigin)
    # We calculate fitness relative to the minimization objective here
    fitness_vals = -fitness_function(positions)
    pbest_positions = positions.clone()
    pbest_fitness = fitness_vals.clone()
    gbest_fitness, gbest_idx = torch.max(pbest_fitness, dim=0)
    gbest_position = pbest_positions[gbest_idx].clone()
    w_start, w_end = w_range

    for iteration in range(max_pso_iterations):
        w = w_start - (w_start - w_end) * (iteration / max_pso_iterations)
        positions, velocities = pso_update(positions, velocities, pbest_positions, gbest_position, w, c1, c2, solution_dim, bounds)
        current_fitness = -fitness_function(positions)
        update_pbest_mask = current_fitness > pbest_fitness
        pbest_positions[update_pbest_mask] = positions[update_pbest_mask].clone()
        pbest_fitness[update_pbest_mask] = current_fitness[update_pbest_mask].clone()
        current_best_fitness, current_best_idx = torch.max(pbest_fitness, dim=0)
        if current_best_fitness > gbest_fitness:
            gbest_fitness = current_best_fitness.clone()
            gbest_position = pbest_positions[current_best_idx].clone()

    # Return the best position found and its fitness (-Rastrigin value)
    return gbest_position, gbest_fitness.item()

# ------------------------------
# 4. Random Normal Initialization Function (Unchanged)
# ------------------------------
def random_normal_initialization(pop_size, solution_dim, mean=0.0, std_dev=1.5):
    population = [torch.randn(solution_dim) * std_dev + mean for _ in range(pop_size)]
    return torch.stack(population)

# ------------------------------
# 5. KL Divergence Function (Modified to focus on std)
# ------------------------------
def kl_divergence_std_only(mean, std): # Renamed for clarity
    """
    Calculates a KL-like divergence term focusing only on the standard deviation,
    comparing it to a standard deviation of 1. This encourages std > 0
    without penalizing the mean.
    Formula: 0.5 * sum(std^2 - 1 - log(std^2)) per sample.
    """
    # mean is unused in this calculation but kept for consistent function signature
    std = std + 1e-8 # Add epsilon for numerical stability before log
    var = std.pow(2)
    log_var = torch.log(var) # log(std^2)
    # Original term: var + mean.pow(2) - 1.0 - log_var
    # Modified term (removing mean.pow(2)):
    kl_div = 0.5 * torch.sum(var - 1.0 - log_var, dim=1)
    # Ensure no NaNs or Infs, though the epsilon should help
    kl_div = torch.nan_to_num(kl_div, nan=0.0, posinf=1e6, neginf=-1e6)
    return kl_div

# Original KL divergence for reference (if needed)
def kl_divergence_normal_standard_normal(mean, std):
    std = std + 1e-8
    var = std.pow(2)
    log_var = torch.log(var)
    kl_div = 0.5 * torch.sum(var + mean.pow(2) - 1.0 - log_var, dim=1)
    return kl_div


# ------------------------------
# 6. Training Function (Adapted for Hybrid Loss)
# ------------------------------
def train_hybrid_loss_pso_variant(
    initialization_type,
    training_iterations,
    best_fitness_history,
    reward_history,
    policy_loss_hist, # Renamed for clarity
    rastrigin_loss_hist,
    kl_loss_hist,
    total_loss_hist,
    # --- NEW Hyperparameters ---
    lambda_rastrigin=0.1, # Weight for initial Rastrigin penalty
    lambda_kl=0.01       # Weight for KL penalty
    ):

    noise_dim = 20
    solution_dim = 10
    swarm_size = 50
    max_pso_iterations = 100
    batch_size = 5 # Generator update frequency

    # PSO hyperparameters
    w_range = (0.9, 0.4)
    c1 = 1.5
    c2 = 1.5
    pso_bounds = None

    # RL reward scaling
    reward_scale = 1.0 # Can be adjusted

    experiment_name = f"{initialization_type.capitalize()}Init_PSO_HybridLoss_StdKL" # Updated name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    generator = None
    optimizer = None
    if initialization_type == "generator":
        generator = Generator(noise_dim, solution_dim).to(device)
        optimizer = optim.Adam(generator.parameters(), lr=1e-4)
        optimizer.zero_grad() # Ensure gradients are cleared initially

    overall_best_fitness = -float('inf') # Track best fitness (-Rastrigin) across all iterations
    accumulated_loss = torch.tensor(0.0, device=device) # Accumulate loss over batch

    for it in range(training_iterations):
        # --- Step A: Generate Initial Swarm Positions ---
        if initialization_type == "generator":
            generator.train() # Ensure generator is in training mode
            z = torch.randn(swarm_size, noise_dim, device=device)
            mean, std = generator(z)
            dist = Normal(mean, std)
            # Sample using reparameterization trick
            epsilon = torch.randn_like(std) # Sample noise
            initial_positions = mean + std * epsilon # Sampled positions

            # --- Calculate terms needed for loss BEFORE PSO ---
            # Log probabilities of the sampled initial positions
            # Ensure log_prob calculation uses the *actual* sampled positions
            log_probs = dist.log_prob(initial_positions).sum(dim=1) # Shape: [swarm_size]

            # Initial Rastrigin values (for penalty term and reward calculation)
            # IMPORTANT: Keep gradient tracking for initial_positions here for the penalty term!
            initial_rastrigin_vals = rastrigin(initial_positions)
            distribution_rastrigin_term = initial_rastrigin_vals.mean() # Avg Rastrigin of initial samples

            # KL divergence term (using the std-only version)
            kl_div = kl_divergence_std_only(mean, std) # USE THE MODIFIED FUNCTION
            distribution_kl_term = kl_div.mean()

            # Initial best fitness (for reward calculation, use detached values)
            initial_fitness_vals = -initial_rastrigin_vals.detach() # Fitness = -Rastrigin
            initial_best_fitness = initial_fitness_vals.max().item()

        elif initialization_type == "random_normal":
            initial_positions = random_normal_initialization(swarm_size, solution_dim).to(device)
            initial_rastrigin_vals = rastrigin(initial_positions)
            initial_fitness_vals = -initial_rastrigin_vals
            initial_best_fitness = initial_fitness_vals.max().item()
            # Dummy values for random init
            log_probs = torch.zeros(swarm_size, device=device)
            distribution_rastrigin_term = torch.tensor(initial_rastrigin_vals.mean().item(), device=device) # Use scalar value
            distribution_kl_term = torch.tensor(0.0, device=device)
        else:
            raise ValueError(f"Unknown initialization type: {initialization_type}")

        # --- Step B: PSO Loop to Refine the Swarm ---
        final_gbest_position, final_gbest_fitness = run_pso_refinement(
            initial_positions.detach().clone(), # PSO operates on detached positions
            rastrigin,
            max_pso_iterations,
            swarm_size,
            solution_dim,
            w_range, c1, c2, pso_bounds
        )

        # Track overall best solution found so far
        if final_gbest_fitness > overall_best_fitness:
            overall_best_fitness = final_gbest_fitness

        # --- Step C: Compute Reward & RL Update (Generator Only) ---
        if initialization_type == "generator":
            # Reward is the improvement achieved by PSO
            # Ensure reward is a scalar tensor on the correct device
            reward = torch.tensor((final_gbest_fitness - initial_best_fitness) * reward_scale, device=device)

            # Calculate loss components
            # Use mean of log_probs for stable gradient estimate
            policy_loss_term = -reward * log_probs.mean() # REINFORCE-like term

            # Total Loss = RL + Initial Rastrigin Penalty + KL(std) Penalty
            total_loss_this_iter = (policy_loss_term
                                    + lambda_rastrigin * distribution_rastrigin_term
                                    + lambda_kl * distribution_kl_term)

            # Accumulate loss for batch update
            accumulated_loss = accumulated_loss + total_loss_this_iter

            # Store history for analysis (use .item() to store numbers, not tensors)
            best_fitness_history.append(final_gbest_fitness) # Best fitness *after* PSO
            reward_history.append(reward.item())
            policy_loss_hist.append(policy_loss_term.item())
            rastrigin_loss_hist.append(distribution_rastrigin_term.item()) # Store avg initial rastrigin
            kl_loss_hist.append(distribution_kl_term.item()) # Store avg KL (std-only)
            total_loss_hist.append(total_loss_this_iter.item())


            # --- Step D: Update Generator Network (Batched) ---
            if (it + 1) % batch_size == 0:
                # Average loss over the batch
                average_loss = accumulated_loss / batch_size
                # Backpropagate the average loss
                average_loss.backward()
                # Optional: Gradient Clipping
                # torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                # Reset accumulated loss
                accumulated_loss = torch.tensor(0.0, device=device)

            if (it + 1) % 100 == 0: # Print less frequently
                print(f"[{experiment_name}] Iter {it+1}/{training_iterations} | "
                      f"PSO Best Fit: {final_gbest_fitness:.3f} | "
                      f"Reward: {reward.item():.3f} | "
                      f"Loss (Tot/Pol/Ras/KL_std): {total_loss_this_iter.item():.3f} / "
                      f"{policy_loss_term.item():.3f} / "
                      f"{lambda_rastrigin * distribution_rastrigin_term.item():.3f} / "
                      f"{lambda_kl * distribution_kl_term.item():.3f}")


        elif initialization_type == "random_normal":
            # Just store performance, no learning
            reward = (final_gbest_fitness - initial_best_fitness) * reward_scale
            best_fitness_history.append(final_gbest_fitness)
            reward_history.append(reward)
            # Append dummy values for loss lists
            policy_loss_hist.append(0.0)
            rastrigin_loss_hist.append(initial_rastrigin_vals.mean().item()) # Log initial rastrigin
            kl_loss_hist.append(0.0)
            total_loss_hist.append(0.0)

            if (it + 1) % 100 == 0: # Print less frequently
                print(f"[{experiment_name}] Iter {it+1}/{training_iterations} | "
                      f"PSO Best Fit: {final_gbest_fitness:.3f} | "
                      f"Reward: {reward:.3f} | Loss: N/A")


    # Final update for any remaining gradients in the generator case
    if initialization_type == "generator" and optimizer is not None and (training_iterations % batch_size != 0):
         if accumulated_loss.requires_grad and accumulated_loss != 0: # Check if there's unapplied loss with grad
             # Average over remaining iterations in the batch
             remaining_iters = training_iterations % batch_size
             if remaining_iters > 0:
                 average_loss = accumulated_loss / remaining_iters
                 average_loss.backward()
                 # Optional: Gradient Clipping
                 # torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                 optimizer.step()
                 optimizer.zero_grad()

    return overall_best_fitness


# ------------------------------
# 7. Run Training for Both Initializations
# ------------------------------
training_iterations = 3000
lambda_r = 0.05 # Weight for initial Rastrigin penalty -> Tune this
lambda_k = 0 # Weight for KL(std) penalty -> Tune this (might need different value than before)

# Lists to store results
bf_hist_gen, rew_hist_gen, pol_loss_gen, ras_loss_gen, kl_loss_gen, tot_loss_gen = [], [], [], [], [], []
bf_hist_rand, rew_hist_rand, pol_loss_rand, ras_loss_rand, kl_loss_rand, tot_loss_rand = [], [], [], [], [], []

print(f"--- Training with Generator + Hybrid Loss (KL on Std Only, lambda_r={lambda_r}, lambda_k={lambda_k}) ---")
final_fitness_gen = train_hybrid_loss_pso_variant(
    "generator", training_iterations,
    bf_hist_gen, rew_hist_gen, pol_loss_gen, ras_loss_gen, kl_loss_gen, tot_loss_gen,
    lambda_rastrigin=lambda_r, lambda_kl=lambda_k
)

print(f"\n--- Training with Random Normal + PSO Refinement ---")
final_fitness_rand = train_hybrid_loss_pso_variant(
    "random_normal", training_iterations,
    bf_hist_rand, rew_hist_rand, pol_loss_rand, ras_loss_rand, kl_loss_rand, tot_loss_rand,
    lambda_rastrigin=lambda_r, lambda_kl=lambda_k # Pass dummy values
)

# ------------------------------
# 8. Plot Training Progress (Unchanged plotting code)
# ------------------------------
plt.figure(figsize=(18, 10)) # Increased figure height
window_size = 50 # Moving average window

# Plot 1: Best Fitness (-Rastrigin) Found After PSO
plt.subplot(2, 3, 1)
plt.plot(bf_hist_gen, label='Gen Hybrid (StdKL)', color='blue', alpha=0.7)
plt.plot(bf_hist_rand, label='Rand Norm', color='green', linestyle='--', alpha=0.7)
if len(bf_hist_gen) > window_size:
    gen_smooth = np.convolve(bf_hist_gen, np.ones(window_size)/window_size, mode='valid')
    plt.plot(np.arange(window_size-1, len(bf_hist_gen)), gen_smooth, color='navy', linewidth=2, label=f'Gen (MA {window_size})')
if len(bf_hist_rand) > window_size:
    rand_smooth = np.convolve(bf_hist_rand, np.ones(window_size)/window_size, mode='valid')
    plt.plot(np.arange(window_size-1, len(bf_hist_rand)), rand_smooth, color='darkgreen', linewidth=2, linestyle='--', label=f'Rand (MA {window_size})')
plt.xlabel("Iteration")
plt.ylabel("Best Fitness (-Rastrigin)")
plt.title("Best Fitness After PSO")
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()

# Plot 2: Reward (PSO Improvement)
plt.subplot(2, 3, 2)
plt.plot(rew_hist_gen, label='Reward (Gen)', color='blue', alpha=0.7)
plt.plot(rew_hist_rand, label='Reward (Rand)', color='green', linestyle='--', alpha=0.7)
if len(rew_hist_gen) > window_size:
    gen_rew_smooth = np.convolve(rew_hist_gen, np.ones(window_size)/window_size, mode='valid')
    plt.plot(np.arange(window_size-1, len(rew_hist_gen)), gen_rew_smooth, color='navy', linewidth=2, label=f'Gen Rew (MA {window_size})')
if len(rew_hist_rand) > window_size:
    rand_rew_smooth = np.convolve(rew_hist_rand, np.ones(window_size)/window_size, mode='valid')
    plt.plot(np.arange(window_size-1, len(rew_hist_rand)), rand_rew_smooth, color='darkgreen', linewidth=2, linestyle='--', label=f'Rand Rew (MA {window_size})')
plt.xlabel("Iteration")
plt.ylabel("Reward (Improvement)")
plt.title("Reward per Iteration")
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()

# Plot 3: Total Loss (Generator Only)
plt.subplot(2, 3, 3)
plt.plot(tot_loss_gen, label='Total Loss (Gen)', color='red', alpha=0.7)
if len(tot_loss_gen) > window_size:
    loss_smooth = np.convolve(tot_loss_gen, np.ones(window_size)/window_size, mode='valid')
    plt.plot(np.arange(window_size-1, len(tot_loss_gen)), loss_smooth, color='darkred', linewidth=2, label=f'Total Loss (MA {window_size})')
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Total Gen Loss per Iteration")
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()

# Plot 4: Policy Loss Component (Generator Only)
plt.subplot(2, 3, 4)
plt.plot(pol_loss_gen, label='Policy Loss Term', color='purple', alpha=0.7)
if len(pol_loss_gen) > window_size:
    pol_loss_smooth = np.convolve(pol_loss_gen, np.ones(window_size)/window_size, mode='valid')
    plt.plot(np.arange(window_size-1, len(pol_loss_gen)), pol_loss_smooth, color='indigo', linewidth=2, label=f'Policy Loss (MA {window_size})')
plt.xlabel("Iteration")
plt.ylabel("Loss Component")
plt.title("Policy Loss (-Rew*logP) Term")
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()


# Plot 5: Average Initial Rastrigin Value (Both Methods)
plt.subplot(2, 3, 5)
# Note: ras_loss_gen stores the raw average rastrigin value (before lambda scaling)
plt.plot(ras_loss_gen, label='Avg Init Rastrigin (Gen)', color='blue', alpha=0.7)
plt.plot(ras_loss_rand, label='Avg Init Rastrigin (Rand)', color='green', linestyle='--', alpha=0.7)
if len(ras_loss_gen) > window_size:
    ras_gen_smooth = np.convolve(ras_loss_gen, np.ones(window_size)/window_size, mode='valid')
    plt.plot(np.arange(window_size-1, len(ras_loss_gen)), ras_gen_smooth, color='navy', linewidth=2, label=f'Gen Init Ras (MA {window_size})')
if len(ras_loss_rand) > window_size:
    ras_rand_smooth = np.convolve(ras_loss_rand, np.ones(window_size)/window_size, mode='valid')
    plt.plot(np.arange(window_size-1, len(ras_loss_rand)), ras_rand_smooth, color='darkgreen', linewidth=2, linestyle='--', label=f'Rand Init Ras (MA {window_size})')
plt.axhline(0, color='black', linestyle=':', label='Ideal Minimum (0.0)')
plt.xlabel("Iteration")
plt.ylabel("Avg Rastrigin Value")
plt.title("Avg Rastrigin of Initial Swarm")
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
#plt.ylim(bottom=-5)

# Plot 6: KL Divergence Component (Generator Only)
plt.subplot(2, 3, 6)
# Note: kl_loss_gen stores the raw KL divergence (before lambda scaling)
plt.plot(kl_loss_gen, label=f'KL Div (Std Only) Term', color='orange', alpha=0.7) # Updated Label
if len(kl_loss_gen) > window_size:
    kl_loss_smooth = np.convolve(kl_loss_gen, np.ones(window_size)/window_size, mode='valid')
    plt.plot(np.arange(window_size-1, len(kl_loss_gen)), kl_loss_smooth, color='darkorange', linewidth=2, label=f'KL Div (Std) (MA {window_size})') # Updated Label
plt.xlabel("Iteration")
plt.ylabel("KL Divergence (Std Only)") # Updated Label
plt.title("KL Divergence (Std Only) Term") # Updated Title
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()


plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# Get the actual solution_dim used in the last run
try:
    sol_dim_plot = generator.solution_dim if generator else solution_dim
except NameError:
     sol_dim_plot = solution_dim # Fallback if generator wasn't created

title = (f"Hybrid Loss (RL + Init Ras + KL_Std) vs Random Init w/ PSO Refinement\n"
         f"(Rastrigin Dim={sol_dim_plot}, $\\lambda_{{Ras}}$={lambda_r}, $\\lambda_{{KL_{{Std}}}}$={lambda_k})") # Updated Title
plt.suptitle(title, y=1.01, fontsize=14)
plt.show()

# ------------------------------
# 9. Report Final Results
# ------------------------------
print("\nOptimization complete.")
print(f"[GeneratorInit_HybridLoss_StdKL] Overall Best Fitness Found: {max(bf_hist_gen):.4f}") # Updated Name
print(f"[RandomNormalInit_PSO] Overall Best Fitness Found: {max(bf_hist_rand):.4f}")

print(f"\nValue at minimum (f(0)=0) -> Fitness = { -rastrigin(torch.zeros(1, sol_dim_plot)).item() }") # Use sol_dim_plot