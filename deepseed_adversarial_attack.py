# -*- coding: utf-8 -*-
"""
Created on Tue May  6 06:39:02 2025

@author: M
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 09:54:22 2025

@author: M
"""

from colorama import Fore, Back, Style, init, deinit
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt

# Initialize colorama for colored terminal output
init()

# ------------------------------
# 0. Configuration
# ------------------------------
TRAINING_MODE = False  # SET TO True FOR TRAINING, False FOR INFERENCE/EVALUATION
LAMBDA_KL = 0.0       # Weight for KL divergence penalty

# ------------------------------
# 1. Device, Target Model & Transforms
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# Use a pre-trained ResNet18 as the target classifier
target_model = models.resnet18(pretrained=True) # Set to False if you don't have internet / want to use cached
# target_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1) # Alternative for newer torchvision
target_model = target_model.to(device)
target_model.eval()  # Freeze target model

# ImageNet normalization parameters and reduced image size for speed
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]
resize_dim = 64       # Resize shorter side to 64
crop_dim   = resize_dim - 8  # Then center-crop

transform = transforms.Compose([
    transforms.Resize(resize_dim),
    transforms.CenterCrop(crop_dim),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

print(f"[INFO] Using transformed image size: {crop_dim}x{crop_dim}")

# Compute per-channel lower and upper bounds in the normalized domain
lower_bound = torch.tensor([(0 - m) / s for m, s in zip(imagenet_mean, imagenet_std)]).view(3, 1, 1).to(device)
upper_bound = torch.tensor([(1 - m) / s for m, s in zip(imagenet_mean, imagenet_std)]).view(3, 1, 1).to(device)

def unnormalize(tensor):
    """Unnormalize a tensor and convert to numpy for display."""
    img = tensor.squeeze(0).cpu().permute(1, 2, 0).detach().numpy()
    img = img * np.array(imagenet_std) + np.array(imagenet_mean)
    return np.clip(img, 0, 1)

# ------------------------------
# 2. Fully Conditional Stochastic Perturbation Generator (Improved)
# ------------------------------
class FullyConditionalStochasticPerturbationGenerator(nn.Module):
    def __init__(self, noise_dim, image_shape, num_classes, class_embedding_dim=16):
        super(FullyConditionalStochasticPerturbationGenerator, self).__init__()
        self.noise_dim = noise_dim
        self.image_shape = image_shape

        self.class_embedding = nn.Embedding(num_classes, class_embedding_dim)

        self.image_branch = nn.Sequential(
            nn.Conv2d(image_shape[0], 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        image_feat_dim = 32
        combined_dim = noise_dim + image_feat_dim + class_embedding_dim

        self.fc = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Linear(512, int(np.prod(image_shape)) * 2)
        )

    def forward(self, z, image, target_class):
        img_feat = self.image_branch(image)
        img_feat = img_feat.view(img_feat.size(0), -1)
        class_embed = self.class_embedding(target_class)
        combined = torch.cat([z, img_feat, class_embed], dim=1)

        x = self.fc(combined)
        batch_size = x.size(0)
        x = x.view(batch_size, 2, *self.image_shape)
        mean = x[:, 0, :, :, :]
        log_std = x[:, 1, :, :, :]

        # Clamp log_std for stability
        log_std = torch.clamp(log_std, -5, 2)
        std = torch.exp(log_std) + 1e-6 # Add epsilon for numerical stability

        eps = torch.randn_like(std)
        sample = mean + std * eps

        # Compute log probability (ensure std isn't exactly zero for division)
        log_prob = -0.5 * (((sample - mean) / (std + 1e-8)) ** 2 + 2 * log_std + np.log(2 * np.pi))
        log_prob = log_prob.sum(dim=[1, 2, 3])
        return sample, log_prob, mean, std # Return mean and std for KL divergence

# Define dimensions and instantiate the generator
noise_dim = 100
image_shape = (3, crop_dim, crop_dim)
num_classes = 1000
generator = FullyConditionalStochasticPerturbationGenerator(noise_dim, image_shape, num_classes).to(device)

# ------------------------------
# 2a. Load Previously Trained Generator (if available)
# ------------------------------
MODEL_SAVE_PATH = "Models/generator_model_updated1249.pth" # Generic name, will be iteration-stamped

if os.path.exists(MODEL_SAVE_PATH):
    print(f"[INFO] Loading trained generator model from: {MODEL_SAVE_PATH}")
    try:
        state_dict = torch.load(MODEL_SAVE_PATH, map_location=device)
        generator.load_state_dict(state_dict)
        print("[INFO] Generator model loaded successfully.")
    except Exception as e:
        print(f"[WARNING] Could not load generator model: {e}. Initializing new generator.")
else:
    print("[INFO] No saved generator model found. Initializing new generator.")

if TRAINING_MODE:
    generator.train()
    gen_optimizer = optim.Adam(generator.parameters(), lr=1e-4)
    print("[INFO] Generator set to TRAINING mode.")
else:
    generator.eval()
    print("[INFO] Generator set to EVALUATION mode.")


# ------------------------------
# 2b. KL Divergence Function
# ------------------------------
def kl_divergence_perturbation_std_only(std_batch):
    """
    Calculates a KL-like divergence term focusing only on the standard deviation,
    comparing it to a standard deviation of 1. This encourages std > 0.
    std_batch shape: [batch_size, C, H, W]
    """
    var = std_batch.pow(2)
    # Add epsilon inside log for stability, though std should already have it
    log_var = torch.log(var + 1e-8)
    # For each pixel: 0.5 * (var - 1 - log_var)
    # This is KL( N(mu, std) || N(mu, 1) ) if mu is the same.
    # Or, more accurately, part of KL( N(0, std) || N(0, 1) ) related to variance.
    kl_div_pixels = 0.5 * (var - 1.0 - log_var)
    kl_div_samples = torch.sum(kl_div_pixels, dim=[1, 2, 3]) # Sum over C, H, W
    return kl_div_samples # Shape: [batch_size]

# ------------------------------
# 3. Genetic Algorithm (GA) Functions
# ------------------------------
def fitness_function(perturbed_images, target_label_idx):
    with torch.no_grad(): # Ensure no gradients for fitness evaluation
        outputs = target_model(perturbed_images)
        probs = F.softmax(outputs, dim=1)
        target_probs = probs[:, target_label_idx]
    return target_probs.cpu().tolist() # Return as list of floats

def create_initial_population_generator(pop_size, image, image_shape, perturb_mag, noise_dim, generator_model, target_class_idx):
    population = []
    log_probs = []
    noise_vectors = []

    z = torch.randn(pop_size, noise_dim, device=device)
    image_batch = image.repeat(pop_size, 1, 1, 1)
    target_tensor = torch.tensor([target_class_idx] * pop_size, dtype=torch.long, device=device)

    # Generator might be in eval mode if not TRAINING_MODE, ensure grads for training
    if generator_model.training:
         samples, lp, means_init, stds_init = generator_model(z, image_batch, target_tensor)
    else: # No grad needed if not training generator, or if called for baseline
        with torch.no_grad():
            samples, lp, means_init, stds_init = generator_model(z, image_batch, target_tensor)

    for i in range(pop_size):
        population.append(samples[i] * perturb_mag) # Scale generator output
        log_probs.append(lp[i])
        noise_vectors.append(z[i])
    return population, log_probs, noise_vectors, means_init, stds_init

def create_initial_population_random(pop_size, image_shape, perturb_mag, device):
    population = []
    for _ in range(pop_size):
        # Perturbations are added to the image, so they are the deltas
        perturb = torch.randn(image_shape, device=device) * perturb_mag
        population.append(perturb)
    return population

def selection(population, fitness_vals, num_parents):
    pop_fit = list(zip(population, fitness_vals))
    pop_fit.sort(key=lambda x: x[1], reverse=True) # Higher fitness is better
    selected_perturbations = [p.clone() for p, _ in pop_fit[:num_parents]]
    return selected_perturbations

def crossover(parents, crossover_rate, image_shape, pop_size):
    offspring = []
    num_parents = len(parents)
    idx_list = list(range(num_parents))

    while len(offspring) < pop_size:
        random.shuffle(idx_list) # Shuffle indices to pick parents
        for i in range(0, num_parents, 2):
            if len(offspring) >= pop_size: break
            p1_idx, p2_idx = idx_list[i], idx_list[(i+1)%num_parents] # Ensure pairs
            parent1, parent2 = parents[p1_idx], parents[p2_idx]

            if random.random() < crossover_rate and i + 1 < num_parents : # Ensure two distinct parents for crossover
                cp_row = random.randint(1, image_shape[1] - 2) # Avoid edges for simplicity
                cp_col = random.randint(1, image_shape[2] - 2)

                child1 = parent1.clone()
                child2 = parent2.clone()

                # Single point crossover on flattened or structured (e.g., quadrant swap)
                # Example: Quadrant swap
                child1_temp_quadrant = parent1[:, :cp_row, :cp_col].clone()
                child1[:, :cp_row, :cp_col] = parent2[:, :cp_row, :cp_col]
                child2[:, :cp_row, :cp_col] = child1_temp_quadrant

                offspring.extend([child1, child2])
            else:
                offspring.append(parent1.clone())
                if len(offspring) < pop_size: # Check before adding second parent
                     offspring.append(parent2.clone())
            if len(offspring) >= pop_size: break

    return offspring[:pop_size]


def mutation(population, mutation_rate, perturb_mag_mutation_noise_scale):
    mutated = []
    for cand_perturb in population:
        if random.random() < mutation_rate:
            # Add small Gaussian noise to the perturbation itself
            noise = torch.randn_like(cand_perturb, device=cand_perturb.device) * perturb_mag_mutation_noise_scale
            mutated.append(cand_perturb + noise)
        else:
            mutated.append(cand_perturb.clone())
    return mutated

def run_ga_attack(initial_population_perturbations, orig_tensor, target_class_idx, max_gens,
                  pop_size, image_shape, base_perturb_mag, # base_perturb_mag for mutation noise scale
                  init_mut_rate, min_mut_rate, cross_rate, patience,
                  lower_b, upper_b, classifier_model, ga_log_prefix="GA"):

    population_perturbations = [p.clone().to(device) for p in initial_population_perturbations]

    best_overall_perturb = population_perturbations[0].clone()
    best_overall_fitness = -float('inf')
    generations_actually_used = max_gens
    attack_succeeded_flag = False

    current_mutation_rate = init_mut_rate
    no_improvement_streak = 0

    for gen_idx in range(max_gens):
        # Add perturbations to original image and clamp
        current_pop_tensor_perturb = torch.stack(population_perturbations)
        orig_img_batch = orig_tensor.repeat(pop_size, 1, 1, 1)
        perturbed_img_batch = torch.clamp(orig_img_batch + current_pop_tensor_perturb, lower_b, upper_b)

        current_fitness_values = fitness_function(perturbed_img_batch, target_class_idx)

        current_gen_best_idx = np.argmax(current_fitness_values)
        current_gen_max_fitness = current_fitness_values[current_gen_best_idx]

        if current_gen_max_fitness > best_overall_fitness:
            best_overall_fitness = current_gen_max_fitness
            best_overall_perturb = population_perturbations[current_gen_best_idx].clone()
            no_improvement_streak = 0
        else:
            no_improvement_streak += 1

        # Check for success with the best perturbation of this generation
        with torch.no_grad():
            test_img_adv = torch.clamp(orig_tensor + population_perturbations[current_gen_best_idx].unsqueeze(0), lower_b, upper_b)
            adv_output = classifier_model(test_img_adv)
            adv_pred_idx = torch.argmax(adv_output, dim=1).item()

        if adv_pred_idx == target_class_idx:
            generations_actually_used = gen_idx + 1
            attack_succeeded_flag = True
            # print(f"[{ga_log_prefix}] Attack succeeded at gen {generations_actually_used} (Target: {target_class_idx}, Pred: {adv_pred_idx}), Fitness: {best_overall_fitness:.4f}")
            # Visualize on success immediately if desired (can be noisy)
            # if TRAINING_MODE: # Only plot if training, otherwise too many plots in eval
            #     orig_disp = unnormalize(orig_tensor)
            #     pert_disp = unnormalize(test_img_adv)
            #     plt.figure(figsize=(6, 3))
            #     plt.subplot(1, 2, 1); plt.imshow(orig_disp); plt.title("Original"); plt.axis("off")
            #     plt.subplot(1, 2, 2); plt.imshow(pert_disp); plt.title(f"{ga_log_prefix} Perturbed"); plt.axis("off")
            #     plt.suptitle(f"{ga_log_prefix} Success Gen {generations_actually_used}"); plt.show()
            break # Exit GA loop on success

        if no_improvement_streak >= patience:
            # print(f"[{ga_log_prefix}] Early stopping at gen {gen_idx + 1} due to no improvement. Best fitness: {best_overall_fitness:.4f}")
            generations_actually_used = gen_idx + 1
            break

        num_parents_to_select = pop_size // 2
        selected_parents = selection(population_perturbations, current_fitness_values, num_parents_to_select)

        # Ensure selected_parents are on device if selection moves them
        selected_parents = [p.to(device) for p in selected_parents]

        offspring_perturbations = crossover(selected_parents, cross_rate, image_shape, pop_size)
        # Mutation noise scale can be smaller than initial perturbation magnitude
        population_perturbations = mutation(offspring_perturbations, current_mutation_rate, perturb_mag_mutation_noise_scale=base_perturb_mag * 0.1) # Smaller noise for mutation

        current_mutation_rate = max(init_mut_rate / (1 + best_overall_fitness), min_mut_rate) # Adapt mutation

    return best_overall_perturb, best_overall_fitness, generations_actually_used, attack_succeeded_flag

# ------------------------------
# 4. Dataset Setup
# ------------------------------
dataset_root = '../dog-and-cat-classification-dataset'
if not os.path.exists(dataset_root):
    print(f"[WARNING] Dataset root {dataset_root} not found. Using a dummy image path.")
    # Create a dummy structure if needed for testing without the actual dataset
    if not os.path.exists("dummy_images"): os.makedirs("dummy_images")
    try:
        from PIL import Image as PILImage
        dummy_img = PILImage.new('RGB', (100, 100), color = 'red')
        dummy_img.save("dummy_images/dummy_image.jpg")
        image_paths = ["dummy_images/dummy_image.jpg"]
    except ImportError:
        image_paths = [] # No images if PIL is not available and path is dummy
else:
    dog_folder = os.path.join(dataset_root, 'Dog')
    if not os.path.exists(dog_folder):
        print(f"[WARNING] Dog folder {dog_folder} not found. Using a dummy image path.")
        image_paths = ["dummy_images/dummy_image.jpg"] if os.path.exists("dummy_images/dummy_image.jpg") else []
    else:
        all_image_files = [fname for fname in os.listdir(dog_folder) if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_paths = [os.path.join(dog_folder, fname) for fname in all_image_files]

if not image_paths:
    print(Back.RED + "[ERROR] No images found. Please check dataset_root or provide dummy images." + Style.RESET_ALL)
    # exit() # Critical error if no images
    # For robust execution, let's try to proceed if in training mode with a warning, or exit if in eval.
    if not TRAINING_MODE:
        print(Back.RED + "[ERROR] Exiting due to no images in EVALUATION mode." + Style.RESET_ALL)
        exit()
    else:
        print(Back.YELLOW + "[WARNING] Proceeding with training loop, but it will likely fail without images." + Style.RESET_ALL)


print(f"[INFO] Found {len(image_paths)} images.")


# ------------------------------
# 5. Training Parameters
# ------------------------------
pop_size = 30              # GA population size (reduced for speed)
max_gens = 500             # Maximum GA generations per image attack (reduced for speed)
init_mut_rate = 0.5
min_mut_rate = 0.01
cross_rate = 0.3
perturb_mag = 0.05         # Magnitude scaling for perturbations from generator / random init

training_iterations = 1000 if TRAINING_MODE else 10 # Total images/attacks
eval_iterations = 300 # Number of images to test in EVALUATION_MODE

# RL/Generator training parameters
reward_scale = 1.0
alpha = 0.1       # Hyperparameter for norm penalty in reward
beta = 0.05       # Hyperparameter for A/B testing (initial pop fitness diff) contribution in reward
patience_ga = 20  # GA early stopping patience
batch_size_gen_update = 5    # Update generator every N attacks

# History tracking lists
gen_counts_rl_hist = []
final_fitness_rl_hist = []
successes_rl_hist = []

gen_counts_baseline_hist = []
final_fitness_baseline_hist = []
successes_baseline_hist = []

# For generator training
policy_gradient_terms_hist = []
kl_div_losses_hist = [] # Raw KL divergence values
norm_penalties_hist = [] # Store norm_penalty applied to reward
rewards_for_gen_update_hist = [] # Store the actual reward value used for generator update
total_generator_losses_hist = []
ab_fitness_diffs_hist = [] # For the A/B part of the reward

example_results = []

# ------------------------------
# 6. Main Loop: GA with RL Updates / Evaluation
# ------------------------------

if TRAINING_MODE and gen_optimizer is not None:
    gen_optimizer.zero_grad()

# Create rl_adv directory if it doesn't exist
if not os.path.exists("rl_adv"):
    os.makedirs("rl_adv")
    print("[INFO] Created directory: rl_adv")


current_iterations = training_iterations if TRAINING_MODE else eval_iterations

for it in range(current_iterations):
    if not image_paths: # Handle case where no images were found earlier
        print(Back.RED + f"[Iteration {it+1}/{current_iterations}] No image available. Skipping." + Style.RESET_ALL)
        if TRAINING_MODE: # Add dummy values to prevent plotting errors if training
            policy_gradient_terms_hist.append(0); kl_div_losses_hist.append(0); norm_penalties_hist.append(0)
            rewards_for_gen_update_hist.append(0); total_generator_losses_hist.append(0); ab_fitness_diffs_hist.append(0)
            gen_counts_rl_hist.append(max_gens); final_fitness_rl_hist.append(0); successes_rl_hist.append(False)
            gen_counts_baseline_hist.append(max_gens); final_fitness_baseline_hist.append(0); successes_baseline_hist.append(False)
        continue


    img_path = random.choice(image_paths)
    try:
        orig_img_pil = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f"[ERROR] Could not open image {img_path}: {e}. Skipping iteration.")
        continue
    orig_tensor = transform(orig_img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        orig_out = target_model(orig_tensor)
        orig_pred = torch.argmax(orig_out, dim=1).item()

    possible_targets = list(range(num_classes))
    if orig_pred in possible_targets:
        possible_targets.remove(orig_pred)
    if not possible_targets: # Should not happen with num_classes=1000
        target_class_idx = (orig_pred + 1) % num_classes
    else:
        target_class_idx = random.choice(possible_targets)

    print(Fore.YELLOW + f"\n--- Iteration {it+1}/{current_iterations} (Orig: {orig_pred}, Target: {target_class_idx}) ---" + Style.RESET_ALL)

    # --- RLAgent (Generator + GA) ---
    initial_pop_rl, log_probs_rl, _, means_gen_init, stds_gen_init = \
        create_initial_population_generator(pop_size, orig_tensor, image_shape, perturb_mag,
                                          noise_dim, generator, target_class_idx)

    # Evaluate initial generator population for A/B comparison and initial fitness
    with torch.no_grad():
        initial_pop_rl_tensor = torch.stack(initial_pop_rl).to(device)
        initial_perturbed_rl = torch.clamp(orig_tensor.repeat(pop_size, 1, 1, 1) + initial_pop_rl_tensor, lower_bound, upper_bound)
        initial_fitness_rl_list = fitness_function(initial_perturbed_rl, target_class_idx)
    initial_max_fitness_rl = np.max(initial_fitness_rl_list) if initial_fitness_rl_list else 0.0
    print(f"[RLAgent] Initial Max Fitness (from Gen): {initial_max_fitness_rl}")

    # Run GA for RLAgent
    best_perturb_rl, final_fitness_rl, gen_used_rl, success_rl = \
        run_ga_attack(initial_pop_rl, orig_tensor, target_class_idx, max_gens, pop_size, image_shape,
                      perturb_mag, init_mut_rate, min_mut_rate, cross_rate, patience_ga,
                      lower_bound, upper_bound, target_model, ga_log_prefix="RLAgent-GA")

    successes_rl_hist.append(success_rl)
    gen_counts_rl_hist.append(gen_used_rl)
    final_fitness_rl_hist.append(final_fitness_rl)
    print(f"[RLAgent-GA] Result: {'Success' if success_rl else 'Failure'} in {gen_used_rl} gens. Final Fitness: {final_fitness_rl}. Pert Norm: {torch.norm(best_perturb_rl).item()}")


    # --- Baseline (Random Init + GA) ---
    initial_pop_baseline = create_initial_population_random(pop_size, image_shape, perturb_mag, device)

    # Evaluate initial random population for A/B comparison
    with torch.no_grad():
        initial_pop_baseline_tensor = torch.stack(initial_pop_baseline).to(device)
        initial_perturbed_baseline = torch.clamp(orig_tensor.repeat(pop_size, 1, 1, 1) + initial_pop_baseline_tensor, lower_bound, upper_bound)
        initial_fitness_baseline_list = fitness_function(initial_perturbed_baseline, target_class_idx)
    initial_max_fitness_baseline = np.max(initial_fitness_baseline_list) if initial_fitness_baseline_list else 0.0
    # print(f"[Baseline] Initial Max Fitness (Random): {initial_max_fitness_baseline:.4f}")


    best_perturb_baseline, final_fitness_baseline, gen_used_baseline, success_baseline = \
        run_ga_attack(initial_pop_baseline, orig_tensor, target_class_idx, max_gens, pop_size, image_shape,
                      perturb_mag, init_mut_rate, min_mut_rate, cross_rate, patience_ga,
                      lower_bound, upper_bound, target_model, ga_log_prefix="Baseline-GA")

    successes_baseline_hist.append(success_baseline)
    gen_counts_baseline_hist.append(gen_used_baseline)
    final_fitness_baseline_hist.append(final_fitness_baseline)
    print(f"[Baseline-GA] Result: {'Success' if success_baseline else 'Failure'} in {gen_used_baseline} gens. Final Fitness: {final_fitness_baseline}. Pert Norm: {torch.norm(best_perturb_baseline).item()}")


    # --- Generator Update (only if TRAINING_MODE is True) ---
    if TRAINING_MODE and gen_optimizer is not None:
        # Delta fitness: improvement by RLAgent's GA over its *own initial* population's best
        # This is the "advantage" gained by the GA refinement starting from generator's samples.
        delta_fitness_for_reward = final_fitness_rl - initial_max_fitness_rl

        # Initial L2 norm of perturbations from generator (for penalty)
        # Ensure initial_pop_rl_tensor is on device and requires_grad is handled by generator output
        initial_gen_norms = torch.stack([torch.norm(p.detach()) for p in initial_pop_rl]) # Detach for norm calculation if p still has grad
        avg_initial_gen_norm = torch.mean(initial_gen_norms)

        # A/B fitness difference (initial generator vs initial random)
        ab_fitness_diff = initial_max_fitness_rl - initial_max_fitness_baseline
        ab_fitness_diffs_hist.append(ab_fitness_diff)

        # Reward for generator update
        # reward = delta_fitness_for_reward - alpha * avg_initial_gen_norm.item() + beta * ab_fitness_diff # Original reward style
        # Let's use final_fitness_rl directly as part of the reward signal, combined with penalties
        # Reward = (achieved_fitness_by_gen_ga) - (penalty_for_large_perturbations) + (bonus_for_better_initial_samples_than_random)
        current_reward = final_fitness_rl - alpha * avg_initial_gen_norm.item() + beta * ab_fitness_diff
        rewards_for_gen_update_hist.append(current_reward)
        norm_penalties_hist.append(alpha * avg_initial_gen_norm.item())

        # Policy gradient term
        # log_probs_rl were for the *initial* samples from the generator
        # We want to encourage initial samples that lead to high reward *after GA refinement*
        if log_probs_rl: # ensure log_probs_rl is not empty
            mean_log_prob = torch.mean(torch.stack(log_probs_rl))
            policy_gradient_term = -current_reward * mean_log_prob * reward_scale
        else: # Should not happen if pop_size > 0
            policy_gradient_term = torch.tensor(0.0, device=device, requires_grad=True)
        policy_gradient_terms_hist.append(policy_gradient_term.item())

        # KL divergence term for generator's initial output std
        if stds_gen_init is not None and stds_gen_init.numel() > 0 : # Check if stds_gen_init is valid
            kl_div_samples = kl_divergence_perturbation_std_only(stds_gen_init)
            kl_loss_component = kl_div_samples.mean()
        else: # Fallback if stds_gen_init is None or empty (e.g. pop_size 0, or error in generator)
            kl_loss_component = torch.tensor(0.0, device=device, requires_grad=True)

        kl_div_losses_hist.append(kl_loss_component.item()) # Store raw KL

        # Total loss for generator
        total_generator_loss = policy_gradient_term + LAMBDA_KL * kl_loss_component
        total_generator_losses_hist.append(total_generator_loss.item())

        # Accumulate gradients for batch updates
        if total_generator_loss.requires_grad: # Ensure loss requires grad before backward
             total_generator_loss.backward()
        else:
            print(Fore.RED + "[WARNING] total_generator_loss does not require grad. Skipping backward pass." + Style.RESET_ALL)


        if (it + 1) % batch_size_gen_update == 0:
            print(Fore.GREEN + f"[INFO] Updating generator at iteration {it+1}." + Style.RESET_ALL)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0) # Optional grad clipping
            gen_optimizer.step()
            gen_optimizer.zero_grad()

        # Save model periodically during training
        if (it + 1) % 50 == 0:
            current_save_path = f"rl_adv/generator_model_iter{it+1}.pth"
            torch.save(generator.state_dict(), current_save_path)
            print(f"[INFO] Generator model saved to: {current_save_path}")

            # Interim Plotting (similar to helper code style)
            plt.figure(figsize=(15, 12)) # Adjusted for potentially 4 rows

            # 1. Total Generator Loss
            plt.subplot(4, 1, 1)
            plt.plot(total_generator_losses_hist, label="Total Generator Loss", color='red')
            if len(total_generator_losses_hist) > 10:
                plt.plot(np.convolve(total_generator_losses_hist, np.ones(10)/10, mode='valid'), label='MA (10) Total Loss', color='darkred')
            plt.title("Total Generator Loss")
            plt.xlabel("Attack Iteration"); plt.ylabel("Loss")
            plt.grid(True); plt.legend()

            # 2. Loss Components
            plt.subplot(4, 1, 2)
            plt.plot(policy_gradient_terms_hist, label="Policy Gradient Term", color='blue', alpha=0.7)
            plt.plot([-n for n in norm_penalties_hist], label="-(Norm Penalty in Reward)", color='green', alpha=0.7) # Plot the penalty part
            if LAMBDA_KL > 0: # Only plot KL if it has a weight
                weighted_kl = [k * LAMBDA_KL for k in kl_div_losses_hist]
                plt.plot(weighted_kl, label=f"Weighted KL Divergence (lambda={LAMBDA_KL})", color='orange', alpha=0.7)
            plt.title("Generator Loss Components")
            plt.xlabel("Attack Iteration"); plt.ylabel("Value")
            plt.grid(True); plt.legend()

            # 3. A/B Fitness Difference (Initial Populations)
            plt.subplot(4, 1, 3)
            plt.plot(ab_fitness_diffs_hist, label="A/B Fitness Diff (Initial Gen - Initial Rand)", color='purple')
            if len(ab_fitness_diffs_hist) > 10:
                plt.plot(np.convolve(ab_fitness_diffs_hist, np.ones(10)/10, mode='valid'), label='MA (10) A/B Diff', color='indigo')
            plt.title("A/B Fitness Difference (Initial Populations)")
            plt.xlabel("Attack Iteration"); plt.ylabel("Fitness Difference")
            plt.grid(True); plt.legend()

            # 4. Moving Average of Success Rates
            plt.subplot(4, 1, 4)
            window = 20 # Moving average window for success rates
            if len(successes_rl_hist) >= window:
                rl_success_rate_ma = np.convolve(np.array(successes_rl_hist).astype(float), np.ones(window)/window, mode='valid')
                plt.plot(range(window-1, len(successes_rl_hist)), rl_success_rate_ma, label=f'RL Agent Success Rate (MA {window})', color='cyan')
            if len(successes_baseline_hist) >= window:
                baseline_success_rate_ma = np.convolve(np.array(successes_baseline_hist).astype(float), np.ones(window)/window, mode='valid')
                plt.plot(range(window-1, len(successes_baseline_hist)), baseline_success_rate_ma, label=f'Baseline Success Rate (MA {window})', color='magenta')
            plt.title("Attack Success Rates (Moving Average)")
            plt.xlabel("Attack Iteration"); plt.ylabel("Success Rate")
            plt.ylim(0, 1.1)
            plt.grid(True); plt.legend()

            plt.tight_layout()
            plt.show()

    # Save one example for final visualization
    if (it < 3 or success_rl) and len(example_results) < 3 : # Save if successful or one of first few
        with torch.no_grad():
            final_perturbed_rl_img = torch.clamp(orig_tensor + best_perturb_rl.unsqueeze(0), lower_bound, upper_bound)
        example_results.append({
            "original": orig_tensor.detach().cpu(),
            "perturbed_rl": final_perturbed_rl_img.detach().cpu(),
            "perturbed_baseline": torch.clamp(orig_tensor + best_perturb_baseline.unsqueeze(0), lower_bound, upper_bound).detach().cpu(),
            "success_rl": success_rl,
            "success_baseline": success_baseline,
            "gens_rl": gen_used_rl,
            "gens_baseline": gen_used_baseline
        })

    # If not training, might break early after a few evaluations
    if not TRAINING_MODE and it >= eval_iterations -1:
        break


# Final generator update if training and iterations not multiple of batch_size
if TRAINING_MODE and gen_optimizer is not None and (current_iterations % batch_size_gen_update != 0) and current_iterations > 0 :
    if any(p.grad is not None for p in generator.parameters()): # Check if there are gradients to apply
        print(Fore.GREEN + "[INFO] Performing final generator update." + Style.RESET_ALL)
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
        gen_optimizer.step()
        gen_optimizer.zero_grad()

# ------------------------------
# 7. Calculate and Log Final Success Rates
# ------------------------------
total_attacks = len(successes_rl_hist)
if total_attacks > 0:
    rl_agent_total_successes = sum(s for s in successes_rl_hist if s) # Summing booleans
    baseline_total_successes = sum(s for s in successes_baseline_hist if s)

    rl_agent_success_rate = rl_agent_total_successes / total_attacks if total_attacks > 0 else 0
    baseline_success_rate = baseline_total_successes / total_attacks if total_attacks > 0 else 0

    print(Fore.CYAN + "\n--- Final Attack Success Rates ---" + Style.RESET_ALL)
    print(f"RL Agent (Generator + GA): {rl_agent_total_successes}/{total_attacks} = {rl_agent_success_rate*100:.2f}%")
    print(f"Baseline (Random Init + GA): {baseline_total_successes}/{total_attacks} = {baseline_success_rate*100:.2f}%")
else:
    print(Fore.YELLOW + "\nNo attacks were run, cannot calculate success rates." + Style.RESET_ALL)

# ------------------------------
# 8. Plot Final Training Progress (if training was done)
# ------------------------------
if TRAINING_MODE and total_attacks > 0:
    print("\n[INFO] Plotting final training progress...")
    num_plots_horizontal = 2
    num_plots_vertical = 3 # For 6 plots
    plt.figure(figsize=(8 * num_plots_horizontal, 5 * num_plots_vertical))

    # Plot 1: Generations Used (RL vs Baseline)
    plt.subplot(num_plots_vertical, num_plots_horizontal, 1)
    plt.plot(gen_counts_rl_hist, label='RL Agent GA Gens', color='blue', alpha=0.6)
    plt.plot(gen_counts_baseline_hist, label='Baseline GA Gens', color='green', alpha=0.6, linestyle='--')
    if len(gen_counts_rl_hist) > 20:
        plt.plot(np.convolve(gen_counts_rl_hist, np.ones(20)/20, mode='valid'), color='navy', label='RL MA(20)')
    if len(gen_counts_baseline_hist) > 20:
        plt.plot(np.convolve(gen_counts_baseline_hist, np.ones(20)/20, mode='valid'), color='darkgreen', label='Baseline MA(20)', linestyle='--')
    plt.title("Generations per Attack")
    plt.xlabel("Attack Iteration"); plt.ylabel("Generations")
    plt.grid(True); plt.legend()

    # Plot 2: Final Fitness (Target Probability) (RL vs Baseline)
    plt.subplot(num_plots_vertical, num_plots_horizontal, 2)
    plt.plot(final_fitness_rl_hist, label='RL Agent Final Fitness', color='blue', alpha=0.6)
    plt.plot(final_fitness_baseline_hist, label='Baseline Final Fitness', color='green', alpha=0.6, linestyle='--')
    if len(final_fitness_rl_hist) > 20:
        plt.plot(np.convolve(final_fitness_rl_hist, np.ones(20)/20, mode='valid'), color='navy', label='RL MA(20)')
    if len(final_fitness_baseline_hist) > 20:
        plt.plot(np.convolve(final_fitness_baseline_hist, np.ones(20)/20, mode='valid'), color='darkgreen', label='Baseline MA(20)', linestyle='--')
    plt.title("Final Fitness (Target Probability)")
    plt.xlabel("Attack Iteration"); plt.ylabel("Fitness (Target Prob)")
    plt.grid(True); plt.legend()

    # Plot 3: Total Generator Loss
    plt.subplot(num_plots_vertical, num_plots_horizontal, 3)
    plt.plot(total_generator_losses_hist, label="Total Gen Loss", color='red', alpha=0.7)
    if len(total_generator_losses_hist) > 20:
        plt.plot(np.convolve(total_generator_losses_hist, np.ones(20)/20, mode='valid'), color='darkred', label='MA(20)')
    plt.title("Total Generator Loss")
    plt.xlabel("Attack Iteration"); plt.ylabel("Loss")
    plt.grid(True); plt.legend()

    # Plot 4: Policy Gradient Term
    plt.subplot(num_plots_vertical, num_plots_horizontal, 4)
    plt.plot(policy_gradient_terms_hist, label="Policy Gradient Term", color='purple', alpha=0.7)
    if len(policy_gradient_terms_hist) > 20:
        plt.plot(np.convolve(policy_gradient_terms_hist, np.ones(20)/20, mode='valid'), color='indigo', label='MA(20)')
    plt.title("Policy Gradient Component of Loss")
    plt.xlabel("Attack Iteration"); plt.ylabel("Value")
    plt.grid(True); plt.legend()

    # Plot 5: KL Divergence (Raw)
    plt.subplot(num_plots_vertical, num_plots_horizontal, 5)
    plt.plot(kl_div_losses_hist, label="KL Divergence (raw)", color='orange', alpha=0.7)
    if len(kl_div_losses_hist) > 20:
        plt.plot(np.convolve(kl_div_losses_hist, np.ones(20)/20, mode='valid'), color='darkorange', label='MA(20)')
    plt.title(f"KL Divergence (Std Dev) (Lambda_KL={LAMBDA_KL})")
    plt.xlabel("Attack Iteration"); plt.ylabel("KL Value")
    plt.grid(True); plt.legend()

    # Plot 6: Rewards used for Generator Update
    plt.subplot(num_plots_vertical, num_plots_horizontal, 6)
    plt.plot(rewards_for_gen_update_hist, label="Reward for Gen Update", color='teal', alpha=0.7)
    if len(rewards_for_gen_update_hist) > 20:
        plt.plot(np.convolve(rewards_for_gen_update_hist, np.ones(20)/20, mode='valid'), color='darkslategrey', label='MA(20)')
    plt.title("Reward Signal for Generator Update")
    plt.xlabel("Attack Iteration"); plt.ylabel("Reward Value")
    plt.grid(True); plt.legend()

    plt.tight_layout()
    plt.suptitle("Final Training Progress Summary", fontsize=16, y=1.02)
    plt.show()
elif not TRAINING_MODE:
    print("\n[INFO] Evaluation mode. Skipping final training progress plots.")
else:
    print("\n[INFO] No training data to plot.")


# ------------------------------
# 9. Visualize Example Original and Perturbed Images
# ------------------------------
if example_results:
    print("\n[INFO] Visualizing example attack results...")
    for idx, res in enumerate(example_results):
        orig_disp = unnormalize(res["original"])
        pert_rl_disp = unnormalize(res["perturbed_rl"])
        pert_baseline_disp = unnormalize(res["perturbed_baseline"])

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(orig_disp)
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(pert_rl_disp)
        plt.title(f"RLAgent: {'Success' if res['success_rl'] else 'Fail'} ({res['gens_rl']} gens)")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(pert_baseline_disp)
        plt.title(f"Baseline: {'Success' if res['success_baseline'] else 'Fail'} ({res['gens_baseline']} gens)")
        plt.axis("off")

        plt.suptitle(f"Example Attack {idx+1}")
        plt.tight_layout()
        plt.show()
else:
    print("[INFO] No example results to visualize.")

# ------------------------------
# 10. Save the Final Generator Model (if training was done)
# ------------------------------
if TRAINING_MODE and gen_optimizer is not None and total_attacks > 0:
    FINAL_MODEL_SAVE_PATH = "rl_adv/generator_model_final.pth"
    torch.save(generator.state_dict(), FINAL_MODEL_SAVE_PATH)
    print(f"\n[INFO] Final generator model saved to: {FINAL_MODEL_SAVE_PATH}")
elif not TRAINING_MODE:
    print("\n[INFO] Evaluation mode. Final generator model not saved from this run.")
else:
    print("\n[INFO] No training performed. Final generator model not saved.")


# Cleanup colorama
deinit()
print("\n[INFO] Script finished.")