import torch
import numpy as np
import matplotlib.pyplot as plt

def kl_divergence_std_only(mean, std):
    """Calculates KL divergence between learned dist N(mean, std) and N(0, 1), considering only std."""
    std = std + 1e-8 # Avoid log(0) or division by zero
    var = std.pow(2)
    log_var = torch.log(var)
    # KL formula component related to variance: 0.5 * (variance + mean^2 - 1 - log(variance))
    # Here, we assume target mean is 0 and ignore the learned mean part, focusing on std matching target std=1 (var=1)
    kl_div = 0.5 * torch.sum(var - 1.0 - log_var, dim=1)
    # Handle potential NaNs or Infs resulting from extreme values
    kl_div = torch.nan_to_num(kl_div, nan=0.0, posinf=1e6, neginf=-1e6) # Replace NaN/Inf
    return kl_div

def plot_distribution(samples, title, filename, bounds, solution_dim):
    """Plots a 2D projection or 1D histogram of the generated samples with enhanced colors."""
    if samples is None or samples.shape[0] == 0:
        print(f"Warning: No samples provided for plotting '{title}'. Skipping.")
        return

    samples_np = samples.cpu().numpy()
    plt.figure(figsize=(10, 10)) # Slightly larger figure
    ax = plt.gca()
    ax.set_facecolor('#e6e6fa') # Light lavender background

    if solution_dim >= 2:
        # Calculate the mean of the first two dimensions (the ones being plotted)
        mean_dim0 = np.mean(samples_np[:, 0])
        mean_dim1 = np.mean(samples_np[:, 1])
        mean_point_2d = np.array([mean_dim0, mean_dim1])

        # Calculate Euclidean distance of each point's first two dimensions from the 2D mean
        points_2d = samples_np[:, :2] # Shape (num_samples, 2)
        distances = np.sqrt(np.sum((points_2d - mean_point_2d)**2, axis=1)) # Shape (num_samples,)

        # Use distances for coloring. 'viridis_r' makes smaller distances (closer to mean) brighter (yellowish).
        scatter_plot = ax.scatter(samples_np[:, 0], samples_np[:, 1],
                                  c=distances, cmap='viridis_r',
                                  alpha=0.75, s=30, edgecolor='black', linewidth=0.2,
                                  label=f'Samples (N={samples.shape[0]})')
        plt.colorbar(scatter_plot, label='Distance to Mean')

        # Plot the mean point itself for reference
        ax.scatter(mean_dim0, mean_dim1, color='red', s=150, edgecolor='black', marker='.', label='Mean of Samples', zorder=5)

        plt.xlabel("Dimension 1", fontsize=12, color='darkslateblue')
        plt.ylabel("Dimension 2", fontsize=12, color='darkslateblue')
        if bounds:
            min_b, max_b = float(bounds[0]), float(bounds[1])
            plt.xlim(min_b, max_b)
            plt.ylim(min_b, max_b)
        plot_title_text = f"{title}\n(Showing first 2 dimensions)"
    elif solution_dim == 1:
        ax.hist(samples_np[:, 0], bins=50, alpha=0.75, color='mediumseagreen', edgecolor='darkgreen', label=f'N={samples.shape[0]}')
        plt.xlabel("Dimension 1", fontsize=12, color='darkslateblue')
        plt.ylabel("Frequency", fontsize=12, color='darkslateblue')
        if bounds:
            min_b, max_b = float(bounds[0]), float(bounds[1])
            plt.xlim(min_b, max_b)
        plot_title_text = f"{title}\n(Showing dimension 1)"
    else:
         plot_title_text = title + " (Invalid Dimension for Plotting)"

    plt.title(plot_title_text, fontsize=16, color='black')
    plt.grid(True, linestyle=':', alpha=0.5, color='grey')
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
    plt.legend(facecolor='whitesmoke', framealpha=0.8, fontsize=10)
    plt.savefig(filename)
    plt.close()
