import os
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from experiments.train import train_hybrid_loss_pso_variant
from src.deepseed.benchmark_functions import BENCHMARK_FUNCTIONS

# --- Configuration ---
NUM_RUNS = 30           # <<< Number of times to repeat the entire experiment
SOLUTION_DIM = 10       # Dimension for the benchmark functions (e.g., 10, 30)
NOISE_DIM = 20          # Generator noise input dimension
TRAINING_ITERATIONS = 2000 # Total training steps per benchmark run (per repetition)
SWARM_SIZE = 50         # Number of particles in the PSO swarm
MAX_PSO_ITERATIONS = 10 # Number of PSO steps per training iteration
BATCH_SIZE = 5          # Generator update frequency
W_RANGE = (0.9, 0.4)    # PSO inertia weight range (start, end)
C1 = 1.5                # PSO cognitive coefficient
C2 = 1.5                # PSO social coefficient
LAMBDA_OBJECTIVE = 0.1  # Weight for initial objective value penalty (Generator loss)
LAMBDA_KL = 0        # Weight for KL divergence penalty (Generator loss)
REWARD_SCALE = 1.0      # Multiplier for the reward signal
BASE_OUTPUT_DIR_MULTI = "results" # Folder to save all run results
WINDOW_SIZE = 50        # Moving average window for smoothing plots
SAVE_PLOTS_PER_RUN = True # Set to False to disable saving plots for each run (saves time/space)
SAVE_DETAILS_PER_RUN = True # Set to False to disable saving detailed CSVs for each run


if __name__ == '__main__':
    multi_run_final_results = {}
    for name in BENCHMARK_FUNCTIONS:
        multi_run_final_results[name] = {
            "generator": [],
            # Other methods removed
        }

    start_time_total = time.time()
    os.makedirs(BASE_OUTPUT_DIR_MULTI, exist_ok=True)

    for run_idx in range(NUM_RUNS):
        run_start_time = time.time()
        print(f"<<<<<<<<<<<<<<<<<<<<<<<<<<< Starting Run {run_idx + 1}/{NUM_RUNS} >>>>>>>>>>>>>>>>>>>>>>>>>>>")

        run_output_dir = os.path.join(BASE_OUTPUT_DIR_MULTI, f"run_{run_idx+1:02d}")
        os.makedirs(run_output_dir, exist_ok=True)

        for name, (func, bounds) in BENCHMARK_FUNCTIONS.items():
            print(f"  -- Running Benchmark: {name} (Dim={SOLUTION_DIM}) for Run {run_idx + 1} --")

            func_output_dir_run = os.path.join(run_output_dir, name)
            os.makedirs(func_output_dir_run, exist_ok=True)

            common_args = {
                "benchmark_func": func,
                "benchmark_name": name,
                "bounds": bounds,
                "solution_dim": SOLUTION_DIM,
                "noise_dim": NOISE_DIM,
                "output_dir": func_output_dir_run,
                "training_iterations": TRAINING_ITERATIONS,
                "swarm_size": SWARM_SIZE,
                "max_pso_iterations": MAX_PSO_ITERATIONS,
                "batch_size": BATCH_SIZE,
                "w_range": W_RANGE, "c1": C1, "c2": C2,
                "lambda_objective": LAMBDA_OBJECTIVE,
                "lambda_kl": LAMBDA_KL,
                "reward_scale": REWARD_SCALE,
                "run_index": run_idx,
                "save_plots": SAVE_PLOTS_PER_RUN
            }

            # --- Run ONLY the Generator Initialization Method ---
            hist_gen, final_best_gen = train_hybrid_loss_pso_variant(initialization_type="generator", **common_args)
            # Calls to other methods (random_normal, latin_hypercube, opposition_based) are removed.

            multi_run_final_results[name]["generator"].append(final_best_gen)
            # Storing results for other methods removed.

            if SAVE_DETAILS_PER_RUN:
                print(f"    [Run {run_idx+1}/{NUM_RUNS}] Saving Detailed Results for {name} (Generator)")
                df_data_run = {
                    'Iteration': list(range(1, TRAINING_ITERATIONS + 1)),
                    'Gen_BestFitness': hist_gen['best_fitness'],
                    'Gen_Reward': hist_gen['reward'],
                    'Gen_PolicyLoss': hist_gen['policy_loss'],
                    'Gen_AvgInitialObjective': hist_gen['objective_loss'],
                    'Gen_KLLoss': hist_gen['kl_loss'],
                    'Gen_TotalLoss': hist_gen['total_loss'],
                    # Columns for other methods removed
                }
                max_len = TRAINING_ITERATIONS
                for key in df_data_run:
                    if key != 'Iteration':
                       current_len = len(df_data_run[key])
                       if current_len < max_len:
                           padding_value = df_data_run[key][-1] if current_len > 0 else 0
                           df_data_run[key].extend([padding_value] * (max_len - current_len))
                       elif current_len > max_len:
                           df_data_run[key] = df_data_run[key][:max_len]
                try:
                    df_run = pd.DataFrame(df_data_run)
                    csv_filename_run = os.path.join(func_output_dir_run, f"{name}_D{SOLUTION_DIM}_Training_Details_Generator.csv")
                    df_run.to_csv(csv_filename_run, index=False)
                    print(f"    Saved run details to: {csv_filename_run}")
                except Exception as e:
                    print(f"    Error saving run DataFrame to CSV for {name} (Run {run_idx+1}): {e}")

            if SAVE_PLOTS_PER_RUN:
                print(f"    [Run {run_idx+1}/{NUM_RUNS}] Plotting Training Metrics for {name} (Generator)")
                plt.figure(figsize=(18, 12))
                def moving_average(data, window_size):
                    if not isinstance(data, (list, np.ndarray)) or len(data) < window_size: return np.array([])
                    try: return np.convolve(np.array(data), np.ones(window_size)/window_size, mode='valid')
                    except ValueError: return np.array([])

                bf_gen = np.array(hist_gen['best_fitness'])
                rew_gen = np.array(hist_gen['reward'])
                loss_tot_gen = np.array(hist_gen['total_loss'])
                loss_pol_gen = np.array(hist_gen['policy_loss'])
                loss_kl_gen = np.array(hist_gen['kl_loss'])
                obj_init_gen = np.array(hist_gen['objective_loss'])
                x_ma = np.arange(WINDOW_SIZE - 1, TRAINING_ITERATIONS)

                # Plot 1: Best Fitness
                plt.subplot(2, 3, 1)
                plt.plot(bf_gen, label='Gen Best Fitness', color='blue', alpha=0.6)
                if x_ma.size > 0 and moving_average(bf_gen, WINDOW_SIZE).size > 0:
                    plt.plot(x_ma, moving_average(bf_gen, WINDOW_SIZE), color='navy', linewidth=1.5, label=f'MA ({WINDOW_SIZE})')
                plt.title("Best Fitness After PSO (Gen)"); plt.ylabel(f"Best Fitness (-{name})"); plt.grid(True, linestyle=':', alpha=0.6); plt.legend()

                # Plot 2: Reward
                plt.subplot(2, 3, 2)
                plt.plot(rew_gen, label='Gen Reward', color='green', alpha=0.6)
                if x_ma.size > 0 and moving_average(rew_gen, WINDOW_SIZE).size > 0:
                    plt.plot(x_ma, moving_average(rew_gen, WINDOW_SIZE), color='darkgreen', linewidth=1.5, label=f'MA ({WINDOW_SIZE})')
                plt.title("Reward per Iteration (Gen)"); plt.ylabel("Reward (Improvement)"); plt.grid(True, linestyle=':', alpha=0.6); plt.legend()

                # Plot 3: Total Loss (Gen Only)
                plt.subplot(2, 3, 3)
                plt.plot(loss_tot_gen, label='Total Loss (Gen)', color='red', alpha=0.7)
                if x_ma.size > 0 and moving_average(loss_tot_gen, WINDOW_SIZE).size > 0:
                    plt.plot(x_ma, moving_average(loss_tot_gen, WINDOW_SIZE), color='darkred', linewidth=1.5, label=f'MA ({WINDOW_SIZE})')
                plt.title("Total Gen Loss"); plt.ylabel("Loss"); plt.grid(True, linestyle=':', alpha=0.6); plt.legend()

                # Plot 4: Policy Loss (Gen Only)
                plt.subplot(2, 3, 4)
                plt.plot(loss_pol_gen, label='Policy Loss Term (Gen)', color='magenta', alpha=0.7)
                if x_ma.size > 0 and moving_average(loss_pol_gen, WINDOW_SIZE).size > 0:
                    plt.plot(x_ma, moving_average(loss_pol_gen, WINDOW_SIZE), color='darkmagenta', linewidth=1.5, label=f'MA ({WINDOW_SIZE})')
                plt.title("Policy Loss (-Rew*logP) Term (Gen)"); plt.ylabel("Loss Component"); plt.grid(True, linestyle=':', alpha=0.6); plt.legend()

                # Plot 5: Avg Initial Objective
                plt.subplot(2, 3, 5)
                plt.plot(obj_init_gen, label='Avg Initial Objective (Gen)', color='darkorange', alpha=0.6)
                if x_ma.size > 0 and moving_average(obj_init_gen, WINDOW_SIZE).size > 0:
                    plt.plot(x_ma, moving_average(obj_init_gen, WINDOW_SIZE), color='saddlebrown', linewidth=1.5, label=f'MA ({WINDOW_SIZE})')
                plt.axhline(known_optimum, color='black', linestyle=':', linewidth=0.8, label=f'Ideal Min ({known_optimum})')
                plt.title(f"Avg Initial Objective ({name}) (Gen)"); plt.ylabel(f"Avg Initial {name} Value"); plt.grid(True, linestyle=':', alpha=0.6); plt.legend()

                # Plot 6: KL Loss (Gen Only)
                plt.subplot(2, 3, 6)
                plt.plot(loss_kl_gen, label=f'KL Div (Std Only) Term (Gen)', color='cyan', alpha=0.7)
                if x_ma.size > 0 and moving_average(loss_kl_gen, WINDOW_SIZE).size > 0:
                    plt.plot(x_ma, moving_average(loss_kl_gen, WINDOW_SIZE), color='teal', linewidth=1.5, label=f'MA ({WINDOW_SIZE})')
                plt.title("KL Divergence (Std Only) Term (Gen)"); plt.ylabel("KL Divergence (Std Only)"); plt.grid(True, linestyle=':', alpha=0.6); plt.legend()

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plot_title = (f"Training Metrics for Generator on {name} (Run {run_idx+1}, Dim={SOLUTION_DIM}, PSO={MAX_PSO_ITERATIONS} iters)\n"
                              f"Gen Params: $\\lambda_{{Obj}}$={LAMBDA_OBJECTIVE}, $\\lambda_{{KL_{{Std}}}}$={LAMBDA_KL}")
                plt.suptitle(plot_title, y=0.99, fontsize=14)
                plot_filename_run = os.path.join(func_output_dir_run, f"{name}_D{SOLUTION_DIM}_Training_Metrics_Generator.png")
                try:
                    plt.savefig(plot_filename_run, bbox_inches='tight')
                    print(f"    Saved run metrics plot to: {plot_filename_run}")
                except Exception as e:
                    print(f"    Error saving run plot for {name} (Run {run_idx+1}): {e}")
                plt.close()

        run_end_time = time.time()
        print(f">>>>>>>>>>>>>>>>>>>>>> Run {run_idx + 1} finished in {run_end_time - run_start_time:.2f} seconds <<<<<<<<<<<<<<<<<<<<<<")


    # ------------------------------
    # 9. Calculate and Report Final Statistics (MODIFIED for Generator-Only)
    # ------------------------------
    print("\n=======================================================================================")
    print(f"Calculating Statistics Across {NUM_RUNS} Runs (Dim={SOLUTION_DIM}) for GENERATOR method")
    print("=======================================================================================")

    summary_stats = {}
    methods = ["generator"] # Only the generator method

    for bench_name in BENCHMARK_FUNCTIONS:
        summary_stats[bench_name] = {}
        print(f"--- Statistics for: {bench_name} ---")
        for method_name in methods: # This loop will only run for "generator"
            results_list = multi_run_final_results[bench_name][method_name]
            if len(results_list) == NUM_RUNS:
                mean_fitness = np.mean(results_list)
                std_fitness = np.std(results_list)
                summary_stats[bench_name][f"{method_name}_mean"] = mean_fitness
                summary_stats[bench_name][f"{method_name}_std"] = std_fitness
                print(f"  {method_name.capitalize():<18}: Mean Best Fitness = {mean_fitness:<15.6f} | Std Dev = {std_fitness:<15.6f}")
            else:
                print(f"  {method_name.capitalize():<18}: Error - Expected {NUM_RUNS} results, found {len(results_list)}")
                summary_stats[bench_name][f"{method_name}_mean"] = np.nan
                summary_stats[bench_name][f"{method_name}_std"] = np.nan

    summary_df_rows = []
    for bench_name, stats in summary_stats.items():
        row = {'Benchmark': bench_name}
        # Only for generator method
        row[f'Generator_Mean'] = stats.get(f"generator_mean", np.nan)
        row[f'Generator_StdDev'] = stats.get(f"generator_std", np.nan)
        summary_df_rows.append(row)

    summary_df = pd.DataFrame(summary_df_rows)
    cols_order = ['Benchmark', 'Generator_Mean', 'Generator_StdDev']
    summary_df = summary_df[cols_order]

    summary_csv_path = os.path.join(BASE_OUTPUT_DIR_MULTI, f"SUMMARY_D{SOLUTION_DIM}_{NUM_RUNS}runs_Stats_Generator.csv")
    try:
        summary_df.to_csv(summary_csv_path, index=False, float_format='%.6f')
        print(f"\nSaved summary statistics to: {summary_csv_path}")
    except Exception as e:
        print(f"\nError saving summary statistics CSV: {e}")

    print("\n==================================================================================================")
    print(f"Overall Summary Statistics for GENERATOR (Best Fitness: Mean ± StdDev over {NUM_RUNS} Runs, Dim={SOLUTION_DIM})")
    print("==================================================================================================")
    header = f"{ 'Benchmark':<15} | {'Generator Performance':<25}"
    print(header)
    print("-" * len(header))

    for bench_name, stats in summary_stats.items():
        mean = stats.get(f"generator_mean", np.nan)
        std = stats.get(f"generator_std", np.nan)
        stats_str = f"{mean:<12.4f} ± {std:<10.4f}"
        print(f"{bench_name:<15} | {stats_str}")

    print("==================================================================================================")

    end_time_total = time.time()
    print(f"\nTotal execution time for {NUM_RUNS} runs (Generator only): {(end_time_total - start_time_total) / 60:.2f} minutes")
    print(f"Results saved in base folder: {BASE_OUTPUT_DIR_MULTI}")
    print("Note: 'Best Fitness' is the maximum of (-Objective Function Value) found by PSO during a run.")
