import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# =====================================================
# PARAMETERS
# =====================================================

N_values = [300, 500, 800]
T = 12000
steady_window = 4000
num_runs = 60
connection_prob = 0.05
initial_rho = 0.8

alpha_values = [0.3, 0.5, 0.8]
p_values = np.linspace(0, 0.8, 30)

master_seed = 2026
master_seq = np.random.SeedSequence(master_seed)

os.makedirs("results", exist_ok=True)

# =====================================================
# THEORY
# =====================================================

def theoretical_rho(alpha, p):
    threshold = alpha / (1 + alpha)
    if p >= threshold:
        return 0
    return 1 - p / (alpha * (1 - p))

# =====================================================
# SINGLE RUN FUNCTION (PARALLEL SAFE)
# =====================================================

def single_run(task):

    N, alpha, p, seed_int = task 
    rng = np.random.default_rng(seed_int)

    A = rng.random((N, N)) < connection_prob
    np.fill_diagonal(A, 0)
    A = np.triu(A)
    A = A + A.T
    A = A.astype(float)

    states = (rng.random(N) < initial_rho).astype(float)
    degrees = A.sum(axis=1)
    rho_history = np.zeros(T)

    for t in range(T):

        neighbor_sum = A @ states

        rho_i = np.divide(
            neighbor_sum,
            degrees,
            out=np.zeros_like(neighbor_sum),
            where=degrees != 0
        )

        noise_mask = rng.random(N) < p
        rho_i[noise_mask] = 1 - rho_i[noise_mask]

        coop_mask = states == 1
        defect_mask = states == 0

        coop_flip = (rng.random(N) < p) & coop_mask
        defect_flip = (rng.random(N) < alpha * rho_i) & defect_mask

        states[coop_flip] = 0
        states[defect_flip] = 1

        rho_history[t] = states.mean()

    steady_mean = np.mean(rho_history[-steady_window:])
    return (N, alpha, p, steady_mean)



# MAIN EXECUTION BLOCK

if __name__ == "__main__":

    multiprocessing.freeze_support() 

    # =====================================================
    # BUILD TASK LIST
    # =====================================================

    tasks = []

    total_tasks = len(N_values) * len(alpha_values) * len(p_values) * num_runs
    child_seeds = master_seq.spawn(total_tasks)

    seed_idx = 0

    for N in N_values:
        for alpha in alpha_values:
            for p in p_values:
                for run in range(num_runs):
                    
                    seed_int = child_seeds[seed_idx].generate_state(1)[0]
                    tasks.append((N, alpha, p, seed_int))
                    seed_idx += 1

    # =====================================================
    # PARALLEL EXECUTION
    # =====================================================

    results_raw = []

    print("Starting parallel simulation...\n")

    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = [executor.submit(single_run, task) for task in tasks]

        for future in tqdm(as_completed(futures), total=len(futures)):
            results_raw.append(future.result())

    print("Simulation complete.\n")

    # =====================================================
    # SAVE RAW CSV
    # =====================================================

    df_raw = pd.DataFrame(results_raw,
                          columns=["N", "alpha", "p", "steady_rho"])

    df_raw.to_csv("results/raw_simulation_data.csv", index=False)

    print("Raw CSV saved.")

    # =====================================================
    # AGGREGATE STATISTICS
    # =====================================================

    grouped = df_raw.groupby(["N", "alpha", "p"])

    df_stats = grouped["steady_rho"].agg(
        mean="mean",
        std="std",
        var="var"
    ).reset_index()

    df_stats["susceptibility"] = df_stats["N"] * df_stats["var"]
    df_stats["theory"] = df_stats.apply(
        lambda row: theoretical_rho(row["alpha"], row["p"]), axis=1
    )

    df_stats.to_csv("results/aggregated_statistics.csv", index=False)

    print("Aggregated statistics saved.")

    # =====================================================
    # GENERATE FIGURES
    # =====================================================

    for N in N_values:
        for alpha in alpha_values:

            subset = df_stats[(df_stats["N"] == N) &
                              (df_stats["alpha"] == alpha)]

            # Phase Diagram
            plt.figure(figsize=(8,6))
            ci = 1.96 * subset["std"] / np.sqrt(num_runs)

            plt.errorbar(subset["p"], subset["mean"],
                         yerr=ci, capsize=3,
                         label="Empirical")

            plt.plot(subset["p"], subset["theory"],
                     linestyle='--', label="Theory")

            plt.xlabel("Noise Probability (p)")
            plt.ylabel("Steady-State Cooperation (ρ*)")
            plt.title(f"Phase Diagram (N={N}, α={alpha})")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"results/Phase_N{N}_alpha{alpha}.png", dpi=600)
            plt.close()

            # Variance
            plt.figure(figsize=(8,6))
            plt.plot(subset["p"], subset["var"])
            plt.xlabel("Noise Probability (p)")
            plt.ylabel("Variance")
            plt.title(f"Variance (N={N}, α={alpha})")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"results/Variance_N{N}_alpha{alpha}.png", dpi=600)
            plt.close()

            # Susceptibility
            plt.figure(figsize=(8,6))
            plt.plot(subset["p"], subset["susceptibility"])
            plt.xlabel("Noise Probability (p)")
            plt.ylabel("Susceptibility χ")
            plt.title(f"Susceptibility (N={N}, α={alpha})")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"results/Susceptibility_N{N}_alpha{alpha}.png", dpi=600)
            plt.close()

    print("All figures generated.")

